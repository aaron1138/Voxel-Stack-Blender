import os
import re
import cv2
import concurrent.futures
import collections
import numpy as np
from typing import List
import subprocess
import datetime
import shutil
import tempfile
import tiledb

from PySide6.QtCore import QThread, Signal

from config import Config, XYBlendOperation, ProcessingMode
import processing_core as core
import processing_core_3d
import xy_blend_processor
from roi_tracker import ROITracker
import uvtools_wrapper
from logger import Logger

class ProcessingPipelineThread(QThread):
    """
    Manages the image processing pipeline in a separate thread to keep the GUI responsive.
    """
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, app_config: Config, max_workers: int):
        super().__init__()
        self.app_config = app_config
        self._is_running = True
        self.error_occurred = False
        self.max_workers = max_workers
        self.logger = Logger()
        self.run_timestamp = self.logger.run_timestamp # Sync timestamps
        self.session_temp_folder = ""
        self.tiledb_temp_dir = ""

    def _run_uvtools_extraction(self) -> str:
        self.status_update.emit("Starting UVTools slice extraction...")
        self.logger.log("Starting UVTools slice extraction.")
        self.session_temp_folder = os.path.join(self.app_config.uvtools_temp_folder, f"{self.app_config.output_file_prefix}{self.run_timestamp}")

        input_folder = uvtools_wrapper.extract_layers(
            self.app_config.uvtools_path,
            self.app_config.uvtools_input_file,
            self.session_temp_folder
        )
        self.status_update.emit("UVTools extraction completed.")
        return input_folder

    def _run_uvtools_repack(self, processed_images_folder: str):
        self.status_update.emit("Generating UVTools operation file...")
        uvtop_filepath = uvtools_wrapper.generate_uvtop_file(
            processed_images_folder,
            self.session_temp_folder,
            self.run_timestamp
        )
        self.status_update.emit("Operation file generated.")

        self.status_update.emit("Repacking slice file with processed layers...")
        final_output_path = uvtools_wrapper.repack_layers(
            self.app_config.uvtools_path,
            self.app_config.uvtools_input_file,
            uvtop_filepath,
            self.app_config.uvtools_output_location,
            self.app_config.uvtools_temp_folder, # FIX: Use the base working folder for output
            self.app_config.output_file_prefix,
            self.run_timestamp
        )
        self.status_update.emit(f"Successfully created: {os.path.basename(final_output_path)}")

    def _setup_tiledb_array(self, image_filenames: List[str], input_path: str, array_uri: str) -> (int, int, int):
        """Creates and populates a TileDB array with image slice data."""
        if not image_filenames:
            raise ValueError("No image files provided to create TileDB array.")

        # 1. Get dimensions from the first image
        first_image_path = os.path.join(input_path, image_filenames[0])
        img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read first image to get dimensions: {first_image_path}")
        height, width = img.shape
        depth = len(image_filenames)
        self.status_update.emit(f"Creating TileDB array with dimensions: {depth}x{height}x{width}")
        self.logger.log(f"Creating TileDB array at {array_uri} with dimensions: {depth}x{height}x{width}")

        # 2. Create TileDB schema and array
        dom = tiledb.Domain(
            tiledb.Dim(name="layer", domain=(0, depth - 1), tile=1, dtype=np.uint32),
            tiledb.Dim(name="y", domain=(0, height - 1), tile=height, dtype=np.uint32),
            tiledb.Dim(name="x", domain=(0, width - 1), tile=width, dtype=np.uint32),
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=[tiledb.Attr(name="pixel_value", dtype=np.uint8)],
            cell_order='row-major',
            tile_order='row-major',
        )
        tiledb.Array.create(array_uri, schema)

        # 3. Ingest data
        with tiledb.open(array_uri, 'w') as A:
            for i, filename in enumerate(image_filenames):
                self.status_update.emit(f"Ingesting slice {i + 1}/{depth} into TileDB array...")
                filepath = os.path.join(input_path, filename)
                img_data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img_data is None:
                    self.logger.log(f"Warning: Could not load image {filepath} during TileDB ingestion. Skipping.")
                    img_data = np.zeros((height, width), dtype=np.uint8) # Fill with black
                A[i, :, :] = img_data

        self.status_update.emit("TileDB array created and populated.")
        self.logger.log("TileDB array creation complete.")
        return depth, height, width

    @staticmethod
    def _process_single_image_task(
        image_data: dict,
        prior_binary_masks_snapshot: List[np.ndarray],
        app_config: Config,
        xy_blend_pipeline_ops: List[XYBlendOperation],
        output_folder: str,
        debug_save: bool
    ) -> str:
        """Processes a single image completely. This function runs in a worker thread."""
        current_binary_image = image_data['binary_image']
        original_image = image_data['original_image']
        filepath = image_data['filepath']

        debug_info = {'output_folder': output_folder, 'base_filename': os.path.splitext(os.path.basename(filepath))[0]} if debug_save else None

        # Prepare the prior mask data based on the blending mode
        if app_config.blending_mode in [ProcessingMode.WEIGHTED_STACK, ProcessingMode.ENHANCED_EDT]:
            # For these modes, pass the list of individual prior masks
            prior_masks_for_blending = list(prior_binary_masks_snapshot)
        else:
            # For other modes, pass the single combined mask
            prior_masks_for_blending = core.find_prior_combined_white_mask(list(prior_binary_masks_snapshot))

        receding_gradient = core.process_z_blending(
            current_binary_image,
            prior_masks_for_blending,
            app_config,
            image_data['classified_rois'],
            debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(original_image, receding_gradient)

        # --- NEW: Apply Orthogonal Blending if enabled ---
        if app_config.use_tiledb and app_config.tiledb_enable_orthogonal_blending:
            if 'tiledb_array_uri' in image_data and image_data['tiledb_array_uri']:
                output_image_from_core = processing_core_3d.blend_orthogonal(
                    tiledb_array_uri=image_data['tiledb_array_uri'],
                    image_shape=(image_data['depth'], image_data['height'], image_data['width']),
                    current_z=image_data['layer_index'],
                    z_blended_image=output_image_from_core,
                    config=app_config
                )

        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        output_filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_filepath, final_processed_image)
        print(f"Successfully wrote output file: {output_filepath}") # Verification print
        return output_filepath

    def run(self):
        """
        The main processing loop.
        """
        self.logger.log("Run started.")
        self.logger.log_config(self.app_config)
        self.status_update.emit("Processing started...")

        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')

        try:
            input_path = ""
            processing_output_path = ""

            if self.app_config.input_mode == "uvtools":
                input_path = self._run_uvtools_extraction()
                self.logger.log("UVTools extraction completed.")
                processing_output_path = os.path.join(self.session_temp_folder, "Output")
                os.makedirs(processing_output_path, exist_ok=True)
            else:
                input_path = self.app_config.input_folder
                processing_output_path = self.app_config.output_folder

            all_image_filenames = sorted(
                [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff'))],
                key=get_numeric_part
            )

            image_filenames_filtered = []
            for f in all_image_filenames:
                numeric_part = get_numeric_part(f)
                if self.app_config.start_index is not None and numeric_part < self.app_config.start_index:
                    continue
                if self.app_config.stop_index is not None and numeric_part > self.app_config.stop_index:
                    continue
                image_filenames_filtered.append(f)

            total_images = len(image_filenames_filtered)
            if total_images == 0:
                error_msg = "No images found in the specified folder or index range."
                self.logger.log(f"ERROR: {error_msg}")
                self.error_signal.emit(error_msg)
                return

            self.logger.log(f"Found {total_images} images to process.")
            tracker = ROITracker()

            if self.app_config.use_tiledb:
                # --- TileDB Processing Path ---
                self.tiledb_temp_dir = tempfile.mkdtemp(prefix="voxel-blend-tiledb-")
                tiledb_array_uri = os.path.join(self.tiledb_temp_dir, "slice_array")
                self.logger.log(f"Using TileDB backend. Temp array at: {tiledb_array_uri}")
                depth, height, width = self._setup_tiledb_array(image_filenames_filtered, input_path, tiledb_array_uri)

                with tiledb.open(tiledb_array_uri, 'r') as A:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        active_futures = set()
                        processed_count = 0
                        max_active_futures = self.max_workers * 2

                        def process_completed_futures(completed_futures):
                            nonlocal processed_count
                            for future in completed_futures:
                                try:
                                    future.result()
                                    processed_count += 1
                                    self.status_update.emit(f"Completed processing images ({processed_count}/{total_images})")
                                    self.progress_update.emit(int((processed_count / total_images) * 100))
                                except Exception as exc:
                                    import traceback
                                    self.error_occurred = True
                                    error_detail = f"An image processing task failed: {exc}\n{traceback.format_exc()}"
                                    self.logger.log(f"ERROR during image processing task: {error_detail}")
                                    self.error_signal.emit(error_detail)
                                    self.stop_processing()
                                active_futures.remove(future)

                        for i, filename in enumerate(image_filenames_filtered):
                            if not self._is_running:
                                self.logger.log("Processing stopped by user.")
                                self.status_update.emit("Processing stopped by user.")
                                break

                            if len(active_futures) >= max_active_futures:
                                done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                                process_completed_futures(done)

                            self.status_update.emit(f"Reading slice {i + 1}/{total_images} from TileDB")

                            start_slice = max(0, i - self.app_config.receding_layers)
                            end_slice = i + 1
                            slice_data = A[start_slice:end_slice]

                            original_image = slice_data['pixel_value'][-1]
                            prior_images_data = slice_data['pixel_value'][:-1]

                            _, binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
                            prior_binary_masks = [cv2.threshold(p, 127, 255, cv2.THRESH_BINARY)[1] for p in prior_images_data]

                            classified_rois = []
                            if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
                                layer_index = get_numeric_part(filename)
                                rois = core.identify_rois(binary_image, self.app_config.roi_params.min_size)
                                classified_rois = tracker.update_and_classify(rois, layer_index, self.app_config)

                            image_data_for_task = {
                                'filepath': os.path.join(input_path, filename),
                                'binary_image': binary_image,
                                'original_image': original_image,
                                'classified_rois': classified_rois,
                                'tiledb_array_uri': tiledb_array_uri,
                                'layer_index': i,
                                'depth': depth,
                                'height': height,
                                'width': width
                            }
                            future = executor.submit(
                                self._process_single_image_task, image_data_for_task,
                                list(reversed(prior_binary_masks)), self.app_config,
                                self.app_config.xy_blend_pipeline, processing_output_path,
                                self.app_config.debug_save
                            )
                            active_futures.add(future)

                        if self._is_running and active_futures:
                            process_completed_futures(concurrent.futures.as_completed(active_futures))
            else:
                # --- Original File-based Processing Path ---
                prior_binary_masks_cache = collections.deque(maxlen=self.app_config.receding_layers)
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    active_futures = set()
                    processed_count = 0
                    max_active_futures = self.max_workers * 2

                    def process_completed_futures(completed_futures):
                        nonlocal processed_count
                        for future in completed_futures:
                            try:
                                future.result()
                                processed_count += 1
                                self.status_update.emit(f"Completed processing images ({processed_count}/{total_images})")
                                self.progress_update.emit(int((processed_count / total_images) * 100))
                            except Exception as exc:
                                import traceback
                                self.error_occurred = True
                                error_detail = f"An image processing task failed: {exc}\n{traceback.format_exc()}"
                                self.logger.log(f"ERROR during image processing task: {error_detail}")
                                self.error_signal.emit(error_detail)
                                self.stop_processing()
                            active_futures.remove(future)

                    for i, filename in enumerate(image_filenames_filtered):
                        if not self._is_running:
                            self.logger.log("Processing stopped by user.")
                            self.status_update.emit("Processing stopped by user.")
                            break

                        if len(active_futures) >= max_active_futures:
                            done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                            process_completed_futures(done)

                        self.status_update.emit(f"Analyzing {filename} ({i + 1}/{total_images})")
                        filepath = os.path.join(input_path, filename)
                        binary_image, original_image = core.load_image(filepath)
                        if binary_image is None:
                            self.status_update.emit(f"Skipping unloadable image: {filename}")
                            total_images = max(1, total_images - 1)
                            continue

                        classified_rois = []
                        if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
                            layer_index = get_numeric_part(filename)
                            rois = core.identify_rois(binary_image, self.app_config.roi_params.min_size)
                            classified_rois = tracker.update_and_classify(rois, layer_index, self.app_config)

                        image_data_for_task = {
                            'filepath': filepath, 'binary_image': binary_image,
                            'original_image': original_image, 'classified_rois': classified_rois
                        }
                        future = executor.submit(
                            self._process_single_image_task, image_data_for_task,
                            list(reversed(prior_binary_masks_cache)), self.app_config,
                            self.app_config.xy_blend_pipeline, processing_output_path,
                            self.app_config.debug_save
                        )
                        active_futures.add(future)
                        prior_binary_masks_cache.append(binary_image)

                    if self._is_running and active_futures:
                        process_completed_futures(concurrent.futures.as_completed(active_futures))

            self.logger.log("Stack blending complete.")
            if self.app_config.input_mode == "uvtools" and not self.error_occurred:
                if self._is_running:
                    self.status_update.emit("All image processing tasks completed.")
                else:
                    self.status_update.emit("Processing stopped by user, repacking completed layers...")

                self.logger.log("Starting UVTools repack.")
                self._run_uvtools_repack(processing_output_path)
                self.logger.log("UVTools repack completed.")

        except Exception as e:
            self.error_occurred = True
            import traceback
            error_info = f"A critical error occurred in the processing pipeline: {e}\n\n{traceback.format_exc()}"
            self.logger.log(f"CRITICAL ERROR in pipeline: {error_info}")
            self.error_signal.emit(error_info)
        finally:
            self.logger.log("Run finalizing.")
            if self.tiledb_temp_dir and os.path.isdir(self.tiledb_temp_dir):
                self.status_update.emit(f"Deleting temporary TileDB directory...")
                self.logger.log(f"Deleting temporary TileDB directory: {self.tiledb_temp_dir}")
                try:
                    shutil.rmtree(self.tiledb_temp_dir)
                    self.status_update.emit("Temporary TileDB directory deleted.")
                    self.logger.log("Temporary TileDB directory deleted successfully.")
                except Exception as e:
                    self.logger.log(f"ERROR: Could not delete temp TileDB folder: {e}")
                    self.error_signal.emit(f"Could not delete temp TileDB folder: {e}")

            if self.app_config.input_mode == "uvtools" and self.app_config.uvtools_delete_temp_on_completion and not self.error_occurred:
                if self.session_temp_folder and os.path.isdir(self.session_temp_folder):
                    self.status_update.emit(f"Deleting temporary folder: {self.session_temp_folder}")
                    self.logger.log("Deleting temporary files.")
                    try:
                        shutil.rmtree(self.session_temp_folder)
                        self.status_update.emit("Temporary files deleted.")
                        self.logger.log("Temporary files deleted successfully.")
                    except Exception as e:
                        self.logger.log(f"ERROR: Could not delete temp folder: {e}")
                        self.error_signal.emit(f"Could not delete temp folder: {e}")

            if not self.error_occurred and self._is_running:
                self.status_update.emit("Processing complete!")
                self.logger.log("Run completed successfully.")
            elif self.error_occurred:
                self.status_update.emit("Processing failed due to an error. Temporary files have been preserved for inspection.")
                self.logger.log("Run failed due to an error.")
            else:
                self.status_update.emit("Processing stopped.")
                self.logger.log("Run was stopped by the user.")

            self.logger.log_total_time()
            self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False
