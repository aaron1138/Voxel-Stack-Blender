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
import sparse
import gc

from PySide6.QtCore import QThread, Signal

from config import Config, XYBlendOperation, ProcessingMode
import processing_core as core
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

    def _load_images_to_sparse_array(self, image_paths: List[str]):
        """
        Loads all images from the given paths into a 3D sparse array in parallel.
        Only stores the binary mask representation.
        """
        total_images = len(image_paths)
        self.status_update.emit(f"Loading {total_images} images into sparse array...")
        self.logger.log(f"Starting parallel load of {total_images} images into a sparse array.")

        coords = []
        data = []
        shape = None

        def load_single_for_sparse(filepath, index):
            binary_img, _ = core.load_image(filepath)
            if binary_img is None:
                return None, None, None
            # Find coordinates of non-zero pixels
            img_coords = np.argwhere(binary_img > 0)
            # Prepend the image index to the coordinates
            full_coords = np.hstack([np.full((img_coords.shape[0], 1), index), img_coords])
            return full_coords, binary_img.shape, binary_img.dtype

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(load_single_for_sparse, path, i): path for i, path in enumerate(image_paths)}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
                path = future_to_path[future]
                try:
                    img_coords, img_shape, img_dtype = future.result()
                    if img_coords is not None:
                        if shape is None: # First successful load determines the shape
                            shape = (total_images, img_shape[0], img_shape[1])
                            self.logger.log(f"Determined stack shape: {shape}")

                        coords.append(img_coords)
                        # All white pixels are 255, so we can just store that value
                        data.append(np.full(img_coords.shape[0], 255, dtype=np.uint8))

                    self.progress_update.emit(int(((i + 1) / total_images) * 100))
                except Exception as exc:
                    self.logger.log(f"ERROR: Generated an exception while loading {path}: {exc}")

        if not coords:
            self.logger.log("ERROR: No images could be loaded into the sparse array.")
            raise ValueError("Could not load any images to process.")

        self.status_update.emit("Concatenating sparse data...")
        all_coords = np.vstack(coords)
        all_data = np.concatenate(data)

        self.status_update.emit("Creating sparse array...")
        sparse_array = sparse.COO(all_coords.T, all_data, shape=shape)

        self.logger.log(f"Sparse array created with shape {sparse_array.shape} and {sparse_array.nnz} non-zero elements.")
        self.status_update.emit("Sparse array loaded successfully.")

        # Clean up memory
        del coords, data, all_coords, all_data
        gc.collect()

        return sparse_array


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

    def _pre_identify_rois_for_sparse(self, sparse_array: sparse.COO, image_filenames: List[str], get_numeric_part_func):
        """
        Iterates through the entire stack to identify and classify all ROIs before processing.
        This is necessary for the ROI_FADE mode to work correctly in the sparse path.
        """
        if self.app_config.blending_mode != ProcessingMode.ROI_FADE:
            # If not in ROI mode, return a list of empty dicts to avoid unnecessary work
            return [{} for _ in range(sparse_array.shape[0])]

        self.status_update.emit("Pre-processing stack for ROI detection...")
        self.logger.log("Starting ROI pre-processing for sparse mode.")

        tracker = ROITracker()
        all_layers_rois = []
        total_images = sparse_array.shape[0]

        # First pass: Identify all ROIs on all layers
        for i in range(total_images):
            if not self._is_running: return []
            self.status_update.emit(f"Identifying ROIs in layer {i+1}/{total_images}")
            binary_image = sparse_array[i].todense()
            rois = core.identify_rois(binary_image, self.app_config.roi_params.min_size)
            all_layers_rois.append(rois)
            self.progress_update.emit(int(((i + 1) / total_images) * 50)) # 0-50% for this pass

        # Second pass: Classify ROIs based on the full context
        classified_rois_by_layer = []
        for i in range(total_images):
            if not self._is_running: return []
            self.status_update.emit(f"Classifying ROIs in layer {i+1}/{total_images}")
            layer_index = get_numeric_part_func(image_filenames[i])
            classified_rois = tracker.update_and_classify(all_layers_rois[i], layer_index, self.app_config)
            classified_rois_by_layer.append(classified_rois)
            self.progress_update.emit(50 + int(((i + 1) / total_images) * 50)) # 50-100% for this pass

        self.logger.log("ROI pre-processing complete.")
        self.status_update.emit("ROI pre-processing complete.")
        return classified_rois_by_layer
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

    @staticmethod
    def _process_single_image_task(
        image_data: dict,
        prior_binary_masks_snapshot: collections.deque,
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
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        output_filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_filepath, final_processed_image)
        print(f"Successfully wrote output file: {output_filepath}") # Verification print
        return output_filepath

    @staticmethod
    def _process_single_image_sparse_task(
        image_data: dict,
        input_sparse_array: sparse.COO,
        output_sparse_array: sparse.DOK,
        app_config: Config,
        xy_blend_pipeline_ops: List[XYBlendOperation],
        debug_save: bool
    ):
        """Processes a single image slice from the sparse array."""
        image_index = image_data['index']
        filepath = image_data['filepath']
        original_image = image_data['original_image']
        classified_rois = image_data['classified_rois']

        # 1. Get current slice and prior slices from the main sparse array
        current_binary_image = input_sparse_array[image_index].todense()
        start_idx = max(0, image_index - app_config.receding_layers)
        prior_binary_masks_snapshot = [input_sparse_array[i].todense() for i in range(start_idx, image_index)]

        debug_info = {'output_folder': app_config.output_folder, 'base_filename': os.path.splitext(os.path.basename(filepath))[0]} if debug_save else None

        # 2. Prepare prior masks for blending function
        if app_config.blending_mode in [ProcessingMode.WEIGHTED_STACK, ProcessingMode.ENHANCED_EDT]:
            prior_masks_for_blending = list(reversed(prior_binary_masks_snapshot))
        else:
            prior_masks_for_blending = core.find_prior_combined_white_mask(prior_binary_masks_snapshot)

        # 3. Run core processing
        receding_gradient = core.process_z_blending(
            current_binary_image,
            prior_masks_for_blending,
            app_config,
            classified_rois,
            debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(original_image, receding_gradient)
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        # 5. Write result to the output sparse array
        # NOTE: This is inefficient as it makes the dense output sparse again.
        # A better implementation might have a different output strategy.
        output_sparse_array[image_index] = final_processed_image


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

            if self.app_config.load_as_sparse_array:
                # --- Sparse Array Processing Path ---
                full_image_paths = [os.path.join(input_path, f) for f in image_filenames_filtered]

                # 1. Load binary masks into a sparse array
                input_sparse_array = self._load_images_to_sparse_array(full_image_paths)

                # 2. Pre-process ROIs if needed
                classified_rois_by_layer = self._pre_identify_rois_for_sparse(input_sparse_array, image_filenames_filtered, get_numeric_part)

                # 3. Create a second sparse array for writing output
                output_sparse_array = sparse.DOK(input_sparse_array.shape, dtype=np.uint8)

                self.status_update.emit("Beginning processing on sparse array.")
                self.logger.log("Beginning processing on sparse array.")
                self.progress_update.emit(0) # Reset progress bar for main processing

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
                                error_detail = f"A sparse image processing task failed: {exc}\n{traceback.format_exc()}"
                                self.logger.log(f"ERROR during sparse processing task: {error_detail}")
                                self.error_signal.emit(error_detail)
                                self.stop_processing()
                            active_futures.remove(future)

                    # Pre-load original images to avoid repeated disk access in workers
                    original_images = [core.load_image(fp)[1] for fp in full_image_paths]

                    for i in range(total_images):
                        if not self._is_running:
                            self.logger.log("Processing stopped by user.")
                            self.status_update.emit("Processing stopped by user.")
                            break

                        if len(active_futures) >= max_active_futures:
                            done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                            process_completed_futures(done)

                        self.status_update.emit(f"Processing slice {i + 1}/{total_images}")

                        image_data_for_task = {
                            'index': i,
                            'filepath': full_image_paths[i],
                            'original_image': original_images[i],
                            'classified_rois': classified_rois_by_layer[i] if classified_rois_by_layer else []
                        }

                        future = executor.submit(
                            self._process_single_image_sparse_task,
                            image_data_for_task, input_sparse_array, output_sparse_array,
                            self.app_config, self.app_config.xy_blend_pipeline,
                            self.app_config.debug_save
                        )
                        active_futures.add(future)

                    if self._is_running and active_futures:
                        process_completed_futures(concurrent.futures.as_completed(active_futures))

                if self._is_running:
                    self._save_images_from_sparse_array(
                        output_sparse_array,
                        image_filenames_filtered,
                        processing_output_path
                    )

                # Clean up large objects
                del input_sparse_array, output_sparse_array
                gc.collect()

            else:
                # --- Original File-Based Processing Path ---
                prior_binary_masks_cache = collections.deque(maxlen=self.app_config.receding_layers)
                tracker = ROITracker()

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

    def _save_images_from_sparse_array(self, output_sparse_array: sparse.DOK, image_filenames: List[str], output_folder: str):
        """
        Saves all processed images from the output sparse array to disk in parallel.
        """
        total_images = len(image_filenames)
        self.status_update.emit(f"Saving {total_images} processed images...")
        self.logger.log(f"Starting parallel save of {total_images} images from sparse array.")

        def save_single_slice(image_index):
            if not self._is_running:
                return

            filename = image_filenames[image_index]
            output_filepath = os.path.join(output_folder, filename)

            # Convert slice to dense numpy array for saving
            image_to_save = output_sparse_array[image_index].todense()

            cv2.imwrite(output_filepath, image_to_save)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(save_single_slice, i) for i in range(total_images)]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    future.result()
                    self.progress_update.emit(int(((i + 1) / total_images) * 100))
                except Exception as exc:
                    self.error_occurred = True
                    error_detail = f"An image saving task failed: {exc}"
                    self.logger.log(f"ERROR during image saving task: {error_detail}")
                    self.error_signal.emit(error_detail)
                    self.stop_processing() # Stop if any save fails
                    break # Exit the loop

        if not self.error_occurred:
            self.logger.log("All processed images saved successfully.")
            self.status_update.emit("All processed images saved.")
