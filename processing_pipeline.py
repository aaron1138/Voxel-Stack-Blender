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

from PySide6.QtCore import QThread, Signal

from config import Config, XYBlendOperation, ProcessingMode
import processing_core as core
import xy_blend_processor
from roi_tracker import ROITracker
import tiledb
import uvtools_wrapper
from logger import Logger
import tiledb_utils
import smaa_engine
import lut_manager

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

            if self.app_config.use_tiledb_backend:
                self.status_update.emit("TileDB Backend Enabled: Ingesting images into array...")
                self.logger.log("Starting TileDB ingestion.")
                try:
                    tiledb_utils.ingest_images_to_tiledb(
                        self.app_config.tiledb_array_uri,
                        image_filenames_filtered,
                        input_path
                    )
                    self.status_update.emit("TileDB ingestion complete.")
                    self.logger.log("TileDB ingestion complete.")
                except Exception as e:
                    import traceback
                    error_info = f"Failed to create or write to TileDB array: {e}\n\n{traceback.format_exc()}"
                    self.logger.log(f"CRITICAL ERROR during TileDB ingestion: {error_info}")
                    self.error_signal.emit(error_info)
                    return

            if self.app_config.blending_mode == ProcessingMode.MORPHOLOGICAL_AA:
                self._run_smaa_pipeline(image_filenames_filtered, processing_output_path)
                return # SMAA pipeline is self-contained

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

                    self.status_update.emit(f"Processing {filename} ({i + 1}/{total_images})")
                    filepath = os.path.join(input_path, filename)

                    if self.app_config.use_tiledb_backend:
                        binary_image, original_image = tiledb_utils.read_xy_slice(self.app_config.tiledb_array_uri, i)
                    else:
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

    def _run_smaa_pipeline(self, image_filenames: List[str], output_folder: str):
        """
        Executes the separable morphological anti-aliasing (SMAA) pipeline.
        """
        if not self.app_config.use_tiledb_backend:
            self.error_signal.emit("Morphological AA requires the TileDB Backend to be enabled.")
            return

        self.status_update.emit("Starting Morphological AA pipeline...")
        self.logger.log("SMAA Pipeline Started.")

        # Get image dimensions from the source array
        try:
            with tiledb.open(self.app_config.tiledb_array_uri, 'r') as A:
                num_layers, height, width = A.shape
        except Exception as e:
            self.error_signal.emit(f"Could not open source TileDB array: {e}")
            return

        # Define URIs for temporary arrays
        temp_uri_xy = f"tiledb_temp_smaa_xy_{self.run_timestamp}"
        temp_uri_xz = f"tiledb_temp_smaa_xz_{self.run_timestamp}"
        temp_uri_yz = f"tiledb_temp_smaa_yz_{self.run_timestamp}"
        temp_uris = [temp_uri_xy, temp_uri_xz, temp_uri_yz]

        try:
            # --- Pass 1: XY Plane ---
            self.status_update.emit("SMAA: Processing XY planes (1/4)...")
            tiledb_utils.create_dense_array_for_slices(temp_uri_xy, height, width, num_layers)
            with tiledb.open(self.app_config.tiledb_array_uri, 'r') as A_in, tiledb.open(temp_uri_xy, 'w') as A_out:
                for i in range(num_layers):
                    if not self._is_running: break
                    self.progress_update.emit(int((i / num_layers) * 25))
                    img_slice = A_in[i, :, :]['pixel_value']
                    smaa_slice = smaa_engine.apply_smaa(img_slice)
                    A_out[i, :, :] = smaa_slice
            if not self._is_running: return

            # --- Pass 2: XZ Plane ---
            self.status_update.emit("SMAA: Processing XZ planes (2/4)...")
            tiledb_utils.create_dense_array_for_slices(temp_uri_xz, num_layers, width, height) # Y becomes the 'num_layers' dim
            z_lut = lut_manager.get_lut_from_params(self.app_config.z_correction_lut)
            with tiledb.open(self.app_config.tiledb_array_uri, 'r') as A_in, tiledb.open(temp_uri_xz, 'w') as A_out:
                for i in range(height): # Iterate over Y dimension
                    if not self._is_running: break
                    self.progress_update.emit(int(25 + (i / height) * 25))
                    img_slice = A_in[:, i, :]['pixel_value'] # This is an XZ slice
                    smaa_slice = smaa_engine.apply_smaa(img_slice)
                    A_out[i, :, :] = z_lut[smaa_slice]
            if not self._is_running: return

            # --- Pass 3: YZ Plane ---
            self.status_update.emit("SMAA: Processing YZ planes (3/4)...")
            # The shape is (width, num_layers, height) because we iterate over X
            # and each slice is a ZY plane.
            tiledb_utils.create_dense_array_for_slices(temp_uri_yz, num_layers, height, width)
            with tiledb.open(self.app_config.tiledb_array_uri, 'r') as A_in, tiledb.open(temp_uri_yz, 'w') as A_out:
                for i in range(width): # Iterate over X dimension
                    if not self._is_running: break
                    self.progress_update.emit(int(50 + (i / width) * 25))
                    img_slice = A_in[:, :, i]['pixel_value'] # This is a YZ slice, shape (num_layers, height)
                    smaa_slice = smaa_engine.apply_smaa(img_slice)
                    A_out[i, :, :] = z_lut[smaa_slice]
            if not self._is_running: return

            # --- Pass 4: Final Blending ---
            self.status_update.emit("SMAA: Blending results (4/4)...")
            with tiledb.open(temp_uri_xy, 'r') as A_xy, \
                 tiledb.open(temp_uri_xz, 'r') as A_xz, \
                 tiledb.open(temp_uri_yz, 'r') as A_yz:

                for i in range(num_layers):
                    if not self._is_running: break
                    self.progress_update.emit(int(75 + (i / num_layers) * 25))

                    # Read components for slice 'i'
                    xy_comp = A_xy[i, :, :]['pixel_value'] # Shape (height, width)

                    # Reconstruct from XZ and YZ passes
                    # A_xz is (y, z, x) -> A_xz[:, i, :] gives a slice over Y and X at Z=i -> shape (height, width)
                    xz_comp = A_xz[:, i, :]['pixel_value']

                    # A_yz is (x, z, y) -> A_yz[:, i, :] gives a slice over X and Y at Z=i -> shape (width, height)
                    yz_comp_raw = A_yz[:, i, :]['pixel_value']
                    yz_comp = yz_comp_raw.T # Transpose from (width, height) to (height, width)

                    # Blend by taking the minimum (darkest) value
                    final_image = np.minimum(xy_comp, xz_comp)
                    final_image = np.minimum(final_image, yz_comp)

                    # Apply XY pipeline post-processing
                    final_image = xy_blend_processor.process_xy_pipeline(final_image, self.app_config.xy_blend_pipeline)

                    # Save the final image
                    output_filename = os.path.basename(image_filenames[i])
                    output_filepath = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_filepath, final_image)

            self.progress_update.emit(100)

        finally:
            # --- Cleanup ---
            self.status_update.emit("SMAA: Cleaning up temporary arrays...")
            for uri in temp_uris:
                try:
                    if tiledb.object_type(uri) == "array":
                        tiledb.remove(uri)
                except Exception as e:
                    self.logger.log(f"Warning: Could not remove temp array {uri}: {e}")
            self.logger.log("SMAA Pipeline Finished.")


    def stop_processing(self):
        self._is_running = False
