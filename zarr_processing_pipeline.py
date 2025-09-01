import os
import re
import cv2
import zarr
import collections
import numpy as np
from typing import List
import subprocess
import datetime
import shutil
import tempfile

from PySide6.QtCore import QThread, Signal

from config import Config, XYBlendOperation, ProcessingMode
import processing_core as core
import xy_blend_processor
from roi_tracker import ROITracker
import uvtools_wrapper
from logger import Logger

class ZarrProcessingPipelineThread(QThread):
    """
    Manages the image processing pipeline using a Zarr data store.
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

    def _get_session_temp_folder(self):
        if self.session_temp_folder and os.path.isdir(self.session_temp_folder):
            return self.session_temp_folder

        # If not in UVTools mode or if the folder wasn't created, use the config setting
        if self.app_config.uvtools_temp_folder and os.path.isdir(self.app_config.uvtools_temp_folder):
            # Create a unique subfolder for this run
            path = os.path.join(self.app_config.uvtools_temp_folder, f"{self.app_config.output_file_prefix}{self.run_timestamp}")
            os.makedirs(path, exist_ok=True)
            self.session_temp_folder = path
            return path

        # As a last resort, create a temp folder in the system's temp directory
        path = os.path.join(tempfile.gettempdir(), "VoxelStackBlender", f"{self.app_config.output_file_prefix}{self.run_timestamp}")
        os.makedirs(path, exist_ok=True)
        self.session_temp_folder = path
        return path

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

            # --- Zarr Initialization ---
            first_image_path = os.path.join(input_path, image_filenames_filtered[0])
            sample_img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
            height, width = sample_img.shape
            stack_shape = (total_images, height, width)

            self.status_update.emit("Creating Zarr data stores...")
            # Use in-memory store for both input and output by default
            input_store = zarr.storage.MemoryStore()
            output_store = zarr.storage.MemoryStore()

            # If saving to disk is enabled, create on-disk stores
            if self.app_config.save_zarr:
                session_folder = self._get_session_temp_folder()
                zarr_input_dir = os.path.join(session_folder, "zarr_input")
                zarr_output_dir = os.path.join(session_folder, "zarr_output")
                input_store = zarr.storage.LocalStore(zarr_input_dir)
                output_store = zarr.storage.LocalStore(zarr_output_dir)
                self.status_update.emit(f"Saving Zarr data to: {session_folder}")

            # Create the Zarr arrays
            z_input = zarr.zeros(stack_shape, chunks=(1, height, width), store=input_store, dtype='u1')
            z_output = zarr.zeros(stack_shape, chunks=(1, height, width), store=output_store, dtype='u1')

            # --- Load images into Zarr array ---
            self.status_update.emit("Loading images into Zarr data store...")
            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running: break
                filepath = os.path.join(input_path, filename)
                _, original_image = core.load_image(filepath)
                if original_image is not None:
                    z_input[i] = original_image
                self.progress_update.emit(int(((i + 1) / total_images) * 50)) # 0-50% for loading

            self.status_update.emit("Image loading complete. Starting processing...")

            # --- Main Processing Loop (Slice by Slice) ---
            prior_binary_masks_cache = collections.deque(maxlen=self.app_config.receding_layers)
            tracker = ROITracker()
            processed_count = 0

            for i in range(total_images):
                if not self._is_running:
                    self.logger.log("Processing stopped by user.")
                    self.status_update.emit("Processing stopped by user.")
                    break

                self.status_update.emit(f"Processing slice {i + 1}/{total_images}")

                # Get current slice data from Zarr
                original_image = z_input[i]
                _, current_binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)

                debug_info = {'output_folder': processing_output_path, 'base_filename': f"layer_{i}"} if self.app_config.debug_save else None

                # --- Z-Blending ---
                if self.app_config.blending_mode in [ProcessingMode.WEIGHTED_STACK, ProcessingMode.ENHANCED_EDT]:
                    prior_masks_for_blending = list(prior_binary_masks_cache)
                else:
                    prior_masks_for_blending = core.find_prior_combined_white_mask(list(prior_binary_masks_cache))

                classified_rois = []
                if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
                    rois = core.identify_rois(current_binary_image, self.app_config.roi_params.min_size)
                    classified_rois = tracker.update_and_classify(rois, i, self.app_config)

                receding_gradient = core.process_z_blending(
                    current_binary_image,
                    prior_masks_for_blending,
                    self.app_config,
                    classified_rois,
                    debug_info=debug_info
                )
                output_image_from_core = core.merge_to_output(original_image, receding_gradient)

                # --- XY-Blending ---
                final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, self.app_config.xy_blend_pipeline)

                # Store result in output Zarr array
                z_output[i] = final_processed_image

                # Update cache for next iteration
                prior_binary_masks_cache.append(current_binary_image)

                processed_count += 1
                self.progress_update.emit(50 + int((processed_count / total_images) * 50)) # 50-100% for processing

            # --- After processing, write output images from Zarr to disk for UVTools ---
            self.status_update.emit("Writing processed slices to disk for repackaging...")
            for i in range(total_images):
                if not self._is_running: break
                output_filename = os.path.basename(image_filenames_filtered[i])
                output_filepath = os.path.join(processing_output_path, output_filename)
                cv2.imwrite(output_filepath, z_output[i])

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
