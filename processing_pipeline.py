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
import tiledb

from PySide6.QtCore import QThread, Signal

from config import Config, XYBlendOperation, ProcessingMode
from data_loader import FileDataLoader, TileDBDataLoader
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
        debug_save: bool,
        layer_index: int = None,
        tiledb_uri: str = None
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
            debug_info=debug_info,
            layer_index=layer_index,
            tiledb_uri=tiledb_uri
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

        try:
            # --- 1. Setup paths and folders ---
            input_path = ""
            processing_output_path = ""
            if self.app_config.input_mode == "uvtools":
                self.session_temp_folder = os.path.join(self.app_config.uvtools_temp_folder, f"{self.app_config.output_file_prefix}{self.run_timestamp}")
                os.makedirs(self.session_temp_folder, exist_ok=True)
                input_path = self._run_uvtools_extraction()
                processing_output_path = os.path.join(self.session_temp_folder, "Output")
                os.makedirs(processing_output_path, exist_ok=True)
            else: # folder mode
                input_path = self.app_config.input_folder
                processing_output_path = self.app_config.output_folder
                if self.app_config.use_tiledb:
                    self.session_temp_folder = os.path.join(self.app_config.output_folder, f"temp_{self.run_timestamp}")
                    os.makedirs(self.session_temp_folder, exist_ok=True)

            # --- 2. Filter image files ---
            numeric_pattern = re.compile(r'(\d+)\.\w+$')
            def get_numeric_part(filename):
                match = numeric_pattern.search(filename)
                return int(match.group(1)) if match else float('inf')

            all_image_filenames = sorted(
                [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff'))],
                key=get_numeric_part
            )
            image_filenames_filtered = [
                f for f in all_image_filenames
                if (self.app_config.start_index is None or get_numeric_part(f) >= self.app_config.start_index)
                and (self.app_config.stop_index is None or get_numeric_part(f) <= self.app_config.stop_index)
            ]

            total_images = len(image_filenames_filtered)
            if total_images == 0:
                self.error_signal.emit("No images found in the specified folder or index range.")
                return
            self.logger.log(f"Found {total_images} images to process.")

            # --- 3. Setup DataLoader ---
            if self.app_config.use_tiledb:
                data_loader = TileDBDataLoader(input_path, image_filenames_filtered, self.session_temp_folder)
            else:
                data_loader = FileDataLoader(input_path, image_filenames_filtered)

            data_loader.setup(self.status_update.emit, self.progress_update.emit)

            # --- 4. Main Processing Loop ---
            tracker = ROITracker()
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

                for i in range(total_images):
                    if not self._is_running:
                        self.logger.log("Processing stopped by user.")
                        break

                    if len(active_futures) >= max_active_futures:
                        done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        process_completed_futures(done)

                    binary_image, original_image, filename, layer_index = data_loader.get_image_data(i)

                    if binary_image is None:
                        self.status_update.emit(f"Skipping unloadable image: {filename}")
                        total_images = max(1, total_images - 1) # Adjust total for progress bar
                        continue

                    self.status_update.emit(f"Analyzing {filename} ({i + 1}/{total_images})")

                    classified_rois = []
                    if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
                        rois = core.identify_rois(binary_image, self.app_config.roi_params.min_size)
                        classified_rois = tracker.update_and_classify(rois, layer_index, self.app_config)

                    image_data_for_task = {
                        'filepath': os.path.join(input_path, filename), # Pass full path for debug name
                        'binary_image': binary_image,
                        'original_image': original_image, 'classified_rois': classified_rois
                    }

                    future = executor.submit(
                        self._process_single_image_task, image_data_for_task,
                        list(reversed(prior_binary_masks_cache)), self.app_config,
                        self.app_config.xy_blend_pipeline, processing_output_path,
                        self.app_config.debug_save,
                        layer_index=layer_index,
                        tiledb_uri=data_loader.get_tiledb_uri()
                    )
                    active_futures.add(future)
                    prior_binary_masks_cache.append(binary_image)

                if self._is_running and active_futures:
                    process_completed_futures(concurrent.futures.as_completed(active_futures))

            # --- 5. Finalization ---
            self.logger.log("Stack blending complete.")
            if self.app_config.input_mode == "uvtools" and not self.error_occurred and self._is_running:
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
            # Cleanup temp folder for both uvtools mode and folder mode with tiledb
            if self.session_temp_folder and os.path.isdir(self.session_temp_folder):
                if self.app_config.input_mode == "uvtools" and self.app_config.uvtools_delete_temp_on_completion and not self.error_occurred:
                    self.logger.log("Deleting temporary files for UVTools.")
                    shutil.rmtree(self.session_temp_folder)
                elif self.app_config.use_tiledb:
                    self.logger.log("Deleting temporary files for TileDB.")
                    shutil.rmtree(self.session_temp_folder)

            if not self.error_occurred and self._is_running:
                self.status_update.emit("Processing complete!")
                self.logger.log("Run completed successfully.")
            elif self.error_occurred:
                self.status_update.emit("Processing failed due to an error.")
                self.logger.log("Run failed due to an error.")
            else:
                self.status_update.emit("Processing stopped.")
                self.logger.log("Run was stopped by the user.")

            self.logger.log_total_time()
            self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False
