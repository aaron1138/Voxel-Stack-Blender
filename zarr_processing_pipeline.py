import os
import re
import cv2
import concurrent.futures
import collections
import numpy as np
import zarr
from typing import List
import subprocess
import datetime
import shutil

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

    @staticmethod
    def _process_single_zarr_slice_task(
        layer_index: int,
        z_input_binary: zarr.Array,
        z_input_original: zarr.Array,
        z_output: zarr.Array,
        tracker: ROITracker,
        app_config: Config,
        xy_blend_pipeline_ops: List[XYBlendOperation],
        debug_save: bool,
        debug_output_folder: str
    ):
        """Processes a single slice from the Zarr store. This function runs in a worker thread."""
        current_binary_image = z_input_binary[layer_index]
        original_image = z_input_original[layer_index]

        # --- Prepare prior masks ---
        start = max(0, layer_index - app_config.receding_layers)
        prior_binary_masks_snapshot = z_input_binary[start:layer_index]

        # Invert the order to match the original pipeline's expectation (closest layer first)
        prior_binary_masks_snapshot = prior_binary_masks_snapshot[::-1]

        debug_info = None
        if debug_save:
            # Create a unique base filename for debug images for this slice
            base_filename = f"zarr_slice_{layer_index:04d}"
            debug_info = {'output_folder': debug_output_folder, 'base_filename': base_filename}

        # --- ROI Classification (if applicable) ---
        classified_rois = []
        if app_config.blending_mode == ProcessingMode.ROI_FADE:
            rois = core.identify_rois(current_binary_image, app_config.roi_params.min_size)
            classified_rois = tracker.update_and_classify(rois, layer_index, app_config)

        # --- Z-Blending ---
        receding_gradient = core.process_z_blending(
            current_binary_image,
            list(prior_binary_masks_snapshot),
            app_config,
            classified_rois,
            debug_info=debug_info
        )

        # --- Merging and XY-Pipeline ---
        output_image_from_core = core.merge_to_output(original_image, receding_gradient)
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops, app_config)

        # --- Write result to output Zarr store ---
        z_output[layer_index] = final_processed_image


    def _load_and_prepare_image(self, filepath):
        """Load an image and return both its binary and original grayscale versions."""
        try:
            binary_image, original_image = core.load_image(filepath)
            if binary_image is None or original_image is None:
                return None, None
            return binary_image, original_image
        except Exception as e:
            self.logger.log(f"Error loading image {os.path.basename(filepath)}: {e}")
            return None, None

    def run(self):
        """
        The main processing loop using Zarr.
        """
        self.logger.log("Zarr pipeline run started.")
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
                raise ValueError("No images found in the specified folder or index range.")

            # --- Zarr Store Initialization ---
            self.status_update.emit("Inspecting image dimensions...")
            first_img_path = os.path.join(input_path, image_filenames_filtered[0])
            first_binary, first_original = self._load_and_prepare_image(first_img_path)
            if first_binary is None:
                raise ValueError(f"Could not load the first image: {image_filenames_filtered[0]}")

            height, width = first_binary.shape
            stack_depth = total_images
            self.logger.log(f"Detected dimensions: {width}x{height}, {stack_depth} layers.")

            # Create Zarr arrays in memory
            # Chunks are set to (1, height, width) for efficient single-slice (XY plane) access
            self.status_update.emit("Creating Zarr data stores in memory...")
            z_input_binary = zarr.zeros((stack_depth, height, width), chunks=(1, height, width), dtype='uint8')
            z_input_original = zarr.zeros((stack_depth, height, width), chunks=(1, height, width), dtype='uint8')
            z_output = zarr.zeros((stack_depth, height, width), chunks=(1, height, width), dtype='uint8')
            self.logger.log("Zarr stores created.")

            # --- Populate Zarr Store ---
            self.status_update.emit(f"Loading {total_images} slices into Zarr store...")
            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running: break
                filepath = os.path.join(input_path, filename)
                binary_img, original_img = self._load_and_prepare_image(filepath)
                if binary_img is not None:
                    z_input_binary[i] = binary_img
                    z_input_original[i] = original_img
                else:
                    self.logger.log(f"Warning: Skipping unloadable image {filename}. A blank slice will be used.")

                progress = int(((i + 1) / total_images) * 100)
                self.status_update.emit(f"Loading slices: {i+1}/{total_images}")
                self.progress_update.emit(progress)

            if not self._is_running:
                self.status_update.emit("Processing stopped by user during data loading.")
                self.logger.log("Processing stopped by user during data loading.")
                return

            self.status_update.emit("All slices loaded into memory store.")
            self.logger.log("Finished populating Zarr stores.")
            self.progress_update.emit(0) # Reset for next phase

            # --- Zarr Slice Processing ---
            self.status_update.emit("Starting Zarr slice processing...")
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
                            self.status_update.emit(f"Completed processing slices ({processed_count}/{stack_depth})")
                            self.progress_update.emit(int((processed_count / stack_depth) * 100))
                        except Exception as exc:
                            import traceback
                            self.error_occurred = True
                            error_detail = f"A slice processing task failed: {exc}\n{traceback.format_exc()}"
                            self.logger.log(f"ERROR during slice processing task: {error_detail}")
                            self.error_signal.emit(error_detail)
                            self.stop_processing()
                        active_futures.remove(future)

                for i in range(stack_depth):
                    if not self._is_running:
                        self.logger.log("Processing stopped by user.")
                        self.status_update.emit("Processing stopped by user.")
                        break

                    if len(active_futures) >= max_active_futures:
                        done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        process_completed_futures(done)

                    self.status_update.emit(f"Processing slice {i + 1}/{stack_depth}")

                    future = executor.submit(
                        self._process_single_zarr_slice_task,
                        layer_index=i,
                        z_input_binary=z_input_binary,
                        z_input_original=z_input_original,
                        z_output=z_output,
                        tracker=tracker,
                        app_config=self.app_config,
                        xy_blend_pipeline_ops=self.app_config.xy_blend_pipeline,
                        debug_save=self.app_config.debug_save,
                        debug_output_folder=processing_output_path
                    )
                    active_futures.add(future)

                if self._is_running and active_futures:
                    process_completed_futures(concurrent.futures.as_completed(active_futures))

            self.logger.log("Zarr stack blending complete.")

            # --- Save Zarr stores to disk if requested (for debugging) ---
            if self.app_config.save_zarr_to_disk and self._is_running:
                self.status_update.emit("Saving Zarr data stores to disk...")
                try:
                    input_zarr_path = os.path.join(self.session_temp_folder or self.app_config.output_folder, "zarr_input_store")
                    output_zarr_path = os.path.join(self.session_temp_folder or self.app_config.output_folder, "zarr_output_store")

                    # Save both input and output stores
                    zarr.save_group(input_zarr_path, input_binary=z_input_binary, input_original=z_input_original)
                    zarr.save(output_zarr_path, z_output)

                    self.logger.log(f"Saved input Zarr store to {input_zarr_path}")
                    self.logger.log(f"Saved output Zarr store to {output_zarr_path}")
                    self.status_update.emit("Zarr data stores saved.")
                except Exception as e:
                    error_msg = f"Failed to save Zarr stores: {e}"
                    self.logger.log(f"ERROR: {error_msg}")
                    self.error_signal.emit(error_msg)

            # --- Write processed slices to disk for UVTools ---
            if self._is_running or self.app_config.input_mode == "uvtools": # Always repack for UVTools
                self.status_update.emit("Writing processed slices to output folder...")
                for i in range(stack_depth):
                    if not self._is_running and self.app_config.input_mode != "uvtools": break

                    output_filename = os.path.basename(image_filenames_filtered[i])
                    output_filepath = os.path.join(processing_output_path, output_filename)
                    cv2.imwrite(output_filepath, z_output[i])
                self.logger.log(f"Finished writing {stack_depth} processed slices.")

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
