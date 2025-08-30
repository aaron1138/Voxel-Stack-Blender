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
import uvtools_wrapper
from logger import Logger
import zarr_engine

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
        layer_index: int,
        original_voxels,
        modified_voxels,
        app_config: Config,
        xy_blend_pipeline_ops: List[XYBlendOperation],
        output_folder: str, # for debug images
        debug_save: bool,
        base_filename: str, # for debug images
        classified_rois: list
    ) -> int:
        """Processes a single slice from Zarr data. This function runs in a worker thread."""
        original_image = original_voxels[layer_index]
        _, current_binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)

        # Get prior masks from Zarr
        look_down = app_config.receding_layers
        start_index = max(0, layer_index - look_down)

        prior_binary_masks_snapshot = []
        if layer_index > 0:
            # Slicing the Zarr array reads the data into memory
            prior_slices = original_voxels[start_index:layer_index]
            for s in prior_slices:
                _, binary_mask = cv2.threshold(s, 127, 255, cv2.THRESH_BINARY)
                prior_binary_masks_snapshot.append(binary_mask)
            # Reverse the list to match the old logic (closest layer first)
            prior_binary_masks_snapshot.reverse()

        debug_info = {'output_folder': output_folder, 'base_filename': base_filename} if debug_save else None

        # Prepare the prior mask data based on the blending mode
        if app_config.blending_mode in [ProcessingMode.WEIGHTED_STACK, ProcessingMode.ENHANCED_EDT]:
            prior_masks_for_blending = prior_binary_masks_snapshot
        else:
            prior_masks_for_blending = core.find_prior_combined_white_mask(prior_binary_masks_snapshot)

        receding_gradient = core.process_z_blending(
            current_binary_image,
            prior_masks_for_blending,
            app_config,
            classified_rois,
            debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(original_image, receding_gradient)
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        modified_voxels[layer_index] = final_processed_image
        return layer_index

    def run(self):
        """
        The main processing loop using Zarr as an intermediate store.
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
            zarr_storage_path = ""

            if self.app_config.input_mode == "uvtools":
                input_path = self._run_uvtools_extraction()
                self.logger.log("UVTools extraction completed.")
                # Define paths for temporary and final output
                self.session_temp_folder = os.path.dirname(input_path)
                processing_output_path = os.path.join(self.session_temp_folder, "Output_PNGs")
                zarr_storage_path = os.path.join(self.session_temp_folder, "zarr_data")
                os.makedirs(processing_output_path, exist_ok=True)
            else:
                input_path = self.app_config.input_folder
                processing_output_path = self.app_config.output_folder
                # Place Zarr data in a subfolder of the output directory
                zarr_storage_path = os.path.join(self.app_config.output_folder, "zarr_data_temp")

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

            # --- Zarr Ingestion Phase ---
            self.status_update.emit("Initializing Zarr storage...")
            self.logger.log("Initializing Zarr storage.")
            first_image_path = os.path.join(input_path, image_filenames_filtered[0])
            _, first_image = core.load_image(first_image_path)
            if first_image is None:
                raise ValueError(f"Could not read first image to determine dimensions: {first_image_path}")
            height, width = first_image.shape

            chunk_size = (1, height, width) # Slice-wise chunking

            original_voxels, modified_voxels = zarr_engine.initialize_storage(
                path=zarr_storage_path, layers=total_images, height=height, width=width, chunk_size=chunk_size
            )
            self.status_update.emit("Streaming image slices into Zarr...")
            self.logger.log("Streaming image slices into Zarr.")
            full_image_paths = [os.path.join(input_path, f) for f in image_filenames_filtered]
            zarr_engine.load_slices_to_zarr(full_image_paths, original_voxels, self.max_workers)
            self.status_update.emit("Finished streaming to Zarr.")
            self.logger.log("Finished streaming to Zarr.")
            # --- End Zarr Ingestion Phase ---

            self.logger.log(f"Processing {total_images} images from Zarr store.")
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
                        break

                    if len(active_futures) >= max_active_futures:
                        done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        process_completed_futures(done)

                    self.status_update.emit(f"Submitting slice {i + 1}/{total_images} for processing.")

                    classified_rois = []
                    if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
                        original_image = original_voxels[i]
                        _, binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
                        layer_index_num = get_numeric_part(filename)
                        rois = core.identify_rois(binary_image, self.app_config.roi_params.min_size)
                        classified_rois = tracker.update_and_classify(rois, layer_index_num, self.app_config)

                    base_filename = os.path.splitext(filename)[0]
                    future = executor.submit(
                        self._process_single_image_task, i,
                        original_voxels, modified_voxels, self.app_config,
                        self.app_config.xy_blend_pipeline, processing_output_path,
                        self.app_config.debug_save, base_filename, classified_rois
                    )
                    active_futures.add(future)

                if self._is_running and active_futures:
                    process_completed_futures(concurrent.futures.as_completed(active_futures))

            if self.error_occurred or not self._is_running:
                 raise InterruptedError("Processing stopped due to error or user request.")

            # --- Zarr Output Phase ---
            self.status_update.emit("Writing output images from Zarr storage...")
            self.logger.log("Writing output images from Zarr storage.")

            def save_slice(args):
                layer_index, zarr_array, output_folder, filename = args
                try:
                    image = zarr_array[layer_index]
                    output_filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(output_filepath, image)
                    return True
                except Exception as e:
                    self.logger.log(f"ERROR: Could not save slice {layer_index}: {e}")
                    return False

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                tasks = [(i, modified_voxels, processing_output_path, filename) for i, filename in enumerate(image_filenames_filtered)]
                results = list(executor.map(save_slice, tasks))
                successful_saves = sum(1 for r in results if r)
                self.logger.log(f"Successfully saved {successful_saves}/{total_images} images.")

            self.status_update.emit("Finished writing output images.")
            # --- End Zarr Output Phase ---

            self.logger.log("Stack blending complete.")
            if self.app_config.input_mode == "uvtools":
                self.logger.log("Starting UVTools repack.")
                self._run_uvtools_repack(processing_output_path)
                self.logger.log("UVTools repack completed.")

        except (Exception, InterruptedError) as e:
            if not isinstance(e, InterruptedError):
                self.error_occurred = True
                import traceback
                error_info = f"A critical error occurred: {e}\n\n{traceback.format_exc()}"
                self.logger.log(f"CRITICAL ERROR in pipeline: {error_info}")
                self.error_signal.emit(error_info)
        finally:
            self.logger.log("Run finalizing.")
            # Cleanup logic for Zarr data if not in debug mode and successful
            if os.path.exists(zarr_storage_path) and not self.app_config.debug_save:
                 try:
                    shutil.rmtree(zarr_storage_path)
                    self.logger.log(f"Cleaned up Zarr temp directory: {zarr_storage_path}")
                 except Exception as e:
                    self.logger.log(f"WARNING: Could not delete Zarr temp folder: {e}")

            if self.app_config.input_mode == "uvtools" and self.app_config.uvtools_delete_temp_on_completion and not self.error_occurred:
                if self.session_temp_folder and os.path.isdir(self.session_temp_folder):
                    self.status_update.emit(f"Deleting temporary folder: {self.session_temp_folder}")
                    # Note: Zarr folder is inside this, so it gets deleted too.
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
                self.status_update.emit("Processing failed. Temp files may be preserved.")
                self.logger.log("Run failed due to an error.")
            else: # Stopped by user
                self.status_update.emit("Processing stopped.")
                self.logger.log("Run was stopped by the user.")

            self.logger.log_total_time()
            self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False
