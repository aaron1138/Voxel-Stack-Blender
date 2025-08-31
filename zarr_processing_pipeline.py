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
import zarr

from PySide6.QtCore import QThread, Signal

from config import Config, XYBlendOperation, ProcessingMode
import processing_core as core
import zarr_core
import zarr_xy_blend_processor
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

    def run(self):
        """
        The main processing loop for the Zarr pipeline.
        """
        self.logger.log("Zarr pipeline run started.")
        self.logger.log_config(self.app_config)
        self.status_update.emit("Zarr pipeline started...")

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

            # --- Zarr Initialization ---
            self.status_update.emit("Inspecting images for Zarr store initialization...")
            first_image_path = os.path.join(input_path, image_filenames_filtered[0])
            _, first_image = core.load_image(first_image_path)
            if first_image is None:
                raise ValueError(f"Could not load the first image: {first_image_path}")

            img_shape = first_image.shape
            stack_shape = (total_images, img_shape[0], img_shape[1])

            self.status_update.emit(f"Creating in-memory Zarr data store with shape {stack_shape}...")
            # Using zarr.zeros for in-memory storage by default
            input_zarr_stack = zarr.zeros(stack_shape, chunks=(1, img_shape[0], img_shape[1]), dtype=first_image.dtype)
            self.logger.log(f"Created in-memory Zarr array with shape {stack_shape}, chunks {input_zarr_stack.chunks}, dtype {input_zarr_stack.dtype}.")

            # --- Populate Zarr Store ---
            self.status_update.emit("Populating Zarr data store with input images...")
            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running:
                    self.logger.log("Processing stopped by user during Zarr population.")
                    break

                filepath = os.path.join(input_path, filename)
                _, image = core.load_image(filepath)
                if image is not None:
                    input_zarr_stack[i] = image
                else:
                    self.logger.log(f"Warning: Could not load image {filename}, filling with zeros.")
                    input_zarr_stack[i] = np.zeros(img_shape, dtype=first_image.dtype)

                progress = int(((i + 1) / total_images) * 100)
                self.progress_update.emit(progress)

            self.status_update.emit("Zarr data store population complete.")
            self.progress_update.emit(100)

            # --- Zarr Processing ---
            self.status_update.emit("Processing Z-blending on Zarr data store...")
            z_blended_stack = zarr_core.process_z_blending_zarr(input_zarr_stack, self.app_config)
            self.status_update.emit("Z-blending complete.")
            self.progress_update.emit(50)

            self.status_update.emit("Processing XY-blending on Zarr data store...")
            final_stack = zarr_xy_blend_processor.process_xy_pipeline_zarr(z_blended_stack, self.app_config)
            self.status_update.emit("XY-blending complete.")
            self.progress_update.emit(75)

            # --- Save Zarr stores if requested ---
            if self.app_config.save_zarr_to_disk:
                self.status_update.emit("Saving Zarr data stores to disk...")
                try:
                    input_save_path = os.path.join(processing_output_path, "input_stack.zarr")
                    output_save_path = os.path.join(processing_output_path, "final_stack.zarr")
                    if os.path.exists(input_save_path): shutil.rmtree(input_save_path)
                    if os.path.exists(output_save_path): shutil.rmtree(output_save_path)
                    zarr.save_array(input_save_path, input_zarr_stack)
                    zarr.save_array(output_save_path, final_stack)
                    self.logger.log(f"Saved input Zarr store to {input_save_path}")
                    self.logger.log(f"Saved final Zarr store to {output_save_path}")
                    self.status_update.emit("Zarr data stores saved.")
                except Exception as e:
                    self.logger.log(f"ERROR: Could not save Zarr stores. {e}")
                    self.status_update.emit(f"Error saving Zarr stores: {e}")

            # --- Finalization: Save slices to PNG ---
            self.status_update.emit("Saving final slices to PNG...")
            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running: break
                output_filepath = os.path.join(processing_output_path, filename)
                cv2.imwrite(output_filepath, final_stack[i, :, :])
                progress = 75 + int(((i + 1) / total_images) * 25)
                self.progress_update.emit(progress)

            self.logger.log("Finished saving final slices.")
            self.status_update.emit("Final slices saved.")

            if self.app_config.input_mode == "uvtools" and not self.error_occurred and self._is_running:
                self.logger.log("Starting UVTools repack.")
                self._run_uvtools_repack(processing_output_path)
                self.logger.log("UVTools repack completed.")

        except Exception as e:
            self.error_occurred = True
            import traceback
            error_info = f"A critical error occurred in the Zarr processing pipeline: {e}\n\n{traceback.format_exc()}"
            self.logger.log(f"CRITICAL ERROR in Zarr pipeline: {error_info}")
            self.error_signal.emit(error_info)
        finally:
            self.logger.log("Zarr run finalizing.")

            if not self.error_occurred and self._is_running:
                self.status_update.emit("Zarr processing complete!")
                self.logger.log("Zarr run completed successfully.")
            elif self.error_occurred:
                self.status_update.emit("Processing failed due to an error.")
                self.logger.log("Zarr run failed due to an error.")
            else:
                self.status_update.emit("Processing stopped.")
                self.logger.log("Zarr run was stopped by the user.")

            self.logger.log_total_time()
            self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False
