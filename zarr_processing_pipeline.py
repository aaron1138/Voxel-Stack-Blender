import os
import re
import cv2
import zarr
import numpy as np
import shutil
from PySide6.QtCore import QThread, Signal

from config import Config
from logger import Logger
import uvtools_wrapper
import processing_core as core
import xy_blend_processor

class ZarrProcessingPipelineThread(QThread):
    """
    Manages the Zarr-based image processing pipeline in a separate thread.
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
        self.run_timestamp = self.logger.run_timestamp
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
            self.app_config.uvtools_temp_folder,
            self.app_config.output_file_prefix,
            self.run_timestamp
        )
        self.status_update.emit(f"Successfully created: {os.path.basename(final_output_path)}")

    def run(self):
        self.logger.log("Zarr pipeline run started.")
        self.logger.log_config(self.app_config)
        self.status_update.emit("Zarr pipeline started...")

        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')

        try:
            input_path = ""
            if self.app_config.input_mode == "uvtools":
                input_path = self._run_uvtools_extraction()
                processing_output_path = os.path.join(self.session_temp_folder, "Output")
                os.makedirs(processing_output_path, exist_ok=True)
            else:
                input_path = self.app_config.input_folder
                processing_output_path = self.app_config.output_folder

            all_image_filenames = sorted(
                [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff'))],
                key=get_numeric_part
            )
            image_filenames_filtered = [f for f in all_image_filenames if (self.app_config.start_index is None or get_numeric_part(f) >= self.app_config.start_index) and (self.app_config.stop_index is None or get_numeric_part(f) <= self.app_config.stop_index)]

            total_images = len(image_filenames_filtered)
            if total_images == 0:
                raise ValueError("No images found in the specified folder or index range.")

            self.status_update.emit(f"Found {total_images} images. Initializing Zarr data store.")
            first_image_path = os.path.join(input_path, image_filenames_filtered[0])
            first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
            if first_image is None: raise IOError(f"Could not read the first image: {first_image_path}")
            height, width = first_image.shape

            zarr_input_stack = zarr.zeros((total_images, height, width), chunks=(1, height, width), dtype='uint8')
            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running: return
                filepath = os.path.join(input_path, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                zarr_input_stack[i, :, :] = img if img is not None else 0
                self.progress_update.emit(int(((i + 1) / total_images) * 33))
            self.status_update.emit("Zarr data ingestion complete.")

            zarr_z_blended_stack = zarr.zeros_like(zarr_input_stack)
            zarr_z_blended_stack[0, :, :] = zarr_input_stack[0, :, :]
            receding_layers = self.app_config.receding_layers
            for z in range(1, total_images):
                if not self._is_running: return
                original_image = zarr_input_stack[z]
                _, current_white_mask = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
                prior_masks = [cv2.threshold(zarr_input_stack[i], 127, 255, cv2.THRESH_BINARY)[1] for i in range(max(0, z - receding_layers), z)]
                prior_masks.reverse()
                gradient = core.process_z_blending(current_white_mask, prior_masks, self.app_config, [], None)
                zarr_z_blended_stack[z, :, :] = core.merge_to_output(original_image, gradient)
                self.progress_update.emit(33 + int((z + 1) / total_images * 33))
            self.status_update.emit("Z-blending complete.")

            zarr_final_stack = zarr.zeros_like(zarr_z_blended_stack)
            for z in range(total_images):
                if not self._is_running: return
                processed_slice = xy_blend_processor.process_xy_pipeline(zarr_z_blended_stack[z], self.app_config.xy_blend_pipeline)
                zarr_final_stack[z, :, :] = processed_slice
                self.progress_update.emit(66 + int((z + 1) / total_images * 34))
            self.status_update.emit("XY-blending complete.")

            if self.app_config.save_zarr_to_disk:
                self.status_update.emit("Saving Zarr data stores to disk...")
                debug_dir = os.path.join(processing_output_path, f"zarr_debug_{self.run_timestamp}")
                os.makedirs(debug_dir, exist_ok=True)
                zarr.save_group(os.path.join(debug_dir, 'zarr_stores.zip'), input=zarr_input_stack, z_blended=zarr_z_blended_stack, final=zarr_final_stack)
                self.logger.log(f"Saved Zarr stores to {debug_dir}")

            self.status_update.emit("Saving final PNG slices...")
            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running: return
                output_filepath = os.path.join(processing_output_path, filename)
                cv2.imwrite(output_filepath, zarr_final_stack[i])

            if self.app_config.input_mode == "uvtools":
                self._run_uvtools_repack(processing_output_path)

        except Exception as e:
            self.error_occurred = True
            import traceback
            error_info = f"A critical error occurred in the Zarr pipeline: {e}\\n\\n{traceback.format_exc()}"
            self.logger.log(f"CRITICAL ERROR in Zarr pipeline: {error_info}")
            self.error_signal.emit(error_info)
        finally:
            self.logger.log("Zarr pipeline run finalizing.")
            if self.app_config.input_mode == "uvtools" and self.app_config.uvtools_delete_temp_on_completion and not self.error_occurred:
                if self.session_temp_folder and os.path.isdir(self.session_temp_folder):
                    try:
                        shutil.rmtree(self.session_temp_folder)
                        self.status_update.emit("Temporary files deleted.")
                    except Exception as e:
                        self.error_signal.emit(f"Could not delete temp folder: {e}")

            status_msg = "Zarr processing complete!" if not self.error_occurred and self._is_running else "Zarr processing failed or stopped."
            self.status_update.emit(status_msg)
            self.logger.log_total_time()
            self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False
