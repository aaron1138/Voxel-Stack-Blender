import os
import re
import cv2
import numpy as np
import zarr
import shutil
from PySide6.QtCore import QThread, Signal

from config import Config
from logger import Logger
import processing_core as core
import xy_blend_processor

class ZarrProcessingPipelineThread(QThread):
    """
    Manages the image processing pipeline using a Zarr data store.
    """
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, app_config: Config):
        super().__init__()
        self.app_config = app_config
        self._is_running = True
        self.error_occurred = False
        self.logger = Logger()
        self.run_timestamp = self.logger.run_timestamp

    def stop_processing(self):
        self._is_running = False

    def run(self):
        """
        The main processing loop for the Zarr pipeline.
        """
        self.logger.log("Zarr Pipeline run started.")
        self.logger.log_config(self.app_config)
        self.status_update.emit("Zarr processing started...")

        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')

        try:
            if self.app_config.input_mode != "folder":
                self.error_signal.emit("Zarr pipeline currently only supports 'Folder Input Mode'.")
                return

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
                self.error_signal.emit("No images found in the specified folder or index range.")
                return

            self.logger.log(f"Found {total_images} images to process.")
            self.status_update.emit(f"Found {total_images} images. Preparing Zarr data store...")

            # --- 1. Load all images into an in-memory Zarr array ---
            first_image_path = os.path.join(input_path, image_filenames_filtered[0])
            try:
                first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
                if first_image is None: raise IOError("First image could not be read.")
                height, width = first_image.shape
            except Exception as e:
                self.error_signal.emit(f"Failed to read first image to determine dimensions: {e}")
                return

            self.status_update.emit(f"Detected dimensions: {width}x{height}. Creating Zarr store...")
            input_zarr_array = zarr.zeros(
                (total_images, height, width),
                chunks=(1, None, None),  # Chunk by individual slice for now
                dtype='uint8',
                store=zarr.MemoryStore()
            )

            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running:
                    self.status_update.emit("Processing stopped by user.")
                    return

                self.status_update.emit(f"Loading slice {i + 1}/{total_images} into memory...")
                filepath = os.path.join(input_path, filename)
                img_slice = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                if img_slice.shape != (height, width):
                    self.error_signal.emit(f"Image '{filename}' has dimensions {img_slice.shape} which do not match the first image's dimensions {(height, width)}.")
                    return

                input_zarr_array[i, :, :] = img_slice
                self.progress_update.emit(int(((i + 1) / total_images) * 50))

            self.logger.log("Finished loading images into in-memory Zarr store.")
            self.status_update.emit("Image loading complete. Voxel store created in memory.")

            # --- 2. Perform 3D Z-Blending ---
            self.status_update.emit("Beginning 3D voxel processing...")
            self.logger.log("Starting 3D Z-blending.")

            # Convert the grayscale Zarr array to a binary numpy volume for processing
            self.status_update.emit("Binarizing voxel data...")
            input_volume_gray = input_zarr_array[:]
            _, input_volume_binary = cv2.threshold(input_volume_gray, 127, 255, cv2.THRESH_BINARY)
            input_volume_binary = (input_volume_binary / 255).astype(np.bool_) # Convert to boolean for scipy
            self.progress_update.emit(60)

            # This function doesn't exist yet, but we are wiring it up.
            # It will contain the new 3D logic.
            # We will need to import 'processing_core as core'
            gradient_volume = core.process_z_blending_3d(input_volume_binary, self.app_config)
            self.progress_update.emit(90)

            self.status_update.emit("Merging gradient with original data...")
            # Ensure gradient is in uint8 format for merging
            gradient_volume_uint8 = gradient_volume.astype(np.uint8)
            z_blended_volume = np.maximum(input_volume_gray, gradient_volume_uint8)
            self.logger.log("Finished 3D Z-blending.")
            self.progress_update.emit(95)

            # --- 3. Apply XY Blend Pipeline ---
            self.status_update.emit("Applying XY Blend Pipeline...")
            self.logger.log("Applying XY Blend Pipeline to volume.")

            processed_slices_list = []
            total_slices = z_blended_volume.shape[0]

            for i in range(total_slices):
                if not self._is_running:
                    self.status_update.emit("Processing stopped by user.")
                    return

                self.status_update.emit(f"Applying XY Blends to slice {i + 1}/{total_slices}")
                slice_to_process = z_blended_volume[i]
                processed_slice = xy_blend_processor.process_xy_pipeline(slice_to_process, self.app_config.xy_blend_pipeline, self.app_config)
                processed_slices_list.append(processed_slice)

            if not self._is_running:
                self.status_update.emit("Processing stopped by user.")
                return

            try:
                final_volume = np.stack(processed_slices_list, axis=0)
            except ValueError as e:
                self.error_signal.emit(f"Could not stack processed slices, likely due to a resize operation. Error: {e}")
                return

            # Create an output zarr array and store the result
            output_zarr_array = zarr.array(final_volume, chunks=(1, None, None), store=zarr.MemoryStore())
            self.logger.log("Finished XY Blending.")

            # --- 4. Save final processed slices to output folder ---
            self.status_update.emit("Saving processed slices to output folder...")
            self.logger.log("Saving final images.")
            for i in range(final_volume.shape[0]):
                output_filename = image_filenames_filtered[i]
                output_filepath = os.path.join(processing_output_path, output_filename)
                cv2.imwrite(output_filepath, final_volume[i])
            self.logger.log("Finished saving images.")
            self.status_update.emit("Voxel processing complete.")

            # --- 5. Save the Zarr data stores to disk if requested ---
            if self.app_config.save_zarr_to_disk:
                self.status_update.emit("Saving Zarr data stores to disk...")
                self.logger.log("Saving Zarr stores to disk for debugging.")

                zarr_output_dir = os.path.join(processing_output_path, f"zarr_debug_{self.run_timestamp}")

                # To be safe, remove any existing directory with the same name
                if os.path.exists(zarr_output_dir):
                    shutil.rmtree(zarr_output_dir)

                os.makedirs(zarr_output_dir, exist_ok=True)

                zarr.save_group(zarr_output_dir, input=input_zarr_array, output=output_zarr_array)

                self.status_update.emit(f"Zarr stores saved to: {zarr_output_dir}")
                self.logger.log(f"Zarr stores successfully saved to {zarr_output_dir}")

        except Exception as e:
            self.error_occurred = True
            import traceback
            error_info = f"A critical error occurred in the Zarr processing pipeline: {e}\n\n{traceback.format_exc()}"
            self.logger.log(f"CRITICAL ERROR in Zarr pipeline: {error_info}")
            self.error_signal.emit(error_info)
        finally:
            self.logger.log("Zarr pipeline run finalizing.")
            if not self.error_occurred and self._is_running:
                self.status_update.emit("Processing complete!")
                self.logger.log("Zarr run completed successfully.")
            elif self.error_occurred:
                self.status_update.emit("Processing failed due to an error.")
                self.logger.log("Zarr run failed due to an error.")
            else:
                self.status_update.emit("Processing stopped.")
                self.logger.log("Zarr run was stopped by the user.")

            self.logger.log_total_time()
            self.finished_signal.emit()
