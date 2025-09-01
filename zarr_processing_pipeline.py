import os
import zarr
import numpy as np
import cv2
from config import Config

class ZarrProcessingPipeline:
    """
    Manages the Zarr-based image processing pipeline.
    This class is not a QThread, to make it easier to test.
    """

    def __init__(self, app_config: Config, max_workers: int, status_callback=None, progress_callback=None):
        self.app_config = app_config
        self.max_workers = max_workers
        self._is_running = True
        self.status_callback = status_callback
        self.progress_callback = progress_callback

    def _status_update(self, msg):
        if self.status_callback:
            self.status_callback(msg)

    def _progress_update(self, val):
        if self.progress_callback:
            self.progress_callback(val)

    def execute(self):
        """
        The main processing loop for the Zarr pipeline.
        """
        import re
        import processing_core as core

        self._status_update("Zarr processing started...")

        try:
            numeric_pattern = re.compile(r'(\d+)\.\w+$')
            def get_numeric_part(filename):
                match = numeric_pattern.search(filename)
                return int(match.group(1)) if match else float('inf')

            input_path = self.app_config.input_folder # For now, only folder mode is supported

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

            # Get image dimensions from the first image
            first_image_path = os.path.join(input_path, image_filenames_filtered[0])
            first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
            if first_image is None:
                raise ValueError(f"Could not load the first image: {first_image_path}")
            height, width = first_image.shape

            # Create an in-memory Zarr array
            self._status_update("Creating in-memory Zarr data store...")
            z_arr = zarr.zeros((total_images, height, width), chunks=(1, height, width), dtype='u1')

            # Load images into the Zarr array
            self._status_update("Loading images into Zarr data store...")
            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running:
                    self._status_update("Processing stopped by user.")
                    return

                self._progress_update(int((i / total_images) * 100))
                filepath = os.path.join(input_path, filename)
                _, binary_img = core.load_image(filepath)
                if binary_img is not None:
                    z_arr[i, :, :] = binary_img

            self._progress_update(100)
            self._status_update("Finished loading images into Zarr data store.")

            output_z_arr = self._perform_z_blending(z_arr)

            if self.app_config.save_zarr_to_disk:
                self._status_update("Saving Zarr data store to disk...")
                zarr_input_save_path = os.path.join(self.app_config.output_folder, "zarr_datastore", "input")
                zarr_output_save_path = os.path.join(self.app_config.output_folder, "zarr_datastore", "output")
                zarr.save(zarr_input_save_path, z_arr[:])
                zarr.save(zarr_output_save_path, output_z_arr[:])
                self._status_update(f"Zarr data store saved to {self.app_config.output_folder}/zarr_datastore")

            self._save_processed_slices(output_z_arr, image_filenames_filtered)

        except Exception as e:
            import traceback
            error_info = f"A critical error occurred in the Zarr processing pipeline: {e}\n\n{traceback.format_exc()}"
            raise RuntimeError(error_info) from e
        finally:
            self._status_update("Zarr processing finished.")

    def _perform_z_blending(self, z_arr):
        import processing_core as core
        from numba import njit, prange
        import concurrent.futures

        self._status_update("Performing Z-blending...")

        output_z_arr = zarr.zeros_like(z_arr)

        if self.app_config.blending_mode == "enhanced_edt":
            self._status_update("Using Enhanced EDT mode.")

            receding_layers = self.app_config.receding_layers

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for i in range(z_arr.shape[0]):
                    if not self._is_running:
                        break

                    current_white_mask = z_arr[i]

                    start = max(0, i - receding_layers)
                    prior_binary_masks = [z_arr[j] for j in range(start, i)]

                    future = executor.submit(self._process_slice_enhanced_edt, current_white_mask, prior_binary_masks, self.app_config)
                    futures.append(future)

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    if not self._is_running:
                        break

                    output_z_arr[i] = future.result()
                    self._progress_update(int((i / z_arr.shape[0]) * 100))

        self._progress_update(100)
        self._status_update("Z-blending finished.")
        return output_z_arr

    @staticmethod
    def _process_slice_enhanced_edt(current_white_mask, prior_binary_masks, config):
        import processing_core as core

        if not prior_binary_masks:
            return current_white_mask

        prior_white_combined_mask = core.find_prior_combined_white_mask(prior_binary_masks)
        if prior_white_combined_mask is None:
            return current_white_mask

        receding_white_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))
        if cv2.countNonZero(receding_white_areas) == 0:
            return current_white_mask

        distance_transform_src = cv2.bitwise_not(current_white_mask)

        # Anisotropic Correction
        if config.anisotropic_params.edt_enabled:
            ap = config.voxel_dimensions
            original_height, original_width = distance_transform_src.shape

            # We use the ratio of the voxel dimensions to get the scaling factors
            x_factor = ap.x / min(ap.x, ap.y, ap.z)
            y_factor = ap.y / min(ap.x, ap.y, ap.z)

            if x_factor != 1.0 or y_factor != 1.0:
                new_width = int(original_width * x_factor)
                new_height = int(original_height * y_factor)

                resized_src = cv2.resize(distance_transform_src, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                resized_dist_map = cv2.distanceTransform(resized_src, cv2.DIST_L2, 5)
                distance_map = cv2.resize(resized_dist_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            else:
                distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)
        else:
            distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

        receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)

        num_labels, labels = cv2.connectedComponents(receding_white_areas)
        if num_labels <= 1:
            return current_white_mask

        fade_distance_limit = config.fixed_fade_distance_receding

        if config.use_numba_jit:
            final_gradient_map = core._calculate_receding_gradient_field_enhanced_edt_numba(
                receding_distance_map, labels.astype(np.int32), num_labels, fade_distance_limit
            )
        else:
            final_gradient_map = core._calculate_receding_gradient_field_enhanced_edt_scipy(
                receding_distance_map, labels, num_labels, fade_distance_limit
            )

        final_gradient_map = cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=receding_white_areas)

        return np.maximum(current_white_mask, final_gradient_map)

    def _save_processed_slices(self, z_arr, original_filenames):
        self._status_update("Saving processed slices...")

        output_folder = self.app_config.output_folder
        os.makedirs(output_folder, exist_ok=True)

        for i, filename in enumerate(original_filenames):
            if not self._is_running:
                self._status_update("Processing stopped by user.")
                return

            self._progress_update(int((i / z_arr.shape[0]) * 100))

            output_filepath = os.path.join(output_folder, filename)
            cv2.imwrite(output_filepath, z_arr[i])

        self._progress_update(100)
        self._status_update("Finished saving processed slices.")

    def stop(self):
        self._is_running = False
