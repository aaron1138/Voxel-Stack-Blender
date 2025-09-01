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
import time
import sparse
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
        """Loads all images in parallel and constructs a sparse array for the stack."""
        self.status_update.emit(f"Starting parallel load of {len(image_paths)} images into sparse array...")
        start_time = time.time()

        coords = []
        data_binary = []
        data_original = []

        # Determine shape from the first image
        first_img_binary, first_img_original = core.load_image(image_paths[0])
        if first_img_binary is None:
            raise IOError(f"Could not load the first image: {image_paths[0]}")
        img_height, img_width = first_img_binary.shape
        stack_depth = len(image_paths)
        shape = (stack_depth, img_height, img_width)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(self._load_single_image_for_sparse, path): i for i, path in enumerate(image_paths)}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_idx)):
                idx = future_to_idx[future]
                try:
                    img_coords, img_data_binary, img_data_original = future.result()
                    if img_coords is not None:
                        # Add the z-index to the coordinates
                        z_coords = np.full((img_coords.shape[0], 1), idx, dtype=np.int32)
                        full_coords = np.hstack((z_coords, img_coords))
                        coords.append(full_coords)
                        data_binary.append(img_data_binary)
                        data_original.append(img_data_original)
                except Exception as exc:
                    self.logger.log(f"ERROR: Image at index {idx} failed to load for sparse array: {exc}")
                    # Continue without this image's data

                if i % 10 == 0 or i == len(image_paths) - 1:
                     self.status_update.emit(f"Loaded {i+1}/{len(image_paths)} images...")

        if not coords:
            return None, None, None

        all_coords = np.vstack(coords)
        all_data_binary = np.concatenate(data_binary)
        all_data_original = np.concatenate(data_original)

        self.status_update.emit("Constructing sparse arrays from loaded data...")
        sparse_binary_stack = sparse.COO(all_coords.T, all_data_binary, shape=shape, fill_value=0)
        sparse_original_stack = sparse.COO(all_coords.T, all_data_original, shape=shape, fill_value=0)

        end_time = time.time()
        self.logger.log(f"Sparse array loading complete in {end_time - start_time:.2f} seconds.")
        self.status_update.emit("Sparse arrays constructed.")

        return sparse_binary_stack, sparse_original_stack, shape

    @staticmethod
    def _load_single_image_for_sparse(filepath: str):
        """Worker function to load one image and return its sparse data."""
        binary_img, original_img = core.load_image(filepath)
        if binary_img is None:
            return None, None, None

        # Find coordinates of non-zero pixels (assuming binary is 0 or 255)
        coords = np.argwhere(binary_img > 0) # Returns (row, col) which is (y, x)

        # Get the data from both images at these coordinates
        data_binary = binary_img[coords[:, 0], coords[:, 1]]
        data_original = original_img[coords[:, 0], coords[:, 1]]

        return coords, data_binary, data_original

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
        """The main processing loop, now with a sparse array mode."""
        self.logger.log("Run started.")
        self.logger.log_config(self.app_config)
        self.status_update.emit("Processing started...")

        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')

        try:
            input_path, processing_output_path = self._setup_io_paths()

            all_image_filenames = sorted(
                [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff'))],
                key=get_numeric_part
            )

            image_filenames_filtered, image_paths_filtered = self._filter_images_by_index(all_image_filenames, input_path)

            total_images = len(image_filenames_filtered)
            if total_images == 0:
                raise ValueError("No images found in the specified folder or index range.")

            self.logger.log(f"Found {total_images} images to process.")

            if self.app_config.use_sparse_array:
                self._run_sparse_mode(image_paths_filtered, processing_output_path)
            else:
                self._run_sliding_window_mode(image_filenames_filtered, input_path, processing_output_path)

            self._finalize_processing(processing_output_path)

        except Exception as e:
            self.error_occurred = True
            import traceback
            error_info = f"A critical error occurred: {e}\n\n{traceback.format_exc()}"
            self.logger.log(f"CRITICAL ERROR: {error_info}")
            self.error_signal.emit(error_info)
        finally:
            self._cleanup_and_finish()

    def _setup_io_paths(self):
        """Determines and creates the necessary input and output directories."""
        if self.app_config.input_mode == "uvtools":
            input_path = self._run_uvtools_extraction()
            self.logger.log("UVTools extraction completed.")
            output_path = os.path.join(self.session_temp_folder, "Output")
            os.makedirs(output_path, exist_ok=True)
        else:
            input_path = self.app_config.input_folder
            output_path = self.app_config.output_folder
        return input_path, output_path

    def _filter_images_by_index(self, filenames, input_path):
        """Filters a list of filenames based on the start/stop index in the config."""
        filenames_filtered = []
        paths_filtered = []
        numeric_pattern = re.compile(r'(\d+)\.\w+$')

        start_idx = self.app_config.start_index if self.app_config.start_index is not None else 0
        stop_idx = self.app_config.stop_index if self.app_config.stop_index is not None else float('inf')

        for f in filenames:
            numeric_part_match = numeric_pattern.search(f)
            if not numeric_part_match: continue

            numeric_part = int(numeric_part_match.group(1))
            if start_idx <= numeric_part <= stop_idx:
                filenames_filtered.append(f)
                paths_filtered.append(os.path.join(input_path, f))

        return filenames_filtered, paths_filtered

    def _run_sparse_mode(self, image_paths: List[str], output_folder: str):
        """Main loop for processing using the sparse array mode."""
        binary_stack, original_stack, shape = self._load_images_to_sparse_array(image_paths)
        if binary_stack is None:
            raise RuntimeError("Failed to create sparse arrays from images.")

        # Create a sparse array for the output, which we'll populate.
        # Using dok format for efficient item assignment.
        output_stack = sparse.DOK(shape, dtype=np.uint8)

        total_images = shape[0]
        tracker = ROITracker() # ROI tracker is still needed for classification logic.

        # Since processing is now serial slice-by-slice (but on in-memory data),
        # we don't use the ThreadPoolExecutor for the main processing loop itself.
        for i in range(total_images):
            if not self._is_running:
                self.logger.log("Processing stopped by user.")
                break

            self.status_update.emit(f"Processing layer {i+1}/{total_images} from sparse stack...")

            current_binary_image = binary_stack[i, :, :].todense()

            # ROI classification logic (if needed)
            classified_rois = []
            if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
                # We need the numeric index, which corresponds to 'i' in this loop
                rois = core.identify_rois(current_binary_image, self.app_config.roi_params.min_size)
                classified_rois = tracker.update_and_classify(rois, i, self.app_config)

            # Get the prior masks from the sparse array
            start_slice = max(0, i - self.app_config.receding_layers)
            prior_binary_masks_sparse = binary_stack[start_slice:i, :, :]
            # Convert to a list of dense arrays for the core function
            prior_binary_masks = [img.todense() for img in prior_binary_masks_sparse]

            # The rest of the processing is the same as the single image task
            debug_info = {'output_folder': output_folder, 'base_filename': f"layer_{i}"} if self.app_config.debug_save else None

            receding_gradient = core.process_z_blending(
                current_binary_image,
                prior_binary_masks,
                self.app_config,
                classified_rois,
                debug_info=debug_info
            )

            original_image = original_stack[i, :, :].todense()
            output_image_from_core = core.merge_to_output(original_image, receding_gradient)
            final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, self.app_config.xy_blend_pipeline)

            # Write the final image to the output folder
            output_filename = os.path.basename(image_paths[i])
            output_filepath = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_filepath, final_processed_image)

            self.progress_update.emit(int(((i + 1) / total_images) * 100))

        self.status_update.emit("Sparse mode processing complete.")

    def _run_sliding_window_mode(self, image_filenames: List[str], input_path: str, output_folder: str):
        """Main loop for processing using the original file-based sliding window mode."""
        total_images = len(image_filenames)
        prior_binary_masks_cache = collections.deque(maxlen=self.app_config.receding_layers)
        tracker = ROITracker()
        numeric_pattern = re.compile(r'(\d+)\.\w+$')

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

            for i, filename in enumerate(image_filenames):
                if not self._is_running:
                    self.logger.log("Processing stopped by user.")
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
                    match = numeric_pattern.search(filename)
                    layer_index = int(match.group(1)) if match else i
                    rois = core.identify_rois(binary_image, self.app_config.roi_params.min_size)
                    classified_rois = tracker.update_and_classify(rois, layer_index, self.app_config)

                image_data_for_task = {
                    'filepath': filepath, 'binary_image': binary_image,
                    'original_image': original_image, 'classified_rois': classified_rois
                }
                future = executor.submit(
                    self._process_single_image_task, image_data_for_task,
                    list(reversed(prior_binary_masks_cache)), self.app_config,
                    self.app_config.xy_blend_pipeline, output_folder,
                    self.app_config.debug_save
                )
                active_futures.add(future)
                prior_binary_masks_cache.append(binary_image)

            if self._is_running and active_futures:
                process_completed_futures(concurrent.futures.as_completed(active_futures))

        self.status_update.emit("Sliding window mode processing complete.")

    def _finalize_processing(self, processed_images_folder: str):
        """Handles post-processing steps like UVTools repacking."""
        self.logger.log("Stack blending complete.")
        if self.app_config.input_mode == "uvtools" and not self.error_occurred:
            if self._is_running:
                self.status_update.emit("All image processing tasks completed.")
            else:
                self.status_update.emit("Processing stopped, repacking completed layers...")

            self.logger.log("Starting UVTools repack.")
            self._run_uvtools_repack(processed_images_folder)
            self.logger.log("UVTools repack completed.")

    def _cleanup_and_finish(self):
        """Handles final cleanup and emits the finished signal."""
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
            self.status_update.emit("Processing failed. Temp files preserved for inspection.")
            self.logger.log("Run failed due to an error.")
        else:
            self.status_update.emit("Processing stopped.")
            self.logger.log("Run was stopped by the user.")

        self.logger.log_total_time()
        self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False
