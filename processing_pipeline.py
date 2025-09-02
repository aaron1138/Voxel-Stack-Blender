import os
import re
import cv2
import concurrent.futures
import collections
import numpy as np
from typing import List, Tuple
import subprocess
import datetime
import shutil
import time

import sparse
from PySide6.QtCore import QThread, Signal

from config import Config, XYBlendOperation, ProcessingMode
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
    def _process_sparse_slice_task(
        z_index: int,
        source_binary_array: sparse.COO,
        source_original_array: sparse.COO,
        original_filename: str,
        app_config: Config,
        xy_blend_pipeline_ops: List[XYBlendOperation],
        output_folder: str,
        debug_save: bool,
        classified_rois: List[dict]
    ) -> Tuple[int, np.ndarray]:
        """
        Processes a single Z-slice from the sparse array and returns the processed data.
        File writing is deferred.
        """
        import processing_core as core
        import xy_blend_processor
        current_binary_slice = source_binary_array[:, :, z_index].todense()
        original_image_slice = source_original_array[:, :, z_index].todense()

        # The concept of prior masks changes. We now have the whole stack.
        start_idx = max(0, z_index - app_config.receding_layers)
        prior_slices_sparse = source_binary_array[:, :, start_idx:z_index]

        # Convert sparse slices to dense for processing
        prior_masks_for_blending = [prior_slices_sparse[:, :, i].todense() for i in range(prior_slices_sparse.shape[2])]

        debug_info = {'output_folder': output_folder, 'base_filename': os.path.splitext(original_filename)[0]} if debug_save else None

        receding_gradient = core.process_z_blending(
            current_binary_slice,
            prior_masks_for_blending,
            app_config,
            classified_rois,
            debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(original_image_slice, receding_gradient)
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        return z_index, final_processed_image


    def _load_single_image_for_sparse(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads a single image and returns coordinates and data for both binary and original
        grayscale sparse arrays. This is to avoid losing grayscale information.
        """
        try:
            binary_image, original_image = core.load_image(filepath)
            if binary_image is None:
                return None, None, None, None

            # Get coords and data for the binary mask (where pixels are > 0)
            binary_coords = np.argwhere(binary_image > 0)
            binary_data = binary_image[binary_coords[:, 0], binary_coords[:, 1]]

            # Get coords and data for the original grayscale image.
            # We can optimize by only storing non-zero pixels, same as the binary mask.
            original_coords = np.argwhere(original_image > 0)
            original_data = original_image[original_coords[:, 0], original_coords[:, 1]]

            return binary_coords, binary_data, original_coords, original_data
        except Exception as e:
            self.logger.log(f"ERROR loading image for sparse array: {filepath}, {e}")
            return None, None, None, None

    def _load_images_to_sparse_array(self, image_filenames: List[str], input_path: str) -> Tuple[sparse.COO, sparse.COO]:
        """Loads all images into two 3D sparse arrays (binary and original grayscale) in parallel."""
        total_images = len(image_filenames)
        self.status_update.emit(f"Loading {total_images} images into sparse arrays...")
        self.logger.log(f"Starting sparse array load for {total_images} images.")
        start_time = time.time()

        all_binary_coords, all_binary_data = [], []
        all_original_coords, all_original_data = [], []
        height, width = 0, 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_z = {
                executor.submit(self._load_single_image_for_sparse, os.path.join(input_path, filename)): z
                for z, filename in enumerate(image_filenames)
            }

            for i, future in enumerate(concurrent.futures.as_completed(future_to_z)):
                z_index = future_to_z[future]
                binary_coords, binary_data, original_coords, original_data = future.result()

                if binary_coords is not None and binary_data is not None:
                    if height == 0 or width == 0:
                        # Determine shape from the first loaded image
                        temp_img, _ = core.load_image(os.path.join(input_path, image_filenames[z_index]))
                        if temp_img is not None:
                            height, width = temp_img.shape
                            self.logger.log(f"Determined stack shape: ({height}, {width}, {total_images})")

                    # Process binary data
                    z_coords = np.full((binary_coords.shape[0], 1), z_index, dtype=binary_coords.dtype)
                    coords_3d = np.hstack([binary_coords, z_coords])
                    all_binary_coords.append(coords_3d)
                    all_binary_data.append(binary_data)

                    # Process original grayscale data
                    if original_coords is not None and original_data is not None:
                        z_coords_orig = np.full((original_coords.shape[0], 1), z_index, dtype=original_coords.dtype)
                        coords_3d_orig = np.hstack([original_coords, z_coords_orig])
                        all_original_coords.append(coords_3d_orig)
                        all_original_data.append(original_data)

                self.progress_update.emit(int(((i + 1) / total_images) * 100))

        if not all_binary_coords:
            raise ValueError("Could not load any images into the sparse array.")

        # Ensure shape is determined even if first image was empty
        if height == 0 or width == 0:
            for filename in image_filenames:
                 temp_img, _ = core.load_image(os.path.join(input_path, filename))
                 if temp_img is not None:
                    height, width = temp_img.shape
                    break
            if height == 0 or width == 0:
                 raise ValueError("Could not determine image dimensions from any file.")

        # Create Binary Array
        final_binary_coords = np.vstack(all_binary_coords).T
        final_binary_data = np.concatenate(all_binary_data)
        s_binary = sparse.COO(final_binary_coords, final_binary_data, shape=(height, width, total_images), fill_value=0)

        # Create Original Grayscale Array
        if all_original_coords:
            final_original_coords = np.vstack(all_original_coords).T
            final_original_data = np.concatenate(all_original_data)
            s_original = sparse.COO(final_original_coords, final_original_data, shape=(height, width, total_images), fill_value=0)
        else: # Handle case where all images are pure black
            s_original = sparse.COO(shape=(height, width, total_images), fill_value=0)

        end_time = time.time()
        load_duration = end_time - start_time
        density = s_binary.density * 100
        self.logger.log(f"Sparse arrays loaded in {load_duration:.2f}s. Binary density: {density:.4f}%.")
        self.status_update.emit("Image stack loaded into RAM.")
        self.progress_update.emit(100)
        return s_binary, s_original

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

            if self.app_config.load_as_sparse_array:
                # --- Sparse Array Processing Path ---
                source_binary_array, source_original_array = self._load_images_to_sparse_array(image_filenames_filtered, input_path)
                # Use DOK format for the working array as it's efficient for item assignment
                working_data_array = sparse.DOK(shape=source_binary_array.shape, fill_value=0)
                tracker = ROITracker()

                self.logger.log(f"Starting sparse array processing for {source_binary_array.shape[2]} slices.")

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # We can't submit all at once anymore because we need to do the ROI classification serially.
                    active_futures = set()

                    for z_index in range(source_binary_array.shape[2]):
                        if not self._is_running:
                            break

                        current_binary_slice = source_binary_array[:, :, z_index].todense()
                        classified_rois = []
                        if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
                            layer_index = get_numeric_part(image_filenames_filtered[z_index])
                            rois = core.identify_rois(current_binary_slice, self.app_config.roi_params.min_size)
                            classified_rois = tracker.update_and_classify(rois, layer_index, self.app_config)

                        future = executor.submit(
                            self._process_sparse_slice_task,
                            z_index,
                            source_binary_array,
                            source_original_array,
                            image_filenames_filtered[z_index],
                            self.app_config,
                            self.app_config.xy_blend_pipeline,
                            processing_output_path,
                            self.app_config.debug_save,
                            classified_rois
                        )
                        active_futures.add(future)

                    processed_count = 0
                    for future in concurrent.futures.as_completed(active_futures):
                        if not self._is_running:
                            self.logger.log("Processing stopped by user.")
                            self.status_update.emit("Processing stopped by user.")
                            break
                        try:
                            z_index, processed_slice = future.result()
                            # Write the dense result to the DOK sparse array.
                            # This is done in the main thread to ensure thread safety.
                            working_data_array[:, :, z_index] = processed_slice

                            processed_count += 1
                            self.status_update.emit(f"Completed processing slices ({processed_count}/{total_images})")
                            self.progress_update.emit(int((processed_count / total_images) * 100))
                        except Exception as exc:
                            import traceback
                            self.error_occurred = True
                            error_detail = f"A sparse slice processing task failed: {exc}\n{traceback.format_exc()}"
                            self.logger.log(f"ERROR during sparse slice processing: {error_detail}")
                            self.error_signal.emit(error_detail)
                            self.stop_processing()

                # After processing, save the results from the working array to files
                if not self.error_occurred and self._is_running:
                    self.status_update.emit("Saving processed slices to disk...")
                    self.logger.log("Converting working array to COO for fast iteration and saving slices.")
                    working_coo = working_data_array.asformat("coo")
                    # This part is inherently serial, but should be fast.
                    for z_index in range(working_coo.shape[2]):
                        filename = image_filenames_filtered[z_index]
                        output_filepath = os.path.join(processing_output_path, filename)
                        processed_slice = working_coo[:, :, z_index].todense()
                        cv2.imwrite(output_filepath, processed_slice)
                    self.logger.log("All processed slices saved.")


            else:
                # --- Original File-Based Processing Path ---
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

                        self.status_update.emit(f"Analyzing {filename} ({i + 1}/{total_images})")
                        filepath = os.path.join(input_path, filename)
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

    def stop_processing(self):
        self._is_running = False
