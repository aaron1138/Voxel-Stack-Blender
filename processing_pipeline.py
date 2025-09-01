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
from scipy.sparse import lil_matrix, csr_matrix

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

    def _process_single_image_sparse_task(
        self,
        image_index: int,
        source_stack_binary: csr_matrix,
        source_stack_original: csr_matrix,
        classified_rois_list: list,
        img_shape: tuple,
        total_images: int,
        app_config: Config,
        xy_blend_pipeline_ops: List[XYBlendOperation],
        debug_save: bool,
        output_folder: str,
        base_filename: str
    ) -> np.ndarray:
        """Processes a single image using data from the sparse stacks."""
        # 1. Extract data from sparse stacks
        binary_image = source_stack_binary[image_index].toarray().reshape(img_shape)
        original_image = source_stack_original[image_index].toarray().reshape(img_shape)

        start_slice = max(0, image_index - app_config.receding_layers)
        end_slice = image_index
        prior_binary_masks_snapshot = [
            source_stack_binary[i].toarray().reshape(img_shape)
            for i in range(start_slice, end_slice)
        ]

        # 2. Prepare data for core processing (similar to the original task)
        debug_info = {'output_folder': output_folder, 'base_filename': base_filename} if debug_save else None

        if app_config.blending_mode in [ProcessingMode.WEIGHTED_STACK, ProcessingMode.ENHANCED_EDT]:
            prior_masks_for_blending = prior_binary_masks_snapshot
        else:
            prior_masks_for_blending = core.find_prior_combined_white_mask(prior_binary_masks_snapshot)

        classified_rois = classified_rois_list[image_index] if classified_rois_list else []

        # 3. Run core processing
        receding_gradient = core.process_z_blending(
            binary_image,
            prior_masks_for_blending,
            app_config,
            classified_rois,
            debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(original_image, receding_gradient)
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        return final_processed_image

    def run_sparse_mode(self, input_path, processing_output_path, image_filenames_filtered, total_images, get_numeric_part):
        self.status_update.emit("Starting Sparse Array Mode...")
        self.logger.log("Running in Sparse Array Mode.")

        # --- 1. Pre-flight and Initialization ---
        self.status_update.emit("Step 1/5: Initializing sparse arrays...")
        first_img_path = os.path.join(input_path, image_filenames_filtered[0])
        _, first_img = core.load_image(first_img_path)
        if first_img is None:
            raise ValueError(f"Failed to load the first image: {first_img_path}")
        img_shape = first_img.shape
        img_size_flat = img_shape[0] * img_shape[1]

        # Use LIL for efficient row-by-row creation, then convert to CSR for fast slicing
        source_stack_binary_lil = lil_matrix((total_images, img_size_flat), dtype=np.uint8)
        source_stack_original_lil = lil_matrix((total_images, img_size_flat), dtype=np.uint8)
        self.logger.log(f"Initialized sparse stacks for {total_images} images of shape {img_shape}.")

        # --- 2. Parallel Image Loading ---
        self.status_update.emit("Step 2/5: Loading all images into RAM...")

        def load_and_store_image(args):
            index, filename = args
            if not self._is_running: return None
            filepath = os.path.join(input_path, filename)
            binary_img, original_img = core.load_image(filepath)
            if binary_img is not None:
                source_stack_binary_lil[index] = binary_img.flatten()
                source_stack_original_lil[index] = original_img.flatten()
            return index

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(load_and_store_image, args) for args in enumerate(image_filenames_filtered)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if not self._is_running: break
                future.result() # Propagate exceptions
                progress = int(((i + 1) / total_images) * 100)
                self.status_update.emit(f"Loading images... ({i + 1}/{total_images})")
                self.progress_update.emit(progress)

        if not self._is_running:
            self.status_update.emit("Processing stopped during image loading.")
            return

        source_stack_binary = source_stack_binary_lil.tocsr()
        source_stack_original = source_stack_original_lil.tocsr()
        self.logger.log("Finished loading images and converted stacks to CSR format.")

        # --- 3. ROI Pre-computation (if needed) ---
        classified_rois_list = []
        if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
            self.status_update.emit("Step 3/5: Pre-calculating all ROIs...")
            tracker = ROITracker()
            # This has to run sequentially
            for i, filename in enumerate(image_filenames_filtered):
                if not self._is_running: break
                self.status_update.emit(f"Analyzing ROIs for {filename} ({i + 1}/{total_images})")
                binary_image = source_stack_binary[i].toarray().reshape(img_shape)
                layer_index = get_numeric_part(filename)
                rois = core.identify_rois(binary_image, self.app_config.roi_params.min_size)
                classified_rois = tracker.update_and_classify(rois, layer_index, self.app_config)
                classified_rois_list.append(classified_rois)
            self.logger.log("Finished pre-calculating ROIs.")

        if not self._is_running:
            self.status_update.emit("Processing stopped during ROI analysis.")
            return

        # --- 4. Parallel Image Processing ---
        self.status_update.emit("Step 4/5: Processing image stack...")
        working_stack_lil = lil_matrix((total_images, img_size_flat), dtype=np.uint8)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_image_sparse_task,
                    i,
                    source_stack_binary,
                    source_stack_original,
                    classified_rois_list,
                    img_shape,
                    total_images,
                    self.app_config,
                    self.app_config.xy_blend_pipeline,
                    self.app_config.debug_save,
                    processing_output_path,
                    os.path.splitext(os.path.basename(filename))[0]
                ): i
                for i, filename in enumerate(image_filenames_filtered)
            }

            processed_count = 0
            for future in concurrent.futures.as_completed(futures):
                if not self._is_running: break
                try:
                    image_index = futures[future]
                    result_image = future.result()
                    working_stack_lil[image_index] = result_image.flatten()
                    processed_count += 1
                    progress = int((processed_count / total_images) * 100)
                    self.status_update.emit(f"Processing images... ({processed_count}/{total_images})")
                    self.progress_update.emit(progress)
                except Exception as exc:
                    import traceback
                    self.error_occurred = True
                    error_detail = f"An image processing task failed: {exc}\n{traceback.format_exc()}"
                    self.logger.log(f"ERROR during sparse image processing task: {error_detail}")
                    self.error_signal.emit(error_detail)
                    self.stop_processing() # Stop other tasks

        if not self._is_running:
            self.status_update.emit("Processing stopped.")
            return

        # --- 5. Save Processed Images ---
        self.status_update.emit("Step 5/5: Saving processed images...")
        working_stack_csr = working_stack_lil.tocsr()
        for i, filename in enumerate(image_filenames_filtered):
            if not self._is_running: break
            output_filepath = os.path.join(processing_output_path, filename)
            processed_image = working_stack_csr[i].toarray().reshape(img_shape)
            cv2.imwrite(output_filepath, processed_image)
            progress = int(((i + 1) / total_images) * 100)
            self.status_update.emit(f"Saving... {filename} ({i + 1}/{total_images})")
            self.progress_update.emit(progress)

        self.logger.log("Sparse mode processing and saving complete.")

    def run(self):
        """
        The main processing loop. Dispatches to sparse or sequential mode.
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

            # <<< --- DISPATCH TO CORRECT MODE --- >>>
            if self.app_config.use_sparse_stack:
                self.run_sparse_mode(input_path, processing_output_path, image_filenames_filtered, total_images, get_numeric_part)
            else:
                self.run_sequential_mode(input_path, processing_output_path, image_filenames_filtered, total_images, get_numeric_part)

            # After processing, handle UVTools repacking if necessary
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

    def run_sequential_mode(self, input_path, processing_output_path, image_filenames_filtered, total_images, get_numeric_part):
        self.logger.log("Running in Sequential (Disk I/O) Mode.")
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
        self.logger.log("Sequential mode stack blending complete.")

    def stop_processing(self):
        self._is_running = False
