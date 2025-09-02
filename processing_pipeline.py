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
from functools import partial

from PySide6.QtCore import QThread, Signal
import sparse

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

    def _load_images_to_sparse_array(self, filepaths: List[str], max_workers: int) -> Tuple[sparse.COO, sparse.COO]:
        """Loads all images in parallel into a sparse array."""
        self.status_update.emit("Loading images into sparse array...")
        if not filepaths:
            raise ValueError("No image filepaths provided to load.")

        # Load the first image to get dimensions
        first_binary, first_original = core.load_image(filepaths[0])
        if first_binary is None:
            raise ValueError(f"Could not load the first image: {filepaths[0]}")
        h, w = first_binary.shape
        dtype = first_original.dtype
        num_images = len(filepaths)

        self.logger.log(f"Stack dimensions: {num_images} layers, {h}x{w} pixels.")

        source_coords, source_data = [], []
        binary_coords, binary_data = [], []

        def _load_and_prep(filepath, index):
            binary_img, original_img = core.load_image(filepath)
            if binary_img is None: return None, None

            # For sparse arrays, we only store non-zero pixels
            source_nz = np.nonzero(original_img)
            s_coords = np.vstack([np.full_like(source_nz[0], index), source_nz[0], source_nz[1]])
            s_data = original_img[source_nz]

            binary_nz = np.nonzero(binary_img)
            b_coords = np.vstack([np.full_like(binary_nz[0], index), binary_nz[0], binary_nz[1]])
            b_data = binary_img[binary_nz]

            return (s_coords, s_data), (b_coords, b_data)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_load_and_prep, fp, i): i for i, fp in enumerate(filepaths)}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_idx)):
                s_res, b_res = future.result()
                if s_res is not None:
                    source_coords.append(s_res[0])
                    source_data.append(s_res[1])
                if b_res is not None:
                    binary_coords.append(b_res[0])
                    binary_data.append(b_res[1])

                progress = int(((i + 1) / num_images) * 100)
                self.progress_update.emit(progress)
                self.status_update.emit(f"Loaded {i+1}/{num_images} images into RAM...")

        source_sparse_array = sparse.COO(np.hstack(source_coords), np.hstack(source_data), shape=(num_images, h, w), fill_value=0)
        binary_sparse_array = sparse.COO(np.hstack(binary_coords), np.hstack(binary_data), shape=(num_images, h, w), fill_value=0)

        self.status_update.emit("Finished loading images to sparse array.")
        self.progress_update.emit(100)
        return source_sparse_array, binary_sparse_array

    def _run_sparse_mode(self, image_filepaths: List[str], processing_output_path: str):
        """Main logic for processing when using sparse arrays."""
        total_images = len(image_filepaths)
        source_array, binary_array = self._load_images_to_sparse_array(image_filepaths, self.max_workers)
        output_array = sparse.DOK(source_array.shape, dtype=source_array.dtype)

        tracker = ROITracker() # ROI tracker might still be useful

        start_idx = self.app_config.start_index if self.app_config.start_index is not None else 0
        stop_idx = self.app_config.stop_index if self.app_config.stop_index is not None else total_images - 1

        self.status_update.emit("Starting sparse processing...")
        for i in range(start_idx, stop_idx + 1):
            if not self._is_running:
                self.logger.log("Processing stopped by user during sparse mode.")
                break

            self.status_update.emit(f"Processing layer {i} from sparse stack...")

            current_binary_image = binary_array[i].todense()
            original_image = source_array[i].todense()

            # --- Prepare prior masks ---
            receding_layers = self.app_config.receding_layers
            prior_start = max(0, i - receding_layers)
            prior_masks_for_blending = [binary_array[j].todense() for j in range(prior_start, i)]

            # --- ROI Classification (optional, but maintained) ---
            classified_rois = []
            if self.app_config.blending_mode == ProcessingMode.ROI_FADE:
                rois = core.identify_rois(current_binary_image, self.app_config.roi_params.min_size)
                classified_rois = tracker.update_and_classify(rois, i, self.app_config)

            # --- Core Processing ---
            base_filename = os.path.splitext(os.path.basename(image_filepaths[i]))[0]
            final_image = self._process_single_image_logic(
                current_binary_image, original_image, prior_masks_for_blending,
                classified_rois, self.app_config, base_filename, processing_output_path
            )

            # Store result in the output sparse array
            output_array[i] = final_image

            processed_count = i - start_idx + 1
            total_to_process = stop_idx - start_idx + 1
            self.progress_update.emit(int((processed_count / total_to_process) * 100))

        # --- Save all processed images from the output array ---
        self.status_update.emit("Saving processed images from sparse stack...")
        for i in range(start_idx, stop_idx + 1):
            if not self._is_running:
                break
            output_filename = os.path.basename(image_filepaths[i])
            output_filepath = os.path.join(processing_output_path, output_filename)
            cv2.imwrite(output_filepath, output_array[i].todense())
        self.status_update.emit("Finished saving all images.")

    def _process_single_image_logic(
        self, current_binary_image, original_image, prior_masks,
        classified_rois, app_config, base_filename, output_folder
    ):
        """The core logic for processing one image, decoupled from file I/O."""
        debug_info = {'output_folder': output_folder, 'base_filename': base_filename} if app_config.debug_save else None

        if app_config.blending_mode not in [ProcessingMode.WEIGHTED_STACK, ProcessingMode.ENHANCED_EDT]:
            prior_masks = core.find_prior_combined_white_mask(prior_masks)

        receding_gradient = core.process_z_blending(
            current_binary_image, prior_masks, app_config, classified_rois, debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(original_image, receding_gradient)
        final_processed_image = xy_blend_processor.process_xy_pipeline(
            output_image_from_core, app_config.xy_blend_pipeline
        )
        return final_processed_image

    def _process_single_image_task(
        self, image_data: dict, prior_binary_masks_snapshot: list, app_config: Config,
        output_folder: str
    ) -> str:
        """Processes a single image completely. This function runs in a worker thread."""
        filepath = image_data['filepath']
        base_filename = os.path.splitext(os.path.basename(filepath))[0]

        final_processed_image = self._process_single_image_logic(
            image_data['binary_image'], image_data['original_image'],
            list(prior_binary_masks_snapshot), image_data['classified_rois'],
            app_config, base_filename, output_folder
        )

        output_filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_filepath, final_processed_image)
        print(f"Successfully wrote output file: {output_filepath}")
        return output_filepath

    def run(self):
        """The main processing loop."""
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

            image_filepaths_full = [os.path.join(input_path, f) for f in all_image_filenames]

            if not image_filepaths_full:
                error_msg = "No images found in the specified folder."
                self.logger.log(f"ERROR: {error_msg}")
                self.error_signal.emit(error_msg)
                return

            # --- Sparse Mode Execution Path ---
            if self.app_config.load_sparse:
                self.logger.log("Starting processing in sparse mode.")
                self._run_sparse_mode(image_filepaths_full, processing_output_path)
            # --- Sequential File-based Execution Path ---
            else:
                self.logger.log("Starting processing in sequential file-based mode.")
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
                    error_msg = "No images found in the specified index range."
                    self.logger.log(f"ERROR: {error_msg}")
                    self.error_signal.emit(error_msg)
                    return

                self.logger.log(f"Found {total_images} images to process.")
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
                            processing_output_path
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
