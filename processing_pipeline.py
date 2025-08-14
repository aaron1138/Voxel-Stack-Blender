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

from config import Config, XYBlendOperation
import processing_core as core
import xy_blend_processor
from roi_tracker import ROITracker
import uvtools_wrapper

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
        self.max_workers = max_workers
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_temp_folder = ""

    def _run_uvtools_extraction(self) -> str:
        self.status_update.emit("Starting UVTools slice extraction...")
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
            self.session_temp_folder,
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
        prior_white_combined_mask = core.find_prior_combined_white_mask(list(prior_binary_masks_snapshot))

        receding_gradient = core.process_z_blending(
            current_binary_image,
            prior_white_combined_mask,
            app_config,
            image_data['classified_rois'],
            debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(original_image, receding_gradient)
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        output_filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_filepath, final_processed_image)
        return output_filepath

    def run(self):
        """
        The main processing loop.
        """
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
                processing_output_path = os.path.join(self.session_temp_folder, "Output")
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
                self.error_signal.emit("No images found in the specified folder or index range.")
                return

            prior_binary_masks_cache = collections.deque(maxlen=self.app_config.receding_layers)
            tracker = ROITracker()

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                active_futures = []

                for i, filename in enumerate(image_filenames_filtered):
                    if not self._is_running:
                        self.status_update.emit("Processing stopped by user.")
                        break

                    self.status_update.emit(f"Analyzing {filename} ({i + 1}/{total_images})")
                    filepath = os.path.join(input_path, filename)

                    binary_image, original_image = core.load_image(filepath)
                    if binary_image is None:
                        self.status_update.emit(f"Skipping unloadable image: {filename}")
                        continue

                    classified_rois = []
                    if self.app_config.blending_mode == 'roi_fade':
                        layer_index = get_numeric_part(filename)
                        rois = core.identify_rois(binary_image, self.app_config.roi_params.min_size)
                        classified_rois = tracker.update_and_classify(rois, layer_index, self.app_config)

                    image_data_for_task = {
                        'filepath': filepath,
                        'binary_image': binary_image,
                        'original_image': original_image,
                        'classified_rois': classified_rois
                    }

                    future = executor.submit(
                        self._process_single_image_task,
                        image_data_for_task,
                        collections.deque(prior_binary_masks_cache),
                        self.app_config,
                        self.app_config.xy_blend_pipeline,
                        processing_output_path,
                        self.app_config.debug_save
                    )
                    active_futures.append(future)

                    prior_binary_masks_cache.append(binary_image)

                processed_count = 0
                for future in concurrent.futures.as_completed(active_futures):
                    if not self._is_running: break
                    try:
                        future.result()
                        processed_count += 1
                        self.status_update.emit(f"Completed processing images ({processed_count}/{total_images})")
                        self.progress_update.emit(int((processed_count / total_images) * 100))
                    except Exception as exc:
                        import traceback
                        error_detail = f"An image processing task failed: {exc}\n{traceback.format_exc()}"
                        self.error_signal.emit(error_detail)
                        self._is_running = False
                        for f in active_futures:
                            f.cancel()
                        break

            self.status_update.emit("All image processing tasks completed.")

            if self.app_config.input_mode == "uvtools" and self._is_running:
                self._run_uvtools_repack(processing_output_path)

        except Exception as e:
            import traceback
            error_info = f"Error in processing thread: {e}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            if self.app_config.input_mode == "uvtools" and self.app_config.uvtools_delete_temp_on_completion:
                if self.session_temp_folder and os.path.isdir(self.session_temp_folder):
                    self.status_update.emit(f"Deleting temporary folder: {self.session_temp_folder}")
                    try:
                        shutil.rmtree(self.session_temp_folder)
                        self.status_update.emit("Temporary files deleted.")
                    except Exception as e:
                        self.error_signal.emit(f"Could not delete temp folder: {e}")

            if self._is_running:
                self.status_update.emit("Processing complete!")
            else:
                self.status_update.emit("Processing stopped by user or error.")
            self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False
