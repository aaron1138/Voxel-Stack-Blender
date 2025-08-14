"""
Copyright (c) 2025 Aaron Baca

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# ui_components.py (Final Touches)

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

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QTabWidget, QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QSettings
from PySide6.QtGui import QIntValidator, QDoubleValidator

from config import app_config as config, Config, XYBlendOperation, LutParameters, DEFAULT_NUM_WORKERS, upgrade_config
import processing_core as core
import xy_blend_processor
import lut_manager
from pyside_xy_blend_tab import XYBlendTab

class ImageProcessorThread(QThread):
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
        """
        Executes UVToolsCmd.exe to extract layers into a timestamped temp folder.
        """
        self.status_update.emit("Starting UVTools slice extraction...")
        
        self.session_temp_folder = os.path.join(self.app_config.uvtools_temp_folder, f"Voxel_Stack_Blender_{self.run_timestamp}")
        input_folder = os.path.join(self.session_temp_folder, "Input")
        output_folder = os.path.join(self.session_temp_folder, "Output")
        
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        command = [
            self.app_config.uvtools_path, "extract", self.app_config.uvtools_input_file,
            input_folder, "--content", "Layers"
        ]
        self.status_update.emit(f"Running command: {' '.join(command)}")
        
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.run(command, capture_output=True, text=True, creationflags=creation_flags)
            print(f"UVToolsCmd.exe (extract) finished with exit code: {process.returncode}")
            if process.returncode not in [0, 1]:
                raise RuntimeError(f"UVTools exited with an error (code {process.returncode}):\n\n{process.stderr}")
            self.status_update.emit("UVTools extraction completed.")
            return input_folder
        except Exception as e:
            raise RuntimeError(f"UVTools extraction failed: {e}")

    def _generate_uvtop_file(self, processed_images_folder: str) -> str:
        """Generates the .uvtop XML file for repacking."""
        self.status_update.emit("Generating UVTools operation file...")
        
        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')
            
        processed_files = sorted(
            [os.path.join(processed_images_folder, f) for f in os.listdir(processed_images_folder) if f.lower().endswith('.png')],
            key=get_numeric_part
        )

        if not processed_files:
            raise RuntimeError("No processed image files found to generate .uvtop file.")

        xml_content = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
        xml_content += '<OperationLayerImport xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n'
        xml_content += '  <LayerRangeSelection>None</LayerRangeSelection>\n'
        xml_content += '  <ImportType>Replace</ImportType>\n'
        xml_content += '  <Files>\n'
        for f_path in processed_files:
            xml_content += '    <GenericFileRepresentation>\n'
            xml_content += f'      <FilePath>{f_path}</FilePath>\n'
            xml_content += '    </GenericFileRepresentation>\n'
        xml_content += '  </Files>\n'
        xml_content += '</OperationLayerImport>\n'

        uvtop_filename = f"repack_operations_{self.run_timestamp}.uvtop"
        uvtop_filepath = os.path.join(self.session_temp_folder, uvtop_filename)
        
        with open(uvtop_filepath, 'w', encoding='utf-8') as f:
            f.write(xml_content)
            
        self.status_update.emit("Operation file generated.")
        return uvtop_filepath

    def _run_uvtools_repack(self, uvtop_filepath: str):
        """Executes UVToolsCmd.exe to repack the processed layers."""
        self.status_update.emit("Repacking slice file with processed layers...")

        original_filename = os.path.basename(self.app_config.uvtools_input_file)
        output_filename = f"Voxel_Blend_Processed_{self.run_timestamp}_{original_filename}"
        
        # NEW: Determine final output directory based on config
        output_directory = ""
        if self.app_config.uvtools_output_location == "input_folder":
            output_directory = os.path.dirname(self.app_config.uvtools_input_file)
        else: # Default to working_folder
            output_directory = self.app_config.uvtools_temp_folder

        final_output_path = os.path.join(output_directory, output_filename)

        command = [
            self.app_config.uvtools_path,
            "run",
            self.app_config.uvtools_input_file,
            uvtop_filepath,
            "--output",
            final_output_path
        ]
        self.status_update.emit(f"Running command: {' '.join(command)}")

        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.run(command, capture_output=True, text=True, creationflags=creation_flags)
            print(f"UVToolsCmd.exe (run) finished with exit code: {process.returncode}")
            if process.returncode not in [0, 1]:
                raise RuntimeError(f"UVTools exited with an error (code {process.returncode}):\n\n{process.stderr}")
            self.status_update.emit(f"Successfully created: {output_filename}")
        except Exception as e:
            raise RuntimeError(f"UVTools repacking failed: {e}")

    @staticmethod
    def _process_single_image_task(
        current_image_filepath: str,
        layer_index: int,
        prior_binary_masks_snapshot: collections.deque,
        app_config: Config,
        xy_blend_pipeline_ops: List[XYBlendOperation],
        output_folder: str,
        debug_save: bool
    ) -> str:
        """Processes a single image completely. This function runs in a worker thread."""
        current_binary_image, original_image = core.load_image(current_image_filepath)
        if current_binary_image is None:
            raise RuntimeError(f"Could not load image: {current_image_filepath}")

        debug_info = {'output_folder': output_folder, 'base_filename': os.path.splitext(os.path.basename(current_image_filepath))[0]} if debug_save else None
        prior_white_combined_mask = core.find_prior_combined_white_mask(list(prior_binary_masks_snapshot))
        
        receding_gradient = core.process_z_blending(
            current_white_mask,
            prior_white_combined_mask,
            app_config,
            layer_index,
            debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(original_image, receding_gradient)
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        output_filename = os.path.basename(current_image_filepath)
        output_filepath = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_filepath, final_processed_image)
        return output_filepath

    def run(self):
        """The main processing loop, orchestrating the full extract-process-repack workflow."""
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
            
            active_futures = set()
            processed_count = 0
            submitted_count = 0

            filename_iterator = iter(image_filenames_filtered)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 1. Prime the pump
                for _ in range(self.max_workers):
                    try:
                        filename = next(filename_iterator)
                        filepath = os.path.join(input_path, filename)
                        
                        current_binary_image_for_cache, _ = core.load_image(filepath)
                        if current_binary_image_for_cache is None:
                            self.status_update.emit(f"Skipping unloadable image: {filename}")
                            continue 
                        
                        numeric_part = get_numeric_part(filename)
                        future = executor.submit(
                            ImageProcessorThread._process_single_image_task,
                            filepath,
                            numeric_part,
                            collections.deque(prior_binary_masks_cache),
                            self.app_config,
                            self.app_config.xy_blend_pipeline,
                            processing_output_path,
                            self.app_config.debug_save
                        )
                        active_futures.add(future)
                        prior_binary_masks_cache.append(current_binary_image_for_cache)
                        submitted_count += 1
                        self.status_update.emit(f"Submitting {filename} ({submitted_count}/{total_images})")
                    except StopIteration:
                        break

                # 2. Main processing loop
                while active_futures:
                    if not self._is_running:
                        self.status_update.emit("Processing stopped by user.")
                        break

                    done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)

                    for future in done:
                        active_futures.remove(future)
                        try:
                            output_path = future.result()
                            processed_count += 1
                            self.status_update.emit(f"Completed: {os.path.basename(output_path)} ({processed_count}/{total_images})")
                            self.progress_update.emit(int((processed_count / total_images) * 100))
                        except Exception as exc:
                            import traceback
                            error_detail = f"An image processing task failed: {exc}\n{traceback.format_exc()}"
                            self.error_signal.emit(error_detail)
                            self._is_running = False
                            break

                    if not self._is_running:
                        break

                    # Submit new tasks
                    try:
                        filename = next(filename_iterator)
                        filepath = os.path.join(input_path, filename)
                        
                        current_binary_image_for_cache, _ = core.load_image(filepath)
                        if current_binary_image_for_cache is None:
                            self.status_update.emit(f"Skipping unloadable image: {filename}")
                            continue
                        
                        numeric_part = get_numeric_part(filename)
                        future = executor.submit(
                            ImageProcessorThread._process_single_image_task,
                            filepath,
                            numeric_part,
                            collections.deque(prior_binary_masks_cache),
                            self.app_config,
                            self.app_config.xy_blend_pipeline,
                            processing_output_path,
                            self.app_config.debug_save
                        )
                        active_futures.add(future)
                        prior_binary_masks_cache.append(current_binary_image_for_cache)
                        submitted_count += 1
                        self.status_update.emit(f"Submitting {filename} ({submitted_count}/{total_images})")
                    except StopIteration:
                        pass
                
                if self._is_running and processed_count == total_images:
                    self.progress_update.emit(100)
            
            self.status_update.emit("All image processing tasks completed.")

            # --- UVTools Repack Step ---
            if self.app_config.input_mode == "uvtools" and self._is_running:
                uvtop_file = self._generate_uvtop_file(processing_output_path)
                self._run_uvtools_repack(uvtop_file)

        except Exception as e:
            import traceback
            error_info = f"Error in processing thread: {e}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            # --- Cleanup Step ---
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

class ImageProcessorApp(QWidget):
    """The main application window, now with a restructured UI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Stack Euclidean Distance Blending & XY Pipeline")
        self.settings = QSettings("YourCompany", "VoxelBlendApp")
        self.processor_thread = None
        self.init_ui()
        self._autodetect_uvtools()
        self.load_settings()
        self._connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.main_processing_tab = QWidget()
        main_processing_layout = QVBoxLayout(self.main_processing_tab)
        self.tab_widget.addTab(self.main_processing_tab, "Main Processing")

        # --- I/O Section ---
        io_group = QGroupBox("I/O")
        io_layout = QVBoxLayout(io_group)
        
        input_mode_layout = QHBoxLayout()
        self.input_mode_group = QButtonGroup(self)
        self.folder_mode_radio = QRadioButton("Folder Input Mode")
        self.folder_mode_radio.setChecked(True)
        self.uvtools_mode_radio = QRadioButton("Use UVTools 5.x+")
        self.input_mode_group.addButton(self.folder_mode_radio, 0)
        self.input_mode_group.addButton(self.uvtools_mode_radio, 1)
        input_mode_layout.addWidget(self.folder_mode_radio)
        input_mode_layout.addWidget(self.uvtools_mode_radio)
        input_mode_layout.addStretch(1)
        io_layout.addLayout(input_mode_layout)

        self.io_stacked_widget = QStackedWidget()
        io_layout.addWidget(self.io_stacked_widget)

        # --- Folder Mode Widget ---
        folder_mode_widget = QWidget()
        folder_mode_layout = QGridLayout(folder_mode_widget)
        folder_mode_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        folder_mode_layout.addWidget(self.input_folder_edit, 0, 1)
        self.input_folder_button = QPushButton("Browse...")
        folder_mode_layout.addWidget(self.input_folder_button, 0, 2)
        
        folder_mode_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        folder_mode_layout.addWidget(self.output_folder_edit, 1, 1)
        self.output_folder_button = QPushButton("Browse...")
        folder_mode_layout.addWidget(self.output_folder_button, 1, 2)

        folder_mode_layout.addWidget(QLabel("Start Index:"), 2, 0)
        self.start_idx_edit = QLineEdit("0")
        folder_mode_layout.addWidget(self.start_idx_edit, 2, 1)
        
        folder_mode_layout.addWidget(QLabel("Stop Index:"), 3, 0)
        self.stop_idx_edit = QLineEdit()
        folder_mode_layout.addWidget(self.stop_idx_edit, 3, 1)
        self.io_stacked_widget.addWidget(folder_mode_widget)

        # --- UVTools Mode Widget ---
        uvtools_mode_widget = QWidget()
        uvtools_mode_layout = QGridLayout(uvtools_mode_widget)
        uvtools_mode_layout.addWidget(QLabel("Path to UVToolsCmd.exe:"), 0, 0)
        self.uvtools_path_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_path_edit, 0, 1)
        self.uvtools_path_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_path_button, 0, 2)
        
        uvtools_mode_layout.addWidget(QLabel("Working Temp Folder:"), 1, 0)
        self.uvtools_temp_folder_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_temp_folder_edit, 1, 1)
        self.uvtools_temp_folder_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_temp_folder_button, 1, 2)
        
        uvtools_mode_layout.addWidget(QLabel("Input Slice File:"), 2, 0)
        self.uvtools_input_file_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_input_file_edit, 2, 1)
        self.uvtools_input_file_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_input_file_button, 2, 2)
        
        # NEW: Horizontal Rule
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        uvtools_mode_layout.addWidget(divider, 3, 0, 1, 3)

        # NEW: Output Location Radios
        uvtools_mode_layout.addWidget(QLabel("Output Completed Slice file:"), 4, 0)
        self.uvtools_output_location_group = QButtonGroup(self)
        self.uvtools_output_working_radio = QRadioButton("To Working Folder")
        self.uvtools_output_input_radio = QRadioButton("To Input Slice Folder")
        self.uvtools_output_location_group.addButton(self.uvtools_output_working_radio, 0)
        self.uvtools_output_location_group.addButton(self.uvtools_output_input_radio, 1)
        uvtools_mode_layout.addWidget(self.uvtools_output_working_radio, 4, 1)
        uvtools_mode_layout.addWidget(self.uvtools_output_input_radio, 5, 1)
        self.uvtools_output_working_radio.setChecked(True)

        output_note_label = QLabel("<i>Output file will be named \"Voxel_Blend_Processed_&lt;date-time&gt;_&lt;original-filename&gt;\"</i>")
        output_note_label.setWordWrap(True)
        uvtools_mode_layout.addWidget(output_note_label, 6, 1, 1, 2)
        
        self.uvtools_cleanup_checkbox = QCheckBox("Delete Temporary Files on Completion")
        self.uvtools_cleanup_checkbox.setChecked(True)
        uvtools_mode_layout.addWidget(self.uvtools_cleanup_checkbox, 7, 1, 1, 2)

        self.io_stacked_widget.addWidget(uvtools_mode_widget)

        main_processing_layout.addWidget(io_group)

        # --- Stack Blending Section ---
        blending_group = QGroupBox("Stack Blending")
        blending_layout = QVBoxLayout(blending_group)

        # --- Blending Mode Selection ---
        blending_mode_layout = QHBoxLayout()
        blending_mode_layout.addWidget(QLabel("Blending Mode:"))
        self.blending_mode_group = QButtonGroup(self)
        self.fixed_fade_mode_radio = QRadioButton("Fixed Fade")
        self.roi_fade_mode_radio = QRadioButton("ROI Fade")
        self.blending_mode_group.addButton(self.fixed_fade_mode_radio, 0)
        self.blending_mode_group.addButton(self.roi_fade_mode_radio, 1)
        blending_mode_layout.addWidget(self.fixed_fade_mode_radio)
        blending_mode_layout.addWidget(self.roi_fade_mode_radio)
        blending_mode_layout.addStretch(1)
        blending_layout.addLayout(blending_mode_layout)

        # --- Common Blending Settings ---
        common_blending_layout = QGridLayout()
        common_blending_layout.addWidget(QLabel("Receding Look Down Layers:"), 0, 0)
        self.receding_layers_edit = QLineEdit("3")
        self.receding_layers_edit.setValidator(QIntValidator(0, 100, self))
        common_blending_layout.addWidget(self.receding_layers_edit, 0, 1)
        common_blending_layout.setColumnStretch(2, 1) # Add stretch
        blending_layout.addLayout(common_blending_layout)

        # --- Blending Settings Stacked Widget ---
        self.blending_stacked_widget = QStackedWidget()
        blending_layout.addWidget(self.blending_stacked_widget)

        # --- Page 0: Fixed Fade Mode ---
        fixed_fade_widget = QWidget()
        fixed_fade_layout = QGridLayout(fixed_fade_widget)
        self.fixed_fade_receding_checkbox = QCheckBox("Use Fixed Fade Distance")
        fixed_fade_layout.addWidget(self.fixed_fade_receding_checkbox, 0, 0)
        self.fade_dist_receding_edit = QLineEdit("10.0")
        self.fade_dist_receding_edit.setValidator(QDoubleValidator(0.1, 1000.0, 2, self))
        fixed_fade_layout.addWidget(self.fade_dist_receding_edit, 0, 1)
        fixed_fade_layout.setColumnStretch(2, 1)
        self.blending_stacked_widget.addWidget(fixed_fade_widget)

        # --- Page 1: ROI Fade Mode ---
        roi_fade_widget = QWidget()
        roi_fade_layout = QVBoxLayout(roi_fade_widget)

        main_roi_layout = QGridLayout()
        main_roi_layout.addWidget(QLabel("Min ROI Size (pixels):"), 0, 0)
        self.roi_min_size_edit = QLineEdit("100")
        self.roi_min_size_edit.setValidator(QIntValidator(1, 1000000, self))
        main_roi_layout.addWidget(self.roi_min_size_edit, 0, 1)
        main_roi_layout.setColumnStretch(2, 1)
        roi_fade_layout.addLayout(main_roi_layout)

        self.raft_support_group = QGroupBox("Raft & Support Handling")
        self.raft_support_group.setCheckable(True)
        raft_support_layout = QGridLayout(self.raft_support_group)
        raft_support_layout.addWidget(QLabel("Raft Layers (from bottom):"), 0, 0)
        self.raft_layer_count_edit = QLineEdit("5")
        self.raft_layer_count_edit.setValidator(QIntValidator(0, 1000))
        raft_support_layout.addWidget(self.raft_layer_count_edit, 0, 1)
        raft_support_layout.addWidget(QLabel("Raft Min Size (pixels):"), 0, 2)
        self.raft_min_size_edit = QLineEdit("10000")
        self.raft_min_size_edit.setValidator(QIntValidator(0, 100000000))
        raft_support_layout.addWidget(self.raft_min_size_edit, 0, 3)
        raft_support_layout.addWidget(QLabel("Support Max Size (pixels):"), 1, 0)
        self.support_max_size_edit = QLineEdit("500")
        self.support_max_size_edit.setValidator(QIntValidator(0, 1000000))
        raft_support_layout.addWidget(self.support_max_size_edit, 1, 1)

        note_label = QLabel("<i>Note: Classified rafts and supports will be ignored (no gradient processing).</i>")
        note_label.setWordWrap(True)
        raft_support_layout.addWidget(note_label, 2, 0, 1, 4)

        roi_fade_layout.addWidget(self.raft_support_group)
        roi_fade_layout.addStretch(1)

        self.blending_stacked_widget.addWidget(roi_fade_widget)

        # --- Overhang (Common) ---
        overhang_layout = QGridLayout()
        overhang_layout.addWidget(QLabel("Overhang Look Up Layers: (Disabled / WiP)"), 0, 0)
        self.overhang_layers_edit = QLineEdit("0")
        self.overhang_layers_edit.setEnabled(False)
        overhang_layout.addWidget(self.overhang_layers_edit, 0, 1)
        self.fixed_fade_overhang_checkbox = QCheckBox("Use Fixed Fade Distance (Disabled / WiP)")
        self.fixed_fade_overhang_checkbox.setEnabled(False)
        overhang_layout.addWidget(self.fixed_fade_overhang_checkbox, 0, 2)
        self.fade_dist_overhang_edit = QLineEdit("10.0")
        self.fade_dist_overhang_edit.setEnabled(False)
        overhang_layout.addWidget(self.fade_dist_overhang_edit, 0, 3)
        blending_layout.addLayout(overhang_layout)

        main_processing_layout.addWidget(blending_group)
        
        # --- General Settings Section ---
        general_group = QGroupBox("General")
        general_layout = QVBoxLayout(general_group)
        
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("Thread Count:"))
        self.thread_count_edit = QLineEdit(str(DEFAULT_NUM_WORKERS))
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        self.thread_count_edit.setFixedWidth(60)
        thread_layout.addWidget(self.thread_count_edit)
        self.ram_estimate_label = QLabel("RAM usage scales with thread count")
        thread_layout.addWidget(self.ram_estimate_label)
        thread_layout.addStretch(1)
        general_layout.addLayout(thread_layout)

        self.debug_checkbox = QCheckBox("Save Intermediate Debug Images")
        general_layout.addWidget(self.debug_checkbox)
        
        config_buttons_layout = QHBoxLayout()
        self.save_config_button = QPushButton("Save Config...")
        config_buttons_layout.addWidget(self.save_config_button)
        self.load_config_button = QPushButton("Load Config...")
        config_buttons_layout.addWidget(self.load_config_button)
        config_buttons_layout.addStretch(1)
        general_layout.addLayout(config_buttons_layout)
        
        main_processing_layout.addWidget(general_group)
        main_processing_layout.addStretch(1)

        self.xy_blend_tab = XYBlendTab(self)
        self.tab_widget.addTab(self.xy_blend_tab, "XY Blend Pipeline")

        self.start_stop_button = QPushButton("Start Processing")
        self.start_stop_button.setMinimumHeight(40)
        main_layout.addWidget(self.start_stop_button)
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True) # NEW: Enable word wrap
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

    def _connect_signals(self):
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.on_input_mode_changed)
        self.blending_mode_group.idClicked.connect(self.on_blending_mode_changed)
        self.save_config_button.clicked.connect(self._save_config_to_file)
        self.load_config_button.clicked.connect(self._load_config_from_file)
        self.start_stop_button.clicked.connect(self.toggle_processing)

    def _autodetect_uvtools(self):
        """Checks for UVTools in the default location and populates the path if found."""
        default_path = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
        if os.path.exists(default_path):
            if not self.uvtools_path_edit.text():
                self.uvtools_path_edit.setText(default_path)

    def on_input_mode_changed(self, stack_index):
        self.io_stacked_widget.setCurrentIndex(stack_index)

    def on_blending_mode_changed(self, stack_index):
        self.blending_stacked_widget.setCurrentIndex(stack_index)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder: line_edit.setText(folder)

    def browse_file(self, line_edit, caption, file_filter="All Files (*)"):
        file, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), file_filter)
        if file: line_edit.setText(file)

    def load_settings(self):
        """Loads settings from the global config object into the UI."""
        self.resize(self.settings.value("window_size", self.size()))
        self.move(self.settings.value("window_position", self.pos()))
        
        self.folder_mode_radio.setChecked(config.input_mode == "folder")
        self.uvtools_mode_radio.setChecked(config.input_mode == "uvtools")
        self.io_stacked_widget.setCurrentIndex(1 if config.input_mode == "uvtools" else 0)
        self.input_folder_edit.setText(config.input_folder)
        self.output_folder_edit.setText(config.output_folder)
        self.start_idx_edit.setText(str(config.start_index) if config.start_index is not None else "")
        self.stop_idx_edit.setText(str(config.stop_index) if config.stop_index is not None else "")
        self.uvtools_path_edit.setText(config.uvtools_path)
        self.uvtools_temp_folder_edit.setText(config.uvtools_temp_folder)
        self.uvtools_input_file_edit.setText(config.uvtools_input_file)
        self.uvtools_cleanup_checkbox.setChecked(config.uvtools_delete_temp_on_completion)
        self.uvtools_output_working_radio.setChecked(config.uvtools_output_location == "working_folder")
        self.uvtools_output_input_radio.setChecked(config.uvtools_output_location == "input_folder")
        self.receding_layers_edit.setText(str(config.receding_layers))
        self.fixed_fade_receding_checkbox.setChecked(config.use_fixed_fade_receding)
        self.fade_dist_receding_edit.setText(str(config.fixed_fade_distance_receding))

        self.fixed_fade_mode_radio.setChecked(config.blending_mode == "fixed_fade")
        self.roi_fade_mode_radio.setChecked(config.blending_mode == "roi_fade")
        self.blending_stacked_widget.setCurrentIndex(1 if config.blending_mode == "roi_fade" else 0)
        self.roi_min_size_edit.setText(str(config.roi_params.min_size))
        self.raft_support_group.setChecked(config.roi_params.enable_raft_support_handling)
        self.raft_layer_count_edit.setText(str(config.roi_params.raft_layer_count))
        self.raft_min_size_edit.setText(str(config.roi_params.raft_min_size))
        self.support_max_size_edit.setText(str(config.roi_params.support_max_size))

        self.overhang_layers_edit.setText(str(config.overhang_layers))
        self.fixed_fade_overhang_checkbox.setChecked(config.use_fixed_fade_overhang)
        self.fade_dist_overhang_edit.setText(str(config.fixed_fade_distance_overhang))
        self.thread_count_edit.setText(str(config.thread_count))
        self.debug_checkbox.setChecked(config.debug_save)
        
        self.xy_blend_tab.apply_settings(config)

    def save_settings(self):
        """Saves current UI settings to the global config object and QSettings."""
        self.settings.setValue("window_size", self.size())
        self.settings.setValue("window_position", self.pos())

        config.input_mode = "uvtools" if self.uvtools_mode_radio.isChecked() else "folder"
        config.input_folder = self.input_folder_edit.text()
        config.output_folder = self.output_folder_edit.text()
        config.start_index = int(s) if (s := self.start_idx_edit.text()) else None
        config.stop_index = int(s) if (s := self.stop_idx_edit.text()) else None
        config.uvtools_path = self.uvtools_path_edit.text()
        config.uvtools_temp_folder = self.uvtools_temp_folder_edit.text()
        config.uvtools_input_file = self.uvtools_input_file_edit.text()
        config.uvtools_delete_temp_on_completion = self.uvtools_cleanup_checkbox.isChecked()
        config.uvtools_output_location = "input_folder" if self.uvtools_output_input_radio.isChecked() else "working_folder"

        config.blending_mode = "roi_fade" if self.roi_fade_mode_radio.isChecked() else "fixed_fade"
        try:
            config.roi_params.min_size = int(self.roi_min_size_edit.text())
        except ValueError:
            config.roi_params.min_size = 100

        config.roi_params.enable_raft_support_handling = self.raft_support_group.isChecked()
        try:
            config.roi_params.raft_layer_count = int(self.raft_layer_count_edit.text())
        except ValueError:
            config.roi_params.raft_layer_count = 5
        try:
            config.roi_params.raft_min_size = int(self.raft_min_size_edit.text())
        except ValueError:
            config.roi_params.raft_min_size = 10000
        try:
            config.roi_params.support_max_size = int(self.support_max_size_edit.text())
        except ValueError:
            config.roi_params.support_max_size = 500

        try: config.receding_layers = int(self.receding_layers_edit.text())
        except ValueError: config.receding_layers = 3
        config.use_fixed_fade_receding = self.fixed_fade_receding_checkbox.isChecked()
        try: config.fixed_fade_distance_receding = float(self.fade_dist_receding_edit.text().replace(',', '.'))
        except ValueError: config.fixed_fade_distance_receding = 10.0
        try: config.thread_count = int(self.thread_count_edit.text())
        except ValueError: config.thread_count = DEFAULT_NUM_WORKERS
        config.debug_save = self.debug_checkbox.isChecked()
        
        config.save("app_config.json")

    def _save_config_to_file(self):
        self.save_settings() 
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "custom_config.json", "JSON Files (*.json)")
        if filepath:
            try:
                config.save(filepath)
                self.show_info_message("Success", "Configuration saved.")
            except Exception as e:
                self.show_error_message("Save Error", f"Failed to save configuration:\n{e}")

    def _load_config_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if filepath:
            try:
                loaded_config = Config.load(filepath)
                upgrade_config(loaded_config)
                config.__dict__.clear()
                config.__dict__.update(loaded_config.__dict__)
                self.load_settings() 
                self.show_info_message("Success", "Configuration loaded.")
            except Exception as e:
                self.show_error_message("Load Error", f"Failed to load configuration:\n{e}")

    def closeEvent(self, event):
        self.save_settings()
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop_processing()
            self.processor_thread.wait(5000)
        event.accept()

    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop_processing()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
        else:
            self.start_processing()

    def start_processing(self):
        """Validates inputs and starts the processing thread."""
        try:
            self.save_settings()
            
            if config.input_mode == "folder":
                if not config.input_folder or not os.path.isdir(config.input_folder):
                    raise ValueError("Input folder must be a valid, existing directory.")
                if not config.output_folder or not os.path.isdir(config.output_folder):
                    raise ValueError("Output folder must be a valid, existing directory.")
            elif config.input_mode == "uvtools":
                if not config.uvtools_path or not os.path.exists(config.uvtools_path):
                    raise ValueError("UVToolsCmd.exe path is not valid.")
                if not config.uvtools_temp_folder or not os.path.isdir(config.uvtools_temp_folder):
                    raise ValueError("Working Temp Folder must be a valid, existing directory.")
                if not config.uvtools_input_file or not os.path.exists(config.uvtools_input_file):
                    raise ValueError("Input Slice File is not valid.")
            
            self.set_ui_enabled(False)
            self.processor_thread = ImageProcessorThread(app_config=config, max_workers=config.thread_count)
            self.processor_thread.status_update.connect(self.update_status)
            self.processor_thread.progress_update.connect(self.progress_bar.setValue)
            self.processor_thread.error_signal.connect(self.show_error)
            self.processor_thread.finished_signal.connect(self.processing_finished)
            self.processor_thread.start()

        except Exception as e:
            self.show_error_message("Input Error", str(e))
            self.processing_finished()

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot(str)
    def show_error(self, message):
        print(f"\n--- PROCESSING ERROR ---\n{message}\n------------------------\n")
        self.show_error_message("Processing Error", message, is_detailed=True)
        self.processing_finished()

    @Slot()
    def processing_finished(self):
        self.status_label.setText("Status: Finished or Stopped.")
        self.set_ui_enabled(True)
        self.processor_thread = None

    def show_error_message(self, title, text, is_detailed=False):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        if is_detailed:
            msg_box.setText("An error occurred. See details for more information.")
            msg_box.setDetailedText(text)
        else:
            msg_box.setText(text)
            msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        msg_box.exec()

    def show_info_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        msg_box.exec()

    def set_ui_enabled(self, enabled):
        """Toggles the enabled state of all UI widgets."""
        self.tab_widget.setEnabled(enabled)
        self.start_stop_button.setEnabled(True) 
        if not enabled:
            self.start_stop_button.setText("Stop Processing")
        else:
            self.start_stop_button.setText("Start Processing")
