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
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QTabWidget, QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import Slot, Qt, QSettings
from PySide6.QtGui import QIntValidator, QDoubleValidator

from config import app_config as config, Config, DEFAULT_NUM_WORKERS, upgrade_config
from pyside_xy_blend_tab import XYBlendTab
from processing_pipeline import ProcessingPipelineThread

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
        
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        uvtools_mode_layout.addWidget(divider, 3, 0, 1, 3)

        uvtools_mode_layout.addWidget(QLabel("Output Completed Slice file:"), 4, 0)
        self.uvtools_output_location_group = QButtonGroup(self)
        self.uvtools_output_working_radio = QRadioButton("To Working Folder")
        self.uvtools_output_input_radio = QRadioButton("To Input Slice Folder")
        self.uvtools_output_location_group.addButton(self.uvtools_output_working_radio, 0)
        self.uvtools_output_location_group.addButton(self.uvtools_output_input_radio, 1)
        uvtools_mode_layout.addWidget(self.uvtools_output_working_radio, 4, 1)
        uvtools_mode_layout.addWidget(self.uvtools_output_input_radio, 5, 1)
        self.uvtools_output_working_radio.setChecked(True)

        uvtools_mode_layout.addWidget(QLabel("Output File Prefix:"), 6, 0)
        self.output_prefix_edit = QLineEdit("Voxel_Blend_Processed_")
        uvtools_mode_layout.addWidget(self.output_prefix_edit, 6, 1, 1, 2)
        
        self.uvtools_cleanup_checkbox = QCheckBox("Delete Temporary Files on Completion")
        self.uvtools_cleanup_checkbox.setChecked(True)
        uvtools_mode_layout.addWidget(self.uvtools_cleanup_checkbox, 7, 1, 1, 2)

        self.io_stacked_widget.addWidget(uvtools_mode_widget)

        main_processing_layout.addWidget(io_group)

        # --- Stack Blending Section ---
        blending_group = QGroupBox("Stack Blending")
        blending_layout = QVBoxLayout(blending_group)

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

        common_blending_layout = QGridLayout()
        common_blending_layout.addWidget(QLabel("Receding Look Down Layers:"), 0, 0)
        self.receding_layers_edit = QLineEdit("3")
        self.receding_layers_edit.setValidator(QIntValidator(0, 100, self))
        common_blending_layout.addWidget(self.receding_layers_edit, 0, 1)
        common_blending_layout.setColumnStretch(2, 1)
        blending_layout.addLayout(common_blending_layout)

        self.blending_stacked_widget = QStackedWidget()
        blending_layout.addWidget(self.blending_stacked_widget)

        fixed_fade_widget = QWidget()
        fixed_fade_layout = QGridLayout(fixed_fade_widget)
        self.fixed_fade_receding_checkbox = QCheckBox("Use Fixed Fade Distance")
        fixed_fade_layout.addWidget(self.fixed_fade_receding_checkbox, 0, 0)
        self.fade_dist_receding_edit = QLineEdit("10.0")
        self.fade_dist_receding_edit.setValidator(QDoubleValidator(0.1, 1000.0, 2, self))
        fixed_fade_layout.addWidget(self.fade_dist_receding_edit, 0, 1)
        fixed_fade_layout.setColumnStretch(2, 1)
        self.blending_stacked_widget.addWidget(fixed_fade_widget)

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
        raft_support_layout.addWidget(QLabel("Max Support Layer:"), 1, 2)
        self.support_max_layer_edit = QLineEdit("1000")
        self.support_max_layer_edit.setValidator(QIntValidator(0, 100000))
        raft_support_layout.addWidget(self.support_max_layer_edit, 1, 3)
        raft_support_layout.addWidget(QLabel("Max Support Growth (%):"), 2, 0)
        self.support_max_growth_edit = QLineEdit("150.0")
        self.support_max_growth_edit.setValidator(QDoubleValidator(0.0, 10000.0, 1))
        raft_support_layout.addWidget(self.support_max_growth_edit, 2, 1)
        note_label = QLabel("<i>Note: Classified rafts/supports are ignored. Growth factor is used to reclassify supports as model.</i>")
        note_label.setWordWrap(True)
        raft_support_layout.addWidget(note_label, 3, 0, 1, 4)
        roi_fade_layout.addWidget(self.raft_support_group)
        roi_fade_layout.addStretch(1)
        self.blending_stacked_widget.addWidget(roi_fade_widget)

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
        self.status_label.setWordWrap(True)
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
        self.output_prefix_edit.setText(config.output_file_prefix)
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
        self.support_max_layer_edit.setText(str(config.roi_params.support_max_layer))
        self.support_max_growth_edit.setText(f"{(config.roi_params.support_max_growth - 1.0) * 100.0:.1f}")

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
        config.output_file_prefix = self.output_prefix_edit.text()
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
        try:
            config.roi_params.support_max_layer = int(self.support_max_layer_edit.text())
        except ValueError:
            config.roi_params.support_max_layer = 1000
        try:
            config.roi_params.support_max_growth = (float(self.support_max_growth_edit.text().replace(',', '.')) / 100.0) + 1.0
        except ValueError:
            config.roi_params.support_max_growth = 2.5

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
            self.processor_thread = ProcessingPipelineThread(app_config=config, max_workers=config.thread_count)
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
