# pyside_file_io_tab.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QFileDialog
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIntValidator

from config import app_config as config, Config # Import Config class explicitly

class FileIOTab(QWidget):
    """
    PySide6 tab for File I/O settings, including input/output directories,
    output options (filename padding), and global controls like Run, Stop, Save, Load.
    Output size and resampling are now handled in the XY Blend tab.
    """
    # Signals to communicate with the main GUI window
    run_requested = Signal()
    stop_requested = Signal()
    save_settings_requested = Signal()
    load_settings_requested = Signal()

    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui # Reference to the main GUI window
        self.config = config # Use the global config instance

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config

    def _setup_ui(self):
        """Sets up the widgets and layout for the File I/O tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- I/O Settings Group ---
        io_group = QGroupBox("I/O Settings")
        io_layout = QVBoxLayout(io_group)
        io_layout.setContentsMargins(10, 10, 10, 10)
        io_layout.setSpacing(5)

        # Input Directory
        input_dir_layout = QHBoxLayout()
        input_dir_layout.addWidget(QLabel("Input Dir:"))
        self.input_dir_edit = QLineEdit()
        input_dir_layout.addWidget(self.input_dir_edit)
        self.input_browse_btn = QPushButton("Browse...")
        input_dir_layout.addWidget(self.input_browse_btn)
        io_layout.addLayout(input_dir_layout)

        # Output Directory
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Dir:"))
        self.output_dir_edit = QLineEdit()
        output_dir_layout.addWidget(self.output_dir_edit)
        self.output_browse_btn = QPushButton("Browse...")
        output_dir_layout.addWidget(self.output_browse_btn)
        io_layout.addLayout(output_dir_layout)

        # File Pattern
        file_pattern_layout = QHBoxLayout()
        file_pattern_layout.addWidget(QLabel("File Pattern:"))
        self.file_pattern_edit = QLineEdit()
        self.file_pattern_edit.setPlaceholderText("e.g., *.png or image_*.tif")
        file_pattern_layout.addWidget(self.file_pattern_edit)
        io_layout.addLayout(file_pattern_layout)

        main_layout.addWidget(io_group)

        # --- Output Options Group ---
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(10, 10, 10, 10)
        output_layout.setSpacing(5)

        # Filename Padding
        pad_layout = QHBoxLayout()
        self.pad_filenames_check = QCheckBox("Pad Filenames")
        pad_layout.addWidget(self.pad_filenames_check)

        pad_layout.addWidget(QLabel("Pad Length:"))
        self.pad_length_edit = QLineEdit()
        self.pad_length_edit.setFixedWidth(60)
        self.pad_length_edit.setValidator(QIntValidator(1, 10, self)) # Max 10 digits for padding
        pad_layout.addWidget(self.pad_length_edit)
        pad_layout.addStretch(1) # Push to left
        output_layout.addLayout(pad_layout)

        main_layout.addWidget(output_group)

        # --- Controls Group (Run/Stop/Save/Load) ---
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)

        self.run_btn = QPushButton("Run")
        controls_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False) # Initially disabled
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addStretch(1) # Push save/load to the right

        self.save_settings_btn = QPushButton("Save Settings")
        controls_layout.addWidget(self.save_settings_btn)

        self.load_settings_btn = QPushButton("Load Settings")
        controls_layout.addWidget(self.load_settings_btn)

        main_layout.addWidget(controls_group)

        main_layout.addStretch(1) # Pushes all content to the top


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.input_browse_btn.clicked.connect(self._browse_input_dir)
        self.output_browse_btn.clicked.connect(self._browse_output_dir)

        # Connect internal buttons to tab's signals, which are then connected
        # to the main GUI's slots in SuperStackerPysideGUI._setup_ui
        self.run_btn.clicked.connect(self.run_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.save_settings_btn.clicked.connect(self.save_settings_requested.emit)
        self.load_settings_btn.clicked.connect(self.load_settings_requested.emit)

        # Input validation for numeric fields
        self.pad_length_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.pad_length_edit, text, int))


    def _browse_input_dir(self):
        """Opens a directory dialog for input path selection."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.input_dir_edit.text())
        if dir_path:
            self.input_dir_edit.setText(dir_path)

    def _browse_output_dir(self):
        """Opens a directory dialog for output path selection."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_dir_edit.text())
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _validate_numeric_input(self, line_edit: QLineEdit, text: str, data_type: type):
        """
        Validates if the input text can be converted to the specified data_type.
        If not, applies red border.
        """
        if not text:
            line_edit.setStyleSheet("") # Clear any error styling for empty string
            return

        try:
            if data_type == int:
                val = int(text)
            elif data_type == float:
                val = float(text)
            line_edit.setStyleSheet("") # Clear any error styling
        except ValueError:
            line_edit.setStyleSheet("border: 1px solid red;")

    def get_config(self) -> dict:
        """
        Collects current settings from this tab's widgets.
        Includes basic validation and conversion.
        """
        config_data = {}
        config_data["input_dir"] = self.input_dir_edit.text()
        config_data["output_dir"] = self.output_dir_edit.text()
        config_data["file_pattern"] = self.file_pattern_edit.text()

        # Numeric fields with validation
        try:
            config_data["pad_length"] = int(self.pad_length_edit.text()) if self.pad_length_edit.text() else 4
        except ValueError as e:
            raise ValueError(f"Invalid numeric input for Pad Length: {e}")

        config_data["pad_filenames"] = self.pad_filenames_check.isChecked()

        return config_data

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        self.input_dir_edit.setText(cfg.input_dir)
        self.output_dir_edit.setText(cfg.output_dir)
        self.file_pattern_edit.setText(cfg.file_pattern)
        self.pad_filenames_check.setChecked(cfg.pad_filenames)
        self.pad_length_edit.setText(str(cfg.pad_length))

    def set_run_button_state(self, enabled: bool):
        """Sets the enabled state of the Run button."""
        self.run_btn.setEnabled(enabled)

    def set_stop_button_state(self, enabled: bool):
        """Sets the enabled state of the Stop button."""
        self.stop_btn.setEnabled(enabled)

