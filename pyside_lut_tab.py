# pyside_lut_tab.py

import sys
import numpy as np
import json
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QLabel, QLineEdit,
    QTextEdit, QComboBox, QInputDialog, QSlider, QSpinBox, QStackedWidget,
    QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator

# Matplotlib imports for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import lut_manager # Import your lut_manager module
from config import app_config as config, Config # Import the global config instance and Config class

# Define a directory for saved LUT versions
LUT_VERSIONS_DIR = "saved_luts"
if not os.path.exists(LUT_VERSIONS_DIR):
    os.makedirs(LUT_VERSIONS_DIR)


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PySide6."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout() # Adjusts plot parameters for a tight layout


class LutTab(QWidget):
    """
    PySide6 tab for editing and managing Z-Axis Look-Up Tables (LUTs).
    Includes text editor, plot visualization, file/version operations,
    and algorithmic LUT generation controls.
    """
    # Signal to notify main GUI if the active LUT was changed (e.g., by loading/applying)
    lut_changed = Signal()

    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config # Use the global config instance
        
        # Initialize lut_manager's config reference (already done in main_gui, but safe to re-set)
        lut_manager.set_config_reference(self.config)

        # Get the initial active LUT from lut_manager, which should already be set by main_gui.py
        self.current_lut = lut_manager.get_current_z_lut()

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config
        self._load_lut_to_text_edit(self.current_lut) # Populate text edit with initial LUT
        self.plot_lut(self.current_lut) # Plot initial LUT
        self._load_lut_versions() # Populate the versions dropdown
        self._apply_initial_state() # Ensure initial control states are correct

    def _setup_ui(self):
        """Sets up the widgets and layout for the LUT tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- LUT Source Selection ---
        lut_source_group = QGroupBox("LUT Source")
        lut_source_layout = QHBoxLayout(lut_source_group)
        lut_source_layout.addWidget(QLabel("Source:"))
        self.lut_source_combo = QComboBox()
        self.lut_source_combo.addItems(["Generated", "File"])
        lut_source_layout.addWidget(self.lut_source_combo)
        lut_source_layout.addStretch(1)
        main_layout.addWidget(lut_source_group)

        # --- Generated LUT Controls ---
        self.generated_lut_group = QGroupBox("Generated LUT Parameters") # Made instance variable
        generated_lut_layout = QVBoxLayout(self.generated_lut_group)
        generated_lut_layout.setContentsMargins(10, 10, 10, 10)
        generated_lut_layout.setSpacing(5)

        # LUT Generation Type
        gen_type_layout = QHBoxLayout()
        gen_type_layout.addWidget(QLabel("Type:"))
        self.lut_generation_type_combo = QComboBox()
        self.lut_generation_type_combo.addItems(["linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard"])
        gen_type_layout.addWidget(self.lut_generation_type_combo)
        gen_type_layout.addStretch(1)
        generated_lut_layout.addLayout(gen_type_layout)

        # Stacked Widget for specific generation parameters
        self.gen_params_stacked_widget = QStackedWidget()
        generated_lut_layout.addWidget(self.gen_params_stacked_widget)

        # --- Linear LUT Params ---
        self.linear_params_widget = QWidget()
        linear_layout = QHBoxLayout(self.linear_params_widget)
        linear_layout.addWidget(QLabel("Min Input:"))
        self.linear_min_input_edit = QLineEdit()
        self.linear_min_input_edit.setFixedWidth(60)
        self.linear_min_input_edit.setValidator(QIntValidator(0, 255, self))
        linear_layout.addWidget(self.linear_min_input_edit)
        linear_layout.addWidget(QLabel("Max Output:"))
        self.linear_max_output_edit = QLineEdit()
        self.linear_max_output_edit.setFixedWidth(60)
        self.linear_max_output_edit.setValidator(QIntValidator(0, 255, self))
        linear_layout.addWidget(self.linear_max_output_edit)
        linear_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.linear_params_widget) # Index 0

        # --- Gamma LUT Params ---
        self.gamma_params_widget = QWidget()
        gamma_layout = QHBoxLayout(self.gamma_params_widget)
        gamma_layout.addWidget(QLabel("Gamma Value:"))
        self.gamma_value_edit = QLineEdit()
        self.gamma_value_edit.setFixedWidth(60)
        self.gamma_value_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        gamma_layout.addWidget(self.gamma_value_edit)
        self.gamma_value_slider = QSlider(Qt.Horizontal)
        self.gamma_value_slider.setRange(1, 1000) # 0.01 to 10.00
        self.gamma_value_slider.setSingleStep(1)
        gamma_layout.addWidget(self.gamma_value_slider)
        gamma_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.gamma_params_widget) # Index 1

        # --- S-Curve LUT Params ---
        self.s_curve_params_widget = QWidget()
        s_curve_layout = QHBoxLayout(self.s_curve_params_widget)
        s_curve_layout.addWidget(QLabel("Contrast:"))
        self.s_curve_contrast_edit = QLineEdit()
        self.s_curve_contrast_edit.setFixedWidth(60)
        self.s_curve_contrast_edit.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        s_curve_layout.addWidget(self.s_curve_contrast_edit)
        self.s_curve_contrast_slider = QSlider(Qt.Horizontal)
        self.s_curve_contrast_slider.setRange(0, 100) # 0.0 to 1.0
        self.s_curve_contrast_slider.setSingleStep(1)
        s_curve_layout.addWidget(self.s_curve_contrast_slider)
        s_curve_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.s_curve_params_widget) # Index 2

        # --- Log LUT Params ---
        self.log_params_widget = QWidget()
        log_layout = QHBoxLayout(self.log_params_widget)
        log_layout.addWidget(QLabel("Param:"))
        self.log_param_edit = QLineEdit()
        self.log_param_edit.setFixedWidth(60)
        self.log_param_edit.setValidator(QDoubleValidator(0.01, 100.0, 2, self))
        log_layout.addWidget(self.log_param_edit)
        self.log_param_slider = QSlider(Qt.Horizontal)
        self.log_param_slider.setRange(1, 1000) # 0.01 to 100.0
        self.log_param_slider.setSingleStep(1)
        log_layout.addWidget(self.log_param_slider)
        log_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.log_params_widget) # Index 3

        # --- Exp LUT Params ---
        self.exp_params_widget = QWidget()
        exp_layout = QHBoxLayout(self.exp_params_widget)
        exp_layout.addWidget(QLabel("Param:"))
        self.exp_param_edit = QLineEdit()
        self.exp_param_edit.setFixedWidth(60)
        self.exp_param_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        exp_layout.addWidget(self.exp_param_edit)
        self.exp_param_slider = QSlider(Qt.Horizontal)
        self.exp_param_slider.setRange(1, 1000) # 0.01 to 10.0
        self.exp_param_slider.setSingleStep(1)
        exp_layout.addWidget(self.exp_param_slider)
        exp_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.exp_params_widget) # Index 4

        # --- Sqrt LUT Params (minimal) ---
        self.sqrt_params_widget = QWidget()
        sqrt_layout = QHBoxLayout(self.sqrt_params_widget)
        sqrt_layout.addWidget(QLabel("Param:")) # Placeholder, currently not used in `lut_manager`
        self.sqrt_param_edit = QLineEdit("1.0")
        self.sqrt_param_edit.setFixedWidth(60)
        self.sqrt_param_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        sqrt_layout.addWidget(self.sqrt_param_edit)
        sqrt_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.sqrt_params_widget) # Index 5

        # --- Rodbard LUT Params (minimal) ---
        self.rodbard_params_widget = QWidget()
        rodbard_layout = QHBoxLayout(self.rodbard_params_widget)
        rodbard_layout.addWidget(QLabel("Param:")) # Placeholder, currently not used in `lut_manager`
        self.rodbard_param_edit = QLineEdit("1.0")
        self.rodbard_param_edit.setFixedWidth(60)
        rodbard_layout.addWidget(self.rodbard_param_edit)
        rodbard_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.rodbard_params_widget) # Index 6

        # Generate Button
        generate_btn_layout = QHBoxLayout()
        generate_btn_layout.addStretch(1)
        self.generate_lut_button = QPushButton("Generate LUT")
        generate_btn_layout.addWidget(self.generate_lut_button)
        generate_btn_layout.addStretch(1)
        generated_lut_layout.addLayout(generate_btn_layout)

        main_layout.addWidget(self.generated_lut_group)

        # --- File-based LUT Controls ---
        self.file_lut_group = QGroupBox("File-based LUT Controls") # Made instance variable
        file_lut_layout = QVBoxLayout(self.file_lut_group)
        file_lut_layout.setContentsMargins(10, 10, 10, 10)
        file_lut_layout.setSpacing(5)

        lut_path_layout = QHBoxLayout()
        lut_path_layout.addWidget(QLabel("Loaded File:"))
        self.lut_filepath_edit = QLineEdit() # This will display the path of loaded/saved LUT
        self.lut_filepath_edit.setReadOnly(True)
        lut_path_layout.addWidget(self.lut_filepath_edit)
        file_lut_layout.addLayout(lut_path_layout)

        file_button_layout = QHBoxLayout()
        self.load_file_button = QPushButton("Load from File...")
        file_button_layout.addWidget(self.load_file_button)
        self.save_file_button = QPushButton("Save to File...")
        file_button_layout.addWidget(self.save_file_button)
        file_button_layout.addStretch(1)
        file_lut_layout.addLayout(file_button_layout)

        main_layout.addWidget(self.file_lut_group)

        # --- LUT Editor (TextEdit) and Plot ---
        editor_plot_group = QGroupBox("Current LUT")
        editor_plot_layout_group = QHBoxLayout(editor_plot_group)
        
        # Text Editor for LUT values
        lut_editor_layout = QVBoxLayout()
        lut_editor_layout.addWidget(QLabel("LUT Values (0-255, one per line):"))
        self.lut_text_edit = QTextEdit()
        self.lut_text_edit.setPlaceholderText("Enter 256 integer values (0-255), one per line.")
        self.lut_text_edit.setMinimumWidth(150)
        self.lut_text_edit.setMaximumWidth(200) # Keep it narrow for single column
        self.lut_text_edit.setMinimumHeight(200)
        lut_editor_layout.addWidget(self.lut_text_edit)
        editor_plot_layout_group.addLayout(lut_editor_layout)

        # Matplotlib Plot
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setMinimumHeight(200)
        editor_plot_layout_group.addWidget(self.canvas)

        main_layout.addWidget(editor_plot_group)

        # --- Version Management ---
        version_group = QGroupBox("Saved Versions")
        version_layout = QHBoxLayout(version_group)
        version_layout.addWidget(QLabel("Select Version:"))
        self.version_combo = QComboBox()
        self.version_combo.setMinimumWidth(150)
        version_layout.addWidget(self.version_combo)

        self.load_version_button = QPushButton("Load Version")
        version_layout.addWidget(self.load_version_button)
        self.save_version_button = QPushButton("Save Current as Version...")
        version_layout.addWidget(self.save_version_button)
        self.delete_version_button = QPushButton("Delete Version")
        version_layout.addWidget(self.delete_version_button)
        version_layout.addStretch(1)
        main_layout.addWidget(version_group)

        # --- Global LUT Controls ---
        global_lut_controls_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Default LUT")
        global_lut_controls_layout.addWidget(self.reset_button)
        global_lut_controls_layout.addStretch(1)
        self.apply_button = QPushButton("Apply LUT to Pipeline")
        global_lut_controls_layout.addWidget(self.apply_button)
        global_lut_controls_layout.addStretch(1)
        main_layout.addLayout(global_lut_controls_layout)

        main_layout.addStretch(1) # Pushes all content to the top


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        # LUT Source and Generation Type
        self.lut_source_combo.currentTextChanged.connect(self._update_lut_source_controls)
        self.lut_generation_type_combo.currentTextChanged.connect(self._update_generated_lut_params_widget)
        self.generate_lut_button.clicked.connect(self._generate_and_display_lut)

        # Generated LUT parameter edits and sliders
        self.linear_min_input_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.linear_min_input_edit, text, int))
        self.linear_max_output_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.linear_max_output_edit, text, int))
        
        self.gamma_value_slider.valueChanged.connect(lambda val: self.gamma_value_edit.setText(f"{val / 100.0:.2f}"))
        self.gamma_value_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.gamma_value_slider, text, 100.0))
        
        self.s_curve_contrast_slider.valueChanged.connect(lambda val: self.s_curve_contrast_edit.setText(f"{val / 100.0:.2f}"))
        self.s_curve_contrast_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.s_curve_contrast_slider, text, 100.0))

        self.log_param_slider.valueChanged.connect(lambda val: self.log_param_edit.setText(f"{val / 10.0:.1f}"))
        self.log_param_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.log_param_slider, text, 10.0))

        self.exp_param_slider.valueChanged.connect(lambda val: self.exp_param_edit.setText(f"{val / 100.0:.2f}"))
        self.exp_param_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.exp_param_slider, text, 100.0))

        # Manual LUT editing and plotting
        self.lut_text_edit.textChanged.connect(self._on_text_edit_changed)
        
        # File Operations
        self.load_file_button.clicked.connect(self._load_lut_from_file)
        self.save_file_button.clicked.connect(self._save_lut_to_file)
        
        # Version Management
        self.load_version_button.clicked.connect(self._load_selected_version)
        self.save_version_button.clicked.connect(self._save_current_as_version)
        self.delete_version_button.clicked.connect(self._delete_selected_version)

        # Global LUT Controls
        self.reset_button.clicked.connect(self._reset_to_default_lut)
        self.apply_button.clicked.connect(self._apply_lut_changes)


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

    def _update_slider_from_edit(self, slider: QSlider, text: str, scale_factor: float):
        """Updates a slider's value based on a QLineEdit's text."""
        try:
            val = float(text)
            slider.setValue(int(val * scale_factor))
        except ValueError:
            pass # Let validator handle invalid input visually

    def _apply_initial_state(self):
        """Applies initial control states based on current config values and active LUT."""
        # Set LUT source and generation type based on config
        self.lut_source_combo.setCurrentText(self.config.lut_source.capitalize())
        self.lut_generation_type_combo.setCurrentText(self.config.lut_generation_type)
        
        # Manually trigger updates for UI enabling/disabling based on source/type
        # These calls should NOT trigger LUT generation/loading themselves during initial setup.
        self.generated_lut_group.setEnabled(self.config.lut_source == "generated")
        self.file_lut_group.setEnabled(self.config.lut_source == "file")
        self._update_generated_lut_params_widget(self.config.lut_generation_type) # Just for stacking widget visibility

        # Populate generated LUT parameter edits and sliders from config
        self.linear_min_input_edit.setText(str(self.config.linear_min_input))
        self.linear_max_output_edit.setText(str(self.config.linear_max_output))
        self.gamma_value_edit.setText(str(self.config.gamma_value))
        self.gamma_value_slider.setValue(int(self.config.gamma_value * 100))
        self.s_curve_contrast_edit.setText(str(self.config.s_curve_contrast))
        self.s_curve_contrast_slider.setValue(int(self.config.s_curve_contrast * 100))
        self.log_param_edit.setText(str(self.config.log_param))
        self.log_param_slider.setValue(int(self.config.log_param * 10))
        self.exp_param_edit.setText(str(self.config.exp_param))
        self.exp_param_slider.setValue(int(self.config.exp_param * 100))
        self.sqrt_param_edit.setText(str(self.config.sqrt_param))
        self.rodbard_param_edit.setText(str(self.config.rodbard_param))

        # Set the fixed LUT path display
        self.lut_filepath_edit.setText(self.config.fixed_lut_path)

        # Load and plot the CURRENTLY ACTIVE LUT from lut_manager
        # This LUT should already be correctly set by main_gui.py's initial call to update_active_lut_from_config()
        self._load_lut_to_text_edit(lut_manager.get_current_z_lut())
        self.plot_lut(lut_manager.get_current_z_lut())


    def _update_lut_source_controls(self, source_text: str):
        """Enables/disables UI elements based on LUT source selection."""
        is_generated = (source_text.lower() == "generated")
        
        self.generated_lut_group.setEnabled(is_generated)
        self.file_lut_group.setEnabled(not is_generated)

        # When source changes, update the config's lut_source
        self.config.lut_source = source_text.lower()

        # If switching to 'generated', regenerate and display the LUT based on current GUI params
        if is_generated:
            self._generate_and_display_lut()
        # If switching to 'file', display the LUT from the currently configured fixed_lut_path
        else:
            filepath = self.lut_filepath_edit.text()
            if filepath and os.path.exists(filepath):
                try:
                    loaded_lut = lut_manager.load_lut(filepath)
                    self._load_lut_to_text_edit(loaded_lut)
                except Exception as e:
                    QMessageBox.warning(self, "LUT Load Error", f"Could not load LUT from '{filepath}': {e}. Displaying default generated LUT.")
                    self.lut_filepath_edit.setText("") # Clear invalid path
                    self._load_lut_to_text_edit(lut_manager.get_default_z_lut()) # Show default linear
            else:
                QMessageBox.information(self, "No LUT File", "No valid LUT file path specified. Displaying default generated LUT.")
                self.lut_filepath_edit.setText("")
                self._load_lut_to_text_edit(lut_manager.get_default_z_lut()) # Show default linear


    def _update_generated_lut_params_widget(self, lut_type: str):
        """Switches the stacked widget to show parameters for the selected LUT generation type."""
        if lut_type == "linear":
            self.gen_params_stacked_widget.setCurrentWidget(self.linear_params_widget)
        elif lut_type == "gamma":
            self.gen_params_stacked_widget.setCurrentWidget(self.gamma_params_widget)
        elif lut_type == "s_curve":
            self.gen_params_stacked_widget.setCurrentWidget(self.s_curve_params_widget)
        elif lut_type == "log":
            self.gen_params_stacked_widget.setCurrentWidget(self.log_params_widget)
        elif lut_type == "exp":
            self.gen_params_stacked_widget.setCurrentWidget(self.exp_params_widget)
        elif lut_type == "sqrt":
            self.gen_params_stacked_widget.setCurrentWidget(self.sqrt_params_widget)
        elif lut_type == "rodbard":
            self.gen_params_stacked_widget.setCurrentWidget(self.rodbard_params_widget)
        else:
            # Default to an empty widget or linear if unknown
            self.gen_params_stacked_widget.setCurrentWidget(self.linear_params_widget) # Fallback

        # Update config's lut_generation_type
        self.config.lut_generation_type = lut_type
        
        # Only generate and display if the source is currently "generated"
        if self.lut_source_combo.currentText().lower() == "generated":
            self._generate_and_display_lut()


    def _generate_and_display_lut(self):
        """
        Generates a LUT based on current GUI settings (not config) and displays it
        in the text editor and plot. This is for previewing.
        It does NOT set the active LUT in lut_manager.
        """
        lut_type = self.lut_generation_type_combo.currentText()
        generated_lut = None
        try:
            # Use .text() to get current values from GUI, not config
            if lut_type == "linear":
                min_in = int(self.linear_min_input_edit.text())
                max_out = int(self.linear_max_output_edit.text())
                generated_lut = lut_manager.generate_linear_lut(min_in, max_out)
            elif lut_type == "gamma":
                gamma = float(self.gamma_value_edit.text())
                generated_lut = lut_manager.generate_gamma_lut(gamma)
            elif lut_type == "s_curve":
                contrast = float(self.s_curve_contrast_edit.text())
                generated_lut = lut_manager.generate_s_curve_lut(contrast)
            elif lut_type == "log":
                param = float(self.log_param_edit.text())
                generated_lut = lut_manager.generate_log_lut(param)
            elif lut_type == "exp":
                param = float(self.exp_param_edit.text())
                generated_lut = lut_manager.generate_exp_lut(param)
            elif lut_type == "sqrt":
                param = float(self.sqrt_param_edit.text()) # Pass param, even if not used in current lut_manager
                generated_lut = lut_manager.generate_sqrt_lut(param)
            elif lut_type == "rodbard":
                param = float(self.rodbard_param_edit.text()) # Pass param, even if not used in current lut_manager
                generated_lut = lut_manager.generate_rodbard_lut(param)
            else:
                generated_lut = lut_manager.get_default_z_lut() # Fallback

            if generated_lut is not None:
                self._load_lut_to_text_edit(generated_lut)
                # self.lut_filepath_edit.setText("") # Do not clear, this is a preview
            else:
                QMessageBox.warning(self, "LUT Generation Error", "Failed to generate LUT. Check parameters.")

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid numeric input for LUT generation: {e}")
            self.lut_text_edit.setStyleSheet("border: 1px solid red;") # Indicate error in text box
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"An error occurred during LUT generation: {e}")
            self.lut_text_edit.setStyleSheet("border: 1px solid red;")

    def _on_text_edit_changed(self):
        """Called when text in the LUT editor is changed."""
        try:
            text_values = self.lut_text_edit.toPlainText().strip().split('\n')
            text_values = [v for v in text_values if v] # Filter out empty strings

            if len(text_values) != 256:
                self.lut_text_edit.setStyleSheet("border: 1px solid orange;")
                self.canvas.axes.clear()
                self.canvas.axes.text(0.5, 0.5, "Invalid LUT Length (Expected 256)",
                                      horizontalalignment='center', verticalalignment='center',
                                      transform=self.canvas.axes.transAxes, color='red')
                self.canvas.draw()
                return

            new_lut = np.zeros(256, dtype=np.uint8)
            for i, val_str in enumerate(text_values):
                val = int(val_str)
                new_lut[i] = max(0, min(255, val)) # Clamp values

            self.current_lut = new_lut
            self.plot_lut(self.current_lut)
            self.lut_text_edit.setStyleSheet("") # Clear any warning style

        except ValueError:
            self.lut_text_edit.setStyleSheet("border: 1px solid red;")
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, "Invalid Input (Non-numeric value)",
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=self.canvas.axes.transAxes, color='red')
            self.canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "LUT Editor Error", f"An unexpected error occurred: {e}")


    def _load_lut_to_text_edit(self, lut_array: np.ndarray):
        """Populates the QTextEdit with values from a NumPy array."""
        if lut_array.dtype != np.uint8 or lut_array.shape != (256,):
            QMessageBox.critical(self, "Load Error", "Invalid LUT array format.")
            return

        # Temporarily disconnect signal to prevent re-triggering during text update
        self.lut_text_edit.textChanged.disconnect(self._on_text_edit_changed)
        
        self.lut_text_edit.setText('\n'.join(map(str, lut_array)))
        self.current_lut = lut_array.copy() # Update internal copy
        self.plot_lut(self.current_lut)
        self.lut_text_edit.setStyleSheet("") # Clear any error/warning styling

        # Reconnect the signal
        self.lut_text_edit.textChanged.connect(self._on_text_edit_changed)


    def plot_lut(self, lut_array: np.ndarray):
        """Updates the matplotlib plot with the given LUT data."""
        self.canvas.axes.clear()
        x_values = np.arange(256)
        y_values = lut_array

        self.canvas.axes.plot(x_values, y_values, 'b-')
        self.canvas.axes.set_title("Current Z-LUT Remapping Curve")
        self.canvas.axes.set_xlabel("Input Value (0-255)")
        self.canvas.axes.set_ylabel("Output Value (0-255)")
        self.canvas.axes.set_xlim(0, 255)
        self.canvas.axes.set_ylim(0, 255)
        self.canvas.axes.grid(True)
        self.canvas.draw()


    def _load_lut_from_file(self):
        """Opens a file dialog to load a LUT from a JSON file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load LUT File", self.config.fixed_lut_path or "", "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            try:
                loaded_lut = lut_manager.load_lut(filepath)
                self._load_lut_to_text_edit(loaded_lut)
                self.lut_filepath_edit.setText(filepath) # Display loaded path
                self.lut_source_combo.setCurrentText("File") # Set source to File
                QMessageBox.information(self, "Load Success", f"LUT loaded from '{filepath}'. Use 'Apply LUT to Pipeline' to make it active.")
                # Do NOT emit lut_changed here; it's only a preview until applied
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load LUT: {e}")

    def _save_lut_to_file(self):
        """Opens a file dialog to save the current LUT to a JSON file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save LUT File", self.lut_filepath_edit.text() or "custom_z_lut.json", "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            try:
                lut_manager.save_lut(filepath, self.current_lut)
                self.lut_filepath_edit.setText(filepath) # Display saved path
                self.lut_source_combo.setCurrentText("File") # Set source to File
                QMessageBox.information(self, "Save Success", f"LUT saved to '{filepath}'. Use 'Apply LUT to Pipeline' to make it active.")
                # Do NOT emit lut_changed here; it's only a save operation
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save LUT: {e}")

    def _load_lut_versions(self):
        """Populates the version combo box with saved LUT files."""
        self.version_combo.clear()
        self.version_combo.addItem("--- Select a version ---")
        lut_files = [f for f in os.listdir(LUT_VERSIONS_DIR) if f.endswith(".json")]
        for f in sorted(lut_files):
            self.version_combo.addItem(f)

    def _load_selected_version(self):
        """Loads the LUT from the selected version in the combo box."""
        selected_file = self.version_combo.currentText()
        if selected_file == "--- Select a version ---" or not selected_file:
            return

        filepath = os.path.join(LUT_VERSIONS_DIR, selected_file)
        if os.path.exists(filepath):
            try:
                loaded_lut = lut_manager.load_lut(filepath)
                self._load_lut_to_text_edit(loaded_lut)
                self.lut_filepath_edit.setText(filepath) # Display loaded path
                self.lut_source_combo.setCurrentText("File") # Set source to File
                QMessageBox.information(self, "Version Loaded", f"LUT version '{selected_file}' loaded. Use 'Apply LUT to Pipeline' to make it active.")
                # Do NOT emit lut_changed here; it's only a preview until applied
            except Exception as e:
                QMessageBox.critical(self, "Load Version Error", f"Failed to load version '{selected_file}': {e}")
        else:
            QMessageBox.warning(self, "File Not Found", f"LUT version file '{selected_file}' not found.")
            self._load_lut_versions() # Refresh list

    def _save_current_as_version(self):
        """Saves the current LUT in the editor as a new version."""
        version_name, ok = QInputDialog.getText(self, "Save LUT Version", "Enter a name for this LUT version:")
        if ok and version_name:
            # Sanitize filename
            version_name = "".join(c for c in version_name if c.isalnum() or c in (' ', '.', '_')).strip()
            if not version_name:
                QMessageBox.warning(self, "Invalid Name", "Version name cannot be empty or contain only invalid characters.")
                return

            filename = f"{version_name}.json"
            filepath = os.path.join(LUT_VERSIONS_DIR, filename)
            
            if os.path.exists(filepath):
                reply = QMessageBox.question(self, "Overwrite Version?",
                                             f"Version '{version_name}' already exists. Overwrite?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return

            try:
                lut_manager.save_lut(filepath, self.current_lut)
                self._load_lut_versions() # Refresh versions dropdown
                self.version_combo.setCurrentText(filename) # Select the newly saved version
                QMessageBox.information(self, "Save Version Success", f"Current LUT saved as version '{version_name}'.")
            except Exception as e:
                QMessageBox.critical(self, "Save Version Error", f"Failed to save LUT version: {e}")

    def _delete_selected_version(self):
        """Deletes the selected LUT version from disk."""
        selected_file = self.version_combo.currentText()
        if selected_file == "--- Select a version ---" or not selected_file:
            QMessageBox.information(self, "No Version Selected", "Please select a LUT version to delete.")
            return

        reply = QMessageBox.question(self, "Delete Version",
                                     f"Are you sure you want to delete LUT version '{selected_file}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            filepath = os.path.join(LUT_VERSIONS_DIR, selected_file)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    self._load_lut_versions() # Refresh versions dropdown
                    QMessageBox.information(self, "Delete Success", f"LUT version '{selected_file}' deleted.")
                else:
                    QMessageBox.warning(self, "File Not Found", f"LUT version file '{selected_file}' not found on disk.")
                    self._load_lut_versions() # Refresh list
            except Exception as e:
                QMessageBox.critical(self, "Delete Error", f"Failed to delete LUT version: {e}")

    def _reset_to_default_lut(self):
        """Resets the LUT in the editor and plot to the module's default."""
        reply = QMessageBox.question(self, "Reset LUT",
                                     "Are you sure you want to reset the LUT to its default values (linear 0-255)? This will not affect the active pipeline LUT until 'Apply' is clicked.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            default_lut = lut_manager.get_default_z_lut()
            self._load_lut_to_text_edit(default_lut)
            self.lut_filepath_edit.setText("") # Clear file path as it's default now
            self.lut_source_combo.setCurrentText("Generated") # Set source to Generated
            self.lut_generation_type_combo.setCurrentText("linear") # Set generation type to linear
            QMessageBox.information(self, "Reset Success", "LUT editor reset to default linear values. Click 'Apply LUT to Pipeline' to make it active.")
            # Do NOT emit lut_changed here; it's only a preview until applied


    def _apply_lut_changes(self):
        """
        Applies the current LUT from the editor to the active LUT in the lut_manager module
        and updates the config based on the selected source/parameters.
        """
        # First, ensure the text edit content is valid and parsed into current_lut
        self._on_text_edit_changed() 
        if self.lut_text_edit.styleSheet() != "": # Check if there's an error style
            QMessageBox.warning(self, "Apply Error", "Cannot apply LUT with invalid input. Please correct the values.")
            return

        try:
            # Update the config based on current GUI state
            self.config.lut_source = self.lut_source_combo.currentText().lower()
            if self.config.lut_source == "generated":
                self.config.lut_generation_type = self.lut_generation_type_combo.currentText()
                # Update all generation parameters in config from GUI edits
                self.config.gamma_value = float(self.gamma_value_edit.text()) if self.gamma_value_edit.text() else 1.0
                self.config.linear_min_input = int(self.linear_min_input_edit.text()) if self.linear_min_input_edit.text() else 0
                self.config.linear_max_output = int(self.linear_max_output_edit.text()) if self.linear_max_output_edit.text() else 255
                self.config.s_curve_contrast = float(self.s_curve_contrast_edit.text()) if self.s_curve_contrast_edit.text() else 0.5
                self.config.log_param = float(self.log_param_edit.text()) if self.log_param_edit.text() else 10.0
                self.config.exp_param = float(self.exp_param_edit.text()) if self.exp_param_edit.text() else 2.0
                self.config.sqrt_param = float(self.sqrt_param_edit.text()) if self.sqrt_param_edit.text() else 1.0
                self.config.rodbard_param = float(self.rodbard_param_edit.text()) if self.rodbard_param_edit.text() else 1.0
                self.config.fixed_lut_path = "" # Clear fixed path if now generated
            elif self.config.lut_source == "file":
                self.config.fixed_lut_path = self.lut_filepath_edit.text()
                # If file source, ensure generation parameters are reset to defaults in config
                # (though they won't be used, it's good practice for clean config state)
                self.config.lut_generation_type = "linear"
                self.config.gamma_value = 1.0
                self.config.linear_min_input = 0
                self.config.linear_max_output = 255
                self.config.s_curve_contrast = 0.5
                self.config.log_param = 10.0
                self.config.exp_param = 2.0
                self.config.sqrt_param = 1.0
                self.config.rodbard_param = 1.0
            
            # Now, tell lut_manager to update its active LUT based on the (now updated) config
            lut_manager.update_active_lut_from_config()
            
            # Finally, update the GUI to reflect the *actual* active LUT from lut_manager
            # (which might have fallen back to default if fixed_lut_path was invalid)
            self._load_lut_to_text_edit(lut_manager.get_current_z_lut())
            self.lut_source_combo.setCurrentText(self.config.lut_source.capitalize()) # Re-set in case of fallback
            self.lut_filepath_edit.setText(self.config.fixed_lut_path) # Re-set in case of fallback
            self.lut_generation_type_combo.setCurrentText(self.config.lut_generation_type) # Re-set in case of fallback

            QMessageBox.information(self, "LUT Applied", "LUT settings applied to the processing pipeline.")
            self.lut_changed.emit() # Signal that the active LUT has been updated

        except ValueError as e:
            QMessageBox.warning(self, "Apply Error", f"Invalid input for LUT parameters: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"An unexpected error occurred while applying LUT: {e}")


    def get_config(self) -> dict:
        """
        Collects current settings from this tab's widgets.
        Note: The config object is updated directly by _apply_lut_changes.
        This method primarily serves to return a dictionary representation
        of the current state of config relevant to this tab.
        """
        # The config object itself is the source of truth, updated by _apply_lut_changes.
        # We just need to return the relevant parts.
        return {
            "lut_source": self.lut_source_combo.currentText().lower(),
            "fixed_lut_path": self.lut_filepath_edit.text(),
            "lut_generation_type": self.lut_generation_type_combo.currentText(),
            "gamma_value": float(self.gamma_value_edit.text()) if self.gamma_value_edit.text() else 1.0,
            "linear_min_input": int(self.linear_min_input_edit.text()) if self.linear_min_input_edit.text() else 0,
            "linear_max_output": int(self.linear_max_output_edit.text()) if self.linear_max_output_edit.text() else 255,
            "s_curve_contrast": float(self.s_curve_contrast_edit.text()) if self.s_curve_contrast_edit.text() else 0.5,
            "log_param": float(self.log_param_edit.text()) if self.log_param_edit.text() else 10.0,
            "exp_param": float(self.exp_param_edit.text()) if self.exp_param_edit.text() else 2.0,
            "sqrt_param": float(self.sqrt_param_edit.text()) if self.sqrt_param_edit.text() else 1.0,
            "rodbard_param": float(self.rodbard_param_edit.text()) if self.rodbard_param_edit.text() else 1.0,
        }

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        # This method is called during initial GUI setup and when loading settings.
        # It should populate the GUI elements with values from the provided Config object.
        
        # Disconnect signals to prevent immediate re-triggering of logic during population
        self.lut_source_combo.currentTextChanged.disconnect(self._update_lut_source_controls)
        self.lut_generation_type_combo.currentTextChanged.disconnect(self._update_generated_lut_params_widget)
        self.lut_text_edit.textChanged.disconnect(self._on_text_edit_changed)
        
        # Set values from config to GUI elements
        self.lut_source_combo.setCurrentText(cfg.lut_source.capitalize())
        self.lut_generation_type_combo.setCurrentText(cfg.lut_generation_type)

        self.linear_min_input_edit.setText(str(cfg.linear_min_input))
        self.linear_max_output_edit.setText(str(cfg.linear_max_output))
        self.gamma_value_edit.setText(str(cfg.gamma_value))
        self.gamma_value_slider.setValue(int(cfg.gamma_value * 100))
        self.s_curve_contrast_edit.setText(str(cfg.s_curve_contrast))
        self.s_curve_contrast_slider.setValue(int(cfg.s_curve_contrast * 100))
        self.log_param_edit.setText(str(cfg.log_param))
        self.log_param_slider.setValue(int(cfg.log_param * 10))
        self.exp_param_edit.setText(str(cfg.exp_param))
        self.exp_param_slider.setValue(int(cfg.exp_param * 100))
        self.sqrt_param_edit.setText(str(cfg.sqrt_param))
        self.rodbard_param_edit.setText(str(cfg.rodbard_param))

        self.lut_filepath_edit.setText(cfg.fixed_lut_path)
        
        # Reconnect signals
        self.lut_source_combo.currentTextChanged.connect(self._update_lut_source_controls)
        self.lut_generation_type_combo.currentTextChanged.connect(self._update_generated_lut_params_widget)
        self.lut_text_edit.textChanged.connect(self._on_text_edit_changed)

        # Manually trigger updates for UI enabling/disabling based on source/type
        self.generated_lut_group.setEnabled(cfg.lut_source == "generated")
        self.file_lut_group.setEnabled(cfg.lut_source == "file")
        self._update_generated_lut_params_widget(cfg.lut_generation_type) # Just for stacking widget visibility

        # Load and plot the LUT based on the *current config state*
        # This will either load from file or generate based on cfg.lut_source and cfg.fixed_lut_path/generation_type
        # This is where the initial display of the LUT in the editor and plot happens.
        # This call will also update lut_manager's active LUT if it's different.
        lut_manager.update_active_lut_from_config() # Ensure lut_manager has the correct active LUT
        self._load_lut_to_text_edit(lut_manager.get_current_z_lut()) # Display it in the UI
