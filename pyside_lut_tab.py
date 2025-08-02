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

import lut_manager
from config import app_config as config, Config, LutConfig

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
        self.fig.tight_layout()


class LutTab(QWidget):
    """
    PySide6 tab for editing and managing Z-Axis Look-Up Tables (LUTs).
    Supports multiple named LUTs (default, receding, overhang).
    """
    lut_changed = Signal()

    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config
        
        self._setup_ui()
        self._connect_signals()
        
        self.apply_settings(self.config) # Applies full config state to UI
        self._load_lut_versions()
        self._load_state_for_target(self._get_current_target_name()) # Initial UI sync

    def _setup_ui(self):
        """Sets up the widgets and layout for the LUT tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- NEW: LUT Target Selection ---
        target_group = QGroupBox("LUT Target")
        target_layout = QHBoxLayout(target_group)
        target_layout.addWidget(QLabel("Editing LUT for:"))
        self.lut_target_combo = QComboBox()
        self.lut_target_combo.addItems(["Default", "Receding Gradient", "Overhang Gradient"])
        target_layout.addWidget(self.lut_target_combo)
        target_layout.addStretch(1)
        main_layout.addWidget(target_group)

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
        self.generated_lut_group = QGroupBox("Generated LUT Parameters")
        generated_lut_layout = QVBoxLayout(self.generated_lut_group)
        
        gen_type_layout = QHBoxLayout()
        gen_type_layout.addWidget(QLabel("Type:"))
        self.lut_generation_type_combo = QComboBox()
        self.lut_generation_type_combo.addItems(["linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard"])
        gen_type_layout.addWidget(self.lut_generation_type_combo)
        gen_type_layout.addStretch(1)
        generated_lut_layout.addLayout(gen_type_layout)

        self.op_params_stacked_widget = QStackedWidget()
        self._create_param_widgets() # Helper to create all param widgets
        generated_lut_layout.addWidget(self.op_params_stacked_widget)

        generate_btn_layout = QHBoxLayout()
        generate_btn_layout.addStretch(1)
        self.generate_lut_button = QPushButton("Generate & Preview LUT")
        generate_btn_layout.addWidget(self.generate_lut_button)
        generate_btn_layout.addStretch(1)
        generated_lut_layout.addLayout(generate_btn_layout)
        main_layout.addWidget(self.generated_lut_group)

        # --- File-based LUT Controls ---
        self.file_lut_group = QGroupBox("File-based LUT Controls")
        file_lut_layout = QVBoxLayout(self.file_lut_group)
        
        lut_path_layout = QHBoxLayout()
        lut_path_layout.addWidget(QLabel("Loaded File:"))
        self.lut_filepath_edit = QLineEdit()
        self.lut_filepath_edit.setReadOnly(True)
        lut_path_layout.addWidget(self.lut_filepath_edit)
        file_lut_layout.addLayout(lut_path_layout)
        
        file_button_layout = QHBoxLayout()
        self.load_file_button = QPushButton("Load from File...")
        file_button_layout.addWidget(self.load_file_button)
        self.save_file_button = QPushButton("Save Current to File...")
        file_button_layout.addWidget(self.save_file_button)
        file_button_layout.addStretch(1)
        file_lut_layout.addLayout(file_button_layout)
        main_layout.addWidget(self.file_lut_group)

        # --- LUT Editor and Plot ---
        editor_plot_group = QGroupBox("Current LUT Preview")
        editor_plot_layout = QHBoxLayout(editor_plot_group)
        
        lut_editor_layout = QVBoxLayout()
        lut_editor_layout.addWidget(QLabel("LUT Values (0-255):"))
        self.lut_text_edit = QTextEdit()
        self.lut_text_edit.setMinimumWidth(150)
        self.lut_text_edit.setMaximumWidth(200)
        lut_editor_layout.addWidget(self.lut_text_edit)
        editor_plot_layout.addLayout(lut_editor_layout)

        self.canvas = MplCanvas(self)
        editor_plot_layout.addWidget(self.canvas)
        main_layout.addWidget(editor_plot_group)

        # --- Global Controls ---
        global_controls_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset Current LUT to Default")
        global_controls_layout.addWidget(self.reset_button)
        global_controls_layout.addStretch(1)
        self.apply_button = QPushButton("Apply All LUTs to Pipeline")
        self.apply_button.setStyleSheet("font-weight: bold;")
        global_controls_layout.addWidget(self.apply_button)
        main_layout.addLayout(global_controls_layout)

        main_layout.addStretch(1)

    def _create_param_widgets(self):
        """Helper method to create all the parameter widgets for the stacked widget."""
        # Linear
        self.linear_params_widget = QWidget()
        linear_layout = QHBoxLayout(self.linear_params_widget)
        linear_layout.addWidget(QLabel("Min Input:"))
        self.linear_min_input_edit = QLineEdit("0", validator=QIntValidator(0, 255, self))
        linear_layout.addWidget(self.linear_min_input_edit)
        linear_layout.addWidget(QLabel("Max Output:"))
        self.linear_max_output_edit = QLineEdit("255", validator=QIntValidator(0, 255, self))
        linear_layout.addWidget(self.linear_max_output_edit)
        linear_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.linear_params_widget)

        # Gamma
        self.gamma_params_widget = QWidget()
        gamma_layout = QHBoxLayout(self.gamma_params_widget)
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_value_edit = QLineEdit("1.0", validator=QDoubleValidator(0.01, 10.0, 2, self))
        gamma_layout.addWidget(self.gamma_value_edit)
        self.gamma_value_slider = QSlider(Qt.Horizontal, minimum=1, maximum=1000, value=100)
        gamma_layout.addWidget(self.gamma_value_slider)
        self.op_params_stacked_widget.addWidget(self.gamma_params_widget)

        # S-Curve
        self.s_curve_params_widget = QWidget()
        s_curve_layout = QHBoxLayout(self.s_curve_params_widget)
        s_curve_layout.addWidget(QLabel("Contrast:"))
        self.s_curve_contrast_edit = QLineEdit("0.5", validator=QDoubleValidator(0.0, 1.0, 2, self))
        s_curve_layout.addWidget(self.s_curve_contrast_edit)
        self.s_curve_contrast_slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=50)
        s_curve_layout.addWidget(self.s_curve_contrast_slider)
        self.op_params_stacked_widget.addWidget(self.s_curve_params_widget)
        
        # Add other param widgets similarly...
        # Log
        self.log_params_widget = QWidget()
        log_layout = QHBoxLayout(self.log_params_widget)
        log_layout.addWidget(QLabel("Param:"))
        self.log_param_edit = QLineEdit("10.0", validator=QDoubleValidator(0.01, 100.0, 2, self))
        log_layout.addWidget(self.log_param_edit)
        self.log_param_slider = QSlider(Qt.Horizontal, minimum=1, maximum=1000, value=100)
        log_layout.addWidget(self.log_param_slider)
        self.op_params_stacked_widget.addWidget(self.log_params_widget)

        # Exp
        self.exp_params_widget = QWidget()
        exp_layout = QHBoxLayout(self.exp_params_widget)
        exp_layout.addWidget(QLabel("Param:"))
        self.exp_param_edit = QLineEdit("2.0", validator=QDoubleValidator(0.01, 10.0, 2, self))
        exp_layout.addWidget(self.exp_param_edit)
        self.exp_param_slider = QSlider(Qt.Horizontal, minimum=1, maximum=1000, value=200)
        exp_layout.addWidget(self.exp_param_slider)
        self.op_params_stacked_widget.addWidget(self.exp_params_widget)

        # Sqrt
        self.sqrt_params_widget = QWidget()
        sqrt_layout = QHBoxLayout(self.sqrt_params_widget)
        sqrt_layout.addWidget(QLabel("Param (unused):"))
        self.sqrt_param_edit = QLineEdit("1.0", validator=QDoubleValidator(0.01, 10.0, 2, self))
        sqrt_layout.addWidget(self.sqrt_param_edit)
        self.op_params_stacked_widget.addWidget(self.sqrt_params_widget)

        # Rodbard
        self.rodbard_params_widget = QWidget()
        rodbard_layout = QHBoxLayout(self.rodbard_params_widget)
        rodbard_layout.addWidget(QLabel("Param (unused):"))
        self.rodbard_param_edit = QLineEdit("1.0", validator=QDoubleValidator(0.01, 10.0, 2, self))
        rodbard_layout.addWidget(self.rodbard_param_edit)
        self.op_params_stacked_widget.addWidget(self.rodbard_params_widget)

    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.lut_target_combo.currentTextChanged.connect(self._on_lut_target_changed)
        self.lut_source_combo.currentTextChanged.connect(self._on_lut_source_changed)
        self.lut_generation_type_combo.currentTextChanged.connect(self._on_gen_type_changed)

        # Connect sliders and edits
        self.gamma_value_slider.valueChanged.connect(lambda val: self.gamma_value_edit.setText(f"{val / 100.0:.2f}"))
        self.gamma_value_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.gamma_value_slider, text, 100.0))
        self.s_curve_contrast_slider.valueChanged.connect(lambda val: self.s_curve_contrast_edit.setText(f"{val / 100.0:.2f}"))
        self.s_curve_contrast_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.s_curve_contrast_slider, text, 100.0))
        self.log_param_slider.valueChanged.connect(lambda val: self.log_param_edit.setText(f"{val / 10.0:.1f}"))
        self.log_param_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.log_param_slider, text, 10.0))
        self.exp_param_slider.valueChanged.connect(lambda val: self.exp_param_edit.setText(f"{val / 100.0:.2f}"))
        self.exp_param_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.exp_param_slider, text, 100.0))
        
        self.generate_lut_button.clicked.connect(self._generate_and_display_lut)
        self.load_file_button.clicked.connect(self._load_lut_from_file)
        self.save_file_button.clicked.connect(self._save_lut_to_file)
        
        self.reset_button.clicked.connect(self._reset_to_default_lut)
        self.apply_button.clicked.connect(self._apply_all_lut_changes)

    def _get_current_target_name(self) -> str:
        """Gets the config-compatible name of the currently selected LUT target."""
        text = self.lut_target_combo.currentText()
        return text.split(' ')[0].lower() # "Default" -> "default", "Receding Gradient" -> "receding"

    def _on_lut_target_changed(self):
        """Handles switching the UI to edit a different named LUT."""
        target_name = self._get_current_target_name()
        self._load_state_for_target(target_name)

    def _load_state_for_target(self, target_name: str):
        """Loads the config and active LUT for the given target into the UI."""
        lut_config = self.config.lut_settings.get(target_name)
        if not lut_config:
            return

        # Block signals to prevent feedback loops
        self._block_all_signals(True)

        # Update UI from the target's LutConfig
        self.lut_source_combo.setCurrentText(lut_config.source.capitalize())
        self.lut_generation_type_combo.setCurrentText(lut_config.generation_type)
        self.lut_filepath_edit.setText(lut_config.fixed_path)
        
        # Populate parameter edits
        self.linear_min_input_edit.setText(str(lut_config.linear_min_input))
        self.linear_max_output_edit.setText(str(lut_config.linear_max_output))
        self.gamma_value_edit.setText(str(lut_config.gamma_value))
        self.s_curve_contrast_edit.setText(str(lut_config.s_curve_contrast))
        self.log_param_edit.setText(str(lut_config.log_param))
        self.exp_param_edit.setText(str(lut_config.exp_param))
        self.sqrt_param_edit.setText(str(lut_config.sqrt_param))
        self.rodbard_param_edit.setText(str(lut_config.rodbard_param))

        # Update sliders
        self.gamma_value_slider.setValue(int(lut_config.gamma_value * 100))
        self.s_curve_contrast_slider.setValue(int(lut_config.s_curve_contrast * 100))
        self.log_param_slider.setValue(int(lut_config.log_param * 10))
        self.exp_param_slider.setValue(int(lut_config.exp_param * 100))
        
        # Update UI visibility
        self._update_ui_visibility()

        # Load the actual active LUT from the manager into the editor/plot
        active_lut = lut_manager.get_current_z_lut(target_name)
        self._load_lut_to_text_edit(active_lut)

        self._block_all_signals(False)

    def _on_lut_source_changed(self):
        """Updates UI visibility when the source (Generated/File) changes."""
        self._update_ui_visibility()
        # When switching to 'generated', immediately preview the result
        if self.lut_source_combo.currentText() == "Generated":
            self._generate_and_display_lut()

    def _on_gen_type_changed(self):
        """Updates the stacked widget and previews the generated LUT."""
        self._update_ui_visibility()
        self._generate_and_display_lut()

    def _update_ui_visibility(self):
        """Central function to set visibility of all dynamic controls."""
        is_generated = self.lut_source_combo.currentText() == "Generated"
        self.generated_lut_group.setVisible(is_generated)
        self.file_lut_group.setVisible(not is_generated)

        if is_generated:
            gen_type = self.lut_generation_type_combo.currentText()
            widget_map = {
                "linear": self.linear_params_widget, "gamma": self.gamma_params_widget,
                "s_curve": self.s_curve_params_widget, "log": self.log_params_widget,
                "exp": self.exp_params_widget, "sqrt": self.sqrt_params_widget,
                "rodbard": self.rodbard_params_widget
            }
            self.op_params_stacked_widget.setCurrentWidget(widget_map.get(gen_type))

    def _generate_and_display_lut(self):
        """Generates a LUT based on current GUI settings and displays it for preview."""
        try:
            lut_type = self.lut_generation_type_combo.currentText()
            # Generate LUT using values directly from UI widgets
            if lut_type == "linear":
                lut = lut_manager.generate_linear_lut(int(self.linear_min_input_edit.text()), int(self.linear_max_output_edit.text()))
            elif lut_type == "gamma":
                lut = lut_manager.generate_gamma_lut(float(self.gamma_value_edit.text()))
            elif lut_type == "s_curve":
                lut = lut_manager.generate_s_curve_lut(float(self.s_curve_contrast_edit.text()))
            elif lut_type == "log":
                lut = lut_manager.generate_log_lut(float(self.log_param_edit.text()))
            elif lut_type == "exp":
                lut = lut_manager.generate_exp_lut(float(self.exp_param_edit.text()))
            elif lut_type == "sqrt":
                lut = lut_manager.generate_sqrt_lut(float(self.sqrt_param_edit.text()))
            elif lut_type == "rodbard":
                lut = lut_manager.generate_rodbard_lut(float(self.rodbard_param_edit.text()))
            else:
                lut = lut_manager.get_default_z_lut()
            
            self._load_lut_to_text_edit(lut)
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "Input Error", f"Invalid numeric input for LUT generation: {e}")

    def _load_lut_to_text_edit(self, lut_array: np.ndarray):
        """Populates the QTextEdit and plot with values from a NumPy array."""
        self.lut_text_edit.blockSignals(True)
        self.lut_text_edit.setText('\n'.join(map(str, lut_array)))
        self.lut_text_edit.blockSignals(False)
        self.plot_lut(lut_array)

    def plot_lut(self, lut_array: np.ndarray):
        """Updates the matplotlib plot with the given LUT data."""
        self.canvas.axes.clear()
        self.canvas.axes.plot(np.arange(256), lut_array, 'b-')
        self.canvas.axes.set_title(f"{self.lut_target_combo.currentText()} LUT Curve")
        self.canvas.axes.set_xlabel("Input Value")
        self.canvas.axes.set_ylabel("Output Value")
        self.canvas.axes.set_xlim(0, 255)
        self.canvas.axes.set_ylim(0, 255)
        self.canvas.axes.grid(True)
        self.canvas.draw()

    def _load_lut_from_file(self):
        """Loads a LUT from a JSON file into the current target's UI state."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load LUT File", "", "JSON Files (*.json)")
        if filepath:
            try:
                loaded_lut = lut_manager.load_lut(filepath)
                self._load_lut_to_text_edit(loaded_lut)
                self.lut_filepath_edit.setText(filepath)
                QMessageBox.information(self, "Load Success", "LUT loaded for preview. Click 'Apply' to save this change.")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load LUT: {e}")

    def _save_lut_to_file(self):
        """Saves the LUT currently in the text editor to a JSON file."""
        try:
            current_lut = self._get_lut_from_text_edit()
            filepath, _ = QFileDialog.getSaveFileName(self, "Save LUT File", "", "JSON Files (*.json)")
            if filepath:
                lut_manager.save_lut(filepath, current_lut)
                self.lut_filepath_edit.setText(filepath)
                QMessageBox.information(self, "Save Success", f"LUT saved to '{filepath}'.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save LUT: {e}")

    def _reset_to_default_lut(self):
        """Resets the currently selected LUT target to the default linear curve."""
        self._load_lut_to_text_edit(lut_manager.get_default_z_lut())
        self.lut_source_combo.setCurrentText("Generated")
        self.lut_generation_type_combo.setCurrentText("linear")
        QMessageBox.information(self, "Reset", "Current LUT preview has been reset. Click 'Apply' to save this change.")

    def _apply_all_lut_changes(self):
        """Saves the current state of the UI to the config, then updates the active LUTs."""
        try:
            # First, save the currently displayed UI state to its corresponding config object
            self._save_current_ui_to_config()
            
            # Now, tell lut_manager to update all its active LUTs from the entire config
            lut_manager.update_active_lut_from_config()
            
            QMessageBox.information(self, "LUTs Applied", "All LUT settings have been applied to the processing pipeline.")
            self.lut_changed.emit()
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"An error occurred while applying LUTs: {e}")

    def _save_current_ui_to_config(self):
        """Reads the current UI state and saves it to the appropriate LutConfig object."""
        target_name = self._get_current_target_name()
        lut_config = self.config.lut_settings.get(target_name)
        if not lut_config: return

        lut_config.source = self.lut_source_combo.currentText().lower()
        lut_config.fixed_path = self.lut_filepath_edit.text()
        lut_config.generation_type = self.lut_generation_type_combo.currentText()
        
        # Save all params, converting text to the correct type
        lut_config.linear_min_input = int(self.linear_min_input_edit.text())
        lut_config.linear_max_output = int(self.linear_max_output_edit.text())
        lut_config.gamma_value = float(self.gamma_value_edit.text())
        lut_config.s_curve_contrast = float(self.s_curve_contrast_edit.text())
        lut_config.log_param = float(self.log_param_edit.text())
        lut_config.exp_param = float(self.exp_param_edit.text())
        lut_config.sqrt_param = float(self.sqrt_param_edit.text())
        lut_config.rodbard_param = float(self.rodbard_param_edit.text())

    def get_config(self) -> dict:
        """
        Collects settings by first saving the current UI state to the main config object,
        then returning the relevant dictionary from it. This ensures consistency.
        """
        self._save_current_ui_to_config()
        return {"lut_settings": {name: lconf.to_dict() for name, lconf in self.config.lut_settings.items()}}

    def apply_settings(self, cfg: Config):
        """Applies settings from a loaded Config object to this tab's widgets."""
        self.config = cfg # Ensure we're using the newly loaded config object
        # The UI will be synced to the currently selected target. If the user wants to see
        # the state of other LUTs, they can switch the target combo box.
        self._load_state_for_target(self._get_current_target_name())

    def _get_lut_from_text_edit(self) -> np.ndarray:
        """Parses the text editor and returns a valid LUT array, raising errors if invalid."""
        text_values = self.lut_text_edit.toPlainText().strip().split('\n')
        if len(text_values) != 256:
            raise ValueError("LUT must contain exactly 256 values.")
        
        lut_array = np.array([int(v) for v in text_values], dtype=np.uint8)
        return np.clip(lut_array, 0, 255)

    def _update_slider_from_edit(self, slider: QSlider, text: str, scale_factor: float):
        """Updates a slider's value based on a QLineEdit's text."""
        try:
            val = float(text)
            slider.setValue(int(val * scale_factor))
        except (ValueError, TypeError):
            pass # Ignore invalid text, validator will handle visual feedback

    def _block_all_signals(self, block: bool):
        """Blocks or unblocks signals for all interactive widgets to prevent feedback loops."""
        widgets = [
            self.lut_target_combo, self.lut_source_combo, self.lut_generation_type_combo,
            self.linear_min_input_edit, self.linear_max_output_edit,
            self.gamma_value_edit, self.gamma_value_slider,
            self.s_curve_contrast_edit, self.s_curve_contrast_slider,
            self.log_param_edit, self.log_param_slider,
            self.exp_param_edit, self.exp_param_slider,
            self.sqrt_param_edit, self.rodbard_param_edit
        ]
        for widget in widgets:
            widget.blockSignals(block)
