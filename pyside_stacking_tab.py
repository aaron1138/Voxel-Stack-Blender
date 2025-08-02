# pyside_stacking_tab.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QSlider, QSpinBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator

from config import app_config as config, Config # Import Config class explicitly

class StackingTab(QWidget):
    """
    PySide6 tab for Z-axis Stacking (Blend) parameters.
    """
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config # Use the global config instance

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config
        self._apply_initial_state() # Ensure initial control states are correct

    def _setup_ui(self):
        """Sets up the widgets and layout for the Stacking tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Stacking (Blend) Group ---
        blend_group = QGroupBox("Z-Axis Stacking (Blend)")
        blend_layout = QVBoxLayout(blend_group)
        blend_layout.setContentsMargins(10, 10, 10, 10)
        blend_layout.setSpacing(5)

        # Primary and Radius
        primary_radius_layout = QHBoxLayout()
        primary_radius_layout.addWidget(QLabel("Primary Layers:"))
        self.primary_edit = QLineEdit()
        self.primary_edit.setFixedWidth(60)
        self.primary_edit.setValidator(QIntValidator(1, 999999, self))
        primary_radius_layout.addWidget(self.primary_edit)

        primary_radius_layout.addWidget(QLabel("Radius:"))
        self.radius_edit = QLineEdit()
        self.radius_edit.setFixedWidth(60)
        self.radius_edit.setValidator(QIntValidator(0, 999999, self))
        primary_radius_layout.addWidget(self.radius_edit)
        primary_radius_layout.addStretch(1)
        blend_layout.addLayout(primary_radius_layout)

        # Blend Mode and Parameter
        blend_mode_param_layout = QHBoxLayout()
        blend_mode_param_layout.addWidget(QLabel("Blend Mode:"))
        self.blend_mode_combo = QComboBox()
        # Ensure these match the blend_mode types in stacking_processor.py
        modes = [
            "gaussian", "linear", "cosine", "exp_decay", "flat",
            "binary_contour", "gradient_contour",
            "z_column_lift", "z_contour_interp",
            # Zarr-backed modes are not implemented in this version, so exclude for now
            # "zarr_binary_shadow", "zarr_binary_overhang", "zarr_binary_contour", "zarr_gradient_contour"
        ]
        self.blend_mode_combo.addItems(modes)
        blend_mode_param_layout.addWidget(self.blend_mode_combo)

        blend_mode_param_layout.addWidget(QLabel("σ / Param:"))
        self.blend_param_edit = QLineEdit()
        self.blend_param_edit.setFixedWidth(60)
        self.blend_param_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        blend_mode_param_layout.addWidget(self.blend_param_edit)

        self.blend_param_slider = QSlider(Qt.Horizontal)
        self.blend_param_slider.setRange(0, 1000) # Scale to allow 0.0 to 100.0 with 2 decimal places
        self.blend_param_slider.setSingleStep(1)
        self.blend_param_slider.setPageStep(10)
        self.blend_param_slider.setTickInterval(100)
        self.blend_param_slider.setTickPosition(QSlider.TicksBelow)
        blend_mode_param_layout.addWidget(self.blend_param_slider)
        blend_layout.addLayout(blend_mode_param_layout)

        # Directional Blend (Z-Bias)
        zbias_layout = QHBoxLayout()
        self.zbias_check = QCheckBox("Directional Blend (Z-Bias)")
        zbias_layout.addWidget(self.zbias_check)

        zbias_layout.addWidget(QLabel("Dir σ:"))
        self.dir_sigma_edit = QLineEdit()
        self.dir_sigma_edit.setFixedWidth(60)
        self.dir_sigma_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        zbias_layout.addWidget(self.dir_sigma_edit)

        self.dir_sigma_slider = QSlider(Qt.Horizontal)
        self.dir_sigma_slider.setRange(0, 1000) # Scale to allow 0.0 to 100.0 with 2 decimal places
        self.dir_sigma_slider.setSingleStep(1)
        self.dir_sigma_slider.setPageStep(10)
        self.dir_sigma_slider.setTickInterval(100)
        self.dir_sigma_slider.setTickPosition(QSlider.TicksBelow)
        zbias_layout.addWidget(self.dir_sigma_slider)
        zbias_layout.addStretch(1)
        blend_layout.addLayout(zbias_layout)

        # Binary/Gradient Thresholds
        thresholds_layout = QHBoxLayout()
        thresholds_layout.addWidget(QLabel("Binary Thres:"))
        self.binary_threshold_edit = QLineEdit()
        self.binary_threshold_edit.setFixedWidth(60)
        self.binary_threshold_edit.setValidator(QIntValidator(0, 255, self))
        thresholds_layout.addWidget(self.binary_threshold_edit)

        thresholds_layout.addWidget(QLabel("Grad Thres:"))
        self.gradient_threshold_edit = QLineEdit()
        self.gradient_threshold_edit.setFixedWidth(60)
        self.gradient_threshold_edit.setValidator(QIntValidator(0, 255, self))
        thresholds_layout.addWidget(self.gradient_threshold_edit)
        thresholds_layout.addStretch(1)
        blend_layout.addLayout(thresholds_layout)

        main_layout.addWidget(blend_group)
        main_layout.addStretch(1)


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.blend_mode_combo.currentTextChanged.connect(self._update_blend_mode_controls)
        
        # Blend Param slider and edit
        self.blend_param_slider.valueChanged.connect(lambda val: self.blend_param_edit.setText(f"{val / 10.0:.1f}"))
        self.blend_param_edit.textChanged.connect(self._update_blend_param_from_edit)

        # Directional Blend checkbox and Dir Sigma slider/edit
        self.zbias_check.stateChanged.connect(self._update_zbias_controls)
        self.dir_sigma_slider.valueChanged.connect(lambda val: self.dir_sigma_edit.setText(f"{val / 10.0:.1f}"))
        self.dir_sigma_edit.textChanged.connect(self._update_dir_sigma_from_edit)

        # Input validation for numeric fields (using textChanged for immediate feedback)
        self.primary_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.primary_edit, text, int))
        self.radius_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.radius_edit, text, int))
        self.binary_threshold_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.binary_threshold_edit, text, int))
        self.gradient_threshold_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.gradient_threshold_edit, text, int))


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

    def _update_blend_param_from_edit(self, text):
        try:
            val = float(text)
            # Scale float value back to slider range (0-100.0 -> 0-1000)
            self.blend_param_slider.setValue(int(val * 10))
        except ValueError:
            pass # Let validator handle invalid input visually

    def _update_dir_sigma_from_edit(self, text):
        try:
            val = float(text)
            # Scale float value back to slider range (0-100.0 -> 0-1000)
            self.dir_sigma_slider.setValue(int(val * 10))
        except ValueError:
            pass # Let validator handle invalid input visually

    def _apply_initial_state(self):
        """Applies initial control states based on current config values."""
        self._update_blend_mode_controls(self.config.blend_mode)
        self._update_zbias_controls(Qt.Checked if self.config.directional_blend else Qt.Unchecked)

    def _update_blend_mode_controls(self, mode: str):
        """
        Enables/disables blend parameter controls based on the selected blend mode.
        Also manages visibility of binary/gradient thresholds.
        """
        # Modes that use blend_param (sigma/exponent)
        param_enabled_modes = ["gaussian", "linear", "cosine", "exp_decay"] # Flat doesn't use param
        disable_param = mode not in param_enabled_modes
        
        self.blend_param_edit.setEnabled(not disable_param)
        self.blend_param_slider.setEnabled(not disable_param)

        # Modes that allow directional blend (Z-Bias)
        zbias_enabled_modes = ["gaussian", "linear", "cosine", "exp_decay"]
        enable_zbias_checkbox = mode in zbias_enabled_modes
        self.zbias_check.setEnabled(enable_zbias_checkbox)
        
        # Dir Sigma controls are enabled only if Z-Bias checkbox is checked AND blend mode allows it
        enable_dir_sigma_controls = enable_zbias_checkbox and self.zbias_check.isChecked()
        self.dir_sigma_edit.setEnabled(enable_dir_sigma_controls)
        self.dir_sigma_slider.setEnabled(enable_dir_sigma_controls)

        # Modes that use binary_threshold
        binary_threshold_modes = ["binary_contour"]
        enable_binary_threshold = mode in binary_threshold_modes
        self.binary_threshold_edit.setEnabled(enable_binary_threshold)

        # Modes that use gradient_threshold
        gradient_threshold_modes = ["gradient_contour"]
        enable_gradient_threshold = mode in gradient_threshold_modes
        self.gradient_threshold_edit.setEnabled(enable_gradient_threshold)

        # Special handling for z_column_lift and z_contour_interp: they don't use blend_param
        # but their logic in stacking_processor might implicitly use blend_param for internal kernel generation
        # or other config values. For GUI, we disable blend_param for them.
        if mode in ["z_column_lift", "z_contour_interp"]:
            self.blend_param_edit.setEnabled(False)
            self.blend_param_slider.setEnabled(False)
            self.zbias_check.setEnabled(False) # No Z-bias for these modes
            self.dir_sigma_edit.setEnabled(False)
            self.dir_sigma_slider.setEnabled(False)


    def _update_zbias_controls(self, state: int):
        """Enables/disables directional sigma controls based on checkbox state."""
        is_checked = (state == Qt.Checked)
        # Only enable if the current blend mode also allows Z-bias
        blend_mode_allows_zbias = self.blend_mode_combo.currentText() in ["gaussian", "linear", "cosine", "exp_decay"]
        
        enable_dir_sigma = is_checked and blend_mode_allows_zbias
        self.dir_sigma_edit.setEnabled(enable_dir_sigma)
        self.dir_sigma_slider.setEnabled(enable_dir_sigma)

    def get_config(self) -> dict:
        """Collects current settings from this tab's widgets."""
        config_data = {}
        config_data["primary"] = int(self.primary_edit.text()) if self.primary_edit.text() else 3
        config_data["radius"] = int(self.radius_edit.text()) if self.radius_edit.text() else 1
        config_data["blend_mode"] = self.blend_mode_combo.currentText()
        config_data["blend_param"] = float(self.blend_param_edit.text()) if self.blend_param_edit.text() else 1.0
        config_data["directional_blend"] = self.zbias_check.isChecked()
        config_data["dir_sigma"] = float(self.dir_sigma_edit.text()) if self.dir_sigma_edit.text() else 1.0
        config_data["binary_threshold"] = int(self.binary_threshold_edit.text()) if self.binary_threshold_edit.text() else 128
        config_data["gradient_threshold"] = int(self.gradient_threshold_edit.text()) if self.gradient_threshold_edit.text() else 128
        
        return config_data

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        self.primary_edit.setText(str(cfg.primary))
        self.radius_edit.setText(str(cfg.radius))
        self.blend_mode_combo.setCurrentText(cfg.blend_mode)
        self.blend_param_edit.setText(str(cfg.blend_param))
        self.blend_param_slider.setValue(int(cfg.blend_param * 10)) # Scale for slider
        self.zbias_check.setChecked(cfg.directional_blend)
        self.dir_sigma_edit.setText(str(cfg.dir_sigma))
        self.dir_sigma_slider.setValue(int(cfg.dir_sigma * 10)) # Scale for slider
        self.binary_threshold_edit.setText(str(cfg.binary_threshold))
        self.gradient_threshold_edit.setText(str(cfg.gradient_threshold))

        # Ensure dynamic controls are updated after setting values
        self._update_blend_mode_controls(cfg.blend_mode)
        self._update_zbias_controls(Qt.Checked if cfg.directional_blend else Qt.Unchecked)

