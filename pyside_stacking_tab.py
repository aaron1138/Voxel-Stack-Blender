from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QSlider, QSpinBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator

from config import app_config as config, Config

class StackingTab(QWidget):
    """
    PySide6 tab for Z-axis Stacking (Blend) parameters, including new vertical blend modes
    and flexible LUT application controls.
    """
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config)

    def _setup_ui(self):
        """Sets up the widgets and layout for the Stacking tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Vertical Blending Group ---
        self.vertical_blend_group = QGroupBox("Vertical Blending")
        self.vertical_blend_group.setCheckable(True) # This is the main toggle
        vertical_blend_layout = QVBoxLayout(self.vertical_blend_group)

        # Mode and LUT controls
        vb_mode_layout = QHBoxLayout()
        self.vb_preprocess_check = QCheckBox("Run as Pre-processor (before stacking)")
        vb_mode_layout.addWidget(self.vb_preprocess_check)
        vb_mode_layout.addStretch(1)
        self.apply_vertical_luts_check = QCheckBox("Apply Receding/Overhang LUTs")
        vb_mode_layout.addWidget(self.apply_vertical_luts_check)
        vertical_blend_layout.addLayout(vb_mode_layout)

        # Receding controls
        receding_layout = QHBoxLayout()
        receding_layout.addWidget(QLabel("Receding Layers:"))
        self.vert_receding_layers_spin = QSpinBox(minimum=0, maximum=100)
        receding_layout.addWidget(self.vert_receding_layers_spin)
        receding_layout.addWidget(QLabel("Fade Distance (px):"))
        self.vert_receding_fade_edit = QLineEdit(validator=QDoubleValidator(0.0, 999.0, 2, self))
        self.vert_receding_fade_edit.setFixedWidth(60)
        receding_layout.addWidget(self.vert_receding_fade_edit)
        receding_layout.addStretch(1)
        vertical_blend_layout.addLayout(receding_layout)
        
        # Overhang controls
        overhang_layout = QHBoxLayout()
        overhang_layout.addWidget(QLabel("Overhang Layers:"))
        self.vert_overhang_layers_spin = QSpinBox(minimum=0, maximum=100)
        overhang_layout.addWidget(self.vert_overhang_layers_spin)
        overhang_layout.addWidget(QLabel("Fade Distance (px):"))
        self.vert_overhang_fade_edit = QLineEdit(validator=QDoubleValidator(0.0, 999.0, 2, self))
        self.vert_overhang_fade_edit.setFixedWidth(60)
        overhang_layout.addWidget(self.vert_overhang_fade_edit)
        overhang_layout.addStretch(1)
        vertical_blend_layout.addLayout(overhang_layout)
        
        # --- NEW: Normalization and Gamma controls ---
        norm_gamma_layout = QHBoxLayout()
        self.use_fixed_fade_check = QCheckBox("Use Fixed Fade Distance")
        self.use_fixed_fade_check.setToolTip("If checked, gradient fades over a fixed distance, preventing a 'halo' on small shapes.")
        norm_gamma_layout.addWidget(self.use_fixed_fade_check)
        norm_gamma_layout.addStretch(1)
        norm_gamma_layout.addWidget(QLabel("Vertical Gamma:"))
        self.vertical_gamma_edit = QLineEdit(validator=QDoubleValidator(0.01, 10.0, 2, self))
        self.vertical_gamma_edit.setToolTip("Gamma value to control the fade profile of the gradient. < 1.0 makes the fade more gradual.")
        self.vertical_gamma_edit.setFixedWidth(60)
        norm_gamma_layout.addWidget(self.vertical_gamma_edit)
        vertical_blend_layout.addLayout(norm_gamma_layout)

        main_layout.addWidget(self.vertical_blend_group)

        # --- Stacking (Blend) Group ---
        self.stacking_group = QGroupBox("Standard Z-Axis Stacking")
        stacking_layout = QVBoxLayout(self.stacking_group)
        
        # LUT control for stacking
        stack_lut_layout = QHBoxLayout()
        stack_lut_layout.addStretch(1)
        self.apply_default_lut_check = QCheckBox("Apply Default LUT After Stacking")
        stack_lut_layout.addWidget(self.apply_default_lut_check)
        stacking_layout.addLayout(stack_lut_layout)

        # Primary and Radius
        primary_radius_layout = QHBoxLayout()
        primary_radius_layout.addWidget(QLabel("Primary Layers:"))
        self.primary_edit = QLineEdit(validator=QIntValidator(1, 999999, self))
        self.primary_edit.setFixedWidth(60)
        primary_radius_layout.addWidget(self.primary_edit)
        primary_radius_layout.addWidget(QLabel("Radius:"))
        self.radius_edit = QLineEdit(validator=QIntValidator(0, 999999, self))
        self.radius_edit.setFixedWidth(60)
        primary_radius_layout.addWidget(self.radius_edit)
        primary_radius_layout.addStretch(1)
        stacking_layout.addLayout(primary_radius_layout)

        # Blend Mode and Parameter
        blend_mode_param_layout = QHBoxLayout()
        blend_mode_param_layout.addWidget(QLabel("Blend Mode:"))
        self.blend_mode_combo = QComboBox()
        modes = [
            "gaussian", "linear", "cosine", "exp_decay", "flat",
            "binary_contour", "gradient_contour",
            "z_column_lift", "z_contour_interp",
        ]
        self.blend_mode_combo.addItems(modes)
        blend_mode_param_layout.addWidget(self.blend_mode_combo)
        self.blend_param_label = QLabel("σ / Param:")
        blend_mode_param_layout.addWidget(self.blend_param_label)
        self.blend_param_edit = QLineEdit(validator=QDoubleValidator(0.0, 100.0, 2, self))
        self.blend_param_edit.setFixedWidth(60)
        blend_mode_param_layout.addWidget(self.blend_param_edit)
        blend_mode_param_layout.addStretch(1)
        stacking_layout.addLayout(blend_mode_param_layout)

        main_layout.addWidget(self.stacking_group)
        main_layout.addStretch(1)

    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.vertical_blend_group.toggled.connect(self._update_controls_visibility)
        self.vb_preprocess_check.stateChanged.connect(self._update_controls_visibility)
        # NEW: Connect the fixed fade checkbox to update UI state
        self.use_fixed_fade_check.stateChanged.connect(self._update_controls_visibility)

    def _update_controls_visibility(self):
        """Enables/disables controls based on the selected modes."""
        is_vb_active = self.vertical_blend_group.isChecked()
        is_preprocess = self.vb_preprocess_check.isChecked()
        use_fixed = self.use_fixed_fade_check.isChecked()

        # The standard stacking group is enabled if VB is off, or if VB is on and in pre-processor mode.
        self.stacking_group.setEnabled(not is_vb_active or is_preprocess)
        
        # The pre-processor checkbox is only relevant if VB is active.
        self.vb_preprocess_check.setEnabled(is_vb_active)
        
        # NEW: Enable/disable fade distance edits based on the fixed fade checkbox
        self.vert_receding_fade_edit.setEnabled(is_vb_active and use_fixed)
        self.vert_overhang_fade_edit.setEnabled(is_vb_active and use_fixed)

    def get_config(self) -> dict:
        """Collects current settings from this tab's widgets."""
        config_data = {}
        # Determine the effective blend mode
        is_vb_active = self.vertical_blend_group.isChecked()
        is_preprocess = self.vb_preprocess_check.isChecked()

        if is_vb_active and not is_preprocess:
            # Substitute mode: use a special blend mode name
            config_data["blend_mode"] = "vertical_combined"
        else:
            # Pre-process mode or no VB: use the standard combo box
            config_data["blend_mode"] = self.blend_mode_combo.currentText()

        config_data["vertical_blend_pre_process"] = is_vb_active and is_preprocess
        
        config_data["primary"] = int(self.primary_edit.text())
        config_data["radius"] = int(self.radius_edit.text())
        config_data["blend_param"] = float(self.blend_param_edit.text()) if self.blend_param_edit.text() else 1.0
        
        config_data["apply_vertical_luts"] = self.apply_vertical_luts_check.isChecked()
        config_data["apply_default_lut_after_stacking"] = self.apply_default_lut_check.isChecked()

        config_data["vertical_receding_layers"] = self.vert_receding_layers_spin.value()
        config_data["vertical_receding_fade_dist"] = float(self.vert_receding_fade_edit.text())
        config_data["vertical_overhang_layers"] = self.vert_overhang_layers_spin.value()
        config_data["vertical_overhang_fade_dist"] = float(self.vert_overhang_fade_edit.text())
        
        # --- NEW: Get values from new controls ---
        config_data["use_fixed_fade"] = self.use_fixed_fade_check.isChecked()
        config_data["vertical_gamma"] = float(self.vertical_gamma_edit.text()) if self.vertical_gamma_edit.text() else 1.0

        # Directional blend and others are not in this simplified UI, but we should provide defaults
        config_data["directional_blend"] = False
        config_data["dir_sigma"] = 1.0

        return config_data

    def apply_settings(self, cfg: Config):
        """Applies settings from a Config object to this tab's widgets."""
        is_vb_substitute = cfg.blend_mode.startswith("vertical_")
        
        self.vertical_blend_group.setChecked(is_vb_substitute or cfg.vertical_blend_pre_process)
        self.vb_preprocess_check.setChecked(cfg.vertical_blend_pre_process)
        
        if not is_vb_substitute:
            self.blend_mode_combo.setCurrentText(cfg.blend_mode)

        self.primary_edit.setText(str(cfg.primary))
        self.radius_edit.setText(str(cfg.radius))
        self.blend_param_edit.setText(str(cfg.blend_param))
        
        self.apply_vertical_luts_check.setChecked(cfg.apply_vertical_luts)
        self.apply_default_lut_check.setChecked(cfg.apply_default_lut_after_stacking)

        self.vert_receding_layers_spin.setValue(cfg.vertical_receding_layers)
        self.vert_receding_fade_edit.setText(str(cfg.vertical_receding_fade_dist))
        self.vert_overhang_layers_spin.setValue(cfg.vertical_overhang_layers)
        self.vert_overhang_fade_edit.setText(str(cfg.vertical_overhang_fade_dist))

        # --- NEW: Apply settings to new controls ---
        self.use_fixed_fade_check.setChecked(cfg.use_fixed_fade)
        self.vertical_gamma_edit.setText(str(cfg.vertical_gamma))

        self._update_controls_visibility()
