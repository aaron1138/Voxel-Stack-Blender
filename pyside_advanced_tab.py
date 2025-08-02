# pyside_advanced_tab.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QSpinBox, QPushButton, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator

from config import app_config as config, Config # Import the global config instance and Config class

class AdvancedTab(QWidget):
    """
    PySide6 tab for advanced settings, including threading, resume/stop at,
    and other miscellaneous toggles.
    Zarr-engine settings are present but disabled for this version.
    """
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config # Use the global config instance

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config
        self._apply_initial_state() # Ensure initial state is correctly set

    def _setup_ui(self):
        """Sets up the widgets and layout for the Advanced tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Performance & Output Group ---
        perf_output_group = QGroupBox("Performance & Run Control")
        perf_output_layout = QVBoxLayout(perf_output_group)
        perf_output_layout.setContentsMargins(10, 10, 10, 10)
        perf_output_layout.setSpacing(5)

        # Threads
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Threads:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, os.cpu_count() or 4) # Max threads based on CPU count
        threads_layout.addWidget(self.threads_spin)
        threads_layout.addStretch(1)
        perf_output_layout.addLayout(threads_layout)

        # Resume From / Stop At
        resume_stop_layout = QHBoxLayout()
        resume_stop_layout.addWidget(QLabel("Resume From Stack Index:"))
        self.resume_from_edit = QLineEdit()
        self.resume_from_edit.setFixedWidth(60)
        self.resume_from_edit.setValidator(QIntValidator(1, 999999, self))
        resume_stop_layout.addWidget(self.resume_from_edit)

        resume_stop_layout.addWidget(QLabel("Stop At Stack Index:"))
        self.stop_at_edit = QLineEdit()
        self.stop_at_edit.setFixedWidth(60)
        self.stop_at_edit.setValidator(QIntValidator(1, 999999, self))
        resume_stop_layout.addWidget(self.stop_at_edit)
        resume_stop_layout.addStretch(1)
        perf_output_layout.addLayout(resume_stop_layout)

        # Cap Layers
        cap_layers_layout = QHBoxLayout()
        cap_layers_layout.addWidget(QLabel("Cap Layers (Blank Slices):"))
        self.cap_layers_spin = QSpinBox()
        self.cap_layers_spin.setRange(0, 100) # Reasonable max for cap layers
        cap_layers_layout.addWidget(self.cap_layers_spin)
        cap_layers_layout.addStretch(1)
        perf_output_layout.addLayout(cap_layers_layout)

        # Scale Bits (for internal integer precision)
        scale_bits_layout = QHBoxLayout()
        scale_bits_layout.addWidget(QLabel("Integer Scale Bits:"))
        self.scale_bits_spin = QSpinBox()
        self.scale_bits_spin.setRange(8, 16) # Typically 8 to 16 bits for precision
        self.scale_bits_spin.setToolTip("Number of bits for internal fixed-point integer precision (e.g., 12 for 12-bit precision).")
        scale_bits_layout.addWidget(self.scale_bits_spin)
        scale_bits_layout.addStretch(1)
        perf_output_layout.addLayout(scale_bits_layout)

        main_layout.addWidget(perf_output_group)

        # --- Zarr-Engine Settings (Present but Disabled) ---
        zarr_group = QGroupBox("Zarr-Engine Settings (Future Use)")
        zarr_layout = QVBoxLayout(zarr_group)
        zarr_layout.setContentsMargins(10, 10, 10, 10)
        zarr_layout.setSpacing(5)

        zarr_store_layout = QHBoxLayout()
        zarr_store_layout.addWidget(QLabel("Zarr Store Path:"))
        self.zarr_store_edit = QLineEdit()
        self.zarr_store_edit.setEnabled(False) # Grayed out for now
        zarr_store_layout.addWidget(self.zarr_store_edit)
        self.zarr_store_browse_btn = QPushButton("Browse...")
        self.zarr_store_browse_btn.setEnabled(False) # Grayed out for now
        zarr_store_layout.addWidget(self.zarr_store_browse_btn)
        zarr_layout.addLayout(zarr_store_layout)

        zarr_cache_layout = QHBoxLayout()
        zarr_cache_layout.addWidget(QLabel("Cache Chunks:"))
        self.zarr_cache_spin = QSpinBox()
        self.zarr_cache_spin.setRange(1, 32)
        self.zarr_cache_spin.setEnabled(False) # Grayed out for now
        zarr_cache_layout.addWidget(self.zarr_cache_spin)
        zarr_cache_layout.addStretch(1)
        zarr_layout.addLayout(zarr_cache_layout)

        main_layout.addWidget(zarr_group)

        # --- Other Advanced Toggles ---
        other_toggles_group = QGroupBox("Other Advanced Toggles")
        other_toggles_layout = QVBoxLayout(other_toggles_group)
        other_toggles_layout.setContentsMargins(10, 10, 10, 10)
        other_toggles_layout.setSpacing(5)

        # Top Surface Smoothing
        top_surface_layout = QHBoxLayout()
        self.top_surface_smoothing_check = QCheckBox("Top Surface Smoothing")
        top_surface_layout.addWidget(self.top_surface_smoothing_check)

        top_surface_layout.addWidget(QLabel("Strength:"))
        self.top_surface_strength_edit = QLineEdit()
        self.top_surface_strength_edit.setFixedWidth(60)
        self.top_surface_strength_edit.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        top_surface_layout.addWidget(self.top_surface_strength_edit)
        top_surface_layout.addStretch(1)
        other_toggles_layout.addLayout(top_surface_layout)

        # Gradient Smooth (kept for consistency, though integer mode might not use directly)
        gradient_smooth_layout = QHBoxLayout()
        self.gradient_smooth_check = QCheckBox("Gradient Smooth")
        gradient_smooth_layout.addWidget(self.gradient_smooth_check)

        gradient_smooth_layout.addWidget(QLabel("Blend Strength:"))
        self.gradient_blend_strength_edit = QLineEdit()
        self.gradient_blend_strength_edit.setFixedWidth(60)
        self.gradient_blend_strength_edit.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        gradient_smooth_layout.addWidget(self.gradient_blend_strength_edit)
        gradient_smooth_layout.addStretch(1)
        other_toggles_layout.addLayout(gradient_smooth_layout)

        main_layout.addWidget(other_toggles_group)
        main_layout.addStretch(1)


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        # Zarr browse button (disabled for now, but connect for future)
        self.zarr_store_browse_btn.clicked.connect(self._browse_zarr_store)

        # Connect top surface smoothing checkbox to enable/disable strength edit
        self.top_surface_smoothing_check.stateChanged.connect(self._update_top_surface_controls)
        # Connect gradient smooth checkbox to enable/disable blend strength edit
        self.gradient_smooth_check.stateChanged.connect(self._update_gradient_smooth_controls)

        # Input validation for numeric fields
        self.resume_from_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.resume_from_edit, text, int))
        self.stop_at_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.stop_at_edit, text, int))
        self.top_surface_strength_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.top_surface_strength_edit, text, float))
        self.gradient_blend_strength_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.gradient_blend_strength_edit, text, float))


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

    def _apply_initial_state(self):
        """Applies initial control states based on default config."""
        self._update_top_surface_controls(Qt.Checked if self.config.top_surface_smoothing else Qt.Unchecked)
        self._update_gradient_smooth_controls(Qt.Checked if self.config.gradient_smooth else Qt.Unchecked)

    def _browse_zarr_store(self):
        """Opens a directory dialog for Zarr store path selection."""
        # This function is currently disabled in the UI
        dir_path = QFileDialog.getExistingDirectory(self, "Select Zarr Store Folder", self.zarr_store_edit.text())
        if dir_path:
            self.zarr_store_edit.setText(dir_path)

    def _update_top_surface_controls(self, state: int):
        """Enables/disables top surface strength edit based on checkbox state."""
        is_checked = (state == Qt.Checked)
        self.top_surface_strength_edit.setEnabled(is_checked)

    def _update_gradient_smooth_controls(self, state: int):
        """Enables/disables gradient blend strength edit based on checkbox state."""
        is_checked = (state == Qt.Checked)
        self.gradient_blend_strength_edit.setEnabled(is_checked)


    def get_config(self) -> dict:
        """Collects current settings from this tab's widgets."""
        config_data = {}
        config_data["threads"] = self.threads_spin.value()
        config_data["resume_from"] = int(self.resume_from_edit.text()) if self.resume_from_edit.text() else 1
        config_data["stop_at"] = int(self.stop_at_edit.text()) if self.stop_at_edit.text() else 999999
        config_data["cap_layers"] = self.cap_layers_spin.value()
        config_data["scale_bits"] = self.scale_bits_spin.value()

        # Zarr settings (disabled in UI, but collect if values exist)
        config_data["zarr_store"] = self.zarr_store_edit.text()
        config_data["zarr_cache_chunks"] = self.zarr_cache_spin.value()

        # Other Advanced Toggles
        config_data["top_surface_smoothing"] = self.top_surface_smoothing_check.isChecked()
        config_data["top_surface_strength"] = float(self.top_surface_strength_edit.text()) if self.top_surface_strength_edit.text() else 0.0
        config_data["gradient_smooth"] = self.gradient_smooth_check.isChecked()
        config_data["gradient_blend_strength"] = float(self.gradient_blend_strength_edit.text()) if self.gradient_blend_strength_edit.text() else 0.0

        return config_data

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        self.threads_spin.setValue(cfg.threads)
        self.resume_from_edit.setText(str(cfg.resume_from))
        self.stop_at_edit.setText(str(cfg.stop_at))
        self.cap_layers_spin.setValue(cfg.cap_layers)
        self.scale_bits_spin.setValue(cfg.scale_bits)

        # Zarr settings (apply even if disabled in UI, for consistency)
        self.zarr_store_edit.setText(cfg.zarr_store)
        self.zarr_cache_spin.setValue(cfg.zarr_cache_chunks)

        # Other Advanced Toggles
        self.top_surface_smoothing_check.setChecked(cfg.top_surface_smoothing)
        self.top_surface_strength_edit.setText(str(cfg.top_surface_strength))
        self.gradient_smooth_check.setChecked(cfg.gradient_smooth)
        self.gradient_blend_strength_edit.setText(str(cfg.gradient_blend_strength))

        # Ensure dynamic controls are updated after setting values
        self._update_top_surface_controls(Qt.Checked if cfg.top_surface_smoothing else Qt.Unchecked)
        self._update_gradient_smooth_controls(Qt.Checked if cfg.gradient_smooth else Qt.Unchecked)

