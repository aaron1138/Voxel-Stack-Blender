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

# pyside_xy_blend_tab.py (Completed with new controls)

import os
import numpy as np
import json
import copy

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QSlider, QSpinBox, QPushButton, QListWidget,
    QListWidgetItem, QStackedWidget, QSizePolicy, QMessageBox,
    QFileDialog, QGridLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator
from typing import Optional
from dataclasses import asdict

# Matplotlib imports for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the new configuration system
from config import app_config as config, Config, XYBlendOperation, LutParameters 
import lut_manager


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PySide6."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout(pad=2)


class XYBlendTab(QWidget):
    """
    PySide6 tab for managing the XY Blending/Processing pipeline.
    """
    
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config
        
        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config)
        self._update_operation_list()
        self._update_selected_operation_details()

    def _setup_ui(self):
        """Sets up the widgets and layout for the XY Blend tab."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Left Panel: Operations List and Controls ---
        left_panel_layout = QVBoxLayout()
        ops_list_group = QGroupBox("Processing Pipeline")
        ops_list_layout = QVBoxLayout(ops_list_group)
        self.ops_list_widget = QListWidget()
        self.ops_list_widget.setDragDropMode(QListWidget.InternalMove)
        self.ops_list_widget.setMinimumWidth(200)
        ops_list_layout.addWidget(self.ops_list_widget)

        op_buttons_layout = QHBoxLayout()
        self.add_op_button = QPushButton("Add Op")
        op_buttons_layout.addWidget(self.add_op_button)
        self.remove_op_button = QPushButton("Remove Op")
        op_buttons_layout.addWidget(self.remove_op_button)
        op_buttons_layout.addStretch(1)
        self.move_up_button = QPushButton("Move Up")
        op_buttons_layout.addWidget(self.move_up_button)
        self.move_down_button = QPushButton("Move Down")
        op_buttons_layout.addWidget(self.move_down_button)
        ops_list_layout.addLayout(op_buttons_layout)
        left_panel_layout.addWidget(ops_list_group)
        left_panel_layout.addStretch(1)
        main_layout.addLayout(left_panel_layout)

        # --- Right Panel: Operation Details ---
        right_panel_layout = QVBoxLayout()
        self.details_group = QGroupBox("Operation Details")
        details_layout = QVBoxLayout(self.details_group)
        
        op_type_layout = QHBoxLayout()
        op_type_layout.addWidget(QLabel("Operation Type:"))
        self.selected_op_type_combo = QComboBox()
        self.selected_op_type_combo.addItems([
            "none", "gaussian_blur", "bilateral_filter", "median_blur", "unsharp_mask", "resize", "apply_lut"
        ])
        op_type_layout.addWidget(self.selected_op_type_combo)
        op_type_layout.addStretch(1)
        details_layout.addLayout(op_type_layout)

        self.op_params_stacked_widget = QStackedWidget()
        details_layout.addWidget(self.op_params_stacked_widget)

        # --- Parameter Widgets for Stacked Widget ---
        self._create_none_params_widget()
        self._create_gaussian_params_widget()
        self._create_bilateral_params_widget()
        self._create_median_params_widget()
        self._create_unsharp_params_widget()
        self._create_resize_params_widget()
        self._create_apply_lut_params_widget() # This creates the complex LUT panel

        details_layout.addStretch(1)
        right_panel_layout.addWidget(self.details_group)
        right_panel_layout.addStretch(1)
        main_layout.addLayout(right_panel_layout)
        main_layout.addStretch(1)

    # --- Widget Creation Methods ---
    
    def _create_slider_combo(self, text_validator, slider_range, scale_factor):
        """Helper to create a linked QLineEdit and QSlider."""
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        line_edit.setFixedWidth(60)
        line_edit.setValidator(text_validator)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(slider_range[0], slider_range[1])
        slider.setSingleStep(1)
        
        layout.addWidget(line_edit)
        layout.addWidget(slider)
        return layout, line_edit, slider

    def _create_none_params_widget(self):
        self.none_params_widget = QWidget()
        layout = QVBoxLayout(self.none_params_widget)
        layout.addWidget(QLabel("No parameters for 'none' operation."))
        layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.none_params_widget)

    def _create_gaussian_params_widget(self):
        self.gaussian_params_widget = QWidget()
        layout = QVBoxLayout(self.gaussian_params_widget)
        ksize_layout = QHBoxLayout()
        ksize_layout.addWidget(QLabel("Kernel Size (X, Y):"))
        self.gaussian_ksize_x_edit = QLineEdit()
        self.gaussian_ksize_x_edit.setFixedWidth(60)
        self.gaussian_ksize_x_edit.setValidator(QIntValidator(1, 99, self))
        ksize_layout.addWidget(self.gaussian_ksize_x_edit)
        self.gaussian_ksize_y_edit = QLineEdit()
        self.gaussian_ksize_y_edit.setFixedWidth(60)
        self.gaussian_ksize_y_edit.setValidator(QIntValidator(1, 99, self))
        ksize_layout.addWidget(self.gaussian_ksize_y_edit)
        ksize_layout.addStretch(1)
        layout.addLayout(ksize_layout)
        
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma (X, Y):"))
        self.gaussian_sigma_x_edit = QLineEdit()
        self.gaussian_sigma_x_edit.setFixedWidth(60)
        self.gaussian_sigma_x_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        sigma_layout.addWidget(self.gaussian_sigma_x_edit)
        self.gaussian_sigma_y_edit = QLineEdit()
        self.gaussian_sigma_y_edit.setFixedWidth(60)
        self.gaussian_sigma_y_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        sigma_layout.addWidget(self.gaussian_sigma_y_edit)
        sigma_layout.addStretch(1)
        layout.addLayout(sigma_layout)
        layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.gaussian_params_widget)

    def _create_bilateral_params_widget(self):
        self.bilateral_params_widget = QWidget()
        layout = QVBoxLayout(self.bilateral_params_widget)
        layout.addWidget(QLabel("Diameter:"))
        self.bilateral_d_edit = QLineEdit()
        self.bilateral_d_edit.setFixedWidth(60)
        self.bilateral_d_edit.setValidator(QIntValidator(1, 99, self))
        layout.addWidget(self.bilateral_d_edit)
        layout.addWidget(QLabel("Sigma Color:"))
        self.bilateral_sigma_color_edit = QLineEdit()
        self.bilateral_sigma_color_edit.setFixedWidth(60)
        self.bilateral_sigma_color_edit.setValidator(QDoubleValidator(0.0, 255.0, 2, self))
        layout.addWidget(self.bilateral_sigma_color_edit)
        layout.addWidget(QLabel("Sigma Space:"))
        self.bilateral_sigma_space_edit = QLineEdit()
        self.bilateral_sigma_space_edit.setFixedWidth(60)
        self.bilateral_sigma_space_edit.setValidator(QDoubleValidator(0.0, 255.0, 2, self))
        layout.addWidget(self.bilateral_sigma_space_edit)
        layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.bilateral_params_widget)

    def _create_median_params_widget(self):
        self.median_params_widget = QWidget()
        layout = QVBoxLayout(self.median_params_widget)
        layout.addWidget(QLabel("Kernel Size:"))
        self.median_ksize_edit = QLineEdit()
        self.median_ksize_edit.setFixedWidth(60)
        self.median_ksize_edit.setValidator(QIntValidator(1, 99, self))
        layout.addWidget(self.median_ksize_edit)
        layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.median_params_widget)

    def _create_unsharp_params_widget(self):
        self.unsharp_params_widget = QWidget()
        layout = QVBoxLayout(self.unsharp_params_widget)
        layout.addWidget(QLabel("Amount:"))
        self.unsharp_amount_edit = QLineEdit()
        self.unsharp_amount_edit.setFixedWidth(60)
        self.unsharp_amount_edit.setValidator(QDoubleValidator(0.0, 5.0, 2, self))
        layout.addWidget(self.unsharp_amount_edit)
        layout.addWidget(QLabel("Threshold:"))
        self.unsharp_threshold_edit = QLineEdit()
        self.unsharp_threshold_edit.setFixedWidth(60)
        self.unsharp_threshold_edit.setValidator(QIntValidator(0, 255, self))
        layout.addWidget(self.unsharp_threshold_edit)
        layout.addWidget(QLabel("Internal Blur KSize:"))
        self.unsharp_blur_ksize_edit = QLineEdit()
        self.unsharp_blur_ksize_edit.setFixedWidth(60)
        self.unsharp_blur_ksize_edit.setValidator(QIntValidator(1, 99, self))
        layout.addWidget(self.unsharp_blur_ksize_edit)
        layout.addWidget(QLabel("Internal Blur Sigma:"))
        self.unsharp_blur_sigma_edit = QLineEdit()
        self.unsharp_blur_sigma_edit.setFixedWidth(60)
        self.unsharp_blur_sigma_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        layout.addWidget(self.unsharp_blur_sigma_edit)
        layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.unsharp_params_widget)

    def _create_resize_params_widget(self):
        self.resize_params_widget = QWidget()
        layout = QVBoxLayout(self.resize_params_widget)
        layout.addWidget(QLabel("Width (px, 0 for auto):"))
        self.resize_width_edit = QLineEdit()
        self.resize_width_edit.setFixedWidth(80)
        self.resize_width_edit.setValidator(QIntValidator(0, 9999, self))
        layout.addWidget(self.resize_width_edit)
        layout.addWidget(QLabel("Height (px, 0 for auto):"))
        self.resize_height_edit = QLineEdit()
        self.resize_height_edit.setFixedWidth(80)
        self.resize_height_edit.setValidator(QIntValidator(0, 9999, self))
        layout.addWidget(self.resize_height_edit)
        layout.addWidget(QLabel("Resampling Mode:"))
        self.resample_mode_combo = QComboBox()
        self.resample_mode_combo.addItems(["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS4", "AREA"])
        layout.addWidget(self.resample_mode_combo)
        layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.resize_params_widget)

    def _create_apply_lut_params_widget(self):
        self.apply_lut_params_widget = QWidget()
        apply_lut_layout = QVBoxLayout(self.apply_lut_params_widget)

        # LUT Source (Generated vs File)
        lut_source_layout = QHBoxLayout()
        lut_source_layout.addWidget(QLabel("LUT Source:"))
        self.lut_source_combo = QComboBox()
        self.lut_source_combo.addItems(["Generated", "File"])
        lut_source_layout.addWidget(self.lut_source_combo)
        lut_source_layout.addStretch(1)
        apply_lut_layout.addLayout(lut_source_layout)

        self.lut_gen_params_stacked_widget = QStackedWidget()
        apply_lut_layout.addWidget(self.lut_gen_params_stacked_widget)

        # --- Generated LUT Panel ---
        self.lut_generated_params_group = QGroupBox("Generated LUT Parameters")
        lut_generated_params_layout = QVBoxLayout(self.lut_generated_params_group)
        
        # Type (linear, gamma, etc.)
        gen_type_layout = QHBoxLayout()
        gen_type_layout.addWidget(QLabel("Type:"))
        self.lut_generation_type_combo = QComboBox()
        self.lut_generation_type_combo.addItems(["linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard"])
        gen_type_layout.addWidget(self.lut_generation_type_combo)
        gen_type_layout.addStretch(1)
        lut_generated_params_layout.addLayout(gen_type_layout)
        
        # --- NEW: Universal Range Controls ---
        range_group = QGroupBox("Input/Output Range")
        range_layout = QGridLayout(range_group)
        range_layout.addWidget(QLabel("Input Min:"), 0, 0)
        self.lut_input_min_edit = QLineEdit("0")
        self.lut_input_min_edit.setValidator(QIntValidator(0, 255, self))
        range_layout.addWidget(self.lut_input_min_edit, 0, 1)
        range_layout.addWidget(QLabel("Input Max:"), 0, 2)
        self.lut_input_max_edit = QLineEdit("255")
        self.lut_input_max_edit.setValidator(QIntValidator(0, 255, self))
        range_layout.addWidget(self.lut_input_max_edit, 0, 3)
        range_layout.addWidget(QLabel("Output Min:"), 1, 0)
        self.lut_output_min_edit = QLineEdit("0")
        self.lut_output_min_edit.setValidator(QIntValidator(0, 255, self))
        range_layout.addWidget(self.lut_output_min_edit, 1, 1)
        range_layout.addWidget(QLabel("Output Max:"), 1, 2)
        self.lut_output_max_edit = QLineEdit("255")
        self.lut_output_max_edit.setValidator(QIntValidator(0, 255, self))
        range_layout.addWidget(self.lut_output_max_edit, 1, 3)
        lut_generated_params_layout.addWidget(range_group)

        # Stacked widget for algorithm-specific params
        self.gen_lut_algo_params_stacked_widget = QStackedWidget()
        lut_generated_params_layout.addWidget(self.gen_lut_algo_params_stacked_widget)

        # --- Algo Param Widgets ---
        # Linear (no specific params, uses range controls)
        self.lut_linear_params_widget = QWidget()
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_linear_params_widget)

        # Gamma
        self.lut_gamma_params_widget = QWidget()
        gamma_layout = QHBoxLayout(self.lut_gamma_params_widget)
        gamma_layout.addWidget(QLabel("Gamma:"))
        g_layout, self.lut_gamma_value_edit, self.lut_gamma_value_slider = self._create_slider_combo(QDoubleValidator(0.01, 10.0, 2), (1, 1000), 100.0)
        gamma_layout.addLayout(g_layout)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_gamma_params_widget)

        # S-Curve
        self.lut_s_curve_params_widget = QWidget()
        s_curve_layout = QHBoxLayout(self.lut_s_curve_params_widget)
        s_curve_layout.addWidget(QLabel("Contrast:"))
        sc_layout, self.lut_s_curve_contrast_edit, self.lut_s_curve_contrast_slider = self._create_slider_combo(QDoubleValidator(0.0, 1.0, 2), (0, 100), 100.0)
        s_curve_layout.addLayout(sc_layout)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_s_curve_params_widget)

        # Log
        self.lut_log_params_widget = QWidget()
        log_layout = QHBoxLayout(self.lut_log_params_widget)
        log_layout.addWidget(QLabel("Param:"))
        l_layout, self.lut_log_param_edit, self.lut_log_param_slider = self._create_slider_combo(QDoubleValidator(0.01, 100.0, 2), (1, 1000), 10.0)
        log_layout.addLayout(l_layout)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_log_params_widget)

        # Exp
        self.lut_exp_params_widget = QWidget()
        exp_layout = QHBoxLayout(self.lut_exp_params_widget)
        exp_layout.addWidget(QLabel("Param:"))
        e_layout, self.lut_exp_param_edit, self.lut_exp_param_slider = self._create_slider_combo(QDoubleValidator(0.01, 10.0, 2), (1, 1000), 100.0)
        exp_layout.addLayout(e_layout)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_exp_params_widget)

        # Sqrt (NEW)
        self.lut_sqrt_params_widget = QWidget()
        sqrt_layout = QHBoxLayout(self.lut_sqrt_params_widget)
        sqrt_layout.addWidget(QLabel("Root:"))
        sq_layout, self.lut_sqrt_param_edit, self.lut_sqrt_param_slider = self._create_slider_combo(QDoubleValidator(0.1, 50.0, 2), (10, 500), 10.0)
        sqrt_layout.addLayout(sq_layout)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_sqrt_params_widget)

        # Rodbard (NEW)
        self.lut_rodbard_params_widget = QWidget()
        rodbard_layout = QHBoxLayout(self.lut_rodbard_params_widget)
        rodbard_layout.addWidget(QLabel("Contrast:"))
        r_layout, self.lut_rodbard_param_edit, self.lut_rodbard_param_slider = self._create_slider_combo(QDoubleValidator(0.0, 2.0, 2), (0, 200), 100.0)
        rodbard_layout.addLayout(r_layout)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_rodbard_params_widget)
        
        lut_generated_params_layout.addStretch(1)
        self.lut_gen_params_stacked_widget.addWidget(self.lut_generated_params_group)

        # --- File LUT Panel ---
        self.lut_file_params_group = QGroupBox("File-based LUT Controls")
        lut_file_params_layout = QVBoxLayout(self.lut_file_params_group)
        lut_path_layout = QHBoxLayout()
        lut_path_layout.addWidget(QLabel("LUT File:"))
        self.lut_filepath_edit = QLineEdit()
        self.lut_filepath_edit.setReadOnly(True)
        lut_path_layout.addWidget(self.lut_filepath_edit)
        lut_file_params_layout.addLayout(lut_path_layout)
        file_button_layout = QHBoxLayout()
        self.lut_load_file_button = QPushButton("Load from File...")
        file_button_layout.addWidget(self.lut_load_file_button)
        self.lut_save_file_button = QPushButton("Save to File...")
        file_button_layout.addWidget(self.lut_save_file_button)
        file_button_layout.addStretch(1)
        lut_file_params_layout.addLayout(file_button_layout)
        lut_file_params_layout.addStretch(1)
        self.lut_gen_params_stacked_widget.addWidget(self.lut_file_params_group)

        # --- LUT Preview Plot ---
        lut_preview_group = QGroupBox("LUT Curve Preview")
        lut_preview_layout = QVBoxLayout(lut_preview_group)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setMinimumHeight(200)
        lut_preview_layout.addWidget(self.canvas)
        apply_lut_layout.addWidget(lut_preview_group)
        
        apply_lut_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.apply_lut_params_widget)

    # --- Signal Connection ---
    
    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.ops_list_widget.currentRowChanged.connect(self._update_selected_operation_details)
        self.ops_list_widget.model().rowsMoved.connect(self._reorder_operations_in_config)

        self.add_op_button.clicked.connect(self._add_operation)
        self.remove_op_button.clicked.connect(self._remove_operation)
        self.move_up_button.clicked.connect(self._move_operation_up)
        self.move_down_button.clicked.connect(self._move_operation_down)

        self.selected_op_type_combo.currentTextChanged.connect(self._on_selected_op_type_changed)

        # Connect standard op params
        self.gaussian_ksize_x_edit.editingFinished.connect(lambda: self._update_param_in_config(self.gaussian_ksize_x_edit, "gaussian_ksize_x", int))
        self.gaussian_ksize_y_edit.editingFinished.connect(lambda: self._update_param_in_config(self.gaussian_ksize_y_edit, "gaussian_ksize_y", int))
        self.gaussian_sigma_x_edit.editingFinished.connect(lambda: self._update_param_in_config(self.gaussian_sigma_x_edit, "gaussian_sigma_x", float))
        self.gaussian_sigma_y_edit.editingFinished.connect(lambda: self._update_param_in_config(self.gaussian_sigma_y_edit, "gaussian_sigma_y", float))
        self.bilateral_d_edit.editingFinished.connect(lambda: self._update_param_in_config(self.bilateral_d_edit, "bilateral_d", int))
        self.bilateral_sigma_color_edit.editingFinished.connect(lambda: self._update_param_in_config(self.bilateral_sigma_color_edit, "bilateral_sigma_color", float))
        self.bilateral_sigma_space_edit.editingFinished.connect(lambda: self._update_param_in_config(self.bilateral_sigma_space_edit, "bilateral_sigma_space", float))
        self.median_ksize_edit.editingFinished.connect(lambda: self._update_param_in_config(self.median_ksize_edit, "median_ksize", int))
        self.unsharp_amount_edit.editingFinished.connect(lambda: self._update_param_in_config(self.unsharp_amount_edit, "unsharp_amount", float))
        self.unsharp_threshold_edit.editingFinished.connect(lambda: self._update_param_in_config(self.unsharp_threshold_edit, "unsharp_threshold", int))
        self.unsharp_blur_ksize_edit.editingFinished.connect(lambda: self._update_param_in_config(self.unsharp_blur_ksize_edit, "unsharp_blur_ksize", int))
        self.unsharp_blur_sigma_edit.editingFinished.connect(lambda: self._update_param_in_config(self.unsharp_blur_sigma_edit, "unsharp_blur_sigma", float))
        self.resize_width_edit.editingFinished.connect(lambda: self._update_param_in_config(self.resize_width_edit, "resize_width", int, allow_none_if_zero=True))
        self.resize_height_edit.editingFinished.connect(lambda: self._update_param_in_config(self.resize_height_edit, "resize_height", int, allow_none_if_zero=True))
        self.resample_mode_combo.currentTextChanged.connect(lambda text: self._update_param_in_config(self.resample_mode_combo, "resample_mode", str))

        # Connect LUT controls
        self.lut_source_combo.currentTextChanged.connect(self._on_lut_source_changed)
        self.lut_generation_type_combo.currentTextChanged.connect(self._on_lut_gen_type_changed)
        self.lut_load_file_button.clicked.connect(self._load_lut_from_file_for_op)
        self.lut_save_file_button.clicked.connect(self._save_lut_to_file_from_op)

        # Connect universal LUT range controls
        self.lut_input_min_edit.editingFinished.connect(lambda: self._update_lut_param_in_config(self.lut_input_min_edit, "input_min", int))
        self.lut_input_max_edit.editingFinished.connect(lambda: self._update_lut_param_in_config(self.lut_input_max_edit, "input_max", int))
        self.lut_output_min_edit.editingFinished.connect(lambda: self._update_lut_param_in_config(self.lut_output_min_edit, "output_min", int))
        self.lut_output_max_edit.editingFinished.connect(lambda: self._update_lut_param_in_config(self.lut_output_max_edit, "output_max", int))

        # Connect algorithm-specific LUT controls
        self._connect_slider_combo(self.lut_gamma_value_edit, self.lut_gamma_value_slider, "gamma_value", 100.0)
        self._connect_slider_combo(self.lut_s_curve_contrast_edit, self.lut_s_curve_contrast_slider, "s_curve_contrast", 100.0)
        self._connect_slider_combo(self.lut_log_param_edit, self.lut_log_param_slider, "log_param", 10.0)
        self._connect_slider_combo(self.lut_exp_param_edit, self.lut_exp_param_slider, "exp_param", 100.0)
        self._connect_slider_combo(self.lut_sqrt_param_edit, self.lut_sqrt_param_slider, "sqrt_param", 10.0)
        self._connect_slider_combo(self.lut_rodbard_param_edit, self.lut_rodbard_param_slider, "rodbard_param", 100.0)

    def _connect_slider_combo(self, line_edit, slider, param_name, scale_factor):
        """Helper to connect signals for a linked QLineEdit and QSlider."""
        slider.valueChanged.connect(lambda val: line_edit.setText(f"{val / scale_factor:.2f}"))
        slider.sliderReleased.connect(lambda: self._update_lut_param_in_config(line_edit, param_name, float, slider=slider, scale_factor=scale_factor))
        line_edit.editingFinished.connect(lambda: self._update_lut_param_in_config(line_edit, param_name, float, slider=slider, scale_factor=scale_factor))

    # --- Core Logic Methods ---

    def _update_operation_list(self):
        current_row = self.ops_list_widget.currentRow()
        self.ops_list_widget.clear()
        for i, op in enumerate(self.config.xy_blend_pipeline):
            item_text = f"{i+1}. {op.type.replace('_', ' ').title()}"
            self.ops_list_widget.addItem(QListWidgetItem(item_text))
        
        if 0 <= current_row < self.ops_list_widget.count():
            self.ops_list_widget.setCurrentRow(current_row)
        elif self.ops_list_widget.count() > 0:
            self.ops_list_widget.setCurrentRow(0)
        else:
            self._update_selected_operation_details()

    def _update_selected_operation_details(self):
        """Updates the details panel based on the selected operation."""
        current_row = self.ops_list_widget.currentRow()
        if 0 <= current_row < len(self.config.xy_blend_pipeline):
            selected_op = self.config.xy_blend_pipeline[current_row]
            self._block_all_param_signals(True)
            
            self.selected_op_type_combo.setCurrentText(selected_op.type)
            self._show_params_for_type(selected_op.type)
            self._populate_params_widgets(selected_op)
            self.details_group.setEnabled(True)
            
            self._block_all_param_signals(False)

            if selected_op.type == "apply_lut":
                self._plot_current_lut_preview(selected_op.lut_params)
        else:
            self.details_group.setEnabled(False)
            self.selected_op_type_combo.setCurrentText("none")
            self.op_params_stacked_widget.setCurrentWidget(self.none_params_widget)
            self.canvas.axes.clear()
            self.canvas.draw()

    def _on_selected_op_type_changed(self, new_type: str):
        """Handles changing the type of the currently selected operation."""
        current_row = self.ops_list_widget.currentRow()
        if not (0 <= current_row < len(self.config.xy_blend_pipeline)):
            return

        old_op = self.config.xy_blend_pipeline[current_row]
        if new_type == old_op.type:
            return

        new_op = XYBlendOperation(type=new_type)
        if new_type == "apply_lut" and old_op.type == "apply_lut":
            new_op.lut_params = copy.deepcopy(old_op.lut_params)

        self.config.xy_blend_pipeline[current_row] = new_op
        
        self._block_all_param_signals(True)
        self.ops_list_widget.currentItem().setText(f"{current_row+1}. {new_type.replace('_', ' ').title()}")
        self._show_params_for_type(new_type)
        self._populate_params_widgets(new_op)
        self._block_all_param_signals(False)

        if new_type == "apply_lut":
            self._plot_current_lut_preview(new_op.lut_params)
        else:
            self.canvas.axes.clear()
            self.canvas.draw()

    def _show_params_for_type(self, op_type: str):
        """Switches the stacked widget to show parameters for the given operation type."""
        widget_map = {
            "none": self.none_params_widget, "gaussian_blur": self.gaussian_params_widget,
            "bilateral_filter": self.bilateral_params_widget, "median_blur": self.median_params_widget,
            "unsharp_mask": self.unsharp_params_widget, "resize": self.resize_params_widget,
            "apply_lut": self.apply_lut_params_widget
        }
        self.op_params_stacked_widget.setCurrentWidget(widget_map.get(op_type, self.none_params_widget))

    def _populate_params_widgets(self, op: XYBlendOperation):
        """Populates UI widgets with values from the given XYBlendOperation object."""
        # Standard Ops
        self.gaussian_ksize_x_edit.setText(str(op.gaussian_ksize_x))
        self.gaussian_ksize_y_edit.setText(str(op.gaussian_ksize_y))
        self.gaussian_sigma_x_edit.setText(str(op.gaussian_sigma_x))
        self.gaussian_sigma_y_edit.setText(str(op.gaussian_sigma_y))
        self.bilateral_d_edit.setText(str(op.bilateral_d))
        self.bilateral_sigma_color_edit.setText(str(op.bilateral_sigma_color))
        self.bilateral_sigma_space_edit.setText(str(op.bilateral_sigma_space))
        self.median_ksize_edit.setText(str(op.median_ksize))
        self.unsharp_amount_edit.setText(str(op.unsharp_amount))
        self.unsharp_threshold_edit.setText(str(op.unsharp_threshold))
        self.unsharp_blur_ksize_edit.setText(str(op.unsharp_blur_ksize))
        self.unsharp_blur_sigma_edit.setText(str(op.unsharp_blur_sigma))
        self.resize_width_edit.setText(str(op.resize_width or 0))
        self.resize_height_edit.setText(str(op.resize_height or 0))
        self.resample_mode_combo.setCurrentText(op.resample_mode)

        # LUT Op
        if op.type == "apply_lut":
            lp = op.lut_params
            self.lut_source_combo.setCurrentText(lp.lut_source.capitalize())
            self._update_lut_source_controls_widget_only(lp.lut_source.capitalize())
            
            self.lut_generation_type_combo.setCurrentText(lp.lut_generation_type)
            self._update_lut_gen_type_controls_widget_only(lp.lut_generation_type)

            self.lut_input_min_edit.setText(str(lp.input_min))
            self.lut_input_max_edit.setText(str(lp.input_max))
            self.lut_output_min_edit.setText(str(lp.output_min))
            self.lut_output_max_edit.setText(str(lp.output_max))

            self.lut_gamma_value_edit.setText(f"{lp.gamma_value:.2f}")
            self.lut_gamma_value_slider.setValue(int(lp.gamma_value * 100))
            self.lut_s_curve_contrast_edit.setText(f"{lp.s_curve_contrast:.2f}")
            self.lut_s_curve_contrast_slider.setValue(int(lp.s_curve_contrast * 100))
            self.lut_log_param_edit.setText(f"{lp.log_param:.2f}")
            self.lut_log_param_slider.setValue(int(lp.log_param * 10))
            self.lut_exp_param_edit.setText(f"{lp.exp_param:.2f}")
            self.lut_exp_param_slider.setValue(int(lp.exp_param * 100))
            self.lut_sqrt_param_edit.setText(f"{lp.sqrt_param:.2f}")
            self.lut_sqrt_param_slider.setValue(int(lp.sqrt_param * 10))
            self.lut_rodbard_param_edit.setText(f"{lp.rodbard_param:.2f}")
            self.lut_rodbard_param_slider.setValue(int(lp.rodbard_param * 100))
            
            self.lut_filepath_edit.setText(lp.fixed_lut_path)

    def _block_all_param_signals(self, block: bool):
        """Blocks/unblocks signals for all parameter widgets to prevent update loops."""
        widgets_to_block = [
            self.selected_op_type_combo, self.gaussian_ksize_x_edit, self.gaussian_ksize_y_edit,
            self.gaussian_sigma_x_edit, self.gaussian_sigma_y_edit, self.bilateral_d_edit,
            self.bilateral_sigma_color_edit, self.bilateral_sigma_space_edit, self.median_ksize_edit,
            self.unsharp_amount_edit, self.unsharp_threshold_edit, self.unsharp_blur_ksize_edit,
            self.unsharp_blur_sigma_edit, self.resize_width_edit, self.resize_height_edit,
            self.resample_mode_combo, self.lut_source_combo, self.lut_generation_type_combo,
            self.lut_input_min_edit, self.lut_input_max_edit, self.lut_output_min_edit,
            self.lut_output_max_edit, self.lut_gamma_value_edit, self.lut_gamma_value_slider,
            self.lut_s_curve_contrast_edit, self.lut_s_curve_contrast_slider, self.lut_log_param_edit,
            self.lut_log_param_slider, self.lut_exp_param_edit, self.lut_exp_param_slider,
            self.lut_sqrt_param_edit, self.lut_sqrt_param_slider, self.lut_rodbard_param_edit,
            self.lut_rodbard_param_slider, self.lut_filepath_edit
        ]
        for widget in widgets_to_block:
            widget.blockSignals(block)

    def _update_param_in_config(self, sender_widget, param_name: str, data_type: type, allow_none_if_zero: bool = False):
        """Updates a standard operation parameter in the config object."""
        current_row = self.ops_list_widget.currentRow()
        if not (0 <= current_row < len(self.config.xy_blend_pipeline)): return
        
        selected_op = self.config.xy_blend_pipeline[current_row]
        text = sender_widget.text() if isinstance(sender_widget, QLineEdit) else sender_widget.currentText()
        
        try:
            value_to_set = None
            if allow_none_if_zero and data_type(text) == 0:
                value_to_set = None
            else:
                value_to_set = data_type(text.replace(',', '.'))
            
            setattr(selected_op, param_name, value_to_set)
            selected_op.__post_init__()
            if hasattr(sender_widget, 'setStyleSheet'): sender_widget.setStyleSheet("")
        except (ValueError, TypeError):
            if hasattr(sender_widget, 'setStyleSheet'): sender_widget.setStyleSheet("border: 1px solid red;")
            return
            
        self._block_all_param_signals(True)
        self._populate_params_widgets(selected_op)
        self._block_all_param_signals(False)

    def _update_lut_param_in_config(self, sender_widget, param_name: str, data_type: type, slider: Optional[QSlider] = None, scale_factor: float = 1.0):
        """Updates a LUT parameter in the config object and refreshes the plot."""
        current_row = self.ops_list_widget.currentRow()
        if not (0 <= current_row < len(self.config.xy_blend_pipeline)): return

        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut": return
        
        text = sender_widget.text() if isinstance(sender_widget, QLineEdit) else sender_widget.currentText()
        
        try:
            value_to_set = data_type(text.replace(',', '.'))
            setattr(selected_op.lut_params, param_name, value_to_set)
            selected_op.lut_params.__post_init__()
            if hasattr(sender_widget, 'setStyleSheet'): sender_widget.setStyleSheet("")
        except (ValueError, TypeError):
            if hasattr(sender_widget, 'setStyleSheet'): sender_widget.setStyleSheet("border: 1px solid red;")
            return

        self._block_all_param_signals(True)
        self._populate_params_widgets(selected_op)
        self._block_all_param_signals(False)
        
        self._plot_current_lut_preview(selected_op.lut_params)

    def _on_lut_source_changed(self, source_text: str):
        self._update_lut_param_in_config(self.lut_source_combo, "lut_source", str)

    def _update_lut_source_controls_widget_only(self, source_text: str):
        is_generated = (source_text.lower() == "generated")
        self.lut_gen_params_stacked_widget.setCurrentIndex(0 if is_generated else 1)

    def _on_lut_gen_type_changed(self, lut_type: str):
        self._update_lut_param_in_config(self.lut_generation_type_combo, "lut_generation_type", str)

    def _update_lut_gen_type_controls_widget_only(self, lut_type: str):
        widget_map = {
            "linear": self.lut_linear_params_widget, "gamma": self.lut_gamma_params_widget,
            "s_curve": self.lut_s_curve_params_widget, "log": self.lut_log_params_widget,
            "exp": self.lut_exp_params_widget, "sqrt": self.lut_sqrt_params_widget,
            "rodbard": self.lut_rodbard_params_widget
        }
        self.gen_lut_algo_params_stacked_widget.setCurrentWidget(widget_map.get(lut_type.lower(), self.lut_linear_params_widget))

    def _plot_current_lut_preview(self, lut_params: LutParameters):
        """Generates/loads the LUT based on parameters and plots it."""
        generated_lut: Optional[np.ndarray] = None
        try:
            if lut_params.lut_source == "generated":
                args = (lut_params.input_min, lut_params.input_max, lut_params.output_min, lut_params.output_max)
                if lut_params.lut_generation_type == "linear":
                    generated_lut = lut_manager.generate_linear_lut(*args)
                elif lut_params.lut_generation_type == "gamma":
                    generated_lut = lut_manager.generate_gamma_lut(lut_params.gamma_value, *args)
                elif lut_params.lut_generation_type == "s_curve":
                    generated_lut = lut_manager.generate_s_curve_lut(lut_params.s_curve_contrast, *args)
                elif lut_params.lut_generation_type == "log":
                    generated_lut = lut_manager.generate_log_lut(lut_params.log_param, *args)
                elif lut_params.lut_generation_type == "exp":
                    generated_lut = lut_manager.generate_exp_lut(lut_params.exp_param, *args)
                elif lut_params.lut_generation_type == "sqrt":
                    generated_lut = lut_manager.generate_sqrt_lut(lut_params.sqrt_param, *args)
                elif lut_params.lut_generation_type == "rodbard":
                    generated_lut = lut_manager.generate_rodbard_lut(lut_params.rodbard_param, *args)
            elif lut_params.lut_source == "file" and lut_params.fixed_lut_path and os.path.exists(lut_params.fixed_lut_path):
                generated_lut = lut_manager.load_lut(lut_params.fixed_lut_path)
            
            if generated_lut is None:
                generated_lut = lut_manager.get_default_z_lut()

        except Exception as e:
            print(f"Error generating/loading LUT for plot preview: {e}")
            generated_lut = lut_manager.get_default_z_lut()

        self.canvas.axes.clear()
        self.canvas.axes.plot(np.arange(256), generated_lut, 'b-')
        self.canvas.axes.set_title("LUT Curve Preview")
        self.canvas.axes.set_xlabel("Input Value (0-255)")
        self.canvas.axes.set_ylabel("Output Value (0-255)")
        self.canvas.axes.set_xlim(0, 255)
        self.canvas.axes.set_ylim(0, 255)
        self.canvas.axes.grid(True)
        self.canvas.draw()
        
    def _add_operation(self):
        new_op = XYBlendOperation(type="none")
        self.config.xy_blend_pipeline.append(new_op)
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(len(self.config.xy_blend_pipeline) - 1)

    def _remove_operation(self):
        current_row = self.ops_list_widget.currentRow()
        if 0 <= current_row < len(self.config.xy_blend_pipeline):
            reply = QMessageBox.question(self, "Remove Operation", "Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.config.xy_blend_pipeline[current_row]
                self._update_operation_list()
                new_row = min(current_row, self.ops_list_widget.count() - 1)
                self.ops_list_widget.setCurrentRow(new_row)

    def _move_operation(self, direction: int):
        current_row = self.ops_list_widget.currentRow()
        new_row = current_row + direction
        if 0 <= new_row < len(self.config.xy_blend_pipeline):
            pipeline = self.config.xy_blend_pipeline
            pipeline[current_row], pipeline[new_row] = pipeline[new_row], pipeline[current_row]
            self._update_operation_list()
            self.ops_list_widget.setCurrentRow(new_row)

    def _move_operation_up(self):
        self._move_operation(-1)

    def _move_operation_down(self):
        self._move_operation(1)

    def _reorder_operations_in_config(self, parent, start, end, destination, row):
        # This signal can be tricky. A robust way is to rebuild the list from the UI.
        new_pipeline = []
        for i in range(self.ops_list_widget.count()):
            item_text = self.ops_list_widget.item(i).text()
            # Find the original operation that corresponds to this text
            # This is complex. A simpler way is to just move the item in the config list.
            pass # The simple move is handled below.

        # The indices provided by the signal are what we need
        moved_op = self.config.xy_blend_pipeline.pop(start)
        self.config.xy_blend_pipeline.insert(row, moved_op)
        
        # After reordering, just update the text and selection
        self._block_all_param_signals(True)
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(row)
        self._block_all_param_signals(False)


    def _load_lut_from_file_for_op(self):
        current_row = self.ops_list_widget.currentRow()
        if not (0 <= current_row < len(self.config.xy_blend_pipeline)): return
        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut": return

        filepath, _ = QFileDialog.getOpenFileName(self, "Load LUT File", "", "JSON Files (*.json);;All Files (*)")
        if filepath:
            try:
                lut_manager.load_lut(filepath) # Validate
                self._block_all_param_signals(True)
                selected_op.lut_params.fixed_lut_path = filepath
                selected_op.lut_params.lut_source = "file"
                self._populate_params_widgets(selected_op)
                self._block_all_param_signals(False)
                self._plot_current_lut_preview(selected_op.lut_params)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load LUT file: {e}")

    def _save_lut_to_file_from_op(self):
        current_row = self.ops_list_widget.currentRow()
        if not (0 <= current_row < len(self.config.xy_blend_pipeline)): return
        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut": return

        # Generate the LUT to be saved
        lut_to_save = None
        try:
            # Re-use the plotting logic to generate the current LUT array
            if selected_op.lut_params.lut_source == "generated":
                lp = selected_op.lut_params
                args = (lp.input_min, lp.input_max, lp.output_min, lp.output_max)
                if lp.lut_generation_type == "linear": lut_to_save = lut_manager.generate_linear_lut(*args)
                elif lp.lut_generation_type == "gamma": lut_to_save = lut_manager.generate_gamma_lut(lp.gamma_value, *args)
                elif lp.lut_generation_type == "s_curve": lut_to_save = lut_manager.generate_s_curve_lut(lp.s_curve_contrast, *args)
                elif lp.lut_generation_type == "log": lut_to_save = lut_manager.generate_log_lut(lp.log_param, *args)
                elif lp.lut_generation_type == "exp": lut_to_save = lut_manager.generate_exp_lut(lp.exp_param, *args)
                elif lp.lut_generation_type == "sqrt": lut_to_save = lut_manager.generate_sqrt_lut(lp.sqrt_param, *args)
                elif lp.lut_generation_type == "rodbard": lut_to_save = lut_manager.generate_rodbard_lut(lp.rodbard_param, *args)
            elif selected_op.lut_params.lut_source == "file" and selected_op.lut_params.fixed_lut_path:
                lut_to_save = lut_manager.load_lut(selected_op.lut_params.fixed_lut_path)

            if lut_to_save is None:
                raise ValueError("Could not generate or load a valid LUT to save.")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to prepare LUT for saving: {e}")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Current LUT", "custom_lut.json", "JSON Files (*.json)")
        if filepath:
            try:
                lut_manager.save_lut(filepath, lut_to_save)
                QMessageBox.information(self, "Save Success", f"LUT saved successfully to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save LUT to file: {e}")
                
    def apply_settings(self, cfg: Config):
        """Applies settings from a Config object to this tab's widgets."""
        self._block_all_param_signals(True)
        self._update_operation_list() 
        self._update_selected_operation_details()
        self._block_all_param_signals(False)

    def get_config(self) -> dict:
        """The config object is updated live, so we just return its dict representation."""
        return {"xy_blend_pipeline": [op.to_dict() for op in self.config.xy_blend_pipeline]}

