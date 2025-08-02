# pyside_xy_blend_tab.py (Modified to fix parameter reset and add copy import)

import os
import numpy as np
import json # Not used directly in this snippet, but common import
import copy # <--- NEW: Import for deepcopy

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QSlider, QSpinBox, QPushButton, QListWidget,
    QListWidgetItem, QStackedWidget, QSizePolicy, QMessageBox,
    QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator
from typing import Optional # Already imported in previous step
from dataclasses import asdict # Already imported in previous step

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
    Allows users to add, remove, reorder, and configure individual
    XYBlendOperation steps (blur, sharpen, resize, apply_lut, etc.).
    """
    
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config # Use the global config instance
        
        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config
        self._update_operation_list() # Populate list with initial operations
        self._update_selected_operation_details() # Show details for first item

    def _setup_ui(self):
        """Sets up the widgets and layout for the XY Blend tab."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Left Panel: Operations List and Controls ---
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(5)

        # Operations List
        ops_list_group = QGroupBox("Processing Pipeline")
        ops_list_layout = QVBoxLayout(ops_list_group)
        ops_list_layout.setContentsMargins(10, 10, 10, 10)
        ops_list_layout.setSpacing(5)

        self.ops_list_widget = QListWidget()
        self.ops_list_widget.setDragDropMode(QListWidget.InternalMove) # Enable reordering
        self.ops_list_widget.setMinimumWidth(200)
        self.ops_list_widget.setMinimumHeight(250)
        ops_list_layout.addWidget(self.ops_list_widget)

        # Add/Remove/Move Buttons
        op_buttons_layout = QHBoxLayout()
        self.add_op_button = QPushButton("Add Op")
        op_buttons_layout.addWidget(self.add_op_button)
        self.remove_op_button = QPushButton("Remove Op")
        op_buttons_layout.addWidget(self.remove_op_button)
        op_buttons_layout.addStretch(1) # Push move buttons to right
        self.move_up_button = QPushButton("Move Up")
        op_buttons_layout.addWidget(self.move_up_button)
        self.move_down_button = QPushButton("Move Down")
        op_buttons_layout.addWidget(self.move_down_button)
        ops_list_layout.addLayout(op_buttons_layout)
        
        left_panel_layout.addWidget(ops_list_group)
        left_panel_layout.addStretch(1) # Push content to top

        main_layout.addLayout(left_panel_layout)

        # --- Right Panel: Operation Details (Stacked Widget) ---
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(5)

        self.details_group = QGroupBox("Operation Details") # Made into an instance variable
        details_layout = QVBoxLayout(self.details_group)
        details_layout.setContentsMargins(10, 10, 10, 10)
        details_layout.setSpacing(5)

        # Operation Type selection for the currently selected item
        op_type_layout = QHBoxLayout()
        op_type_layout.addWidget(QLabel("Operation Type:"))
        self.selected_op_type_combo = QComboBox()
        self.selected_op_type_combo.addItems([
            "none", "gaussian_blur", "bilateral_filter", "median_blur", "unsharp_mask", "resize", "apply_lut" # Added apply_lut
        ])
        op_type_layout.addWidget(self.selected_op_type_combo)
        op_type_layout.addStretch(1)
        details_layout.addLayout(op_type_layout)

        # Stacked Widget for specific operation parameters
        self.op_params_stacked_widget = QStackedWidget()
        details_layout.addWidget(self.op_params_stacked_widget)

        # --- 0. None Op Params (Placeholder) ---
        self.none_params_widget = QWidget()
        none_layout = QVBoxLayout(self.none_params_widget)
        none_layout.addWidget(QLabel("No parameters for 'none' operation."))
        none_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.none_params_widget) # Index 0

        # --- 1. Gaussian Blur Params ---
        self.gaussian_params_widget = QWidget()
        gaussian_layout = QVBoxLayout(self.gaussian_params_widget)
        gaussian_layout.addWidget(QLabel("Kernel Size (X, Y):"))
        ksize_layout = QHBoxLayout()
        self.gaussian_ksize_x_edit = QLineEdit()
        self.gaussian_ksize_x_edit.setFixedWidth(60)
        self.gaussian_ksize_x_edit.setValidator(QIntValidator(1, 99, self))
        ksize_layout.addWidget(self.gaussian_ksize_x_edit)
        self.gaussian_ksize_y_edit = QLineEdit()
        self.gaussian_ksize_y_edit.setFixedWidth(60)
        self.gaussian_ksize_y_edit.setValidator(QIntValidator(1, 99, self))
        ksize_layout.addWidget(self.gaussian_ksize_y_edit)
        ksize_layout.addStretch(1)
        gaussian_layout.addLayout(ksize_layout)

        gaussian_layout.addWidget(QLabel("Sigma (X, Y): (0 for auto)"))
        sigma_layout = QHBoxLayout()
        self.gaussian_sigma_x_edit = QLineEdit()
        self.gaussian_sigma_x_edit.setFixedWidth(60)
        self.gaussian_sigma_x_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        sigma_layout.addWidget(self.gaussian_sigma_x_edit)
        self.gaussian_sigma_y_edit = QLineEdit()
        self.gaussian_sigma_y_edit.setFixedWidth(60)
        self.gaussian_sigma_y_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        sigma_layout.addWidget(self.gaussian_sigma_y_edit)
        sigma_layout.addStretch(1)
        gaussian_layout.addLayout(sigma_layout)
        gaussian_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.gaussian_params_widget) # Index 1

        # --- 2. Bilateral Filter Params ---
        self.bilateral_params_widget = QWidget()
        bilateral_layout = QVBoxLayout(self.bilateral_params_widget)
        bilateral_layout.addWidget(QLabel("Diameter:"))
        self.bilateral_d_edit = QLineEdit()
        self.bilateral_d_edit.setFixedWidth(60)
        self.bilateral_d_edit.setValidator(QIntValidator(1, 99, self))
        bilateral_layout.addWidget(self.bilateral_d_edit)

        bilateral_layout.addWidget(QLabel("Sigma Color:"))
        self.bilateral_sigma_color_edit = QLineEdit()
        self.bilateral_sigma_color_edit.setFixedWidth(60)
        self.bilateral_sigma_color_edit.setValidator(QDoubleValidator(0.0, 255.0, 2, self))
        bilateral_layout.addWidget(self.bilateral_sigma_color_edit)

        bilateral_layout.addWidget(QLabel("Sigma Space:"))
        self.bilateral_sigma_space_edit = QLineEdit()
        self.bilateral_sigma_space_edit.setFixedWidth(60)
        self.bilateral_sigma_space_edit.setValidator(QDoubleValidator(0.0, 255.0, 2, self))
        bilateral_layout.addWidget(self.bilateral_sigma_space_edit)
        bilateral_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.bilateral_params_widget) # Index 2

        # --- 3. Median Blur Params ---
        self.median_params_widget = QWidget()
        median_layout = QVBoxLayout(self.median_params_widget)
        median_layout.addWidget(QLabel("Kernel Size:"))
        self.median_ksize_edit = QLineEdit()
        self.median_ksize_edit.setFixedWidth(60)
        self.median_ksize_edit.setValidator(QIntValidator(1, 99, self))
        median_layout.addWidget(self.median_ksize_edit)
        median_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.median_params_widget) # Index 3

        # --- 4. Unsharp Mask Params ---
        self.unsharp_params_widget = QWidget()
        unsharp_layout = QVBoxLayout(self.unsharp_params_widget)
        unsharp_layout.addWidget(QLabel("Amount:"))
        self.unsharp_amount_edit = QLineEdit()
        self.unsharp_amount_edit.setFixedWidth(60)
        self.unsharp_amount_edit.setValidator(QDoubleValidator(0.0, 5.0, 2, self))
        unsharp_layout.addWidget(self.unsharp_amount_edit)

        unsharp_layout.addWidget(QLabel("Threshold:"))
        self.unsharp_threshold_edit = QLineEdit()
        self.unsharp_threshold_edit.setFixedWidth(60)
        self.unsharp_threshold_edit.setValidator(QIntValidator(0, 255, self))
        unsharp_layout.addWidget(self.unsharp_threshold_edit)

        unsharp_layout.addWidget(QLabel("Internal Blur KSize:"))
        self.unsharp_blur_ksize_edit = QLineEdit()
        self.unsharp_blur_ksize_edit.setFixedWidth(60)
        self.unsharp_blur_ksize_edit.setValidator(QIntValidator(1, 99, self))
        unsharp_layout.addWidget(self.unsharp_blur_ksize_edit)

        unsharp_layout.addWidget(QLabel("Internal Blur Sigma:"))
        self.unsharp_blur_sigma_edit = QLineEdit()
        self.unsharp_blur_sigma_edit.setFixedWidth(60)
        self.unsharp_blur_sigma_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        unsharp_layout.addWidget(self.unsharp_blur_sigma_edit)
        unsharp_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.unsharp_params_widget) # Index 4

        # --- 5. Resize Params ---
        self.resize_params_widget = QWidget()
        resize_layout = QVBoxLayout(self.resize_params_widget)
        resize_layout.addWidget(QLabel("Width (px, 0 for auto):"))
        self.resize_width_edit = QLineEdit()
        self.resize_width_edit.setFixedWidth(80)
        self.resize_width_edit.setValidator(QIntValidator(0, 9999, self)) # Max 9999 for now
        resize_layout.addWidget(self.resize_width_edit)

        resize_layout.addWidget(QLabel("Height (px, 0 for auto):"))
        self.resize_height_edit = QLineEdit()
        self.resize_height_edit.setFixedWidth(80)
        self.resize_height_edit.setValidator(QIntValidator(0, 9999, self))
        resize_layout.addWidget(self.resize_height_edit)

        resize_layout.addWidget(QLabel("Resampling Mode:"))
        self.resample_mode_combo = QComboBox()
        self.resample_mode_combo.addItems(["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS4", "AREA"])
        resize_layout.addWidget(self.resample_mode_combo)
        resize_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.resize_params_widget) # Index 5

        # --- 6. Apply LUT Params (NEW) ---
        self.apply_lut_params_widget = QWidget()
        apply_lut_layout = QVBoxLayout(self.apply_lut_params_widget)
        apply_lut_layout.setContentsMargins(10, 10, 10, 10)
        apply_lut_layout.setSpacing(5)

        # LUT Source Selection
        lut_source_layout = QHBoxLayout()
        lut_source_layout.addWidget(QLabel("LUT Source:"))
        self.lut_source_combo = QComboBox()
        self.lut_source_combo.addItems(["Generated", "File"])
        lut_source_layout.addWidget(self.lut_source_combo)
        lut_source_layout.addStretch(1)
        apply_lut_layout.addLayout(lut_source_layout)

        # Stacked Widget for specific LUT generation parameters
        self.lut_gen_params_stacked_widget = QStackedWidget()
        apply_lut_layout.addWidget(self.lut_gen_params_stacked_widget)

        # --- LUT: Generated Type selection ---
        self.lut_generated_params_group = QGroupBox("Generated LUT Parameters")
        lut_generated_params_layout = QVBoxLayout(self.lut_generated_params_group)
        gen_type_layout = QHBoxLayout()
        gen_type_layout.addWidget(QLabel("Type:"))
        self.lut_generation_type_combo = QComboBox()
        self.lut_generation_type_combo.addItems(["linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard"])
        gen_type_layout.addWidget(self.lut_generation_type_combo)
        gen_type_layout.addStretch(1)
        lut_generated_params_layout.addLayout(gen_type_layout)
        
        # Sub-stacked widget for generated LUT specific params
        self.gen_lut_algo_params_stacked_widget = QStackedWidget()
        lut_generated_params_layout.addWidget(self.gen_lut_algo_params_stacked_widget)

        # --- LUT: Linear Algo Params ---
        self.lut_linear_params_widget = QWidget()
        linear_layout = QHBoxLayout(self.lut_linear_params_widget)
        linear_layout.addWidget(QLabel("Min Input:"))
        self.lut_linear_min_input_edit = QLineEdit()
        self.lut_linear_min_input_edit.setFixedWidth(60)
        self.lut_linear_min_input_edit.setValidator(QIntValidator(0, 255, self))
        linear_layout.addWidget(self.lut_linear_min_input_edit)
        linear_layout.addWidget(QLabel("Max Output:"))
        self.lut_linear_max_output_edit = QLineEdit()
        self.lut_linear_max_output_edit.setFixedWidth(60)
        self.lut_linear_max_output_edit.setValidator(QIntValidator(0, 255, self))
        linear_layout.addWidget(self.lut_linear_max_output_edit)
        linear_layout.addStretch(1)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_linear_params_widget) # Index 0

        # --- LUT: Gamma Algo Params ---
        self.lut_gamma_params_widget = QWidget()
        gamma_layout = QHBoxLayout(self.lut_gamma_params_widget)
        gamma_layout.addWidget(QLabel("Gamma Value:"))
        self.lut_gamma_value_edit = QLineEdit()
        self.lut_gamma_value_edit.setFixedWidth(60)
        self.lut_gamma_value_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        gamma_layout.addWidget(self.lut_gamma_value_edit)
        self.lut_gamma_value_slider = QSlider(Qt.Horizontal)
        self.lut_gamma_value_slider.setRange(1, 1000) # 0.01 to 10.00
        self.lut_gamma_value_slider.setSingleStep(1)
        gamma_layout.addWidget(self.lut_gamma_value_slider)
        gamma_layout.addStretch(1)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_gamma_params_widget) # Index 1

        # --- LUT: S-Curve Algo Params ---
        self.lut_s_curve_params_widget = QWidget()
        s_curve_layout = QHBoxLayout(self.lut_s_curve_params_widget)
        s_curve_layout.addWidget(QLabel("Contrast:"))
        self.lut_s_curve_contrast_edit = QLineEdit()
        self.lut_s_curve_contrast_edit.setFixedWidth(60)
        self.lut_s_curve_contrast_edit.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        s_curve_layout.addWidget(self.lut_s_curve_contrast_edit)
        self.lut_s_curve_contrast_slider = QSlider(Qt.Horizontal)
        self.lut_s_curve_contrast_slider.setRange(0, 100) # 0.0 to 1.0
        self.lut_s_curve_contrast_slider.setSingleStep(1)
        s_curve_layout.addWidget(self.lut_s_curve_contrast_slider)
        s_curve_layout.addStretch(1)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_s_curve_params_widget) # Index 2

        # --- LUT: Log Algo Params ---
        self.lut_log_params_widget = QWidget()
        log_layout = QHBoxLayout(self.lut_log_params_widget)
        log_layout.addWidget(QLabel("Param:"))
        self.lut_log_param_edit = QLineEdit()
        self.lut_log_param_edit.setFixedWidth(60)
        self.lut_log_param_edit.setValidator(QDoubleValidator(0.01, 100.0, 2, self))
        log_layout.addWidget(self.lut_log_param_edit)
        self.lut_log_param_slider = QSlider(Qt.Horizontal)
        self.lut_log_param_slider.setRange(1, 1000) # 0.01 to 100.0
        self.lut_log_param_slider.setSingleStep(1)
        log_layout.addWidget(self.lut_log_param_slider)
        log_layout.addStretch(1)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_log_params_widget) # Index 3

        # --- LUT: Exp Algo Params ---
        self.lut_exp_params_widget = QWidget()
        exp_layout = QHBoxLayout(self.lut_exp_params_widget)
        exp_layout.addWidget(QLabel("Param:"))
        self.lut_exp_param_edit = QLineEdit()
        self.lut_exp_param_edit.setFixedWidth(60)
        self.lut_exp_param_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        exp_layout.addWidget(self.lut_exp_param_edit)
        self.lut_exp_param_slider = QSlider(Qt.Horizontal)
        self.lut_exp_param_slider.setRange(1, 1000) # 0.01 to 10.0
        self.lut_exp_param_slider.setSingleStep(1)
        exp_layout.addWidget(self.lut_exp_param_slider)
        exp_layout.addStretch(1)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_exp_params_widget) # Index 4

        # --- LUT: Sqrt Algo Params (minimal) ---
        self.lut_sqrt_params_widget = QWidget()
        sqrt_layout = QHBoxLayout(self.lut_sqrt_params_widget)
        sqrt_layout.addWidget(QLabel("Param:")) # Placeholder, currently not used in `lut_manager`
        self.lut_sqrt_param_edit = QLineEdit("1.0")
        self.lut_sqrt_param_edit.setFixedWidth(60)
        self.lut_sqrt_param_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        sqrt_layout.addWidget(self.lut_sqrt_param_edit)
        sqrt_layout.addStretch(1)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_sqrt_params_widget) # Index 5

        # --- LUT: Rodbard Algo Params (minimal) ---
        self.lut_rodbard_params_widget = QWidget()
        rodbard_layout = QHBoxLayout(self.lut_rodbard_params_widget)
        rodbard_layout.addWidget(QLabel("Param:")) # Placeholder, currently not used in `lut_manager`
        self.lut_rodbard_param_edit = QLineEdit("1.0")
        rodbard_layout.addWidget(self.lut_rodbard_param_edit)
        rodbard_layout.addStretch(1)
        self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_rodbard_params_widget) # Index 6

        lut_generated_params_layout.addStretch(1) # Push content to top
        self.lut_gen_params_stacked_widget.addWidget(self.lut_generated_params_group) # Index 0 for generated group

        # --- LUT: File-based Params ---
        self.lut_file_params_group = QGroupBox("File-based LUT Controls")
        lut_file_params_layout = QVBoxLayout(self.lut_file_params_group)
        lut_path_layout = QHBoxLayout()
        lut_path_layout.addWidget(QLabel("LUT File:"))
        self.lut_filepath_edit = QLineEdit() # This will display the path of loaded/saved LUT
        self.lut_filepath_edit.setReadOnly(True)
        lut_path_layout.addWidget(self.lut_filepath_edit)
        lut_file_params_layout.addLayout(lut_path_layout)

        file_button_layout = QHBoxLayout()
        self.lut_load_file_button = QPushButton("Load from File...")
        file_button_layout.addWidget(self.lut_load_file_button)
        self.lut_save_file_button = QPushButton("Save to File...") # This button saves the current previewed LUT
        file_button_layout.addWidget(self.lut_save_file_button)
        file_button_layout.addStretch(1)
        lut_file_params_layout.addLayout(file_button_layout)
        lut_file_params_layout.addStretch(1)
        self.lut_gen_params_stacked_widget.addWidget(self.lut_file_params_group) # Index 1 for file group

        # --- LUT Preview Plot (NEW) ---
        lut_preview_group = QGroupBox("LUT Curve Preview")
        lut_preview_layout = QVBoxLayout(lut_preview_group)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setMinimumHeight(200)
        lut_preview_layout.addWidget(self.canvas)
        apply_lut_layout.addWidget(lut_preview_group)

        apply_lut_layout.addStretch(1) # Push content to top of apply_lut_params_widget
        self.op_params_stacked_widget.addWidget(self.apply_lut_params_widget) # Index 6 for apply_lut

        details_layout.addStretch(1) # Push content to top of details group
        right_panel_layout.addWidget(self.details_group)
        right_panel_layout.addStretch(1) # Push content to top of right panel

        main_layout.addLayout(right_panel_layout)
        main_layout.addStretch(1) # Push all content to left


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.ops_list_widget.currentRowChanged.connect(self._update_selected_operation_details)
        self.ops_list_widget.model().rowsMoved.connect(self._reorder_operations_in_config)

        self.add_op_button.clicked.connect(self._add_operation)
        self.remove_op_button.clicked.connect(self._remove_operation)
        self.move_up_button.clicked.connect(self._move_operation_up)
        self.move_down_button.clicked.connect(self._move_operation_down)

        self.selected_op_type_combo.currentTextChanged.connect(self._on_selected_op_type_changed)

        # Connect parameter edits to update config when text changes
        # Gaussian
        self.gaussian_ksize_x_edit.textChanged.connect(lambda text: self._update_param_in_config("gaussian_ksize_x", text, int))
        self.gaussian_ksize_y_edit.textChanged.connect(lambda text: self._update_param_in_config("gaussian_ksize_y", text, int))
        self.gaussian_sigma_x_edit.textChanged.connect(lambda text: self._update_param_in_config("gaussian_sigma_x", text, float))
        self.gaussian_sigma_y_edit.textChanged.connect(lambda text: self._update_param_in_config("gaussian_sigma_y", text, float))
        
        # Bilateral
        self.bilateral_d_edit.textChanged.connect(lambda text: self._update_param_in_config("bilateral_d", text, int))
        self.bilateral_sigma_color_edit.textChanged.connect(lambda text: self._update_param_in_config("bilateral_sigma_color", text, float))
        self.bilateral_sigma_space_edit.textChanged.connect(lambda text: self._update_param_in_config("bilateral_sigma_space", text, float))

        # Median
        self.median_ksize_edit.textChanged.connect(lambda text: self._update_param_in_config("median_ksize", text, int))

        # Unsharp
        self.unsharp_amount_edit.textChanged.connect(lambda text: self._update_param_in_config("unsharp_amount", text, float))
        self.unsharp_threshold_edit.textChanged.connect(lambda text: self._update_param_in_config("unsharp_threshold", text, int))
        self.unsharp_blur_ksize_edit.textChanged.connect(lambda text: self._update_param_in_config("unsharp_blur_ksize", text, int))
        self.unsharp_blur_sigma_edit.textChanged.connect(lambda text: self._update_param_in_config("unsharp_blur_sigma", text, float))

        # Resize
        self.resize_width_edit.textChanged.connect(lambda text: self._update_param_in_config("resize_width", text, int, allow_none_if_zero=True))
        self.resize_height_edit.textChanged.connect(lambda text: self._update_param_in_config("resize_height", text, int, allow_none_if_zero=True))
        self.resample_mode_combo.currentTextChanged.connect(lambda text: self._update_param_in_config("resample_mode", text, str))

        # LUT Parameters (NEW connections)
        self.lut_source_combo.currentTextChanged.connect(self._on_lut_source_changed)
        self.lut_generation_type_combo.currentTextChanged.connect(self._on_lut_gen_type_changed)

        self.lut_linear_min_input_edit.textChanged.connect(lambda text: self._update_lut_param_in_config("linear_min_input", text, int))
        self.lut_linear_max_output_edit.textChanged.connect(lambda text: self._update_lut_param_in_config("linear_max_output", text, int))
        
        self.lut_gamma_value_edit.textChanged.connect(lambda text: self._update_lut_param_in_config("gamma_value", text, float, slider=self.lut_gamma_value_slider, scale_factor=100.0))
        self.lut_gamma_value_slider.valueChanged.connect(lambda val: self.lut_gamma_value_edit.setText(f"{val / 100.0:.2f}"))

        self.lut_s_curve_contrast_edit.textChanged.connect(lambda text: self._update_lut_param_in_config("s_curve_contrast", text, float, slider=self.lut_s_curve_contrast_slider, scale_factor=100.0))
        self.lut_s_curve_contrast_slider.valueChanged.connect(lambda val: self.lut_s_curve_contrast_edit.setText(f"{val / 100.0:.2f}"))

        self.lut_log_param_edit.textChanged.connect(lambda text: self._update_lut_param_in_config("log_param", text, float, slider=self.lut_log_param_slider, scale_factor=10.0))
        self.lut_log_param_slider.valueChanged.connect(lambda val: self.lut_log_param_edit.setText(f"{val / 10.0:.1f}"))

        self.lut_exp_param_edit.textChanged.connect(lambda text: self._update_lut_param_in_config("exp_param", text, float, slider=self.lut_exp_param_slider, scale_factor=100.0))
        self.lut_exp_param_slider.valueChanged.connect(lambda val: self.lut_exp_param_edit.setText(f"{val / 100.0:.2f}"))

        self.lut_sqrt_param_edit.textChanged.connect(lambda text: self._update_lut_param_in_config("sqrt_param", text, float))
        self.lut_rodbard_param_edit.textChanged.connect(lambda text: self._update_lut_param_in_config("rodbard_param", text, float))

        self.lut_load_file_button.clicked.connect(self._load_lut_from_file_for_op)
        self.lut_save_file_button.clicked.connect(self._save_lut_to_file_from_op)


    def _update_operation_list(self):
        """Populates the QListWidget from the config's xy_blend_pipeline."""
        self.ops_list_widget.clear()
        for i, op in enumerate(self.config.xy_blend_pipeline):
            item = QListWidgetItem(f"{i+1}. {op.type.replace('_', ' ').title()}")
            self.ops_list_widget.addItem(item)
        
        # Select the first item if list is not empty
        if self.ops_list_widget.count() > 0:
            self.ops_list_widget.setCurrentRow(0)
        else:
            self.selected_op_type_combo.setCurrentText("none") # Reset type combo
            self.op_params_stacked_widget.setCurrentWidget(self.none_params_widget) # Show none params


    def _update_selected_operation_details(self):
        """
        Updates the details panel (combo box and stacked widget)
        based on the currently selected item in the operations list.
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.config.xy_blend_pipeline):
            selected_op = self.config.xy_blend_pipeline[current_row]
            
            # Temporarily block signals to avoid re-triggering updates during UI population
            self._block_all_param_signals(True)

            self.selected_op_type_combo.setCurrentText(selected_op.type)
            self._show_params_for_type(selected_op.type)
            self._populate_params_widgets(selected_op)
            self.details_group.setEnabled(True) # Enable details group
            
            self._block_all_param_signals(False)

            # After populating, if it's an apply_lut, update the plot
            if selected_op.type == "apply_lut":
                self._plot_current_lut_preview(selected_op.lut_params)

        else:
            # No item selected or list is empty
            self.details_group.setEnabled(False) # Disable details group
            self.selected_op_type_combo.setCurrentText("none") # Reset type combo
            self.op_params_stacked_widget.setCurrentWidget(self.none_params_widget) # Show none params
            self.canvas.axes.clear() # Clear LUT plot if no op selected
            self.canvas.draw()


    def _on_selected_op_type_changed(self, new_type: str):
        """
        Called when the operation type combo box for the selected item changes.
        Replaces the current operation in the config with a new one of the chosen type,
        preserving specific attributes if relevant (e.g., if changing from one LUT type to another).
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row < 0 or current_row >= len(self.config.xy_blend_pipeline):
            return

        old_op = self.config.xy_blend_pipeline[current_row]

        if new_type == old_op.type:
            # Type is not actually changing, just repopulate UI from existing values
            # This handles cases where user clicks on the same type in the combobox
            # or when initial UI population triggers this signal.
            self._update_selected_operation_details() # Re-read current state and plot
            return

        # Type is changing, create a new operation with defaults for the new type
        new_op = XYBlendOperation(type=new_type)

        # Special handling for preserving some state if relevant across type changes
        if new_type == "apply_lut" and old_op.type == "apply_lut":
             # If changing between apply_lut operations, preserve the lut_params
             new_op.lut_params = copy.deepcopy(old_op.lut_params)

        # Replace the old operation object in the pipeline with the new one
        self.config.xy_blend_pipeline[current_row] = new_op
        
        # Block signals temporarily to prevent redundant updates during UI population
        self._block_all_param_signals(True)

        # Update the list item text to reflect the new type
        self.ops_list_widget.currentItem().setText(f"{current_row+1}. {new_type.replace('_', ' ').title()}")

        # Show the correct parameters widget and populate with new defaults/copied values
        self._show_params_for_type(new_type)
        self._populate_params_widgets(new_op) # Pass the new_op to populate from its values
        
        self._block_all_param_signals(False)

        # If the new type is apply_lut, update the plot
        if new_type == "apply_lut":
            self._plot_current_lut_preview(new_op.lut_params) # Plot from the new op's lut_params
        else:
            self.canvas.axes.clear() # Clear LUT plot if not apply_lut
            self.canvas.draw()


    def _show_params_for_type(self, op_type: str):
        """Switches the stacked widget to show parameters for the given operation type."""
        widget_map = {
            "none": self.none_params_widget,
            "gaussian_blur": self.gaussian_params_widget,
            "bilateral_filter": self.bilateral_params_widget,
            "median_blur": self.median_params_widget,
            "unsharp_mask": self.unsharp_params_widget,
            "resize": self.resize_params_widget,
            "apply_lut": self.apply_lut_params_widget # New LUT params widget
        }
        self.op_params_stacked_widget.setCurrentWidget(widget_map.get(op_type, self.none_params_widget))
        

    def _populate_params_widgets(self, op: XYBlendOperation):
        """Populates the parameter QLineEdits with values from the given XYBlendOperation."""
        # Disconnect signals temporarily to avoid triggering _update_param_in_config during population
        self._block_all_param_signals(True)

        # Clear all edits first to ensure fresh state
        all_line_edits = [
            self.gaussian_ksize_x_edit, self.gaussian_ksize_y_edit, self.gaussian_sigma_x_edit, self.gaussian_sigma_y_edit,
            self.bilateral_d_edit, self.bilateral_sigma_color_edit, self.bilateral_sigma_space_edit,
            self.median_ksize_edit,
            self.unsharp_amount_edit, self.unsharp_threshold_edit, self.unsharp_blur_ksize_edit, self.unsharp_blur_sigma_edit,
            self.resize_width_edit, self.resize_height_edit,
            # LUT related edits
            self.lut_linear_min_input_edit, self.lut_linear_max_output_edit,
            self.lut_gamma_value_edit, self.lut_s_curve_contrast_edit,
            self.lut_log_param_edit, self.lut_exp_param_edit,
            self.lut_sqrt_param_edit, self.lut_rodbard_param_edit,
            self.lut_filepath_edit
        ]
        for edit in all_line_edits:
            edit.setText("") # Clear text
            edit.setStyleSheet("") # Clear any error styling

        # Populate based on type
        if op.type == "gaussian_blur":
            self.gaussian_ksize_x_edit.setText(str(op.gaussian_ksize_x))
            self.gaussian_ksize_y_edit.setText(str(op.gaussian_ksize_y))
            self.gaussian_sigma_x_edit.setText(str(op.gaussian_sigma_x))
            self.gaussian_sigma_y_edit.setText(str(op.gaussian_sigma_y))
        elif op.type == "bilateral_filter":
            self.bilateral_d_edit.setText(str(op.bilateral_d))
            self.bilateral_sigma_color_edit.setText(str(op.bilateral_sigma_color))
            self.bilateral_sigma_space_edit.setText(str(op.bilateral_sigma_space))
        elif op.type == "median_blur":
            self.median_ksize_edit.setText(str(op.median_ksize))
        elif op.type == "unsharp_mask":
            self.unsharp_amount_edit.setText(str(op.unsharp_amount))
            self.unsharp_threshold_edit.setText(str(op.unsharp_threshold))
            self.unsharp_blur_ksize_edit.setText(str(op.unsharp_blur_ksize))
            self.unsharp_blur_sigma_edit.setText(str(op.unsharp_blur_sigma))
        elif op.type == "resize":
            # Convert None to 0 for display in QLineEdit for resize dimensions
            self.resize_width_edit.setText(str(op.resize_width or 0))
            self.resize_height_edit.setText(str(op.resize_height or 0))
            self.resample_mode_combo.setCurrentText(op.resample_mode)
        elif op.type == "apply_lut":
            # Populate LUT specific parameters
            self.lut_source_combo.setCurrentText(op.lut_params.lut_source.capitalize())
            
            # Trigger source update first to set correct sub-widget visibility
            self._update_lut_source_controls_widget_only(op.lut_params.lut_source.capitalize())
            
            # Populate generated LUT parameters
            self.lut_generation_type_combo.setCurrentText(op.lut_params.lut_generation_type) # Triggers update for algo params
            self._update_lut_gen_type_controls_widget_only(op.lut_params.lut_generation_type) # Ensure correct algo sub-widget

            self.lut_linear_min_input_edit.setText(str(op.lut_params.linear_min_input))
            self.lut_linear_max_output_edit.setText(str(op.lut_params.linear_max_output))
            self.lut_gamma_value_edit.setText(str(op.lut_params.gamma_value))
            self.lut_gamma_value_slider.setValue(int(op.lut_params.gamma_value * 100))
            self.lut_s_curve_contrast_edit.setText(str(op.lut_params.s_curve_contrast))
            self.lut_s_curve_contrast_slider.setValue(int(op.lut_params.s_curve_contrast * 100))
            self.lut_log_param_edit.setText(str(op.lut_params.log_param))
            self.lut_log_param_slider.setValue(int(op.lut_params.log_param * 10))
            self.lut_exp_param_edit.setText(str(op.lut_params.exp_param))
            self.lut_exp_param_slider.setValue(int(op.lut_params.exp_param * 100))
            self.lut_sqrt_param_edit.setText(str(op.lut_params.sqrt_param))
            self.lut_rodbard_param_edit.setText(str(op.lut_params.rodbard_param))

            # Populate file-based LUT parameter
            self.lut_filepath_edit.setText(op.lut_params.fixed_lut_path)
            

        self._block_all_param_signals(False) # Reconnect signals

    def _block_all_param_signals(self, block: bool):
        """Blocks/unblocks signals for all parameter QLineEdits and QComboBoxes/QSliders."""
        # List all widgets whose signals should be blocked during UI population
        widgets_to_block = [
            self.gaussian_ksize_x_edit, self.gaussian_ksize_y_edit, self.gaussian_sigma_x_edit, self.gaussian_sigma_y_edit,
            self.bilateral_d_edit, self.bilateral_sigma_color_edit, self.bilateral_sigma_space_edit,
            self.median_ksize_edit,
            self.unsharp_amount_edit, self.unsharp_threshold_edit, self.unsharp_blur_ksize_edit, self.unsharp_blur_sigma_edit,
            self.resize_width_edit, self.resize_height_edit, self.resample_mode_combo,
            # LUT related widgets
            self.lut_source_combo, self.lut_generation_type_combo,
            self.lut_linear_min_input_edit, self.lut_linear_max_output_edit,
            self.lut_gamma_value_edit, self.lut_gamma_value_slider,
            self.lut_s_curve_contrast_edit, self.lut_s_curve_contrast_slider,
            self.lut_log_param_edit, self.lut_log_param_slider,
            self.lut_exp_param_edit, self.lut_exp_param_slider,
            self.lut_sqrt_param_edit, self.lut_rodbard_param_edit,
            self.lut_filepath_edit,
            # self.lut_load_file_button, self.lut_save_file_button # Buttons should also be blocked during update if they trigger change events
        ]
        for widget in widgets_to_block:
            widget.blockSignals(block)

    def _update_param_in_config(self, param_name: str, text: str, data_type: type, allow_none_if_zero: bool = False):
        """
        Updates a parameter directly on the currently selected XYBlendOperation object.
        Applies basic type conversion and handles post_init validation feedback.
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row < 0 or current_row >= len(self.config.xy_blend_pipeline):
            return # No operation selected

        selected_op = self.config.xy_blend_pipeline[current_row]
        
        sender_widget = self.sender()
        if not (isinstance(sender_widget, QLineEdit) or isinstance(sender_widget, QComboBox)):
            return # Ensure it's a widget we expect

        value_to_set = None
        if not text and allow_none_if_zero:
            value_to_set = None
            sender_widget.setStyleSheet("")
        else:
            try:
                if data_type == int:
                    value_to_set = int(text)
                elif data_type == float:
                    value_to_set = float(text)
                else: # For string types like resample_mode
                    value_to_set = text
                sender_widget.setStyleSheet("") # Clear any error styling
            except ValueError:
                sender_widget.setStyleSheet("border: 1px solid red;")
                return # Do not update config with invalid value

        # Update the parameter directly on the selected operation object
        setattr(selected_op, param_name, value_to_set)
        
        # Re-run __post_init__ to apply validation/corrections (e.g., odd ksize)
        selected_op.__post_init__() 
        
        # If the value was corrected by __post_init__, update the UI
        corrected_value = getattr(selected_op, param_name)
        if corrected_value != value_to_set and isinstance(sender_widget, QLineEdit):
            # Temporarily block signals to avoid recursion during UI update
            sender_widget.blockSignals(True)
            sender_widget.setText(str(corrected_value))
            sender_widget.blockSignals(False)


    def _update_lut_param_in_config(self, param_name: str, text: str, data_type: type, slider: Optional[QSlider] = None, scale_factor: float = 1.0):
        """
        Updates a parameter in the currently selected XYBlendOperation's LutParameters.
        Handles type conversion, validation, and slider synchronization.
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row < 0 or current_row >= len(self.config.xy_blend_pipeline):
            return # No operation selected

        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut":
            return # Only update if it's an apply_lut operation

        sender_widget = self.sender()
        if not (isinstance(sender_widget, QLineEdit) or isinstance(sender_widget, QComboBox) or isinstance(sender_widget, QSlider)):
            return

        value_to_set = None
        try:
            if data_type == int:
                value_to_set = int(text)
            elif data_type == float:
                value_to_set = float(text)
            else:
                value_to_set = text # String (e.g., lut source, type)

            if isinstance(sender_widget, QLineEdit):
                sender_widget.setStyleSheet("") # Clear any error styling
            
            # Update slider if applicable and signal came from QLineEdit
            if slider and isinstance(sender_widget, QLineEdit):
                slider.blockSignals(True)
                slider.setValue(int(value_to_set * scale_factor))
                slider.blockSignals(False)

        except ValueError:
            if isinstance(sender_widget, QLineEdit):
                sender_widget.setStyleSheet("border: 1px solid red;")
            # No return here, allow invalid text to stay in the QLineEdit with red border.
            # The actual LUT generation will gracefully handle errors.
            pass 

        # Update the parameter directly on the LutParameters object, BUT ONLY if it's a valid number.
        # This prevents setting attributes to None if the text is invalid or empty string.
        if value_to_set is not None or (not text and param_name == "fixed_lut_path"): # Allow empty path for file source
            setattr(selected_op.lut_params, param_name, value_to_set)
        
        # Re-run LutParameters.__post_init__ to apply validation/corrections
        selected_op.lut_params.__post_init__()

        # If the value was corrected by __post_init__, update the UI
        corrected_value = getattr(selected_op.lut_params, param_name)
        if corrected_value != value_to_set and isinstance(sender_widget, QLineEdit):
            sender_widget.blockSignals(True)
            sender_widget.setText(str(corrected_value))
            sender_widget.blockSignals(False)
            if slider and isinstance(sender_widget, QLineEdit):
                slider.blockSignals(True)
                slider.setValue(int(corrected_value * scale_factor))
                slider.blockSignals(False)

        # Handle source/type specific widget visibility after parameter update
        # This ensures the correct stacked widget is visible even if value was corrected
        if param_name == "lut_source":
            self._update_lut_source_controls_widget_only(selected_op.lut_params.lut_source.capitalize())
        elif param_name == "lut_generation_type":
            self._update_lut_gen_type_controls_widget_only(selected_op.lut_params.lut_generation_type)

        # Always update the plot after any LUT parameter changes
        self._plot_current_lut_preview(selected_op.lut_params)


    def _on_lut_source_changed(self, source_text: str):
        """Called when LUT source combo box changes for an 'apply_lut' op."""
        current_row = self.ops_list_widget.currentRow()
        if current_row < 0 or current_row >= len(self.config.xy_blend_pipeline):
            return

        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut":
            return
        
        self._block_all_param_signals(True) # Block signals during update to prevent recursion
        selected_op.lut_params.lut_source = source_text.lower()
        selected_op.lut_params.__post_init__() # Re-validate lut_params
        self._update_lut_source_controls_widget_only(source_text)
        self._block_all_param_signals(False) # Reconnect signals

        # Update the plot after change
        self._plot_current_lut_preview(selected_op.lut_params)


    def _update_lut_source_controls_widget_only(self, source_text: str):
        """Sets visibility for LUT source (generated vs. file) specific controls."""
        is_generated = (source_text.lower() == "generated")
        self.lut_gen_params_stacked_widget.setCurrentIndex(0 if is_generated else 1)
        # 0 for Generated group, 1 for File group


    def _on_lut_gen_type_changed(self, lut_type: str):
        """Called when LUT generation type combo box changes for an 'apply_lut' op."""
        current_row = self.ops_list_widget.currentRow()
        if current_row < 0 or current_row >= len(self.config.xy_blend_pipeline):
            return

        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut":
            return
        
        self._block_all_param_signals(True) # Block signals during update
        selected_op.lut_params.lut_generation_type = lut_type.lower()
        selected_op.lut_params.__post_init__() # Re-validate lut_params
        self._update_lut_gen_type_controls_widget_only(lut_type)
        # When generation type changes, re-populate params for this type with defaults
        self._populate_params_widgets(selected_op) # This will re-populate all controls for the op
        self._block_all_param_signals(False) # Reconnect signals

        # Update the plot after change
        self._plot_current_lut_preview(selected_op.lut_params)


    def _update_lut_gen_type_controls_widget_only(self, lut_type: str):
        """Switches the sub-stacked widget to show parameters for the selected LUT generation type."""
        widget_map = {
            "linear": self.lut_linear_params_widget,
            "gamma": self.lut_gamma_params_widget,
            "s_curve": self.lut_s_curve_params_widget,
            "log": self.lut_log_params_widget,
            "exp": self.lut_exp_params_widget,
            "sqrt": self.lut_sqrt_params_widget,
            "rodbard": self.lut_rodbard_params_widget
        }
        self.gen_lut_algo_params_stacked_widget.setCurrentWidget(widget_map.get(lut_type.lower(), self.lut_linear_params_widget))


    def _plot_current_lut_preview(self, lut_params: LutParameters):
        """
        Generates/loads the LUT based on the provided LutParameters and plots it.
        """
        generated_lut: Optional[np.ndarray] = None
        try:
            if lut_params.lut_source == "generated":
                if lut_params.lut_generation_type == "linear":
                    generated_lut = lut_manager.generate_linear_lut(lut_params.linear_min_input, lut_params.linear_max_output)
                elif lut_params.lut_generation_type == "gamma":
                    generated_lut = lut_manager.generate_gamma_lut(lut_params.gamma_value)
                elif lut_params.lut_generation_type == "s_curve":
                    generated_lut = lut_manager.generate_s_curve_lut(lut_params.s_curve_contrast)
                elif lut_params.lut_generation_type == "log":
                    generated_lut = lut_manager.generate_log_lut(lut_params.log_param)
                elif lut_params.lut_generation_type == "exp":
                    generated_lut = lut_manager.generate_exp_lut(lut_params.exp_param)
                elif lut_params.lut_generation_type == "sqrt":
                    generated_lut = lut_manager.generate_sqrt_lut(lut_params.sqrt_param)
                elif lut_params.lut_generation_type == "rodbard":
                    generated_lut = lut_manager.generate_rodbard_lut(lut_params.rodbard_param)
                else:
                    generated_lut = lut_manager.get_default_z_lut() # Fallback
            elif lut_params.lut_source == "file":
                if lut_params.fixed_lut_path and os.path.exists(lut_params.fixed_lut_path):
                    generated_lut = lut_manager.load_lut(lut_params.fixed_lut_path)
                else:
                    generated_lut = lut_manager.get_default_z_lut() # Fallback if file not found/path empty
            else:
                generated_lut = lut_manager.get_default_z_lut() # Fallback for unknown source

            if generated_lut is None: # Should not happen with fallbacks, but safety
                generated_lut = lut_manager.get_default_z_lut()

        except Exception as e:
            print(f"Error generating/loading LUT for plot preview: {e}")
            generated_lut = lut_manager.get_default_z_lut() # Use default on error

        self.canvas.axes.clear()
        x_values = np.arange(256)
        y_values = generated_lut

        self.canvas.axes.plot(x_values, y_values, 'b-')
        self.canvas.axes.set_title("LUT Curve Preview")
        self.canvas.axes.set_xlabel("Input Value (0-255)")
        self.canvas.axes.set_ylabel("Output Value (0-255)")
        self.canvas.axes.set_xlim(0, 255)
        self.canvas.axes.set_ylim(0, 255)
        self.canvas.axes.grid(True)
        self.canvas.draw()


    def _add_operation(self):
        """Adds a new 'none' operation to the pipeline."""
        new_op = XYBlendOperation(type="none")
        self.config.xy_blend_pipeline.append(new_op)
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(len(self.config.xy_blend_pipeline) - 1) # Select the new item


    def _remove_operation(self):
        """Removes the currently selected operation from the pipeline."""
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.config.xy_blend_pipeline):
            reply = QMessageBox.question(self, "Remove Operation",
                                         f"Are you sure you want to remove operation {current_row+1}?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.config.xy_blend_pipeline[current_row]
                self._update_operation_list()
                # Select the item before the removed one, or the new first item
                if self.ops_list_widget.count() > 0:
                    new_row = min(current_row, self.ops_list_widget.count() - 1)
                    self.ops_list_widget.setCurrentRow(new_row)
                else:
                    self._update_selected_operation_details() # Clear details if list is empty


    def _move_operation_up(self):
        """Moves the selected operation up in the pipeline list."""
        current_row = self.ops_list_widget.currentRow()
        if current_row > 0:
            # Swap in the config list
            self.config.xy_blend_pipeline[current_row], self.config.xy_blend_pipeline[current_row - 1] = \
                self.config.xy_blend_pipeline[current_row - 1], self.config.xy_blend_pipeline[current_row]
            self._update_operation_list()
            self.ops_list_widget.setCurrentRow(current_row - 1) # Keep selection on the moved item


    def _move_operation_down(self):
        """Moves the selected operation down in the pipeline list."""
        current_row = self.ops_list_widget.currentRow()
        if current_row < len(self.config.xy_blend_pipeline) - 1:
            # Swap in the config list
            self.config.xy_blend_pipeline[current_row], self.config.xy_blend_pipeline[current_row + 1] = \
                self.config.xy_blend_pipeline[current_row + 1], self.config.xy_blend_pipeline[current_row]
            self._update_operation_list()
            self.ops_list_widget.setCurrentRow(current_row + 1) # Keep selection on the moved item

    def _reorder_operations_in_config(self, parent, start, end, destination, row):
        """
        Slot for QListWidget.model().rowsMoved.
        Updates the config's xy_blend_pipeline to reflect the reordering.
        """
        # This signal is emitted *before* the model is actually changed.
        # So, we need to get the item being moved *before* the move,
        # and then re-insert it at the new position.
        
        # Get the item that was moved
        moved_item_index = start
        moved_op = self.config.xy_blend_pipeline.pop(moved_item_index)

        # Calculate the new insertion index
        # If moving down, and the destination is after the original position,
        # the destination index will be effectively one less due to the pop.
        new_index = row
        if destination == Qt.BottomToTop: # Moving up
            if new_index > moved_item_index:
                new_index -= 1
        else: # Moving down
            if new_index < moved_item_index:
                new_index += 1

        self.config.xy_blend_pipeline.insert(new_index, moved_op)
        
        # After reordering, re-populate the list widget to ensure numbering is correct
        # and re-select the moved item.
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(new_index)

    def _load_lut_from_file_for_op(self):
        """
        Opens a file dialog to load a LUT for the currently selected 'apply_lut' operation.
        Updates the operation's lut_params.fixed_lut_path and the UI.
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row < 0 or current_row >= len(self.config.xy_blend_pipeline):
            QMessageBox.warning(self, "No Operation Selected", "Please select an 'Apply LUT' operation first.")
            return

        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut":
            QMessageBox.warning(self, "Invalid Operation Type", "Please select an 'Apply LUT' operation to load a LUT file.")
            return

        initial_path = selected_op.lut_params.fixed_lut_path or ""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load LUT File for Operation", initial_path, "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            try:
                # Validate the LUT file content by trying to load it
                temp_lut = lut_manager.load_lut(filepath) 
                
                # Update the config object directly
                self._block_all_param_signals(True) # Block signals during this update
                selected_op.lut_params.fixed_lut_path = filepath
                selected_op.lut_params.lut_source = "file"
                selected_op.lut_params.__post_init__() # Re-validate lut_params

                # Update the UI
                self.lut_filepath_edit.setText(filepath)
                self.lut_source_combo.setCurrentText("File")
                self._block_all_param_signals(False) # Reconnect signals

                # Update the plot
                self._plot_current_lut_preview(selected_op.lut_params)
                QMessageBox.information(self, "Load Success", f"LUT file set for operation: '{os.path.basename(filepath)}'.")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load/validate LUT file: {e}")
                self.lut_filepath_edit.setText("") # Clear invalid path
                # Plot default LUT on error
                self._plot_current_lut_preview(selected_op.lut_params) # This will default to linear


    def _save_lut_to_file_from_op(self):
        """
        Opens a file dialog to save the currently defined LUT for the selected 'apply_lut' operation.
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row < 0 or current_row >= len(self.config.xy_blend_pipeline):
            QMessageBox.warning(self, "No Operation Selected", "Please select an 'Apply LUT' operation first.")
            return

        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut":
            QMessageBox.warning(self, "Invalid Operation Type", "Please select an 'Apply LUT' operation to save its LUT.")
            return

        # Generate the LUT temporarily from the operation's parameters
        generated_lut_array = None
        try:
            if selected_op.lut_params.lut_source == "generated":
                lut_params = selected_op.lut_params
                if lut_params.lut_generation_type == "linear":
                    generated_lut_array = lut_manager.generate_linear_lut(lut_params.linear_min_input, lut_params.linear_max_output)
                elif lut_params.lut_generation_type == "gamma":
                    generated_lut_array = lut_manager.generate_gamma_lut(lut_params.gamma_value)
                elif lut_params.lut_generation_type == "s_curve":
                    generated_lut_array = lut_manager.generate_s_curve_lut(lut_params.s_curve_contrast)
                elif lut_params.lut_generation_type == "log":
                    generated_lut_array = lut_manager.generate_log_lut(lut_params.log_param)
                elif lut_params.lut_generation_type == "exp":
                    generated_lut_array = lut_manager.generate_exp_lut(lut_params.exp_param)
                elif lut_params.lut_generation_type == "sqrt":
                    generated_lut_array = lut_manager.generate_sqrt_lut(lut_params.sqrt_param)
                elif lut_params.lut_generation_type == "rodbard":
                    generated_lut_array = lut_manager.generate_rodbard_lut(lut_params.rodbard_param)
                else:
                    raise ValueError(f"Unknown LUT generation type: {lut_params.lut_generation_type}")
            elif selected_op.lut_params.lut_source == "file":
                if selected_op.lut_params.fixed_lut_path and os.path.exists(selected_op.lut_params.fixed_lut_path):
                    generated_lut_array = lut_manager.load_lut(selected_op.lut_params.fixed_lut_path)
                else:
                    QMessageBox.warning(self, "LUT File Not Found", "No valid LUT file specified to save, or file does not exist. Cannot save.")
                    return
            else:
                raise ValueError(f"Unknown LUT source: {selected_op.lut_params.lut_source}")

            if generated_lut_array is None:
                raise ValueError("Could not generate or load LUT for saving.")

        except Exception as e:
            QMessageBox.critical(self, "LUT Save Error", f"Failed to prepare LUT for saving: {e}")
            return

        # Proceed with saving the generated/loaded LUT
        default_filename = os.path.basename(selected_op.lut_params.fixed_lut_path) if selected_op.lut_params.fixed_lut_path else "custom_lut_op.json"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save LUT for Operation", default_filename, "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            try:
                lut_manager.save_lut(filepath, generated_lut_array)
                QMessageBox.information(self, "Save Success", f"LUT saved to '{filepath}'.")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save LUT: {e}")


    def get_config(self) -> dict:
        """
        Collects current settings for the XY blend pipeline.
        The XYBlendOperation objects in config.xy_blend_pipeline are already
        kept up-to-date by the _update_param_in_config and _on_selected_op_type_changed methods.
        So, we just need to return a representation of the pipeline.
        """
        # The config.xy_blend_pipeline is already the source of truth.
        # We just need to ensure it's correctly serialized by Config.to_dict().
        # No direct collection needed here, as changes are applied immediately.
        return {"xy_blend_pipeline": [op.to_dict() for op in self.config.xy_blend_pipeline]}

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        # This method is called during initial GUI setup and when loading settings.
        # It should populate the GUI elements with values from the provided Config object.
        
        # Disconnect all signals to prevent immediate re-triggering of logic during population
        self._block_all_param_signals(True)
        
        # This will also ensure correct stacked widget visibility
        self._update_operation_list() 
        self._update_selected_operation_details() # This will populate all fields and plot
        
        self._block_all_param_signals(False) # Reconnect all signals