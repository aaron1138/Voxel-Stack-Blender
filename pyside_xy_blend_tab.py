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

# pyside_xy_blend_tab.py (Corrected)

import os
import numpy as np
import json
import copy

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QPushButton, QListWidget,
    QListWidgetItem, QStackedWidget, QMessageBox, QFileDialog,
    QAbstractItemView, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator
from typing import Optional

from config import app_config as config, Config, XYBlendOperation, LutParameters 
import lut_manager
from lut_editor_widget import LutEditorWidget

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
        
        # --- Left Panel: Operations List and Controls ---
        left_panel_layout = QVBoxLayout()
        ops_list_group = QGroupBox("Processing Pipeline")
        ops_list_layout = QVBoxLayout(ops_list_group)
        self.ops_list_widget = QListWidget()
        self.ops_list_widget.setDragDropMode(QListWidget.InternalMove)
        self.ops_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ops_list_widget.setMinimumWidth(200)
        ops_list_layout.addWidget(self.ops_list_widget)

        op_buttons_layout = QHBoxLayout()
        self.add_op_button = QPushButton("Add Op"); op_buttons_layout.addWidget(self.add_op_button)
        self.remove_op_button = QPushButton("Remove Op"); op_buttons_layout.addWidget(self.remove_op_button)
        op_buttons_layout.addStretch(1)
        self.move_up_button = QPushButton("Move Up"); op_buttons_layout.addWidget(self.move_up_button)
        self.move_down_button = QPushButton("Move Down"); op_buttons_layout.addWidget(self.move_down_button)
        ops_list_layout.addLayout(op_buttons_layout)
        
        self.collapse_luts_button = QPushButton("Collapse Selected LUTs")
        ops_list_layout.addWidget(self.collapse_luts_button)

        left_panel_layout.addWidget(ops_list_group)
        left_panel_layout.addStretch(1)
        main_layout.addLayout(left_panel_layout)

        # --- Center Panel: Operation Details ---
        self.details_group = QGroupBox("Operation Details")
        details_layout = QVBoxLayout(self.details_group)
        
        op_type_layout = QHBoxLayout()
        op_type_layout.addWidget(QLabel("Operation Type:"))
        self.selected_op_type_combo = QComboBox()
        self.selected_op_type_combo.addItems(["none", "gaussian_blur", "bilateral_filter", "median_blur", "unsharp_mask", "resize", "apply_lut"])
        op_type_layout.addWidget(self.selected_op_type_combo)
        op_type_layout.addStretch(1)
        details_layout.addLayout(op_type_layout)

        self.op_params_stacked_widget = QStackedWidget()
        details_layout.addWidget(self.op_params_stacked_widget)

        self._create_parameter_widgets()
        
        details_layout.addStretch(1)
        main_layout.addWidget(self.details_group, 1)
        
        # --- Right Panel: Collapsible LUT Table ---
        self.lut_table_group = QGroupBox("LUT Values (Input -> Output)")
        lut_table_layout = QVBoxLayout(self.lut_table_group)
        self.lut_table_widget = QTableWidget(256, 2)
        self.lut_table_widget.setHorizontalHeaderLabels(["Input", "Output"])
        self.lut_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lut_table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        for i in range(256):
            self.lut_table_widget.setItem(i, 0, QTableWidgetItem(str(i)))
        lut_table_layout.addWidget(self.lut_table_widget)
        
        self.lut_table_group.setVisible(False) # Hide by default
        main_layout.addWidget(self.lut_table_group, 0)

    def _create_parameter_widgets(self):
        """Creates all the parameter widgets for the stacked widget."""
        self.none_params_widget = QWidget(); layout = QVBoxLayout(self.none_params_widget); layout.addWidget(QLabel("No parameters for 'none' operation.")); layout.addStretch(1); self.op_params_stacked_widget.addWidget(self.none_params_widget)
        self.gaussian_params_widget = QWidget(); layout = QVBoxLayout(self.gaussian_params_widget); ksize_layout = QHBoxLayout(); ksize_layout.addWidget(QLabel("Kernel Size (X, Y):")); self.gaussian_ksize_x_edit = QLineEdit(); self.gaussian_ksize_x_edit.setFixedWidth(60); self.gaussian_ksize_x_edit.setValidator(QIntValidator(1, 99, self)); ksize_layout.addWidget(self.gaussian_ksize_x_edit); self.gaussian_ksize_y_edit = QLineEdit(); self.gaussian_ksize_y_edit.setFixedWidth(60); self.gaussian_ksize_y_edit.setValidator(QIntValidator(1, 99, self)); ksize_layout.addWidget(self.gaussian_ksize_y_edit); ksize_layout.addStretch(1); layout.addLayout(ksize_layout); sigma_layout = QHBoxLayout(); sigma_layout.addWidget(QLabel("Sigma (X, Y):")); self.gaussian_sigma_x_edit = QLineEdit(); self.gaussian_sigma_x_edit.setFixedWidth(60); self.gaussian_sigma_x_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self)); sigma_layout.addWidget(self.gaussian_sigma_x_edit); self.gaussian_sigma_y_edit = QLineEdit(); self.gaussian_sigma_y_edit.setFixedWidth(60); self.gaussian_sigma_y_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self)); sigma_layout.addWidget(self.gaussian_sigma_y_edit); sigma_layout.addStretch(1); layout.addLayout(sigma_layout); layout.addStretch(1); self.op_params_stacked_widget.addWidget(self.gaussian_params_widget)
        self.bilateral_params_widget = QWidget(); layout = QVBoxLayout(self.bilateral_params_widget); layout.addWidget(QLabel("Diameter:")); self.bilateral_d_edit = QLineEdit(); self.bilateral_d_edit.setFixedWidth(60); self.bilateral_d_edit.setValidator(QIntValidator(1, 99, self)); layout.addWidget(self.bilateral_d_edit); layout.addWidget(QLabel("Sigma Color:")); self.bilateral_sigma_color_edit = QLineEdit(); self.bilateral_sigma_color_edit.setFixedWidth(60); self.bilateral_sigma_color_edit.setValidator(QDoubleValidator(0.0, 255.0, 2, self)); layout.addWidget(self.bilateral_sigma_color_edit); layout.addWidget(QLabel("Sigma Space:")); self.bilateral_sigma_space_edit = QLineEdit(); self.bilateral_sigma_space_edit.setFixedWidth(60); self.bilateral_sigma_space_edit.setValidator(QDoubleValidator(0.0, 255.0, 2, self)); layout.addWidget(self.bilateral_sigma_space_edit); layout.addStretch(1); self.op_params_stacked_widget.addWidget(self.bilateral_params_widget)
        self.median_params_widget = QWidget(); layout = QVBoxLayout(self.median_params_widget); layout.addWidget(QLabel("Kernel Size:")); self.median_ksize_edit = QLineEdit(); self.median_ksize_edit.setFixedWidth(60); self.median_ksize_edit.setValidator(QIntValidator(1, 99, self)); layout.addWidget(self.median_ksize_edit); layout.addStretch(1); self.op_params_stacked_widget.addWidget(self.median_params_widget)
        self.unsharp_params_widget = QWidget(); layout = QVBoxLayout(self.unsharp_params_widget); layout.addWidget(QLabel("Amount:")); self.unsharp_amount_edit = QLineEdit(); self.unsharp_amount_edit.setFixedWidth(60); self.unsharp_amount_edit.setValidator(QDoubleValidator(0.0, 5.0, 2, self)); layout.addWidget(self.unsharp_amount_edit); layout.addWidget(QLabel("Threshold:")); self.unsharp_threshold_edit = QLineEdit(); self.unsharp_threshold_edit.setFixedWidth(60); self.unsharp_threshold_edit.setValidator(QIntValidator(0, 255, self)); layout.addWidget(self.unsharp_threshold_edit); layout.addWidget(QLabel("Internal Blur KSize:")); self.unsharp_blur_ksize_edit = QLineEdit(); self.unsharp_blur_ksize_edit.setFixedWidth(60); self.unsharp_blur_ksize_edit.setValidator(QIntValidator(1, 99, self)); layout.addWidget(self.unsharp_blur_ksize_edit); layout.addWidget(QLabel("Internal Blur Sigma:")); self.unsharp_blur_sigma_edit = QLineEdit(); self.unsharp_blur_sigma_edit.setFixedWidth(60); self.unsharp_blur_sigma_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self)); layout.addWidget(self.unsharp_blur_sigma_edit); layout.addStretch(1); self.op_params_stacked_widget.addWidget(self.unsharp_params_widget)
        self.resize_params_widget = QWidget(); layout = QVBoxLayout(self.resize_params_widget); layout.addWidget(QLabel("Width (px, 0 for auto):")); self.resize_width_edit = QLineEdit(); self.resize_width_edit.setFixedWidth(80); self.resize_width_edit.setValidator(QIntValidator(0, 9999, self)); layout.addWidget(self.resize_width_edit); layout.addWidget(QLabel("Height (px, 0 for auto):")); self.resize_height_edit = QLineEdit(); self.resize_height_edit.setFixedWidth(80); self.resize_height_edit.setValidator(QIntValidator(0, 9999, self)); layout.addWidget(self.resize_height_edit); layout.addWidget(QLabel("Resampling Mode:")); self.resample_mode_combo = QComboBox(); self.resample_mode_combo.addItems(["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS4", "AREA"]); layout.addWidget(self.resample_mode_combo); layout.addStretch(1); self.op_params_stacked_widget.addWidget(self.resize_params_widget)
        self.apply_lut_widget = LutEditorWidget(self); self.op_params_stacked_widget.addWidget(self.apply_lut_widget)

    def _connect_signals(self):
        self.ops_list_widget.currentRowChanged.connect(self._update_selected_operation_details)
        self.ops_list_widget.model().rowsMoved.connect(self._reorder_operations_in_config)
        self.add_op_button.clicked.connect(self._add_operation)
        self.remove_op_button.clicked.connect(self._remove_operation)
        self.move_up_button.clicked.connect(self._move_operation_up)
        self.move_down_button.clicked.connect(self._move_operation_down)
        self.collapse_luts_button.clicked.connect(self._collapse_selected_luts)
        self.selected_op_type_combo.currentTextChanged.connect(self._on_selected_op_type_changed)
        
        # FIX: Restore all signal connections
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

        self.apply_lut_widget.lut_params_changed.connect(self._on_lut_params_changed)
        self.apply_lut_widget.toggle_table_visibility_requested.connect(self._toggle_lut_table_panel)

    def _toggle_lut_table_panel(self, checked: bool):
        self.lut_table_group.setVisible(checked)
        button = self.apply_lut_widget.toggle_table_button
        if checked:
            button.setText("Hide LUT Values <")
        else:
            button.setText("Show LUT Values >")

    def _on_lut_params_changed(self):
        current_row = self.ops_list_widget.currentRow()
        if not (0 <= current_row < len(self.config.xy_blend_pipeline)): return
        selected_op = self.config.xy_blend_pipeline[current_row]
        if selected_op.type != "apply_lut": return
        self.apply_lut_widget.plot_current_lut()

    def _update_operation_list(self):
        current_row = self.ops_list_widget.currentRow()
        self.ops_list_widget.clear()
        for i, op in enumerate(self.config.xy_blend_pipeline):
            self.ops_list_widget.addItem(f"{i+1}. {op.type.replace('_', ' ').title()}")
        if 0 <= current_row < self.ops_list_widget.count():
            self.ops_list_widget.setCurrentRow(current_row)
        elif self.ops_list_widget.count() > 0:
            self.ops_list_widget.setCurrentRow(0)
        else:
            self._update_selected_operation_details()

    def _update_selected_operation_details(self):
        current_row = self.ops_list_widget.currentRow()
        if 0 <= current_row < len(self.config.xy_blend_pipeline):
            selected_op = self.config.xy_blend_pipeline[current_row]
            self.details_group.setEnabled(True)
            self.selected_op_type_combo.blockSignals(True)
            self.selected_op_type_combo.setCurrentText(selected_op.type)
            self.selected_op_type_combo.blockSignals(False)
            self._show_params_for_type(selected_op.type)
            self._populate_params_widgets(selected_op)
        else:
            self.details_group.setEnabled(False)
            self.selected_op_type_combo.setCurrentText("none")
            self.op_params_stacked_widget.setCurrentWidget(self.none_params_widget)
            self._update_lut_table(np.arange(256, dtype=np.uint8))

    def _on_selected_op_type_changed(self, new_type: str):
        current_row = self.ops_list_widget.currentRow()
        if not (0 <= current_row < len(self.config.xy_blend_pipeline)): return
        old_op = self.config.xy_blend_pipeline[current_row]
        if new_type == old_op.type: return
        new_op = XYBlendOperation(type=new_type)
        if new_type == "apply_lut" and old_op.type == "apply_lut":
            new_op.lut_params = copy.deepcopy(old_op.lut_params)
        self.config.xy_blend_pipeline[current_row] = new_op
        self.ops_list_widget.currentItem().setText(f"{current_row+1}. {new_type.replace('_', ' ').title()}")
        self._show_params_for_type(new_type)
        self._populate_params_widgets(new_op)

    def _show_params_for_type(self, op_type: str):
        widget_map = {"none": self.none_params_widget, "gaussian_blur": self.gaussian_params_widget, "bilateral_filter": self.bilateral_params_widget, "median_blur": self.median_params_widget, "unsharp_mask": self.unsharp_params_widget, "resize": self.resize_params_widget, "apply_lut": self.apply_lut_widget}
        self.op_params_stacked_widget.setCurrentWidget(widget_map.get(op_type, self.none_params_widget))

    def _populate_params_widgets(self, op: XYBlendOperation):
        # FIX: Restore population of all parameter widgets
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
        if op.type == "apply_lut":
            self.apply_lut_widget.set_lut_params(op.lut_params)

    def _update_param_in_config(self, sender_widget, param_name: str, data_type: type, allow_none_if_zero: bool = False):
        # FIX: Restore full update logic
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
        
        self._populate_params_widgets(selected_op)

    def _add_operation(self):
        self.config.xy_blend_pipeline.append(XYBlendOperation(type="none"))
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(len(self.config.xy_blend_pipeline) - 1)

    def _remove_operation(self):
        selected_rows = [item.row() for item in self.ops_list_widget.selectedIndexes()]
        if not selected_rows: return
        reply = QMessageBox.question(self, "Remove Operation(s)", f"Are you sure you want to remove {len(selected_rows)} operation(s)?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for index in sorted(selected_rows, reverse=True):
                del self.config.xy_blend_pipeline[index]
            self._update_operation_list()

    def _move_operation(self, direction: int):
        current_row = self.ops_list_widget.currentRow()
        new_row = current_row + direction
        if 0 <= new_row < len(self.config.xy_blend_pipeline):
            p = self.config.xy_blend_pipeline
            p[current_row], p[new_row] = p[new_row], p[current_row]
            self._update_operation_list()
            self.ops_list_widget.setCurrentRow(new_row)

    def _move_operation_up(self): self._move_operation(-1)
    def _move_operation_down(self): self._move_operation(1)

    def _reorder_operations_in_config(self, parent, start, end, destination, row):
        if row > start: row -=1
        moved_op = self.config.xy_blend_pipeline.pop(start)
        self.config.xy_blend_pipeline.insert(row, moved_op)
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(row)

    def _collapse_selected_luts(self):
        selected_items = self.ops_list_widget.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.information(self, "Collapse LUTs", "Please select at least two consecutive 'Apply LUT' operations to collapse.")
            return

        indices = sorted([self.ops_list_widget.row(item) for item in selected_items])

        for i in range(len(indices) - 1):
            if indices[i+1] != indices[i] + 1:
                QMessageBox.warning(self, "Selection Error", "Please select a consecutive block of operations.")
                return
            if self.config.xy_blend_pipeline[indices[i]].type != "apply_lut":
                QMessageBox.warning(self, "Selection Error", "All selected items must be 'Apply LUT' operations.")
                return
        if self.config.xy_blend_pipeline[indices[-1]].type != "apply_lut":
             QMessageBox.warning(self, "Selection Error", "All selected items must be 'Apply LUT' operations.")
             return

        final_lut = np.arange(256, dtype=np.uint8)
        for index in indices:
            op = self.config.xy_blend_pipeline[index]
            current_lut = self.apply_lut_widget._get_lut_from_params(op.lut_params)
            if current_lut is None:
                QMessageBox.critical(self, "Collapse Error", f"Could not generate LUT for operation at index {index+1}.")
                return
            final_lut = current_lut[final_lut]

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Collapsed LUT", "collapsed_lut.json", "JSON Files (*.json)")
        if not filepath: return

        try:
            lut_manager.save_lut(filepath, final_lut)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save collapsed LUT: {e}")
            return

        new_op = XYBlendOperation(type="apply_lut", lut_params=LutParameters(lut_source="file", fixed_lut_path=filepath))

        for index in reversed(indices):
            del self.config.xy_blend_pipeline[index]
        
        self.config.xy_blend_pipeline.insert(indices[0], new_op)
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(indices[0])
              
    def apply_settings(self, cfg: Config):
        self._update_operation_list() 
        self._update_selected_operation_details()

    def get_config(self) -> dict:
        return {"xy_blend_pipeline": [op.to_dict() for op in self.config.xy_blend_pipeline]}

    def _update_lut_table(self, lut_array: np.ndarray):
        self.lut_table_widget.blockSignals(True)
        for i in range(256):
            self.lut_table_widget.setItem(i, 1, QTableWidgetItem(str(lut_array[i])))
        self.lut_table_widget.blockSignals(False)
