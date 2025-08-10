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

# lut_editor_widget.py (Corrected)

import os
import numpy as np
import json
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QPushButton, QStackedWidget, QMessageBox, QFileDialog,
    QGridLayout, QSlider
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from config import LutParameters
import lut_manager

class InteractiveMplCanvas(FigureCanvas):
    """
    An interactive Matplotlib canvas that displays the resulting LUT curve and allows 
    adding, dragging, and deleting spline control points as an overlay.
    """
    points_changed = Signal()

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(InteractiveMplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout(pad=3)

        self._curve_line = Line2D([], [], color='blue', linestyle='-')
        self.axes.add_line(self._curve_line)
        
        self._points_line = Line2D([], [], marker='o', color='red', linestyle='None')
        self.axes.add_line(self._points_line)

        self._points = []
        self._selected_point_index = None
        self._is_interactive = False

        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)

    def set_interactive(self, interactive: bool):
        self._is_interactive = interactive
        self._points_line.set_visible(interactive)
        self.draw()

    def update_plot(self, curve_data: Optional[np.ndarray], points_data: Optional[List[List[int]]]):
        if curve_data is not None:
            self._curve_line.set_data(np.arange(len(curve_data)), curve_data)
        else:
            self._curve_line.set_data([], [])

        if points_data:
            self._points = sorted(points_data, key=lambda p: p[0])
            x = [p[0] for p in self._points]
            y = [p[1] for p in self._points]
            self._points_line.set_data(x, y)
        else:
            self._points = []
            self._points_line.set_data([], [])
            
        self.axes.set_title("LUT Curve Preview")
        self.axes.set_xlabel("Input Value (0-255)")
        self.axes.set_ylabel("Output Value (0-255)")
        self.axes.set_xlim(0, 255)
        self.axes.set_ylim(0, 255)
        self.axes.grid(True)
        self.draw()

    def get_points(self) -> List[List[int]]:
        return self._points

    def _get_point_at_event(self, event):
        if event.xdata is None or event.ydata is None: return None
        tolerance = 10
        for i, point in enumerate(self._points):
            dist = np.sqrt((point[0] - event.xdata)**2 + (point[1] - event.ydata)**2)
            if dist < tolerance:
                return i
        return None

    def _on_press(self, event):
        if not self._is_interactive or event.inaxes != self.axes: return
        self._selected_point_index = self._get_point_at_event(event)
        if event.button == 3 and self._selected_point_index is not None:
            if self._points[self._selected_point_index][0] not in [0, 255]:
                del self._points[self._selected_point_index]
                self._selected_point_index = None
                self.points_changed.emit()
        elif event.dblclick:
            self._points.append([int(event.xdata), int(event.ydata)])
            self._points = sorted(self._points, key=lambda p: p[0])
            self.points_changed.emit()

    def _on_release(self, event):
        if not self._is_interactive or self._selected_point_index is None: return
        self._selected_point_index = None
        self.points_changed.emit()

    def _on_motion(self, event):
        if not self._is_interactive or self._selected_point_index is None or event.inaxes != self.axes: return
        x, y = int(event.xdata), int(event.ydata)
        x = max(0, min(255, x))
        y = max(0, min(255, y))
        if self._points[self._selected_point_index][0] == 0: x = 0
        elif self._points[self._selected_point_index][0] == 255: x = 255
        self._points[self._selected_point_index] = [x, y]
        self._points = sorted(self._points, key=lambda p: p[0])
        self.points_changed.emit()


class LutEditorWidget(QWidget):
    """A self-contained widget for all LUT creation and editing functionality."""
    lut_params_changed = Signal()
    toggle_table_visibility_requested = Signal(bool)

    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab = parent_tab
        self._lut_params = None
        self._setup_ui()
        self._connect_signals()

    def set_lut_params(self, lut_params: LutParameters):
        self._lut_params = lut_params
        self.populate_controls()
        self.plot_current_lut()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        controls_group = QGroupBox("LUT Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        lut_source_layout = QHBoxLayout()
        lut_source_layout.addWidget(QLabel("LUT Source:"))
        self.lut_source_combo = QComboBox()
        self.lut_source_combo.addItems(["Generated", "File"])
        lut_source_layout.addWidget(self.lut_source_combo)
        lut_source_layout.addStretch(1)
        controls_layout.addLayout(lut_source_layout)

        self.lut_gen_params_stacked_widget = QStackedWidget()
        controls_layout.addWidget(self.lut_gen_params_stacked_widget)
        
        self.lut_generated_params_group = QWidget()
        lut_generated_params_layout = QVBoxLayout(self.lut_generated_params_group)
        
        gen_type_layout = QHBoxLayout()
        gen_type_layout.addWidget(QLabel("Type:"))
        self.lut_generation_type_combo = QComboBox()
        self.lut_generation_type_combo.addItems(["linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard", "spline"])
        gen_type_layout.addWidget(self.lut_generation_type_combo)
        gen_type_layout.addStretch(1)
        lut_generated_params_layout.addLayout(gen_type_layout)
        
        range_group = QGroupBox("Input/Output Range")
        range_layout = QGridLayout(range_group)
        range_layout.addWidget(QLabel("Input Min:"), 0, 0); self.lut_input_min_edit = QLineEdit("0"); self.lut_input_min_edit.setValidator(QIntValidator(0, 255, self)); range_layout.addWidget(self.lut_input_min_edit, 0, 1)
        range_layout.addWidget(QLabel("Input Max:"), 0, 2); self.lut_input_max_edit = QLineEdit("255"); self.lut_input_max_edit.setValidator(QIntValidator(0, 255, self)); range_layout.addWidget(self.lut_input_max_edit, 0, 3)
        range_layout.addWidget(QLabel("Output Min:"), 1, 0); self.lut_output_min_edit = QLineEdit("0"); self.lut_output_min_edit.setValidator(QIntValidator(0, 255, self)); range_layout.addWidget(self.lut_output_min_edit, 1, 1)
        range_layout.addWidget(QLabel("Output Max:"), 1, 2); self.lut_output_max_edit = QLineEdit("255"); self.lut_output_max_edit.setValidator(QIntValidator(0, 255, self)); range_layout.addWidget(self.lut_output_max_edit, 1, 3)
        lut_generated_params_layout.addWidget(range_group)

        self.gen_lut_algo_params_stacked_widget = QStackedWidget()
        lut_generated_params_layout.addWidget(self.gen_lut_algo_params_stacked_widget)

        self.lut_linear_params_widget = QWidget(); self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_linear_params_widget)
        self.lut_gamma_params_widget = QWidget(); gamma_layout = QHBoxLayout(self.lut_gamma_params_widget); gamma_layout.addWidget(QLabel("Gamma:")); g_layout, self.lut_gamma_value_edit, self.lut_gamma_value_slider = self._create_slider_combo(QDoubleValidator(0.01, 10.0, 2), (1, 1000), 100.0); gamma_layout.addLayout(g_layout); self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_gamma_params_widget)
        self.lut_s_curve_params_widget = QWidget(); s_curve_layout = QHBoxLayout(self.lut_s_curve_params_widget); s_curve_layout.addWidget(QLabel("Contrast:")); sc_layout, self.lut_s_curve_contrast_edit, self.lut_s_curve_contrast_slider = self._create_slider_combo(QDoubleValidator(0.0, 1.0, 2), (0, 100), 100.0); s_curve_layout.addLayout(sc_layout); self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_s_curve_params_widget)
        self.lut_log_params_widget = QWidget(); log_layout = QHBoxLayout(self.lut_log_params_widget); log_layout.addWidget(QLabel("Param:")); l_layout, self.lut_log_param_edit, self.lut_log_param_slider = self._create_slider_combo(QDoubleValidator(0.01, 100.0, 2), (1, 1000), 10.0); log_layout.addLayout(l_layout); self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_log_params_widget)
        self.lut_exp_params_widget = QWidget(); exp_layout = QHBoxLayout(self.lut_exp_params_widget); exp_layout.addWidget(QLabel("Param:")); e_layout, self.lut_exp_param_edit, self.lut_exp_param_slider = self._create_slider_combo(QDoubleValidator(0.01, 10.0, 2), (1, 1000), 100.0); exp_layout.addLayout(e_layout); self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_exp_params_widget)
        self.lut_sqrt_params_widget = QWidget(); sqrt_layout = QHBoxLayout(self.lut_sqrt_params_widget); sqrt_layout.addWidget(QLabel("Root:")); sq_layout, self.lut_sqrt_param_edit, self.lut_sqrt_param_slider = self._create_slider_combo(QDoubleValidator(0.1, 50.0, 2), (10, 500), 10.0); sqrt_layout.addLayout(sq_layout); self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_sqrt_params_widget)
        self.lut_rodbard_params_widget = QWidget(); rodbard_layout = QHBoxLayout(self.lut_rodbard_params_widget); rodbard_layout.addWidget(QLabel("Contrast:")); r_layout, self.lut_rodbard_param_edit, self.lut_rodbard_param_slider = self._create_slider_combo(QDoubleValidator(0.0, 2.0, 2), (0, 200), 100.0); rodbard_layout.addLayout(r_layout); self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_rodbard_params_widget)
        self.lut_spline_params_widget = QWidget(); spline_layout = QVBoxLayout(self.lut_spline_params_widget); spline_layout.addWidget(QLabel("On graph: Double-click to add, Right-click to delete, Drag to move.")); self.gen_lut_algo_params_stacked_widget.addWidget(self.lut_spline_params_widget)

        self.save_generated_lut_button = QPushButton("Save This Generated LUT to File...")
        lut_generated_params_layout.addWidget(self.save_generated_lut_button)
        self.lut_gen_params_stacked_widget.addWidget(self.lut_generated_params_group)

        self.lut_file_params_group = QWidget()
        lut_file_params_layout = QVBoxLayout(self.lut_file_params_group)
        lut_path_layout = QHBoxLayout(); lut_path_layout.addWidget(QLabel("LUT File:")); self.lut_filepath_edit = QLineEdit(); self.lut_filepath_edit.setReadOnly(True); lut_path_layout.addWidget(self.lut_filepath_edit); lut_file_params_layout.addLayout(lut_path_layout)
        self.lut_load_file_button = QPushButton("Load from File...")
        lut_file_params_layout.addWidget(self.lut_load_file_button)
        self.lut_gen_params_stacked_widget.addWidget(self.lut_file_params_group)
        
        main_layout.addWidget(controls_group)

        self.preview_canvas = InteractiveMplCanvas(self, width=6, height=5)
        self.preview_canvas.setMinimumHeight(250)
        main_layout.addWidget(self.preview_canvas)

        self.toggle_table_button = QPushButton("Show LUT Values >")
        self.toggle_table_button.setCheckable(True)
        self.toggle_table_button.setChecked(False)
        main_layout.addWidget(self.toggle_table_button)

        main_layout.addStretch(1)

    def _create_slider_combo(self, text_validator, slider_range, scale_factor):
        layout = QHBoxLayout(); line_edit = QLineEdit(); line_edit.setFixedWidth(60); line_edit.setValidator(text_validator); slider = QSlider(Qt.Horizontal); slider.setRange(slider_range[0], slider_range[1]); slider.setSingleStep(1); layout.addWidget(line_edit); layout.addWidget(slider); return layout, line_edit, slider

    def _connect_signals(self):
        self.lut_source_combo.currentTextChanged.connect(self._on_source_changed)
        self.lut_generation_type_combo.currentTextChanged.connect(self._on_gen_type_changed)
        
        # FIX: Pass the widget itself to the lambda to avoid relying on self.sender()
        self.lut_input_min_edit.editingFinished.connect(lambda: self._update_param("input_min", int, self.lut_input_min_edit))
        self.lut_input_max_edit.editingFinished.connect(lambda: self._update_param("input_max", int, self.lut_input_max_edit))
        self.lut_output_min_edit.editingFinished.connect(lambda: self._update_param("output_min", int, self.lut_output_min_edit))
        self.lut_output_max_edit.editingFinished.connect(lambda: self._update_param("output_max", int, self.lut_output_max_edit))
        
        self._connect_slider_combo(self.lut_gamma_value_edit, self.lut_gamma_value_slider, "gamma_value", 100.0)
        self._connect_slider_combo(self.lut_s_curve_contrast_edit, self.lut_s_curve_contrast_slider, "s_curve_contrast", 100.0)
        self._connect_slider_combo(self.lut_log_param_edit, self.lut_log_param_slider, "log_param", 10.0)
        self._connect_slider_combo(self.lut_exp_param_edit, self.lut_exp_param_slider, "exp_param", 100.0)
        self._connect_slider_combo(self.lut_sqrt_param_edit, self.lut_sqrt_param_slider, "sqrt_param", 10.0)
        self._connect_slider_combo(self.lut_rodbard_param_edit, self.lut_rodbard_param_slider, "rodbard_param", 100.0)
        
        self.preview_canvas.points_changed.connect(self._on_spline_points_changed)
        self.save_generated_lut_button.clicked.connect(self._save_generated_lut)
        self.lut_load_file_button.clicked.connect(self._load_lut_from_file)
        self.toggle_table_button.toggled.connect(self.toggle_table_visibility_requested.emit)

    def _connect_slider_combo(self, line_edit, slider, param_name, scale_factor):
        slider.valueChanged.connect(lambda val: line_edit.setText(f"{val / scale_factor:.2f}"))
        slider.sliderReleased.connect(lambda: self._update_param(param_name, float, line_edit))
        line_edit.editingFinished.connect(lambda: self._update_param(param_name, float, line_edit))

    def _on_source_changed(self): self._update_param("lut_source", str, self.lut_source_combo)
    def _on_gen_type_changed(self): self._update_param("lut_generation_type", str, self.lut_generation_type_combo)

    def _on_spline_points_changed(self):
        if not self._lut_params: return
        self._lut_params.spline_points = self.preview_canvas.get_points()
        self.lut_params_changed.emit()

    def _block_all_signals(self, block: bool):
        widgets = [self.lut_source_combo, self.lut_generation_type_combo, self.lut_input_min_edit, self.lut_input_max_edit, self.lut_output_min_edit, self.lut_output_max_edit, self.lut_gamma_value_edit, self.lut_gamma_value_slider, self.lut_s_curve_contrast_edit, self.lut_s_curve_contrast_slider, self.lut_log_param_edit, self.lut_log_param_slider, self.lut_exp_param_edit, self.lut_exp_param_slider, self.lut_sqrt_param_edit, self.lut_sqrt_param_slider, self.lut_rodbard_param_edit, self.lut_rodbard_param_slider, self.preview_canvas, self.lut_filepath_edit, self.toggle_table_button]
        for widget in widgets: widget.blockSignals(block)

    def _update_param(self, param_name, data_type, source_widget=None):
        if not self._lut_params: return
        sender = source_widget or self.sender()
        # The sender check is now safe because we explicitly pass the source_widget
        text = sender.text() if isinstance(sender, QLineEdit) else sender.currentText()
        try:
            value = data_type(text.replace(',', '.'))
            setattr(self._lut_params, param_name, value)
            self._lut_params.__post_init__()
            if hasattr(sender, 'setStyleSheet'): sender.setStyleSheet("")
        except (ValueError, TypeError):
            if hasattr(sender, 'setStyleSheet'): sender.setStyleSheet("border: 1px solid red;")
            return
        self._block_all_signals(True)
        self.populate_controls()
        self._block_all_signals(False)
        self.lut_params_changed.emit()

    def populate_controls(self):
        if not self._lut_params: return
        lp = self._lut_params
        self.lut_source_combo.setCurrentText(lp.lut_source.capitalize())
        self.lut_gen_params_stacked_widget.setCurrentIndex(0 if lp.lut_source == "generated" else 1)
        self.lut_generation_type_combo.setCurrentText(lp.lut_generation_type)
        self._update_lut_gen_type_controls_widget_only(lp.lut_generation_type)
        self.lut_input_min_edit.setText(str(lp.input_min)); self.lut_input_max_edit.setText(str(lp.input_max)); self.lut_output_min_edit.setText(str(lp.output_min)); self.lut_output_max_edit.setText(str(lp.output_max))
        self.lut_gamma_value_edit.setText(f"{lp.gamma_value:.2f}"); self.lut_gamma_value_slider.setValue(int(lp.gamma_value * 100))
        self.lut_s_curve_contrast_edit.setText(f"{lp.s_curve_contrast:.2f}"); self.lut_s_curve_contrast_slider.setValue(int(lp.s_curve_contrast * 100))
        self.lut_log_param_edit.setText(f"{lp.log_param:.2f}"); self.lut_log_param_slider.setValue(int(lp.log_param * 10))
        self.lut_exp_param_edit.setText(f"{lp.exp_param:.2f}"); self.lut_exp_param_slider.setValue(int(lp.exp_param * 100))
        self.lut_sqrt_param_edit.setText(f"{lp.sqrt_param:.2f}"); self.lut_sqrt_param_slider.setValue(int(lp.sqrt_param * 10))
        self.lut_rodbard_param_edit.setText(f"{lp.rodbard_param:.2f}"); self.lut_rodbard_param_slider.setValue(int(lp.rodbard_param * 100))
        self.lut_filepath_edit.setText(lp.fixed_lut_path)

    def _update_lut_gen_type_controls_widget_only(self, lut_type: str):
        widget_map = {"linear": self.lut_linear_params_widget, "gamma": self.lut_gamma_params_widget, "s_curve": self.lut_s_curve_params_widget, "log": self.lut_log_params_widget, "exp": self.lut_exp_params_widget, "sqrt": self.lut_sqrt_params_widget, "rodbard": self.lut_rodbard_params_widget, "spline": self.lut_spline_params_widget}
        self.gen_lut_algo_params_stacked_widget.setCurrentWidget(widget_map.get(lut_type.lower(), self.lut_linear_params_widget))

    def _get_lut_from_params(self, lut_params: LutParameters) -> Optional[np.ndarray]:
        if not lut_params: return None
        try:
            if lut_params.lut_source == "generated":
                args = (lut_params.input_min, lut_params.input_max, lut_params.output_min, lut_params.output_max)
                # FIX: Restore the full if/elif block for LUT generation
                if lut_params.lut_generation_type == "spline": return lut_manager.generate_spline_lut(lut_params.spline_points, *args)
                if lut_params.lut_generation_type == "linear": return lut_manager.generate_linear_lut(*args)
                if lut_params.lut_generation_type == "gamma": return lut_manager.generate_gamma_lut(lut_params.gamma_value, *args)
                if lut_params.lut_generation_type == "s_curve": return lut_manager.generate_s_curve_lut(lut_params.s_curve_contrast, *args)
                if lut_params.lut_generation_type == "log": return lut_manager.generate_log_lut(lut_params.log_param, *args)
                if lut_params.lut_generation_type == "exp": return lut_manager.generate_exp_lut(lut_params.exp_param, *args)
                if lut_params.lut_generation_type == "sqrt": return lut_manager.generate_sqrt_lut(lut_params.sqrt_param, *args)
                if lut_params.lut_generation_type == "rodbard": return lut_manager.generate_rodbard_lut(lut_params.rodbard_param, *args)
            elif lut_params.lut_source == "file" and lut_params.fixed_lut_path and os.path.exists(lut_params.fixed_lut_path):
                return lut_manager.load_lut(lut_params.fixed_lut_path)
        except Exception as e:
            print(f"Error generating LUT: {e}")
        return None

    def plot_current_lut(self):
        if not self._lut_params: return
        generated_lut = self._get_lut_from_params(self._lut_params)
        if generated_lut is None: generated_lut = lut_manager.get_default_z_lut()
        is_spline_mode = (self._lut_params.lut_source == "generated" and self._lut_params.lut_generation_type == "spline")
        points_to_show = self._lut_params.spline_points if is_spline_mode else None
        self.preview_canvas.set_interactive(is_spline_mode)
        self.preview_canvas.update_plot(curve_data=generated_lut, points_data=points_to_show)
        self.parent_tab._update_lut_table(generated_lut)

    def _load_lut_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load LUT File", "", "JSON Files (*.json)")
        if filepath:
            try:
                lut_manager.load_lut(filepath)
                self._lut_params.fixed_lut_path = filepath
                self._lut_params.lut_source = "file"
                self.populate_controls()
                self.lut_params_changed.emit()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load LUT file: {e}")

    def _save_generated_lut(self):
        if not self._lut_params or self._lut_params.lut_source != "generated":
            QMessageBox.warning(self, "Save Error", "Can only save a LUT when source is 'Generated'.")
            return
        lut_to_save = self._get_lut_from_params(self._lut_params)
        if lut_to_save is None:
            QMessageBox.critical(self, "Save Error", "Could not generate a valid LUT to save.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Generated LUT", "custom_lut.json", "JSON Files (*.json)")
        if filepath:
            try:
                lut_manager.save_lut(filepath, lut_to_save)
                QMessageBox.information(self, "Save Success", f"LUT saved successfully to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save LUT to file: {e}")
