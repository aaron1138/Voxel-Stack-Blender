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

# config.py (Spline Update)

from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Union, Any
import json
import os
import copy

DEFAULT_NUM_WORKERS = max(1, os.cpu_count() - 1)

@dataclass
class LutParameters:
    """
    Parameters for generating or loading a Look-Up Table (LUT).
    """
    lut_source: str = "generated"
    lut_generation_type: str = "linear"
    input_min: int = 0
    input_max: int = 255
    output_min: int = 0
    output_max: int = 255
    gamma_value: float = 1.0
    s_curve_contrast: float = 0.5
    log_param: float = 10.0
    exp_param: float = 2.0
    sqrt_param: float = 2.0
    rodbard_param: float = 1.0
    
    # NEW: Field to store control points for spline curves.
    # Format: [[x1, y1], [x2, y2], ...] where x and y are 0-255
    spline_points: List[List[int]] = field(default_factory=lambda: [[0, 0], [255, 255]])

    fixed_lut_path: str = ""

    def __post_init__(self):
        # (Validation logic remains the same as previous version)
        self.lut_source = self.lut_source.lower()
        if self.lut_source not in ["generated", "file"]: self.lut_source = "generated"
        if self.lut_generation_type not in ["linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard", "spline"]: self.lut_generation_type = "linear"
        self.input_min = max(0, min(255, self.input_min))
        self.input_max = max(0, min(255, self.input_max))
        if self.input_min > self.input_max: self.input_min, self.input_max = self.input_max, self.input_min
        self.output_min = max(0, min(255, self.output_min))
        self.output_max = max(0, min(255, self.output_max))
        if self.output_min > self.output_max: self.output_min, self.output_max = self.output_max, self.output_min
        self.gamma_value = max(0.01, min(10.0, self.gamma_value))
        self.s_curve_contrast = max(0.0, min(1.0, self.s_curve_contrast))
        self.log_param = max(0.01, min(100.0, self.log_param))
        self.exp_param = max(0.01, min(10.0, self.exp_param))
        self.sqrt_param = max(0.1, min(50.0, self.sqrt_param))
        self.rodbard_param = max(0.0, min(2.0, self.rodbard_param))

@dataclass
class XYBlendOperation:
    """
    Represents a single operation in the XY image processing pipeline.
    """
    type: str = "none"
    gaussian_ksize_x: int = 3
    gaussian_ksize_y: int = 3
    gaussian_sigma_x: float = 0.0
    gaussian_sigma_y: float = 0.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    median_ksize: int = 5
    unsharp_amount: float = 1.0
    unsharp_threshold: int = 0
    unsharp_blur_ksize: int = 5
    unsharp_blur_sigma: float = 0.0
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    resample_mode: str = "LANCZOS4"
    lut_params: LutParameters = field(default_factory=LutParameters)

    def __post_init__(self):
        # (Validation logic remains the same as previous version)
        self.type = self.type.lower()
        if self.type == "gaussian_blur":
            self.gaussian_ksize_x = self._ensure_odd_positive_ksize(self.gaussian_ksize_x)
            self.gaussian_ksize_y = self._ensure_odd_positive_ksize(self.gaussian_ksize_y)
        elif self.type == "median_blur":
            self.median_ksize = self._ensure_odd_positive_ksize(self.median_ksize)
        elif self.type == "unsharp_mask":
            self.unsharp_blur_ksize = self._ensure_odd_positive_ksize(self.unsharp_blur_ksize)
        if self.type == "resize":
            if self.resize_width is not None: self.resize_width = max(0, self.resize_width)
            if self.resize_height is not None: self.resize_height = max(0, self.resize_height)
            if self.resize_width == 0: self.resize_width = None
            if self.resize_height == 0: self.resize_height = None
        if isinstance(self.lut_params, dict):
            self.lut_params = self.from_dict_to_lut_params(self.lut_params)

    def _ensure_odd_positive_ksize(self, ksize: int) -> int:
        if ksize <= 0: return 1
        return ksize if ksize % 2 != 0 else ksize + 1
        
    @staticmethod
    def from_dict_to_lut_params(data: dict) -> LutParameters:
        lut_field_names = {f.name for f in fields(LutParameters)}
        filtered_lut_data = {k: v for k, v in data.items() if k in lut_field_names}
        return LutParameters(**filtered_lut_data)

@dataclass
class RoiParameters:
    """
    Parameters for ROI (Region of Interest) processing.
    """
    min_size: int = 100

    # Raft/Support detection settings
    enable_raft_support_handling: bool = False
    raft_layer_count: int = 5
    raft_min_size: int = 10000
    support_max_size: int = 500
    support_max_layer: int = 1000
    support_max_growth: float = 2.5


@dataclass
class Config:
    """
    Main application configuration, updated with new UI fields.
    """
    # --- I/O Settings ---
    output_file_prefix: str = "Voxel_Blend_Processed_"
    input_mode: str = "folder" # "folder" or "uvtools"
    input_folder: str = ""
    output_folder: str = ""
    start_index: Optional[int] = 0
    stop_index: Optional[int] = None
    
    # --- UVTools Mode Settings ---
    uvtools_path: str = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
    uvtools_temp_folder: str = ""
    uvtools_input_file: str = ""
    uvtools_output_location: str = "working_folder"
    uvtools_delete_temp_on_completion: bool = True

    # --- Stack Blending Settings ---
    blending_mode: str = "fixed_fade"  # "fixed_fade" or "roi_fade"
    receding_layers: int = 3
    use_fixed_fade_receding: bool = False
    fixed_fade_distance_receding: float = 10.0
    
    # --- Overhang Settings (for future use) ---
    overhang_layers: int = 0
    use_fixed_fade_overhang: bool = False
    fixed_fade_distance_overhang: float = 10.0

    # --- ROI Mode Settings ---
    roi_params: RoiParameters = field(default_factory=RoiParameters)

    # --- General Settings ---
    thread_count: int = DEFAULT_NUM_WORKERS
    debug_save: bool = False
    xy_blend_pipeline: List[XYBlendOperation] = field(default_factory=lambda: [XYBlendOperation("none")])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        # (This robust loading method remains the same)
        config_instance = cls()
        field_map = {f.name: f for f in fields(cls)}
        for key, value in data.items():
            if key in field_map:
                field_obj = field_map[key]
                if key == 'xy_blend_pipeline':
                    pipeline_list = []
                    if isinstance(value, list):
                        for op_data in value:
                            if not isinstance(op_data, dict): continue
                            op_field_names = {f.name for f in fields(XYBlendOperation)}
                            filtered_op_data = {k: v for k, v in op_data.items() if k in op_field_names}
                            if 'lut_params' in filtered_op_data and isinstance(filtered_op_data['lut_params'], dict):
                                filtered_op_data['lut_params'] = XYBlendOperation.from_dict_to_lut_params(filtered_op_data['lut_params'])
                            pipeline_list.append(XYBlendOperation(**filtered_op_data))
                    setattr(config_instance, key, pipeline_list)
                elif key == 'roi_params':
                    if isinstance(value, dict):
                        roi_field_names = {f.name for f in fields(RoiParameters)}
                        filtered_roi_data = {k: v for k, v in value.items() if k in roi_field_names}
                        setattr(config_instance, key, RoiParameters(**filtered_roi_data))
                else:
                    if field_obj.type is bool and isinstance(value, str):
                        value = value.lower() in ('true', '1', 't', 'y')
                    try:
                        setattr(config_instance, key, value)
                    except (TypeError, ValueError):
                        print(f"Warning: Could not assign value '{value}' to '{key}'. Using default.")
            else:
                print(f"Warning: Unrecognized config key '{key}' found. Skipping.")
        return config_instance

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "Config":
        if not os.path.exists(filepath):
            default_config = cls()
            try:
                default_config.save(filepath)
            except Exception as e:
                print(f"Error saving default config: {e}")
            return default_config
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error loading config '{filepath}': {e}. Using default.")
            return cls()

def upgrade_config(cfg: Config):
    if hasattr(cfg, 'n_layers'):
        cfg.receding_layers = cfg.n_layers
        delattr(cfg, 'n_layers')
    if hasattr(cfg, 'use_fixed_norm'):
        cfg.use_fixed_fade_receding = cfg.use_fixed_norm
        delattr(cfg, 'use_fixed_norm')
    if hasattr(cfg, 'fixed_fade_distance'):
        cfg.fixed_fade_distance_receding = cfg.fixed_fade_distance
        delattr(cfg, 'fixed_fade_distance')

_CONFIG_FILE = "app_config.json"
app_config = Config.load(_CONFIG_FILE)
upgrade_config(app_config)
