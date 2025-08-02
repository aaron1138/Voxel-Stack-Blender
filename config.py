# config.py (Corrected from_dict for robust loading)

from dataclasses import dataclass, field, asdict, fields # NEW: Import 'fields'
from typing import List, Optional, Union, Any
import json
import os
import copy

# Define a sensible default for thread count, using system core count
DEFAULT_NUM_WORKERS = max(1, os.cpu_count() - 1)

# --- Data Classes for Pipeline Operations ---

@dataclass
class LutParameters:
    """
    Parameters for generating or loading a Look-Up Table (LUT).
    These parameters will be nested within an XYBlendOperation of type 'apply_lut'.
    """
    lut_source: str = "generated"
    
    # Parameters for 'generated' LUTs
    lut_generation_type: str = "linear"
    gamma_value: float = 1.0
    linear_min_input: int = 0
    linear_max_output: int = 255
    s_curve_contrast: float = 0.5
    log_param: float = 10.0
    exp_param: float = 2.0
    sqrt_param: float = 1.0
    rodbard_param: float = 1.0

    # Parameters for 'file' LUTs
    fixed_lut_path: str = ""

    def __post_init__(self):
        self.lut_source = self.lut_source.lower()
        self.lut_generation_type = self.lut_generation_type.lower()
        self.gamma_value = max(0.01, self.gamma_value)
        self.linear_min_input = max(0, min(255, self.linear_min_input))
        self.linear_max_output = max(0, min(255, self.linear_max_output))
        self.s_curve_contrast = max(0.0, min(1.0, self.s_curve_contrast))
        self.log_param = max(0.01, self.log_param)
        self.exp_param = max(0.01, self.exp_param)
        self.sqrt_param = max(0.01, self.sqrt_param)
        self.rodbard_param = max(0.01, self.rodbard_param)


@dataclass
class XYBlendOperation:
    """
    Represents a single operation in the XY image processing pipeline.
    Parameters for each operation type are defined here.
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
        self.type = self.type.lower()
        if self.type in ["gaussian_blur", "median_blur", "unsharp_mask"]:
            if self.type == "gaussian_blur":
                self.gaussian_ksize_x = self._ensure_odd_positive_ksize(self.gaussian_ksize_x)
                self.gaussian_ksize_y = self._ensure_odd_positive_ksize(self.gaussian_ksize_y)
            elif self.type == "median_blur":
                self.median_ksize = self._ensure_odd_positive_ksize(self.median_ksize)
            elif self.type == "unsharp_mask":
                self.unsharp_blur_ksize = self._ensure_odd_positive_ksize(self.unsharp_blur_ksize)

        if self.type == "resize":
            if self.resize_width is not None:
                self.resize_width = max(0, self.resize_width)
            if self.resize_height is not None:
                self.resize_height = max(0, self.resize_height)
            if self.resize_width == 0: self.resize_width = None
            if self.resize_height == 0: self.resize_height = None

    def _ensure_odd_positive_ksize(self, ksize: int) -> int:
        if ksize <= 0:
            return 1
        return ksize if ksize % 2 != 0 else ksize + 1


@dataclass
class Config:
    """
    Main application configuration.
    This dataclass will hold all settings for the program.
    """
    n_layers: int = 3
    start_index: Optional[int] = 0
    stop_index: Optional[int] = None
    debug_save: bool = False
    thread_count: int = DEFAULT_NUM_WORKERS

    use_fixed_norm: bool = False
    fixed_fade_distance: float = 10.0

    input_folder: str = ""
    output_folder: str = ""

    xy_blend_pipeline: List[XYBlendOperation] = field(default_factory=lambda: [XYBlendOperation("none")])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        config_instance = cls() # Start with default config instance
        
        # Create a mapping of field names to Field objects for easy lookup
        # This resolves the 'tuple indices must be integers or slices, not str' error
        field_map = {f.name: f for f in fields(cls)}

        for key, value in data.items():
            if key in field_map: # Check if key is a valid field name
                field_obj = field_map[key]
                
                if key == 'xy_blend_pipeline':
                    pipeline_list = []
                    for op_data in value:
                        # Reconstruct XYBlendOperation, filtering data to match its fields
                        op_field_names = {f.name for f in fields(XYBlendOperation)}
                        filtered_op_data = {k: v for k, v in op_data.items() if k in op_field_names}
                        
                        # Handle nested LutParameters
                        if 'lut_params' in filtered_op_data and isinstance(filtered_op_data['lut_params'], dict):
                            lut_field_names = {f.name for f in fields(LutParameters)}
                            filtered_lut_data = {k: v for k, v in filtered_op_data['lut_params'].items() if k in lut_field_names}
                            filtered_op_data['lut_params'] = LutParameters(**filtered_lut_data)
                        
                        pipeline_list.append(XYBlendOperation(**filtered_op_data))
                    setattr(config_instance, key, pipeline_list)
                else:
                    # For other fields, attempt to set the attribute directly.
                    # Special handling for boolean values that might be saved as strings "true"/"false"
                    if field_obj.type is bool and isinstance(value, str):
                        value = value.lower() in ('true', '1', 't', 'y')
                    
                    try:
                        # Attempt to assign directly, dataclasses' __init__ handles basic type coercion
                        setattr(config_instance, key, value)
                    except (TypeError, ValueError) as e:
                        print(f"Warning: Failed to assign value '{value}' to field '{key}' (expected type {field_obj.type}). Error: {e}. Using default.")
                        # If assignment fails, the default value (from config_instance = cls()) is preserved.
            else:
                print(f"Warning: Unrecognized config key '{key}' found in loaded data. Skipping.")
        return config_instance

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "Config":
        if not os.path.exists(filepath):
            print(f"Config file not found: {filepath}. Creating default config and saving it.")
            default_config = cls()
            try:
                default_config.save(filepath)
            except Exception as e:
                print(f"Error saving default config to {filepath}: {e}")
            return default_config
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from config file '{filepath}': {e}. Using default config.")
            return cls()
        except Exception as e:
            print(f"An unexpected error occurred loading config from '{filepath}': {e}. Using default config.")
            return cls()

# Global instance of the configuration, to be used throughout the application.
_CONFIG_FILE = "app_config.json"
app_config = Config.load(_CONFIG_FILE)