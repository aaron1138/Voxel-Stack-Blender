# config.py (Modified)

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Any
import json
import os
import copy # Import copy for deepcopy if needed for objects in pipeline

# Define a sensible default for thread count, using system core count
DEFAULT_NUM_WORKERS = max(1, os.cpu_count() - 1)

# --- Data Classes for Pipeline Operations ---

@dataclass
class LutParameters:
    """
    Parameters for generating or loading a Look-Up Table (LUT).
    These parameters will be nested within an XYBlendOperation of type 'apply_lut'.
    """
    lut_source: str = "generated"  # "generated" or "file"
    
    # Parameters for 'generated' LUTs
    lut_generation_type: str = "linear" # "linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard"
    gamma_value: float = 1.0
    linear_min_input: int = 0
    linear_max_output: int = 255
    s_curve_contrast: float = 0.5
    log_param: float = 10.0
    exp_param: float = 2.0
    sqrt_param: float = 1.0 # Currently a placeholder in lut_manager for algos that don't use it
    rodbard_param: float = 1.0 # Currently a placeholder

    # Parameters for 'file' LUTs
    fixed_lut_path: str = ""

    def __post_init__(self):
        # Basic validation and type coercion for LutParameters
        self.lut_source = self.lut_source.lower()
        self.lut_generation_type = self.lut_generation_type.lower()
        self.gamma_value = max(0.01, self.gamma_value) # Gamma > 0
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
    type: str = "none" # "none", "gaussian_blur", "bilateral_filter", "median_blur", "unsharp_mask", "resize", "apply_lut"

    # Gaussian Blur Parameters
    gaussian_ksize_x: int = 3
    gaussian_ksize_y: int = 3
    gaussian_sigma_x: float = 0.0
    gaussian_sigma_y: float = 0.0

    # Bilateral Filter Parameters
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Median Blur Parameters
    median_ksize: int = 5

    # Unsharp Mask Parameters
    unsharp_amount: float = 1.0
    unsharp_threshold: int = 0
    unsharp_blur_ksize: int = 5
    unsharp_blur_sigma: float = 0.0

    # Resize Parameters
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    resample_mode: str = "LANCZOS4" # "NEAREST", "BILINEAR", "BICUBIC", "LANCZOS4", "AREA"

    # LUT Application Parameters (nested dataclass)
    lut_params: LutParameters = field(default_factory=LutParameters)

    def __post_init__(self):
        self.type = self.type.lower()
        # Ensure kernel sizes are odd and positive
        if self.type in ["gaussian_blur", "median_blur", "unsharp_mask"]:
            if self.type == "gaussian_blur":
                self.gaussian_ksize_x = self._ensure_odd_positive_ksize(self.gaussian_ksize_x)
                self.gaussian_ksize_y = self._ensure_odd_positive_ksize(self.gaussian_ksize_y)
            elif self.type == "median_blur":
                self.median_ksize = self._ensure_odd_positive_ksize(self.median_ksize)
            elif self.type == "unsharp_mask":
                self.unsharp_blur_ksize = self._ensure_odd_positive_ksize(self.unsharp_blur_ksize)

        # Basic validation for resize dimensions
        if self.type == "resize":
            if self.resize_width is not None:
                self.resize_width = max(0, self.resize_width)
            if self.resize_height is not None:
                self.resize_height = max(0, self.resize_height)
            if self.resize_width == 0: self.resize_width = None
            if self.resize_height == 0: self.resize_height = None
            if self.resize_width is None and self.resize_height is None:
                # print(f"Warning: Resize operation '{self.type}' has no dimensions. It will be a no-op.")
                pass # This is fine, just means no-op.

    def _ensure_odd_positive_ksize(self, ksize: int) -> int:
        """Ensures a kernel size is an odd positive integer."""
        if ksize <= 0:
            return 1 # Default to 1 for no-op or minimal valid
        return ksize if ksize % 2 != 0 else ksize + 1


# --- Main Application Configuration ---

@dataclass
class Config:
    """
    Main application configuration.
    This dataclass will hold all settings for the program.
    """
    # General Processing Core Settings
    n_layers: int = 3
    start_index: Optional[int] = 0
    stop_index: Optional[int] = None
    debug_save: bool = False
    thread_count: int = DEFAULT_NUM_WORKERS # NEW: User-configurable thread count

    # Receding Gradient Settings
    use_fixed_norm: bool = False
    fixed_fade_distance: float = 10.0

    # Input/Output Folders
    input_folder: str = ""
    output_folder: str = ""

    # XY Blend Pipeline (list of operations)
    xy_blend_pipeline: List[XYBlendOperation] = field(default_factory=lambda: [XYBlendOperation("none")])

    # Method to convert Config object to a dictionary for serialization
    def to_dict(self) -> dict:
        """Converts the Config instance and its nested dataclasses to a dictionary."""
        return asdict(self)

    # Method to load Config object from a dictionary (e.g., from JSON)
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Creates a Config instance from a dictionary."""
        config_instance = cls() # Start with default config instance
        
        for key, value in data.items():
            if hasattr(config_instance, key):
                if key == 'xy_blend_pipeline':
                    # Manually reconstruct XYBlendOperation objects, including nested LutParameters
                    pipeline_list = []
                    for op_data in value:
                        if 'lut_params' in op_data and isinstance(op_data['lut_params'], dict):
                            op_data['lut_params'] = LutParameters(**op_data['lut_params'])
                        pipeline_list.append(XYBlendOperation(**op_data))
                    setattr(config_instance, key, pipeline_list)
                else:
                    # For other fields, attempt direct assignment, with type conversion if needed
                    current_field_value = getattr(config_instance, key)
                    if isinstance(current_field_value, (int, float, bool, str)) and not isinstance(value, type(current_field_value)):
                        try:
                            if isinstance(current_field_value, int):
                                converted_value = int(value)
                            elif isinstance(current_field_value, float):
                                converted_value = float(value)
                            elif isinstance(current_field_value, bool):
                                # Handle string to bool conversion
                                converted_value = str(value).lower() in ('true', '1', 't', 'y')
                            else: # string
                                converted_value = str(value)
                            setattr(config_instance, key, converted_value)
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert config value for '{key}' from '{value}' to expected type {type(current_field_value)}. Using default.")
                            # Keep default value
                    else:
                        setattr(config_instance, key, value)
            else:
                print(f"Warning: Unrecognized config key '{key}' found in loaded data. Skipping.")
        return config_instance

    def save(self, filepath: str):
        """Saves the current configuration to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Loads configuration from a JSON file."""
        if not os.path.exists(filepath):
            print(f"Config file not found: {filepath}. Creating default config and saving it.")
            default_config = cls()
            try:
                default_config.save(filepath) # Save a default config for future use
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
app_config = Config.load(_CONFIG_FILE) # Load on startup, or create default if not exists