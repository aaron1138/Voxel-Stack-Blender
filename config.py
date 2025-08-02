# config.py

"""
Defines the canonical Config data structure for the new Modular-Stacker pipeline.
All parameters are declared here with sensible defaults.
This class will be a singleton to ensure a single source of truth for settings.
"""

import os
import json
import math
from dataclasses import dataclass, field, fields
from typing import Optional, Callable, Dict, Any, List

@dataclass
class XYBlendOperation:
    """
    Represents a single XY blending/smoothing/sharpening/resizing operation in the pipeline.
    Contains the operation type and its specific parameters.
    """
    type: str = "none" # "none", "gaussian_blur", "bilateral_filter", "median_blur", "unsharp_mask", "resize"
    
    # Parameters for Gaussian Blur
    gaussian_ksize_x: int = 3
    gaussian_ksize_y: int = 3
    gaussian_sigma_x: float = 0.0
    gaussian_sigma_y: float = 0.0

    # Parameters for Bilateral Filter
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Parameters for Median Blur
    median_ksize: int = 5

    # Parameters for Unsharp Masking
    unsharp_amount: float = 1.0
    unsharp_threshold: int = 0
    unsharp_blur_ksize: int = 5
    unsharp_blur_sigma: float = 0.0

    # Parameters for Resize operation
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    resample_mode: str = "LANCZOS4" # NEAREST, BILINEAR, BICUBIC, LANCZOS4, AREA

    def __post_init__(self):
        # Ensure kernel sizes are odd and positive where applicable
        if self.type in ["gaussian_blur", "median_blur", "unsharp_mask"]:
            self.gaussian_ksize_x = self._ensure_odd_positive(self.gaussian_ksize_x)
            self.gaussian_ksize_y = self._ensure_odd_positive(self.gaussian_ksize_y)
            self.median_ksize = self._ensure_odd_positive(self.median_ksize)
            self.unsharp_blur_ksize = self._ensure_odd_positive(self.unsharp_blur_ksize)

    def _ensure_odd_positive(self, val: int) -> int:
        """Ensures a kernel size is an odd positive integer, adjusting if necessary."""
        if val <= 0:
            return 1
        return val if val % 2 != 0 else val + 1

    def to_dict(self) -> Dict[str, Any]:
        """Converts the XYBlendOperation object to a dictionary for serialization."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'XYBlendOperation':
        """Creates an XYBlendOperation instance from a dictionary."""
        known_fields = {f.name for f in fields(cls)}
        filtered_d = {k: v for k, v in d.items() if k in known_fields}
        instance = cls(**filtered_d)
        instance.__post_init__()
        return instance

@dataclass
class LutConfig:
    """
    Encapsulates all settings for a single named LUT.
    """
    source: str = "generated"
    fixed_path: str = ""
    generation_type: str = "linear"
    gamma_value: float = 1.0
    linear_min_input: int = 0
    linear_max_output: int = 255
    s_curve_contrast: float = 0.5
    log_param: float = 10.0
    exp_param: float = 2.0
    sqrt_param: float = 1.0
    rodbard_param: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts the LutConfig object to a dictionary for serialization."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LutConfig':
        """Creates a LutConfig instance from a dictionary."""
        known_fields = {f.name for f in fields(cls)}
        filtered_d = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered_d)

@dataclass
class Config:
    """
    Application configuration for Modular-Stacker.
    """
    _config_file: str = "modular_stacker_config.json"

    # --- I/O settings ---
    input_dir: str = ""
    file_pattern: str = "*.png"
    output_dir: str = ""
    run_log_file: str = "modular_stacker_runs.log"
    
    # --- Stacking (blend) parameters ---
    primary: int = 3
    radius: int = 1
    blend_mode: str = "gaussian"
    blend_param: float = 1.0
    directional_blend: bool = False
    dir_sigma: float = 1.0

    # --- Vertical Blending Parameters ---
    vertical_blend_pre_process: bool = False # NEW: If True, run VB before stacking. If False, VB is a substitute for stacking.
    vertical_receding_layers: int = 3
    vertical_receding_fade_dist: float = 10.0
    vertical_overhang_layers: int = 3
    vertical_overhang_fade_dist: float = 10.0

    # --- LUT Management ---
    apply_vertical_luts: bool = True # NEW: Apply receding/overhang LUTs post-vertical blend.
    apply_default_lut_after_stacking: bool = True # NEW: Apply default LUT post-stacking.
    lut_settings: Dict[str, LutConfig] = field(default_factory=lambda: {
        "default": LutConfig(),
        "receding": LutConfig(),
        "overhang": LutConfig()
    })

    # --- XY Blending Pipeline ---
    xy_blend_pipeline: List[XYBlendOperation] = field(default_factory=lambda: [XYBlendOperation()])

    # --- Output & threading ---
    threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    pad_filenames: bool = False
    pad_length: int = 4

    # --- Advanced toggles ---
    cap_layers: int = 0
    resume_from: int = 1
    stop_at: int = 999999
    scale_bits: int = 12
    binary_threshold: int = 128
    gradient_threshold: int = 128
    top_surface_smoothing: bool = False
    top_surface_strength: float = 0.0
    gradient_smooth: bool = True
    gradient_blend_strength: float = 0.0

    # --- Callbacks (not serialized) ---
    stop_requested: bool = field(default=False, compare=False, repr=False)
    progress_callback: Optional[Callable[[int, int], None]] = field(default=None, compare=False, repr=False)
    stop_callback: Optional[Callable[[], bool]] = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.threads < 1:
            self.threads = 1
        
        self.xy_blend_pipeline = [
            op if isinstance(op, XYBlendOperation) else XYBlendOperation.from_dict(op)
            for op in self.xy_blend_pipeline
        ]
        for op in self.xy_blend_pipeline:
            op.__post_init__()

        self.lut_settings = {
            name: (conf if isinstance(conf, LutConfig) else LutConfig.from_dict(conf))
            for name, conf in self.lut_settings.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Config object to a dictionary for JSON serialization."""
        data = {}
        for f in fields(self):
            if f.compare is False or f.name.startswith('_'):
                continue
            
            value = getattr(self, f.name)
            if f.name == "xy_blend_pipeline":
                data[f.name] = [op.to_dict() for op in value]
            elif f.name == "lut_settings":
                data[f.name] = {name: conf.to_dict() for name, conf in value.items()}
            else:
                data[f.name] = value
        return data

    def load(self) -> None:
        """Loads configuration from the JSON file."""
        if not os.path.exists(self._config_file):
            return

        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file '{self._config_file}': {e}. Using defaults.")
            return

        for key, value in loaded_data.items():
            if hasattr(self, key):
                if key == "xy_blend_pipeline" and isinstance(value, list):
                    setattr(self, key, [XYBlendOperation.from_dict(op_dict) for op_dict in value])
                elif key == "lut_settings" and isinstance(value, dict):
                    default_luts = {
                        "default": LutConfig(), "receding": LutConfig(), "overhang": LutConfig()
                    }
                    for name, conf_dict in value.items():
                        if name in default_luts:
                            default_luts[name] = LutConfig.from_dict(conf_dict)
                    setattr(self, key, default_luts)
                else:
                    setattr(self, key, value)
        
        self.__post_init__()

    def save(self) -> None:
        """Saves the current configuration to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self._config_file) or ".", exist_ok=True)
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4)
        except IOError as e:
            print(f"Error saving config file '{self._config_file}': {e}")

# --- Singleton Instance Management ---
_app_config_instance: Optional[Config] = None

def get_app_config() -> Config:
    """Returns the singleton instance of the Config, creating and loading it if necessary."""
    global _app_config_instance
    if _app_config_instance is None:
        _app_config_instance = Config()
        _app_config_instance.load()
    return _app_config_instance

# Global instance of the configuration
app_config = get_app_config()
