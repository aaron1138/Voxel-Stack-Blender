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
    
    # Parameters for Gaussian Blur (relevant if type is "gaussian_blur")
    gaussian_ksize_x: int = 3
    gaussian_ksize_y: int = 3
    gaussian_sigma_x: float = 0.0
    gaussian_sigma_y: float = 0.0

    # Parameters for Bilateral Filter (relevant if type is "bilateral_filter")
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Parameters for Median Blur (relevant if type is "median_blur")
    median_ksize: int = 5

    # Parameters for Unsharp Masking (relevant if type is "unsharp_mask")
    unsharp_amount: float = 1.0
    unsharp_threshold: int = 0
    unsharp_blur_ksize: int = 5
    unsharp_blur_sigma: float = 0.0

    # Parameters for Resize operation (relevant if type is "resize")
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
        data = {"type": self.type}
        params = {}
        if self.type == "gaussian_blur":
            params = {
                "gaussian_ksize_x": self.gaussian_ksize_x,
                "gaussian_ksize_y": self.gaussian_ksize_y,
                "gaussian_sigma_x": self.gaussian_sigma_x,
                "gaussian_sigma_y": self.gaussian_sigma_y,
            }
        elif self.type == "bilateral_filter":
            params = {
                "bilateral_d": self.bilateral_d,
                "bilateral_sigma_color": self.bilateral_sigma_color,
                "bilateral_sigma_space": self.bilateral_sigma_space,
            }
        elif self.type == "median_blur":
            params = {
                "median_ksize": self.median_ksize,
            }
        elif self.type == "unsharp_mask":
            params = {
                "unsharp_amount": self.unsharp_amount,
                "unsharp_threshold": self.unsharp_threshold,
                "unsharp_blur_ksize": self.unsharp_blur_ksize,
                "unsharp_blur_sigma": self.unsharp_blur_sigma,
            }
        elif self.type == "resize":
            params = {
                "resize_width": self.resize_width,
                "resize_height": self.resize_height,
                "resample_mode": self.resample_mode,
            }
        data["params"] = params
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'XYBlendOperation':
        """Creates an XYBlendOperation instance from a dictionary."""
        op_type = d.get("type", "none")
        params = d.get("params", {})
        
        # Create a default instance and then override its attributes with loaded params
        instance = cls(type=op_type)
        for key, value in params.items():
            if hasattr(instance, key):
                # Special handling for resize_width/height: convert 0 to None if that's the default behavior
                if key in ["resize_width", "resize_height"] and value == 0:
                    setattr(instance, key, None)
                else:
                    setattr(instance, key, value)
        instance.__post_init__() # Ensure validation logic runs after setting attributes
        return instance


@dataclass
class Config:
    """
    Application configuration for Modular-Stacker.
    This is a dataclass defining all configuration parameters.
    The singleton pattern is managed externally at the module level.
    """
    _config_file: str = "modular_stacker_config.json"

    # --- I/O settings ---
    input_dir: str = ""  # Path to folder of numbered PNGs
    file_pattern: str = "*.png" # Pattern to match input files (e.g., "*.png", "image_*.tif")
    output_dir: str = "" # Where to write output PNGs
    run_log_file: str = "modular_stacker_runs.log" # File to log run settings and timestamps
    
    # --- Stacking (blend) parameters ---
    primary: int = 3     # Number of primary layers per output
    radius: int = 1      # Number of neighbors on each side
    blend_mode: str = "gaussian" # Weight curve: gaussian, linear, flat, exp_decay, cosine,
                                 # binary_contour, gradient_contour, z_column_lift, z_contour_interp
    blend_param: float = 1.0     # σ or exponent for blend_mode
    directional_blend: bool = False # Enable Z-bias directionality
    dir_sigma: float = 1.0       # σ for directional bias

    # --- LUT Management (replaces old remap, gamma, normalize, clamp, preserve_black, floor/ceil) ---
    # This defines how the active Z-LUT is generated or loaded.
    # The actual LUT data is managed by the lut_manager module.
    lut_source: str = "generated" # "generated" or "file"
    fixed_lut_path: str = ""       # Path to a custom Z-LUT JSON file if lut_source is "file"
    
    # Parameters for algorithmic LUT generation (if lut_source is "generated")
    lut_generation_type: str = "linear" # "linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard"
    gamma_value: float = 1.0            # For "gamma" generation type
    linear_min_input: int = 0           # For "linear" generation type
    linear_max_output: int = 255        # For "linear" generation type
    s_curve_contrast: float = 0.5       # For "s_curve" generation type (0.0 to 1.0)
    log_param: float = 10.0             # For "log" generation type (e.g., base for log, or strength)
    exp_param: float = 2.0              # For "exp" generation type (e.g., exponent)
    sqrt_param: float = 1.0             # For "sqrt" generation type (e.g., strength, currently simplified)
    rodbard_param: float = 1.0          # For "rodbard" generation type (e.g., strength, currently simplified)

    # --- XY Blending / Smoothing / Sharpening / Resizing Pipeline ---
    # This is a list of XYBlendOperation objects, allowing for a sequence of operations.
    xy_blend_pipeline: List[XYBlendOperation] = field(default_factory=lambda: [XYBlendOperation()])

    # --- Output & threading (global output resize removed, now per-slot in XY pipeline) ---
    threads: int = field(default_factory=lambda: os.cpu_count() or 4) # number of worker threads
    pad_filenames: bool = False          # zero-pad output filenames
    pad_length: int = 4                  # digits for filename padding

    # --- Advanced toggles ---
    integer_mode: bool = True            # use integer engine (always true for new app)
    cap_layers: int = 0                  # extra blank “cap” slices
    resume_from: int = 1                 # first stack index to process
    stop_at: int = 999999                # last stack index to process
    scale_bits: int = 12                 # Fixed-point scaling bits for integer mode (for internal 16-bit processing)

    # --- Zarr-Engine Settings ---
    zarr_store: str = ""                 # Path to Zarr store (future use)
    zarr_cache_chunks: int = 1           # Number of Zarr chunks to cache (future use)

    # --- Callbacks (not serialized, handled by runtime) ---
    stop_requested: bool = field(default=False, compare=False) # internal cancellation flag
    progress_callback: Optional[Callable[[int, int], None]] = field(default=None, compare=False) # receives (current, total) tuples
    stop_callback: Optional[Callable[[], bool]] = field(default=None, compare=False) # if set, can abort the run

    # --- Binary/Gradient contour thresholds ---
    binary_threshold: int = 128          # B/W cutoff for shadow/overhang/contour
    gradient_threshold: int = 128        # Cutoff for gradient-based contour

    # --- Other Advanced Toggles (now only these, clamp/preserve_black removed) ---
    top_surface_smoothing: bool = False # boost trailing-edge pixels
    top_surface_strength: float = 0.0 # 0.0–1.0 strength of surface smooth
    gradient_smooth: bool = True         # enable XY gradient smoothing (for float mode, but kept for consistency)
    gradient_blend_strength: float = 0.0 # 0.0–1.0 strength of gradient smooth

    def __post_init__(self):
        # Ensure threads is at least 1
        if self.threads < 1:
            self.threads = 1
        
        # Apply post_init logic for each XYBlendOperation in the pipeline
        # Ensure xy_blend_pipeline is a list of XYBlendOperation objects
        if not isinstance(self.xy_blend_pipeline, list):
            self.xy_blend_pipeline = [XYBlendOperation()] # Default to one 'none' operation
        else:
            # Ensure elements are XYBlendOperation instances, converting from dicts if necessary
            self.xy_blend_pipeline = [
                op if isinstance(op, XYBlendOperation) else XYBlendOperation.from_dict(op)
                for op in self.xy_blend_pipeline
            ]
        for op in self.xy_blend_pipeline:
            op.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Config object to a dictionary, suitable for JSON serialization."""
        data = {}
        for f in fields(self):
            # Skip non-serializable fields like Callables and internal singleton state
            if f.name in ["_config_file", "progress_callback", "stop_callback"]: # Removed _instance, _initialized
                continue
            value = getattr(self, f.name)
            if f.name == "xy_blend_pipeline":
                data[f.name] = [op.to_dict() for op in value]
            else:
                data[f.name] = value
        return data

    def load(self) -> None:
        """Loads configuration from the JSON file if it exists."""
        print("Config: Attempting to load config from file...")
        loaded_data = {}
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Error decoding config file '{self._config_file}': {e}. Using default settings.")
            except Exception as e:
                print(f"Warning: An unexpected error occurred while loading config '{self._config_file}': {e}. Using default settings.")
        
        # Apply loaded data, overriding current defaults set by dataclass __init__
        for key, value in loaded_data.items():
            if hasattr(self, key): # Only set attributes that are defined in the dataclass
                if key == "xy_blend_pipeline":
                    # Reconstruct XYBlendOperation objects from dicts
                    if isinstance(value, list):
                        # Use object.__setattr__ to bypass any custom setters if they existed
                        object.__setattr__(self, key, [XYBlendOperation.from_dict(op_dict) for op_dict in value])
                    else:
                        print(f"Warning: Expected list for '{key}' in config, but got {type(value)}. Retaining default.")
                elif key in ["resize_width", "resize_height"] and value == 0:
                    object.__setattr__(self, key, None) # Convert 0 to None for Optional[int] fields
                else:
                    object.__setattr__(self, key, value)
            else:
                print(f"Warning: Loaded config contains unknown key '{key}'. Skipping.")
        
        print(f"Config: After loading, xy_blend_pipeline exists: {'xy_blend_pipeline' in self.__dict__}")
        # __post_init__ is called automatically by the dataclass after super().__init__()
        # and after all default fields are set. We do NOT call it explicitly here.

    def save(self) -> None:
        """Saves the current configuration to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self._config_file) or ".", exist_ok=True)
            with open(self._config_file, 'w', encoding='utf-8') as f:
                serializable_data = self.to_dict()
                json.dump(serializable_data, f, indent=4)
        except Exception as e:
            print(f"Error saving config file '{self._config_file}': {e}")

# Global instance management for Config
_app_config_instance: Optional[Config] = None

def get_app_config() -> Config:
    """
    Returns the singleton instance of the Config.
    Initializes it if it doesn't exist.
    """
    global _app_config_instance
    if _app_config_instance is None:
        print("Config: Creating new singleton instance.")
        _app_config_instance = Config()
        _app_config_instance.load() # Load settings from file after creation
    return _app_config_instance

# Global instance of the configuration
app_config = get_app_config()
