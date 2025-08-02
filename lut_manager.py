# lut_manager.py (Modified)

"""
Provides functions to generate LUTs based on various algorithms (linear, gamma, S-curve, etc.)
or load them from a file. LUTs are no longer stored as a global singleton within this module.
Functions now operate on and return LUT arrays directly.
"""

import numpy as np
import json
import os
import math
from typing import Optional # Removed Any as _config_ref is gone

# The default Z Remapping LUT (linear 0-255) is now a constant, not part of mutable global state
_DEFAULT_Z_REMAP_LUT_ARRAY = np.arange(256, dtype=np.uint8)

# Removed:
# _current_z_remap_lut: np.ndarray = _DEFAULT_Z_REMAP_LUT_ARRAY.copy()
# _config_ref: Optional[Any] = None

# Removed:
# def set_config_reference(config_instance: Any):
#     """Sets the reference to the global Config instance."""
#     global _config_ref
#     _config_ref = config_instance

def get_default_z_lut() -> np.ndarray:
    """Returns a copy of the default Z-remapping LUT (linear)."""
    return _DEFAULT_Z_REMAP_LUT_ARRAY.copy()

# Removed:
# def get_current_z_lut() -> np.ndarray:
#     """Returns the currently active Z-remapping LUT."""
#     return _current_z_remap_lut.copy()
#
# def set_current_z_lut(new_lut: np.ndarray):
#     """Sets the currently active Z-remapping LUT.
#     Args:
#         new_lut (np.ndarray): A 256-entry NumPy array of dtype uint8.
#     """
#     if not isinstance(new_lut, np.ndarray) or new_lut.dtype != np.uint8 or new_lut.shape != (256,):
#         raise ValueError("New LUT must be a 256-entry NumPy array of dtype uint8.")
#     global _current_z_remap_lut
#     _current_z_remap_lut = new_lut.copy()

def apply_z_lut(image_array: np.ndarray, lut_array: np.ndarray) -> np.ndarray:
    """
    Applies a given LUT to an 8-bit grayscale image (NumPy array).

    Args:
        image_array (np.ndarray): An 8-bit grayscale NumPy array (uint8).
                                  Expected values are 0-255.
        lut_array (np.ndarray): A 256-entry NumPy array of dtype uint8 representing the LUT.
                                This is the LUT to be applied.
    Returns:
        np.ndarray: A new NumPy array with the LUT applied,
                    remapped to 0-255 uint8 values.
    """
    if image_array.dtype != np.uint8:
        raise TypeError("Input image_array for apply_z_lut must be of type np.uint8.")
    
    if not isinstance(lut_array, np.ndarray) or lut_array.dtype != np.uint8 or lut_array.shape != (256,):
        raise ValueError("Provided lut_array must be a 256-entry NumPy array of dtype uint8.")

    # Apply the LUT using direct indexing (most efficient way for 0-255 range)
    remapped_array = lut_array[image_array]
    
    return remapped_array

def save_lut(filepath: str, lut_array: np.ndarray):
    """Saves a LUT array to a JSON file."""
    if lut_array.dtype != np.uint8 or lut_array.shape != (256,):
        raise ValueError("LUT must be a 256-entry NumPy array of dtype uint8 to save.")
    
    try:
        with open(filepath, 'w') as f:
            # Convert NumPy array to Python list for JSON serialization
            json.dump(lut_array.tolist(), f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to save LUT to '{filepath}': {e}")

def load_lut(filepath: str) -> np.ndarray:
    """Loads a LUT array from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"LUT file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            lut_list = json.load(f)
        
        # Validate and convert to NumPy array
        if not isinstance(lut_list, list) or len(lut_list) != 256:
            raise ValueError("Invalid LUT file format: Expected a list of 256 numbers.")
        
        loaded_lut = np.array(lut_list, dtype=np.uint8)
        # Ensure loaded values are within 0-255 if they somehow exceed (though uint8 handles this)
        return np.clip(loaded_lut, 0, 255).astype(np.uint8)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in LUT file '{filepath}': {e}")
    except Exception as e:
        raise IOError(f"Failed to load LUT from '{filepath}': {e}")

# --- Algorithmic LUT Generation Functions ---

def generate_linear_lut(min_input: int, max_output: int) -> np.ndarray:
    """Generates a linear LUT that maps input range [0, 255] to [min_input, max_output]."""
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        # Scale input i (0-255) to a normalized 0-1 range
        normalized_input = i / 255.0
        # Then map this normalized value to the desired output range [min_input, max_output]
        output_val = min_input + normalized_input * (max_output - min_input)
        lut[i] = np.clip(output_val, 0, 255) # Ensure output is within 0-255
    return lut.astype(np.uint8)

def generate_gamma_lut(gamma_value: float) -> np.ndarray:
    """Generates a gamma correction LUT."""
    if gamma_value <= 0:
        raise ValueError("Gamma value must be positive.")
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized_input = i / 255.0
        output_val = np.power(normalized_input, 1.0 / gamma_value) * 255.0
        lut[i] = np.clip(output_val, 0, 255)
    return lut.astype(np.uint8)

def generate_s_curve_lut(contrast: float) -> np.ndarray:
    """Generates an S-curve (contrast) LUT."""
    if not (0.0 <= contrast <= 1.0):
        raise ValueError("Contrast must be between 0.0 and 1.0.")
    
    lut = np.zeros(256, dtype=np.float32)
    x = np.linspace(0, 1, 256) # Normalized input values

    midpoint = 0.5
    if contrast == 0.0: # Linear
        y = x
    elif contrast == 1.0: # Max contrast (hard clip)
        y = np.where(x < midpoint, 0.0, 1.0)
    else:
        # A more flexible S-curve can be achieved with a cubic or similar function
        # For simplicity, we'll use a power-based approach for now, similar to what's often seen.
        # This one is more like a gamma curve applied to halves.
        gamma_factor = 1.0 / (1.0 + contrast * 4) # Adjust this factor for desired curve
        y = np.where(x < midpoint,
                     np.power(x / midpoint, gamma_factor) * midpoint,
                     1.0 - np.power((1.0 - x) / midpoint, gamma_factor) * midpoint)
        
    lut = np.clip(y * 255.0, 0, 255)
    return lut.astype(np.uint8)

def generate_log_lut(param: float) -> np.ndarray:
    """Generates a logarithmic LUT."""
    if param <= 0:
        raise ValueError("Log parameter must be positive.")
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized_input = i / 255.0
        # Avoid log(0)
        if normalized_input == 0:
            output_val = 0.0
        else:
            output_val = np.log1p(normalized_input * param) / np.log1p(param) * 255.0
        lut[i] = np.clip(output_val, 0, 255)
    return lut.astype(np.uint8)

def generate_exp_lut(param: float) -> np.ndarray:
    """Generates an exponential LUT."""
    if param <= 0:
        raise ValueError("Exp parameter must be positive.")
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized_input = i / 255.0
        output_val = np.power(normalized_input, param) * 255.0
        lut[i] = np.clip(output_val, 0, 255)
    return lut.astype(np.uint8)

def generate_sqrt_lut(param: float) -> np.ndarray:
    """Generates a square root LUT."""
    if param <= 0: # Param currently unused, but for future scaling
        param = 1.0
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized_input = i / 255.0
        output_val = np.sqrt(normalized_input) * 255.0 # Simple sqrt
        lut[i] = np.clip(output_val, 0, 255)
    return lut.astype(np.uint8)

def generate_rodbard_lut(param: float) -> np.ndarray:
    """Generates an ACES-style Rodbard contrast LUT."""
    # Parameters are constants tuned for specific curve. 'param' can be used for scaling.
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        x = i / 255.0 # Normalized input
        num = x * (a * x + b)
        den = x * (c * x + d) + e
        y = np.clip(num / den, 0.0, 1.0)
        lut[i] = np.clip(y * 255.0, 0, 255)
    return lut.astype(np.uint8)

# Removed:
# def update_active_lut_from_config():
# Initial setup: ensure a default LUT is active (no longer needed here)
# set_current_z_lut(get_default_z_lut())

# Removed example usage __main__ block