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

# lut_manager.py (Completed with range controls and new algorithms)

import numpy as np
import json
import os
import math
from typing import Optional, Callable

_DEFAULT_Z_REMAP_LUT_ARRAY = np.arange(256, dtype=np.uint8)

def get_default_z_lut() -> np.ndarray:
    """Returns a copy of the default Z-remapping LUT (linear pass-through)."""
    return _DEFAULT_Z_REMAP_LUT_ARRAY.copy()

def _generate_curve_in_range(
    curve_func: Callable[[np.ndarray], np.ndarray],
    input_min: int, input_max: int,
    output_min: int, output_max: int
) -> np.ndarray:
    """
    Helper function to apply a normalized (0-1) curve function within a specific
    input range and scale it to a specific output range.
    
    Values outside the [input_min, input_max] range are passed through unchanged.
    """
    # Start with a linear pass-through LUT. Values outside the active range will keep this value.
    lut = np.arange(256, dtype=np.float32) 

    # Handle the case where the input range is zero or invalid.
    if input_min >= input_max:
        # If the range is flat, it means no curve is applied. The initial linear LUT is correct.
        return lut.astype(np.uint8)

    # Create a normalized ramp (0.0 to 1.0) over the specified input range.
    # This represents the 'x-axis' for the normalized curve function.
    input_ramp = np.linspace(0.0, 1.0, num=(input_max - input_min + 1))
    
    # Apply the provided normalized curve function (e.g., gamma, sqrt) to the ramp.
    curved_ramp = curve_func(input_ramp)
    
    # Scale the result (which is in the 0-1 range) to the desired output range.
    output_range_size = output_max - output_min
    scaled_curve = curved_ramp * output_range_size + output_min
    
    # Place the calculated curve segment into the correct part of the main LUT.
    lut[input_min:input_max+1] = scaled_curve
    
    # Clip the final LUT to ensure all values are valid 8-bit integers and return.
    return np.clip(lut, 0, 255).astype(np.uint8)

def apply_z_lut(image_array: np.ndarray, lut_array: np.ndarray) -> np.ndarray:
    """Applies a given LUT to an 8-bit grayscale image."""
    if image_array.dtype != np.uint8:
        raise TypeError("Input image_array for apply_z_lut must be of type np.uint8.")
    if not isinstance(lut_array, np.ndarray) or lut_array.dtype != np.uint8 or lut_array.shape != (256,):
        raise ValueError("Provided lut_array must be a 256-entry NumPy array of dtype uint8.")
    return lut_array[image_array]

def save_lut(filepath: str, lut_array: np.ndarray):
    """Saves a LUT array to a JSON file."""
    if not isinstance(lut_array, np.ndarray) or lut_array.dtype != np.uint8 or lut_array.shape != (256,):
        raise ValueError("LUT must be a 256-entry NumPy array of dtype uint8 to save.")
    try:
        with open(filepath, 'w') as f:
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
        if not isinstance(lut_list, list) or len(lut_list) != 256:
            raise ValueError("Invalid LUT file format: Expected a list of 256 numbers.")
        return np.array(lut_list, dtype=np.uint8)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in LUT file '{filepath}': {e}")
    except Exception as e:
        raise IOError(f"Failed to load LUT from '{filepath}': {e}")

# --- Algorithmic LUT Generation Functions ---

def generate_linear_lut(input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates a linear LUT that maps a specific input range to a specific output range."""
    # A linear curve is a function where y = x.
    return _generate_curve_in_range(lambda x: x, input_min, input_max, output_min, output_max)

def generate_gamma_lut(gamma_value: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates a gamma correction LUT within a specified range."""
    if gamma_value <= 0: gamma_value = 0.01
    inv_gamma = 1.0 / gamma_value
    curve_func = lambda x: np.power(x, inv_gamma)
    return _generate_curve_in_range(curve_func, input_min, input_max, output_min, output_max)

def generate_s_curve_lut(contrast: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates a sigmoid-based S-curve (contrast) LUT within a specified range."""
    # Map contrast (0-1) to a strength parameter 'k'. Avoid k=0.
    k = (contrast * 10.0) + 0.001 
    # A common sigmoid function: 1 / (1 + exp(-k * (x - 0.5)))
    # We must normalize it to ensure it still maps 0->0 and 1->1
    def sigmoid(x):
        y = 1 / (1 + np.exp(-k * (x - 0.5)))
        # Rescale to fit 0-1 range
        y0 = 1 / (1 + np.exp(k * 0.5))
        y1 = 1 / (1 + np.exp(-k * 0.5))
        return (y - y0) / (y1 - y0)
    return _generate_curve_in_range(sigmoid, input_min, input_max, output_min, output_max)

def generate_log_lut(param: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates a logarithmic LUT within a specified range."""
    if param <= 0: param = 0.01
    # Use np.log1p(x) which is log(1+x) for numerical stability near zero.
    curve_func = lambda x: np.log1p(x * param) / np.log1p(param)
    return _generate_curve_in_range(curve_func, input_min, input_max, output_min, output_max)

def generate_exp_lut(param: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates an exponential LUT within a specified range."""
    if param <= 0: param = 0.01
    curve_func = lambda x: np.power(x, param)
    return _generate_curve_in_range(curve_func, input_min, input_max, output_min, output_max)

def generate_sqrt_lut(root_value: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates a root LUT (e.g., square root, cube root) within a specified range."""
    if root_value <= 0: root_value = 0.1
    inv_root = 1.0 / root_value
    curve_func = lambda x: np.power(x, inv_root)
    return _generate_curve_in_range(curve_func, input_min, input_max, output_min, output_max)

def generate_rodbard_lut(contrast: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates an ACES-style Rodbard contrast LUT within a specified range."""
    # Define the base Rodbard curve function which maps 0-1 to 0-1
    def rodbard_curve(x):
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        num = x * (a * x + b)
        den = x * (c * x + d) + e
        # Use np.divide to handle potential division by zero safely, returning 0 in that case
        return np.divide(num, den, out=np.zeros_like(x), where=den!=0)

    # The curve function passed to the helper will blend between linear (y=x) and the full Rodbard curve
    # The 'contrast' parameter controls the blend amount.
    curve_func = lambda x: (1 - contrast) * x + contrast * rodbard_curve(x)
    return _generate_curve_in_range(curve_func, input_min, input_max, output_min, output_max)
