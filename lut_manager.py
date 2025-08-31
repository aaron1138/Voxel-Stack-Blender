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

# lut_manager.py (Spline Update)

import numpy as np
import json
import os
import math
from typing import Optional, Callable, List
from scipy.interpolate import CubicSpline # NEW: Add SciPy for spline interpolation

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
    """
    lut = np.arange(256, dtype=np.float32) 

    if input_min >= input_max:
        return lut.astype(np.uint8)

    input_ramp = np.linspace(0.0, 1.0, num=(input_max - input_min + 1))
    curved_ramp = curve_func(input_ramp)
    
    output_range_size = output_max - output_min
    scaled_curve = curved_ramp * output_range_size + output_min
    
    lut[input_min:input_max+1] = scaled_curve
    
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
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(lut_array.tolist(), f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to save LUT to '{filepath}': {e}")

def load_lut(filepath: str) -> np.ndarray:
    """Loads a LUT array from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"LUT file not found: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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
    return _generate_curve_in_range(lambda x: x, input_min, input_max, output_min, output_max)

def generate_gamma_lut(gamma_value: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates a gamma correction LUT within a specified range."""
    if gamma_value <= 0: gamma_value = 0.01
    inv_gamma = 1.0 / gamma_value
    curve_func = lambda x: np.power(x, inv_gamma)
    return _generate_curve_in_range(curve_func, input_min, input_max, output_min, output_max)

def generate_s_curve_lut(contrast: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates the original power-based S-curve (contrast) LUT within a specified range."""
    contrast = max(0.0, min(1.0, contrast))
    def original_s_curve(x):
        midpoint = 0.5
        if contrast == 0.0: return x
        elif contrast == 1.0: return np.where(x < midpoint, 0.0, 1.0)
        else:
            gamma_factor = 1.0 / (1.0 + contrast * 4)
            return np.where(x < midpoint, np.power(x / midpoint, gamma_factor) * midpoint, 1.0 - np.power((1.0 - x) / midpoint, gamma_factor) * midpoint)
    return _generate_curve_in_range(original_s_curve, input_min, input_max, output_min, output_max)

def generate_log_lut(param: float, input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """Generates a logarithmic LUT within a specified range."""
    if param <= 0: param = 0.01
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
    def rodbard_curve(x):
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        num = x * (a * x + b)
        den = x * (c * x + d) + e
        return np.divide(num, den, out=np.zeros_like(x), where=den!=0)
    curve_func = lambda x: (1 - contrast) * x + contrast * rodbard_curve(x)
    return _generate_curve_in_range(curve_func, input_min, input_max, output_min, output_max)

def generate_spline_lut(control_points: List[List[int]], input_min: int, input_max: int, output_min: int, output_max: int) -> np.ndarray:
    """
    Generates a LUT from a series of control points using cubic spline interpolation.
    """
    if len(control_points) < 2:
        # Not enough points for a spline, return a linear ramp
        return generate_linear_lut(input_min, input_max, output_min, output_max)

    # Sort points by x-coordinate and normalize to 0-1 range
    points = sorted(control_points, key=lambda p: p[0])
    x_coords = np.array([p[0] for p in points]) / 255.0
    y_coords = np.array([p[1] for p in points]) / 255.0
    
    # Ensure start and end points exist if they weren't provided
    if x_coords[0] > 0:
        x_coords = np.insert(x_coords, 0, 0)
        y_coords = np.insert(y_coords, 0, y_coords[0])
    if x_coords[-1] < 1.0:
        x_coords = np.append(x_coords, 1.0)
        y_coords = np.append(y_coords, y_coords[-1])

    # Create the spline function
    spline = CubicSpline(x_coords, y_coords, bc_type='clamped')

    # The curve function is now our spline
    return _generate_curve_in_range(spline, input_min, input_max, output_min, output_max)
