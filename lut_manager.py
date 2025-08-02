# lut_manager.py

"""
Manages the active Z-axis Look-Up Table (LUT) for Modular-Stacker.
Provides functions to generate LUTs based on various algorithms (linear, gamma, S-curve, etc.)
or load them from a file. The active LUT is a global singleton.
"""

import numpy as np
import json
import os
import math
from typing import Optional, Any # Import Any here

# Default Z Remapping LUT (linear 0-255)
_DEFAULT_Z_REMAP_LUT_ARRAY = np.arange(256, dtype=np.uint8)

# Currently Active Z Remapping LUT
_current_z_remap_lut: np.ndarray = _DEFAULT_Z_REMAP_LUT_ARRAY.copy()

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

def get_default_z_lut() -> np.ndarray:
    """Returns a copy of the default Z-remapping LUT (linear)."""
    return _DEFAULT_Z_REMAP_LUT_ARRAY.copy()

def get_current_z_lut() -> np.ndarray:
    """Returns the currently active Z-remapping LUT."""
    return _current_z_remap_lut.copy()

def set_current_z_lut(new_lut: np.ndarray):
    """Sets the currently active Z-remapping LUT.
    Args:
        new_lut (np.ndarray): A 256-entry NumPy array of dtype uint8.
    """
    if not isinstance(new_lut, np.ndarray) or new_lut.dtype != np.uint8 or new_lut.shape != (256,):
        raise ValueError("New LUT must be a 256-entry NumPy array of dtype uint8.")
    global _current_z_remap_lut
    _current_z_remap_lut = new_lut.copy()

def apply_z_lut(image_array: np.ndarray) -> np.ndarray:
    """
    Applies the currently active Z-REMAP_LUT to an 8-bit grayscale image (NumPy array).
    Args:
        image_array (np.ndarray): An 8-bit grayscale NumPy array (uint8).
                                  Expected values are 0-255.
    Returns:
        np.ndarray: A new NumPy array with the LUT applied,
                    remapped to 0-255 uint8 values.
    """
    if image_array.dtype != np.uint8:
        # If input is float, convert to uint8 before applying LUT
        # This assumes the float values are in the 0-255 range or need to be scaled
        # If the float image is normalized 0-1, it should be scaled to 0-255 first
        # before calling this function.
        raise TypeError("Input image_array for apply_z_lut must be of type np.uint8.")
    
    # Apply the LUT using direct indexing (most efficient way for 0-255 range)
    remapped_array = _current_z_remap_lut[image_array]
    
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
        return loaded_lut
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
        # This assumes min_input and max_output are within 0-255
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

    # Adjust the S-curve formula based on contrast
    # A common S-curve formula uses a logistic function or a power function around a midpoint.
    # For simplicity, we can use a power function based on contrast.
    # Higher contrast makes the curve steeper, lower makes it flatter.
    
    # Example S-curve approximation using a piecewise power function
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


def update_active_lut_from_config():
    """
    Updates the global active LUT based on the current settings in the Config.
    This should be called whenever config settings related to LUT generation change.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in lut_manager. Cannot update active LUT.")
        return

    cfg = _config_ref # Use the globally set config instance

    if cfg.lut_source == "file":
        if cfg.fixed_lut_path and os.path.exists(cfg.fixed_lut_path):
            try:
                loaded_lut = load_lut(cfg.fixed_lut_path)
                set_current_z_lut(loaded_lut)
                print(f"LUT Manager: Loaded LUT from file: {cfg.fixed_lut_path}")
            except Exception as e:
                print(f"Error loading LUT from file '{cfg.fixed_lut_path}': {e}. Falling back to default linear LUT.")
                set_current_z_lut(get_default_z_lut())
        else:
            print("LUT Manager: Fixed LUT path not specified or file not found. Falling back to default linear LUT.")
            set_current_z_lut(get_default_z_lut())
    elif cfg.lut_source == "generated":
        generated_lut = None
        if cfg.lut_generation_type == "linear":
            generated_lut = generate_linear_lut(cfg.linear_min_input, cfg.linear_max_output)
        elif cfg.lut_generation_type == "gamma":
            generated_lut = generate_gamma_lut(cfg.gamma_value)
        elif cfg.lut_generation_type == "s_curve":
            generated_lut = generate_s_curve_lut(cfg.s_curve_contrast)
        elif cfg.lut_generation_type == "log":
            generated_lut = generate_log_lut(cfg.log_param)
        elif cfg.lut_generation_type == "exp":
            generated_lut = generate_exp_lut(cfg.exp_param)
        elif cfg.lut_generation_type == "sqrt":
            generated_lut = generate_sqrt_lut(cfg.sqrt_param)
        elif cfg.lut_generation_type == "rodbard":
            generated_lut = generate_rodbard_lut(cfg.rodbard_param)
        else:
            print(f"Warning: Unknown LUT generation type '{cfg.lut_generation_type}'. Falling back to default linear LUT.")
            generated_lut = get_default_z_lut()
        
        if generated_lut is not None:
            set_current_z_lut(generated_lut)
            print(f"LUT Manager: Generated '{cfg.lut_generation_type}' LUT.")
        else:
            print("LUT Manager: Failed to generate LUT. Falling back to default linear LUT.")
            set_current_z_lut(get_default_z_lut())
    else:
        print(f"Warning: Unknown LUT source '{cfg.lut_source}'. Falling back to default linear LUT.")
        set_current_z_lut(get_default_z_lut())

# Initial setup: ensure a default LUT is active
set_current_z_lut(get_default_z_lut())

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- LUT Manager Module Test ---")
    
    # Dummy Config for testing
    class MockConfig:
        def __init__(self):
            self.lut_source = "generated"
            self.lut_generation_type = "linear"
            self.gamma_value = 2.2
            self.linear_min_input = 50
            self.linear_max_output = 200
            self.s_curve_contrast = 0.7
            self.log_param = 50.0
            self.exp_param = 0.5
            self.sqrt_param = 1.0 # Not actively used in sqrt generation, but for consistency
            self.rodbard_param = 1.0 # Not actively used in rodbard generation, but for consistency
            self.fixed_lut_path = ""
    
    mock_config = MockConfig()
    set_config_reference(mock_config) # Set the mock config

    # Test linear LUT generation
    mock_config.lut_generation_type = "linear"
    mock_config.linear_min_input = 50
    mock_config.linear_max_output = 200
    update_active_lut_from_config()
    linear_lut = get_current_z_lut()
    print(f"\nLinear LUT (first 10): {linear_lut[:10]}")
    print(f"Linear LUT (last 10): {linear_lut[-10:]}")
    
    # Test gamma LUT generation
    mock_config.lut_generation_type = "gamma"
    mock_config.gamma_value = 0.5 # Brightening gamma
    update_active_lut_from_config()
    gamma_lut = get_current_z_lut()
    print(f"\nGamma LUT (gamma=0.5, first 10): {gamma_lut[:10]}")
    print(f"Gamma LUT (gamma=0.5, last 10): {gamma_lut[-10:]}")

    # Test S-curve LUT generation
    mock_config.lut_generation_type = "s_curve"
    mock_config.s_curve_contrast = 0.8
    update_active_lut_from_config()
    s_curve_lut = get_current_z_lut()
    print(f"\nS-Curve LUT (contrast=0.8, first 10): {s_curve_lut[:10]}")
    print(f"S-Curve LUT (contrast=0.8, last 10): {s_curve_lut[-10:]}")

    # Test saving and loading a LUT
    test_file = "test_generated_lut.json"
    try:
        save_lut(test_file, linear_lut)
        print(f"\nGenerated linear LUT saved to {test_file}")
        loaded_lut = load_lut(test_file)
        print(f"Loaded LUT from {test_file} (first 10): {loaded_lut[:10]}")
        if np.array_equal(linear_lut, loaded_lut):
            print("Loaded LUT matches saved LUT.")
        else:
            print("Loaded LUT DOES NOT match saved LUT.")
    except Exception as e:
        print(f"Error during LUT file operations: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up {test_file}")

    # Test applying LUT to a dummy image
    dummy_image = np.array([[0, 50, 100, 150, 200, 255],
                            [10, 60, 110, 160, 210, 240]], dtype=np.uint8)
    
    mock_config.lut_generation_type = "linear"
    mock_config.linear_min_input = 0
    mock_config.linear_max_output = 255
    update_active_lut_from_config() # Ensure linear LUT is active
    
    remapped_image = apply_z_lut(dummy_image)
    print("\nOriginal dummy image:\n", dummy_image)
    print("Remapped dummy image (linear 0-255):\n", remapped_image)

    mock_config.lut_generation_type = "gamma"
    mock_config.gamma_value = 2.2 # Darkening gamma
    update_active_lut_from_config() # Ensure gamma LUT is active
    
    remapped_image_gamma = apply_z_lut(dummy_image)
    print("\nRemapped dummy image (gamma=2.2):\n", remapped_image_gamma)
