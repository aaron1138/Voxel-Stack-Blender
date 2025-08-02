# lut_manager.py

"""
Manages the active Z-axis Look-Up Tables (LUTs) for Modular-Stacker.
Provides functions to generate LUTs based on various algorithms (linear, gamma, S-curve, etc.)
or load them from a file. The active LUTs are stored in a global dictionary.
"""

import numpy as np
import json
import os
from typing import Optional, Any, Dict

# Default Z Remapping LUT (linear 0-255)
_DEFAULT_Z_REMAP_LUT_ARRAY = np.arange(256, dtype=np.uint8)

# --- RESTRUCTURED: Manages multiple named LUTs ---
_active_luts: Dict[str, np.ndarray] = {
    "default": _DEFAULT_Z_REMAP_LUT_ARRAY.copy(),
    "receding": _DEFAULT_Z_REMAP_LUT_ARRAY.copy(),
    "overhang": _DEFAULT_Z_REMAP_LUT_ARRAY.copy()
}

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

def get_default_z_lut() -> np.ndarray:
    """Returns a copy of the default Z-remapping LUT (linear)."""
    return _DEFAULT_Z_REMAP_LUT_ARRAY.copy()

def get_current_z_lut(lut_name: str = "default") -> np.ndarray:
    """Returns the currently active Z-remapping LUT for the given name."""
    return _active_luts.get(lut_name, _DEFAULT_Z_REMAP_LUT_ARRAY).copy()

def set_current_z_lut(new_lut: np.ndarray, lut_name: str = "default"):
    """
    Sets the currently active Z-remapping LUT for the given name.
    Args:
        new_lut (np.ndarray): A 256-entry NumPy array of dtype uint8.
        lut_name (str): The name of the LUT to set ('default', 'receding', 'overhang').
    """
    if not isinstance(new_lut, np.ndarray) or new_lut.dtype != np.uint8 or new_lut.shape != (256,):
        raise ValueError("New LUT must be a 256-entry NumPy array of dtype uint8.")
    
    global _active_luts
    _active_luts[lut_name] = new_lut.copy()

def apply_z_lut(image_array: np.ndarray, lut_name: str = "default") -> np.ndarray:
    """
    Applies the specified active Z-REMAP_LUT to an 8-bit grayscale image.
    Args:
        image_array (np.ndarray): An 8-bit grayscale NumPy array (uint8).
        lut_name (str): The name of the LUT to apply.
    Returns:
        np.ndarray: A new NumPy array with the LUT applied.
    """
    if image_array.dtype != np.uint8:
        raise TypeError("Input image_array for apply_z_lut must be of type np.uint8.")
    
    active_lut = _active_luts.get(lut_name, _DEFAULT_Z_REMAP_LUT_ARRAY)
    return active_lut[image_array]

def save_lut(filepath: str, lut_array: np.ndarray):
    """Saves a LUT array to a JSON file."""
    if lut_array.dtype != np.uint8 or lut_array.shape != (256,):
        raise ValueError("LUT must be a 256-entry NumPy array of dtype uint8 to save.")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(lut_array.tolist(), f, indent=4)
    except IOError as e:
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
    except IOError as e:
        raise IOError(f"Failed to load LUT from '{filepath}': {e}")

# --- Algorithmic LUT Generation Functions ---

def generate_linear_lut(min_input: int, max_output: int) -> np.ndarray:
    """Generates a linear LUT that maps input range [0, 255] to [min_input, max_output]."""
    lut = np.linspace(min_input, max_output, 256)
    return np.clip(lut, 0, 255).astype(np.uint8)

def generate_gamma_lut(gamma_value: float) -> np.ndarray:
    """Generates a gamma correction LUT."""
    if gamma_value <= 0:
        raise ValueError("Gamma value must be positive.")
    inv_gamma = 1.0 / gamma_value
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)])
    return np.clip(lut, 0, 255).astype(np.uint8)

def generate_s_curve_lut(contrast: float) -> np.ndarray:
    """Generates an S-curve (contrast) LUT using a sigmoid function."""
    if not (0.0 <= contrast <= 1.0):
        raise ValueError("Contrast must be between 0.0 and 1.0.")
    
    steepness = 5 + (contrast * 10)
    x = np.linspace(-1, 1, 256)
    y = 1 / (1 + np.exp(-steepness * x))
    lut = y * 255
    return np.clip(lut, 0, 255).astype(np.uint8)

def generate_log_lut(param: float) -> np.ndarray:
    """Generates a logarithmic LUT."""
    if param <= 0:
        raise ValueError("Log parameter must be positive.")
    lut = np.array([np.log1p(i * param) / np.log1p(255 * param) * 255 for i in range(256)])
    return np.clip(lut, 0, 255).astype(np.uint8)

def generate_exp_lut(param: float) -> np.ndarray:
    """Generates an exponential LUT."""
    if param <= 0:
        raise ValueError("Exp parameter must be positive.")
    lut = np.array([((i / 255.0) ** param) * 255 for i in range(256)])
    return np.clip(lut, 0, 255).astype(np.uint8)

def generate_sqrt_lut(param: float) -> np.ndarray:
    """Generates a square root LUT."""
    lut = np.array([np.sqrt(i / 255.0) * 255 for i in range(256)])
    return np.clip(lut, 0, 255).astype(np.uint8)

def generate_rodbard_lut(param: float) -> np.ndarray:
    """Generates an ACES-style Rodbard contrast LUT."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.linspace(0, 1, 256)
    num = x * (a * x + b)
    den = x * (c * x + d) + e
    y = num / den
    lut = y * 255
    return np.clip(lut, 0, 255).astype(np.uint8)

# --- Main Update Function ---

def update_active_lut_from_config():
    """
    Updates the global active LUTs based on the current settings in the Config.
    Iterates through all named LUTs defined in config.lut_settings.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in lut_manager. Cannot update active LUTs.")
        return

    cfg = _config_ref

    for lut_name, lut_config in cfg.lut_settings.items():
        try:
            if lut_config.source == "file":
                if lut_config.fixed_path and os.path.exists(lut_config.fixed_path):
                    loaded_lut = load_lut(lut_config.fixed_path)
                    set_current_z_lut(loaded_lut, lut_name)
                else:
                    print(f"Warning for '{lut_name}' LUT: File path not found. Using default linear LUT.")
                    set_current_z_lut(get_default_z_lut(), lut_name)
            
            elif lut_config.source == "generated":
                gen_type = lut_config.generation_type
                if gen_type == "linear":
                    lut = generate_linear_lut(lut_config.linear_min_input, lut_config.linear_max_output)
                elif gen_type == "gamma":
                    lut = generate_gamma_lut(lut_config.gamma_value)
                elif gen_type == "s_curve":
                    lut = generate_s_curve_lut(lut_config.s_curve_contrast)
                elif gen_type == "log":
                    lut = generate_log_lut(lut_config.log_param)
                elif gen_type == "exp":
                    lut = generate_exp_lut(lut_config.exp_param)
                elif gen_type == "sqrt":
                    lut = generate_sqrt_lut(lut_config.sqrt_param)
                elif gen_type == "rodbard":
                    lut = generate_rodbard_lut(lut_config.rodbard_param)
                else:
                    print(f"Warning for '{lut_name}' LUT: Unknown generation type. Using default linear LUT.")
                    lut = get_default_z_lut()
                
                set_current_z_lut(lut, lut_name)
            
            else:
                print(f"Warning for '{lut_name}' LUT: Unknown source '{lut_config.source}'. Using default.")
                set_current_z_lut(get_default_z_lut(), lut_name)

        except Exception as e:
            print(f"Error updating '{lut_name}' LUT: {e}. Falling back to default linear LUT.")
            set_current_z_lut(get_default_z_lut(), lut_name)

# Initial setup
update_active_lut_from_config()

# Example usage (for testing purposes)
if __name__ == '__main__':
    print("--- LUT Manager Module Test ---")
    
    # Import necessary mock objects from the updated config structure
    from config import Config, LutConfig

    # --- Mock Config Setup ---
    class MockConfig(Config):
        def __init__(self):
            # Bypass the singleton's file loading for a clean test environment
            super().__init__() 
            self.lut_settings = {
                "default": LutConfig(source="generated", generation_type="linear", linear_min_input=50, linear_max_output=200),
                "receding": LutConfig(source="generated", generation_type="gamma", gamma_value=0.5),
                "overhang": LutConfig(source="file", fixed_path="test_overhang_lut.json")
            }

    mock_config = MockConfig()
    
    # Create a dummy LUT file for the 'overhang' test case
    test_file = "test_overhang_lut.json"
    dummy_overhang_lut = np.flip(np.arange(256, dtype=np.uint8)) # An inverted LUT
    save_lut(test_file, dummy_overhang_lut)
    print(f"Created dummy LUT file: {test_file}")

    # Set the config reference for the lut_manager
    set_config_reference(mock_config)

    # --- Test Execution ---
    print("\nUpdating active LUTs from mock config...")
    update_active_lut_from_config()

    # Verify each LUT was set correctly
    default_lut = get_current_z_lut("default")
    receding_lut = get_current_z_lut("receding")
    overhang_lut = get_current_z_lut("overhang")

    print(f"\n'default' LUT (linear 50-200) start: {default_lut[:5]}")
    assert default_lut[0] == 50, "Default LUT min value is incorrect"
    assert default_lut[255] == 200, "Default LUT max value is incorrect"

    print(f"'receding' LUT (gamma 0.5) start: {receding_lut[:5]}")
    expected_gamma_lut = generate_gamma_lut(0.5)
    assert np.array_equal(receding_lut, expected_gamma_lut), "Receding LUT is not a correct gamma 0.5 LUT"

    print(f"'overhang' LUT (loaded from file) start: {overhang_lut[:5]}")
    assert np.array_equal(overhang_lut, dummy_overhang_lut), "Overhang LUT did not load correctly from file"

    print("\n--- Testing LUT Application ---")
    dummy_image = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    print(f"Original image values: {dummy_image}")

    applied_default = apply_z_lut(dummy_image, "default")
    print(f"Applied 'default' LUT: {applied_default}")
    assert applied_default[0] == 50, "Default LUT application failed"

    applied_receding = apply_z_lut(dummy_image, "receding")
    print(f"Applied 'receding' LUT: {applied_receding}")
    assert applied_receding[1] > 64, "Receding (brightening gamma) LUT application failed"

    applied_overhang = apply_z_lut(dummy_image, "overhang")
    print(f"Applied 'overhang' LUT: {applied_overhang}")
    assert applied_overhang[0] == 255 and applied_overhang[-1] == 0, "Overhang (inverted) LUT application failed"

    # --- Cleanup ---
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\nCleaned up {test_file}")

    print("\n--- LUT Manager Test Complete ---")
