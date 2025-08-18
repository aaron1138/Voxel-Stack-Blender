import sys
import os
import numpy as np
import cv2
import pytest

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, ProcessingMode
from processing_core import _calculate_weighted_receding_gradient_field

@pytest.fixture
def base_config():
    """Fixture to create a base config object for tests."""
    cfg = Config()
    cfg.blending_mode = ProcessingMode.WEIGHTED_STACK
    cfg.fixed_fade_distance_receding = 20.0  # Use a fixed distance for predictable gradients
    return cfg

def test_weighted_blending_logic_single_layer(base_config):
    """
    Tests the core gradient calculation for a single prior layer to isolate bugs.
    """
    # 1. Setup
    cfg = base_config
    cfg.manual_weights = [100]
    cfg.fade_distances_receding = [20.0]

    current_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(current_mask, (48, 48), (52, 52), 255, -1)

    prior_mask = np.zeros_like(current_mask)
    cv2.rectangle(prior_mask, (40, 40), (60, 60), 255, -1)

    prior_masks = [prior_mask]

    # 2. Execution
    debug_output_dir = "test_debug_output"
    os.makedirs(debug_output_dir, exist_ok=True)
    debug_info = {'output_folder': debug_output_dir, 'base_filename': 'single_layer_test'}
    gradient = _calculate_weighted_receding_gradient_field(current_mask, prior_masks, cfg, debug_info)

    # 3. Assertions
    # Point is 7 pixels away from the edge of the prior mask (at x=40)
    point_to_test = (41, 50)
    val = gradient[point_to_test]

    # Expected value:
    # dist = 7. fade_dist = 20. weight = 100. total_weight = 100.
    # accum = (1 - 7/20) * 100 = 65.
    # final_norm = 65 / 100 = 0.65.
    # final_val = 0.65 * 255 = 165.75
    print(f"Value at test point (should be ~165): {val}")
    assert 160 < val < 170, "Gradient for a single layer is incorrect"

def test_empty_inputs(base_config):
    """Tests that the function handles empty inputs gracefully."""
    current_mask = np.zeros((100, 100), dtype=np.uint8)

    # Test with no prior masks
    gradient_no_masks = _calculate_weighted_receding_gradient_field(current_mask, [], base_config)
    assert np.sum(gradient_no_masks) == 0

    # Test with no weights
    base_config.manual_weights = []
    prior_mask = np.ones_like(current_mask) * 255
    gradient_no_weights = _calculate_weighted_receding_gradient_field(current_mask, [prior_mask], base_config)
    assert np.sum(gradient_no_weights) == 0
