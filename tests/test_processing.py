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

def test_weighted_blending_logic(base_config):
    """
    Tests the core logic of the weighted stack blending.
    """
    # 1. Setup
    cfg = base_config
    cfg.manual_weights = [100, 50]  # Higher weight for the first prior mask

    # Create image masks
    # current_mask is a 10x10 square in the middle
    current_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(current_mask, (45, 45), (55, 55), 255, -1)

    # prior_mask_1 (weight 100) is a 20x20 square
    prior_mask_1 = np.zeros_like(current_mask)
    cv2.rectangle(prior_mask_1, (40, 40), (60, 60), 255, -1)

    # prior_mask_2 (weight 50) is a 30x30 square
    prior_mask_2 = np.zeros_like(current_mask)
    cv2.rectangle(prior_mask_2, (35, 35), (65, 65), 255, -1)

    prior_masks = [prior_mask_1, prior_mask_2]

    # 2. Execution
    gradient = _calculate_weighted_receding_gradient_field(current_mask, prior_masks, cfg)

    # 3. Assertions
    assert gradient is not None
    assert gradient.dtype == np.uint8
    assert cv2.countNonZero(gradient) > 0, "Gradient should not be empty"

    # Define points to test
    # Point A is in the receding area of ONLY prior_mask_1 (and 2), but closer to the edge
    # This should be influenced by both weights, but dominated by the higher weight.
    # It's in the area of mask 2, but also mask 1.
    point_a = (42, 50)

    # Point B is in the receding area of ONLY prior_mask_2.
    # It should have a lower value than a point at a similar distance influenced by mask 1.
    point_b = (37, 50)

    # Point C is inside the current mask, should be black (0)
    point_c = (50, 50)

    val_a = gradient[point_a]
    val_b = gradient[point_b]
    val_c = gradient[point_c]

    print(f"Pixel values: A={val_a}, B={val_b}, C={val_c}")

    # Based on manual calculation with the corrected logic:
    # val_a should be approx (1 - (3/20)*100 + (1-(3/20))*50)/150 * 255 = 216
    # val_b should be approx (1 - (8/20)*100 + (1-(8/20))*50)/150 * 255 = 153 <- My test point was wrong, it is in both zones.
    # Let's pick a new point B that is ONLY in the second zone.
    point_b_new = (32, 50) # x=32 is outside mask 1 (edge at 40) but inside mask 2 (edge at 35)
    val_b_new = gradient[point_b_new]
    print(f"New B value: {val_b_new}")

    # For point_b_new, dist is 13. accum = (1-13/20)*50 = 17.5. norm = 17.5/150=0.116. final = 0.116*255=29

    assert val_a > val_b_new, "Point influenced by two weights should be brighter than a point influenced by one, weaker weight"
    assert val_c == 0, "Point inside the current mask should have zero gradient"
    # Let's assert the general relationship, which is more robust than exact values.
    assert val_a > 150 # Should be bright
    assert val_b_new < 100 # Should be dim

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
