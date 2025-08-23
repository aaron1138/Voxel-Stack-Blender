import sys
import os
import numpy as np
import cv2
import pytest

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, ProcessingMode
from processing_core import _calculate_weighted_receding_gradient_field, _calculate_receding_gradient_field_orthogonal_1d
import gradient_table_manager

@pytest.fixture
def base_config():
    """Fixture to create a base config object for tests."""
    cfg = Config()
    cfg.blending_mode = ProcessingMode.WEIGHTED_STACK
    cfg.fixed_fade_distance_receding = 20.0  # Use a fixed distance for predictable gradients
    return cfg

def test_weighted_blending_logic(base_config):
    """
    Tests the final, corrected logic for weighted blending.
    """
    # 1. Setup
    cfg = base_config
    # Newest to oldest
    cfg.manual_weights = [100, 50]
    cfg.fade_distances_receding = [20.0, 10.0]

    current_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(current_mask, (45, 45), (55, 55), 255, -1)

    # Oldest mask (will be at index 1 after reversal)
    prior_mask_oldest = np.zeros_like(current_mask)
    cv2.rectangle(prior_mask_oldest, (35, 35), (65, 65), 255, -1)

    # Newest mask (will be at index 0 after reversal)
    prior_mask_newest = np.zeros_like(current_mask)
    cv2.rectangle(prior_mask_newest, (40, 40), (60, 60), 255, -1)

    # This is the order they appear in the cache (oldest to newest)
    prior_masks_in_cache = [prior_mask_oldest, prior_mask_newest]
    # The pipeline will reverse it to [prior_mask_newest, prior_mask_oldest]
    prior_masks_for_func = list(reversed(prior_masks_in_cache))

    # 2. Execution
    gradient = _calculate_weighted_receding_gradient_field(current_mask, prior_masks_for_func, cfg)

    # 3. Assertions
    # This point is in the receding area of BOTH masks
    point_a = (42, 50)
    # This point is ONLY in the receding area of the OLDEST mask
    point_b = (38, 50)

    val_a = gradient[point_a]
    val_b = gradient[point_b]

    # Manual trace with final correct logic:
    # point_a (dist 3 from current):
    #   - contrib from newest (w:100, d:20): (1 - 3/20)*100 = 85
    #   - contrib from oldest (w:50, d:10): (1 - 3/10)*50 = 35
    #   - total=120. norm=120/150=0.8. val=0.8*255=204
    # point_b (dist 7 from current):
    #   - contrib from newest (w:100, d:20): not in this receding area -> 0
    #   - contrib from oldest (w:50, d:10): (1 - 7/10)*50 = 15
    #   - total=15. norm=15/150=0.1. val=0.1*255=25

    print(f"Values: A={val_a} (~204), B={val_b} (~25)")

    assert 200 < val_a < 210
    assert 20 < val_b < 30

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

def test_orthogonal_1d_gradient(base_config):
    """
    Tests the orthogonal 1D gradient logic with a simple case.
    """
    # 1. Setup
    cfg = base_config
    cfg.blending_mode = ProcessingMode.ORTHOGONAL_1D_GRADIENT

    # Create a 10px wide receding border around a central square
    current_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(current_mask, (40, 40), (60, 60), 255, -1)

    prior_mask = np.zeros_like(current_mask)
    cv2.rectangle(prior_mask, (30, 30), (70, 70), 255, -1)

    prior_masks_for_func = [prior_mask]

    # 2. Execution
    gradient = _calculate_receding_gradient_field_orthogonal_1d(current_mask, prior_masks_for_func, cfg)

    # 3. Assertions
    # Get the expected gradient ramp for a 10-pixel distance
    grad_table = gradient_table_manager.generate_linear_table()
    ramp_10px = grad_table[10]

    # Point on the left receding edge, right next to the source feature
    # Expected value should be the highest in the ramp
    point_left_near = (50, 39)
    val_left_near = gradient[point_left_near]
    assert val_left_near == ramp_10px[0]

    # Point on the left receding edge, furthest from the source feature
    # Expected value should be the lowest in the ramp
    point_left_far = (50, 30)
    val_left_far = gradient[point_left_far]
    assert val_left_far == ramp_10px[9]

    # Point on the top receding edge, right next to the source feature
    point_top_near = (39, 50)
    val_top_near = gradient[point_top_near]
    assert val_top_near == ramp_10px[0]

    # Point on the top receding edge, furthest from the source feature
    point_top_far = (30, 50)
    val_top_far = gradient[point_top_far]
    assert val_top_far == ramp_10px[9]

    # Point in a corner. Should be the max of the horizontal and vertical gradients.
    # At (39, 39), the horizontal and vertical ramps both want to place ramp_10px[0].
    point_corner = (39, 39)
    val_corner = gradient[point_corner]
    assert val_corner == ramp_10px[0]

    # Point inside the original shape should be zero
    point_inside = (50, 50)
    val_inside = gradient[point_inside]
    assert val_inside == 0

    # Point outside all masks should be zero
    point_outside = (10, 10)
    val_outside = gradient[point_outside]
    assert val_outside == 0
