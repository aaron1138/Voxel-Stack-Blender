import sys
import os
import numpy as np
import cv2
import pytest

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, ProcessingMode
from processing_core import (
    _calculate_weighted_receding_gradient_field, find_prior_combined_white_mask,
    _calculate_receding_gradient_field_enhanced_edt_scipy,
    _calculate_receding_gradient_field_enhanced_edt_numba
)

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

def edt_setup(current_white_mask, prior_binary_masks):
    """Helper to perform the common setup for EDT tests."""
    prior_white_combined_mask = find_prior_combined_white_mask(prior_binary_masks)
    receding_white_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))
    distance_transform_src = cv2.bitwise_not(current_white_mask)
    distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)
    receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)
    num_labels, labels = cv2.connectedComponents(receding_white_areas)
    return receding_distance_map, labels, num_labels, receding_white_areas

@pytest.mark.parametrize("edt_function_name", [
    "scipy",
    "numba"
])
def test_enhanced_edt_max_fade_limit(base_config, edt_function_name):
    """
    Tests that the Enhanced EDT mode correctly uses the fade distance
    as a "Max Fade" limit on its adaptive normalization. This test runs
    for both the SciPy and Numba implementations.
    """
    # 1. Setup
    cfg = base_config
    fade_distance_limit = 10.0

    current_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(current_mask, (45, 45), (55, 55), 255, -1)

    prior_mask = np.zeros_like(current_mask)
    cv2.rectangle(prior_mask, (25, 40), (45, 60), 255, -1)
    cv2.rectangle(prior_mask, (55, 40), (60, 60), 255, -1)

    # 2. Execution
    receding_dist_map, labels, num_labels, mask = edt_setup(current_mask, [prior_mask])

    if edt_function_name == "scipy":
        gradient = _calculate_receding_gradient_field_enhanced_edt_scipy(
            receding_dist_map, labels, num_labels, fade_distance_limit
        )
    else: # numba
        gradient = _calculate_receding_gradient_field_enhanced_edt_numba(
            receding_dist_map, labels.astype(np.int32), num_labels, fade_distance_limit
        )

    gradient = cv2.bitwise_and(gradient, gradient, mask=mask)

    # 3. Assertions
    point_large_gap = (50, 35)
    val_large_gap = gradient[point_large_gap]
    assert val_large_gap == 0, f"Failed on {edt_function_name} implementation"

    point_small_gap = (50, 57)
    val_small_gap = gradient[point_small_gap]
    assert 180 < val_small_gap < 185, f"Failed on {edt_function_name} implementation"

    point_large_far = (50, 25)
    val_large_far = gradient[point_large_far]
    assert val_large_far == 0, f"Failed on {edt_function_name} implementation"

    point_small_far = (50, 59)
    val_small_far = gradient[point_small_far]
    assert 108 < val_small_far < 112, f"Failed on {edt_function_name} implementation"

def test_anisotropic_correction(base_config):
    """
    Tests that the Z-anisotropic correction correctly penalizes distances
    from further Z-layers in the Enhanced EDT mode.
    """
    # 1. Setup
    cfg = base_config
    cfg.blending_mode = ProcessingMode.ENHANCED_EDT
    cfg.edt_enable_anisotropic_correction = True
    cfg.anisotropic_voxel_dimensions.x_um = 10
    cfg.anisotropic_voxel_dimensions.y_um = 10
    cfg.anisotropic_voxel_dimensions.z_um = 100 # Z is 10x "taller" than X/Y
    cfg.fixed_fade_distance_receding = 1000.0 # Use a large fade distance to not interfere

    current_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(current_mask, (45, 45), (55, 55), 255, -1)

    # A prior mask that is 1 layer away
    prior_mask_z1 = np.zeros_like(current_mask)
    cv2.rectangle(prior_mask_z1, (40, 40), (60, 60), 255, -1)

    # 2. Execution
    # We need to call the full dispatcher function to test the new logic
    from processing_core import _calculate_receding_gradient_field_enhanced_edt

    # Case 1: No anisotropy
    cfg.edt_enable_anisotropic_correction = False
    gradient_no_anisotropy = _calculate_receding_gradient_field_enhanced_edt(current_mask, [prior_mask_z1], cfg)

    # Case 2: With Z-anisotropy enabled
    cfg.edt_enable_anisotropic_correction = True
    gradient_with_anisotropy = _calculate_receding_gradient_field_enhanced_edt(current_mask, [prior_mask_z1], cfg)

    # 3. Assertions
    test_point = (42, 50) # A point inside the receding area

    val_no_anisotropy = gradient_no_anisotropy[test_point]
    val_with_anisotropy = gradient_with_anisotropy[test_point]

    # The Z-distance penalty (z_dist * (z_um/x_um) = 1 * (100/10) = 10) is added to the XY distance.
    # This makes the effective distance much larger for the anisotropic case.
    # A larger distance results in a smaller (weaker) gradient value.
    assert val_with_anisotropy < val_no_anisotropy

    # And check that both have some gradient
    assert val_with_anisotropy > 0
    assert val_no_anisotropy > 0
