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
    Tests that the anisotropic correction correctly scales the distance field.
    """
    # 1. Setup
    cfg = base_config
    cfg.anisotropic_params.enabled = True
    cfg.anisotropic_params.x_factor = 2.0  # Stretch distances twice as much on X-axis
    cfg.anisotropic_params.y_factor = 1.0  # Keep Y-axis normal
    cfg.fixed_fade_distance_receding = 50.0 # Use a large fade distance to not interfere

    # Create a central black square on a white background
    # The distance transform will be calculated from the edges of this square
    current_mask = np.ones((100, 100), dtype=np.uint8) * 255
    cv2.rectangle(current_mask, (45, 45), (54, 54), 0, -1)

    # The entire area is a receding area for this test
    prior_mask = np.ones_like(current_mask) * 255

    # 2. Execution
    # We need to call the full dispatcher function to test the resize logic
    from processing_core import _calculate_receding_gradient_field_enhanced_edt
    gradient = _calculate_receding_gradient_field_enhanced_edt(cv2.bitwise_not(current_mask), [prior_mask], cfg)

    # 3. Assertions
    # Point 10 pixels to the left of the black box
    point_x_dir = (50, 35)
    # Point 10 pixels above the black box
    point_y_dir = (35, 50)

    val_x = gradient[point_x_dir]
    val_y = gradient[point_y_dir]

    # Because the X-axis distance is stretched by 2x, the gradient value
    # at the same physical distance should be STRONGER (closer to white/255)
    # than the Y-axis value, because its "perceived" distance is smaller.
    # E.g., a physical distance of 10 on X is now ~20 in the stretched space.
    # A larger distance means a smaller (weaker) gradient value.
    assert val_x < val_y

    # Check that some gradient exists
    assert val_x > 0
    assert val_y > 0

def test_sparse_vs_sliding_window_equivalence(tmp_path, base_config):
    """
    Tests that the output of the sparse array mode is identical to the
    sliding window mode for a simple case.
    """
    # 1. Setup: Create dummy images and config
    input_dir = tmp_path / "input"
    output_sparse_dir = tmp_path / "output_sparse"
    output_sliding_dir = tmp_path / "output_sliding"
    input_dir.mkdir()
    output_sparse_dir.mkdir()
    output_sliding_dir.mkdir()

    # Create 5 simple images
    image_paths = []
    for i in range(5):
        img = np.zeros((50, 50), dtype=np.uint8)
        # Make a white square that moves
        cv2.rectangle(img, (10 + i, 10), (20 + i, 20), 255, -1)
        filename = f"test_{i}.png"
        filepath = input_dir / filename
        cv2.imwrite(str(filepath), img)
        image_paths.append(str(filepath))

    # Import the pipeline here to avoid circular import issues at top level
    from processing_pipeline import ProcessingPipelineThread

    # 2. Run Sparse Mode
    cfg_sparse = base_config
    cfg_sparse.input_folder = str(input_dir)
    cfg_sparse.output_folder = str(output_sparse_dir)
    cfg_sparse.use_sparse_array = True
    cfg_sparse.blending_mode = ProcessingMode.ENHANCED_EDT
    cfg_sparse.receding_layers = 3
    cfg_sparse.use_numba_jit = True # Test with Numba on

    pipeline_sparse = ProcessingPipelineThread(app_config=cfg_sparse, max_workers=1)
    pipeline_sparse.run() # Run in the same thread for testing simplicity

    # 3. Run Sliding Window Mode
    cfg_sliding = base_config
    cfg_sliding.input_folder = str(input_dir)
    cfg_sliding.output_folder = str(output_sliding_dir)
    cfg_sliding.use_sparse_array = False # The only difference
    cfg_sliding.blending_mode = ProcessingMode.ENHANCED_EDT
    cfg_sliding.receding_layers = 3
    cfg_sliding.use_numba_jit = True

    pipeline_sliding = ProcessingPipelineThread(app_config=cfg_sliding, max_workers=1)
    pipeline_sliding.run()

    # 4. Compare results
    sparse_files = sorted(os.listdir(output_sparse_dir))
    sliding_files = sorted(os.listdir(output_sliding_dir))

    assert len(sparse_files) > 0, "Sparse mode produced no output files."
    assert sparse_files == sliding_files, "Output filenames do not match between modes."

    for filename in sparse_files:
        img_sparse = cv2.imread(str(output_sparse_dir / filename), cv2.IMREAD_GRAYSCALE)
        img_sliding = cv2.imread(str(output_sliding_dir / filename), cv2.IMREAD_GRAYSCALE)

        assert img_sparse is not None
        assert img_sliding is not None

        diff = cv2.absdiff(img_sparse, img_sliding)
        assert np.sum(diff) == 0, f"Mismatch found in file {filename}"
