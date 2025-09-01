import os
import shutil
import cv2
import numpy as np
import pytest

from config import Config, ProcessingMode
from processing_pipeline import ProcessingPipelineThread

@pytest.fixture
def image_stack(tmp_path):
    """Creates a temporary directory with a few sample images."""
    stack_dir = tmp_path / "image_stack"
    os.makedirs(stack_dir)
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)

    # Create 3 simple images
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img1[2:5, 2:5] = 255
    cv2.imwrite(str(stack_dir / "img_0.png"), img1)

    img2 = np.zeros((10, 10), dtype=np.uint8)
    img2[3:6, 3:6] = 255
    cv2.imwrite(str(stack_dir / "img_1.png"), img2)

    img3 = np.zeros((10, 10), dtype=np.uint8)
    img3[4:7, 4:7] = 255
    cv2.imwrite(str(stack_dir / "img_2.png"), img3)

    return str(stack_dir), str(output_dir)

def test_sparse_array_pipeline(image_stack):
    """Tests the processing pipeline with the sparse array option enabled."""
    input_dir, output_dir = image_stack

    config = Config()
    config.input_folder = input_dir
    config.output_folder = output_dir
    config.use_sparse_array = True
    config.blending_mode = ProcessingMode.ENHANCED_EDT
    config.receding_layers = 1
    config.fixed_fade_distance_receding = 2.0
    config.use_numba_jit = False # Use scipy for predictability in test
    config.anisotropic_params.enabled = False # Disable for this test

    # Run the pipeline synchronously for testing purposes
    pipeline = ProcessingPipelineThread(app_config=config, max_workers=1)
    pipeline.run() # This will block until finished

    # 1. Check if output files were created
    output_files = sorted(os.listdir(output_dir))
    assert len(output_files) == 3
    assert "img_0.png" in output_files
    assert "img_1.png" in output_files
    assert "img_2.png" in output_files

    # 2. Check the content of a processed image
    # We'll check img_1, which should be affected by img_0
    processed_img = cv2.imread(os.path.join(output_dir, "img_1.png"), cv2.IMREAD_GRAYSCALE)
    assert processed_img is not None

    # The area from (2,2) to (2,4) in img_1 should have a gradient
    # because it's a receding area from img_0. The pixel at (2,3) is not the
    # furthest point in the receding area, so it should have a non-zero gradient.
    gradient_pixel = processed_img[2, 3]
    assert 0 < gradient_pixel < 255

    # The area that is white in img_1 should still be white
    assert processed_img[4, 4] == 255
