import os
import cv2
import numpy as np
import pytest
import shutil

from config import app_config, Config, ProcessingMode
from processing_pipeline import ProcessingPipelineThread

# Define a fixture to create a temporary directory for test images
@pytest.fixture(scope="module")
def image_stack_directory():
    """Creates a temporary directory with a stack of test images."""
    temp_dir = "test_image_stack"
    os.makedirs(temp_dir, exist_ok=True)
    # Create a series of simple images
    for i in range(10):
        img = np.zeros((100, 100), dtype=np.uint8)
        # Draw a white rectangle that moves across the screen
        cv2.rectangle(img, (i * 10, 40), (i * 10 + 20, 60), 255, -1)
        cv2.imwrite(os.path.join(temp_dir, f"test_{i}.png"), img)
    yield temp_dir
    # Clean up the directory after tests are done
    shutil.rmtree(temp_dir)

def run_processing_and_compare(config, image_stack_directory):
    """Helper function to run the processing and compare the output."""
    output_dir_sparse = "test_output_sparse"
    output_dir_sequential = "test_output_sequential"
    os.makedirs(output_dir_sparse, exist_ok=True)
    os.makedirs(output_dir_sequential, exist_ok=True)

    # Run with sparse mode enabled
    config.use_sparse_stack = True
    config.input_folder = image_stack_directory
    config.output_folder = output_dir_sparse
    sparse_thread = ProcessingPipelineThread(app_config=config, max_workers=2)
    sparse_thread.run()

    # Run with sparse mode disabled
    config.use_sparse_stack = False
    config.output_folder = output_dir_sequential
    sequential_thread = ProcessingPipelineThread(app_config=config, max_workers=2)
    sequential_thread.run()

    # Compare the output images
    sparse_files = sorted(os.listdir(output_dir_sparse))
    sequential_files = sorted(os.listdir(output_dir_sequential))
    assert sparse_files == sequential_files

    for sparse_file, sequential_file in zip(sparse_files, sequential_files):
        sparse_img = cv2.imread(os.path.join(output_dir_sparse, sparse_file), cv2.IMREAD_GRAYSCALE)
        sequential_img = cv2.imread(os.path.join(output_dir_sequential, sequential_file), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_allclose(sparse_img, sequential_img, atol=1)

    shutil.rmtree(output_dir_sparse)
    shutil.rmtree(output_dir_sequential)

def test_sparse_vs_sequential_enhanced_edt(image_stack_directory):
    """Tests if sparse mode produces the same output as sequential mode for Enhanced EDT."""
    config = Config()
    config.blending_mode = ProcessingMode.ENHANCED_EDT
    config.receding_layers = 4
    config.fixed_fade_distance_receding = 10.0
    run_processing_and_compare(config, image_stack_directory)

def test_sparse_vs_sequential_fixed_fade(image_stack_directory):
    """Tests if sparse mode produces the same output as sequential mode for Fixed Fade."""
    config = Config()
    config.blending_mode = ProcessingMode.FIXED_FADE
    config.receding_layers = 4
    config.fixed_fade_distance_receding = 10.0
    run_processing_and_compare(config, image_stack_directory)
