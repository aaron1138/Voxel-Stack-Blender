import os
import sys
import shutil
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from zarr_processing_pipeline import ZarrProcessingPipeline

@pytest.fixture
def setup_test_environment(tmpdir):
    """
    Set up a test environment with a config object and test images.
    """
    # Create a temporary output directory
    output_dir = tmpdir.mkdir("output")

    # Create a config object
    config = Config()
    config.input_folder = "tests/test_images"
    config.output_folder = str(output_dir)
    config.use_zarr = True
    config.blending_mode = "enhanced_edt"
    config.receding_layers = 2
    config.fixed_fade_distance_receding = 10
    config.use_numba_jit = True # Test with Numba

    yield config, output_dir

    # Teardown
    # No need to remove tmpdir, pytest handles it

def test_zarr_pipeline_runs_without_errors(setup_test_environment):
    """
    Test that the Zarr pipeline runs to completion without raising exceptions.
    """
    config, output_dir = setup_test_environment

    # Run the Zarr pipeline
    pipeline = ZarrProcessingPipeline(app_config=config, max_workers=1)
    pipeline.execute()

    # Check that output files were created
    output_files = os.listdir(str(output_dir))
    assert "0001.png" in output_files
    assert "0002.png" in output_files
    assert "0003.png" in output_files

def test_zarr_save_to_disk(setup_test_environment):
    """
    Test that the Zarr data store is saved to disk when the option is enabled.
    """
    config, output_dir = setup_test_environment
    config.save_zarr_to_disk = True

    # Run the Zarr pipeline
    pipeline = ZarrProcessingPipeline(app_config=config, max_workers=1)
    pipeline.execute()

    # Check that the Zarr datastore was saved
    zarr_datastore_path = os.path.join(str(output_dir), "zarr_datastore")
    assert os.path.isdir(zarr_datastore_path)
    assert os.path.exists(os.path.join(zarr_datastore_path, "input", "zarr.json"))
    assert os.path.exists(os.path.join(zarr_datastore_path, "output", "zarr.json"))
