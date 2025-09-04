import os
import shutil
from config import Config, ProcessingMode
from processing_pipeline import ProcessingPipelineThread

def run_test(use_tiledb, mode, output_suffix):
    print(f"--- Running test with TileDB {'enabled' if use_tiledb else 'disabled'}, mode: {mode.value} ---")

    # --- Setup Config ---
    config = Config()
    config.input_folder = "test_images"
    config.output_folder = f"output_images_{output_suffix}"
    config.use_tiledb = use_tiledb
    config.blending_mode = mode
    config.receding_layers = 4
    config.fixed_fade_distance_receding = 10.0
    config.use_numba_jit = True # Use Numba for the default pipeline
    config.roi_params.min_size = 10 # Lower for test images

    if os.path.exists(config.output_folder):
        shutil.rmtree(config.output_folder)
    os.makedirs(config.output_folder)

    # --- Run Processing ---
    # We can't use the QThread directly without a QApplication,
    # so we'll call the run method directly for this test.
    thread = ProcessingPipelineThread(app_config=config, max_workers=1)

    # Mock the signals
    thread.status_update.connect(lambda msg: print(f"Status: {msg}"))
    thread.error_signal.connect(lambda msg: print(f"Error: {msg}"))
    thread.finished_signal.connect(lambda: print("Finished."))

    thread.run()

    print(f"--- Test finished ---")

if __name__ == "__main__":
    # Test ENHANCED_EDT mode
    run_test(use_tiledb=False, mode=ProcessingMode.ENHANCED_EDT, output_suffix="default_edt")
    run_test(use_tiledb=True, mode=ProcessingMode.ENHANCED_EDT, output_suffix="tiledb_edt")

    # Test ROI_FADE mode
    run_test(use_tiledb=False, mode=ProcessingMode.ROI_FADE, output_suffix="default_roi")
    run_test(use_tiledb=True, mode=ProcessingMode.ROI_FADE, output_suffix="tiledb_roi")

    print("\n--- Verification ---")
    print("Check the output directories. The edt outputs should be different from the roi outputs.")
    print("The default and tiledb outputs for each mode should be visually similar.")
