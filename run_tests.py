import os
import shutil
from config import Config, ProcessingMode
from processing_pipeline import ProcessingPipelineThread

def run_test(use_tiledb):
    print(f"--- Running test with TileDB {'enabled' if use_tiledb else 'disabled'} ---")

    # --- Setup Config ---
    config = Config()
    config.input_folder = "test_images"
    config.output_folder = f"output_images_{'tiledb' if use_tiledb else 'default'}"
    config.use_tiledb = use_tiledb
    config.blending_mode = ProcessingMode.ENHANCED_EDT
    config.receding_layers = 4
    config.fixed_fade_distance_receding = 10.0
    config.use_numba_jit = True # Use Numba for the default pipeline

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
    # Run with default pipeline
    run_test(use_tiledb=False)

    # Run with TileDB pipeline
    run_test(use_tiledb=True)

    print("\n--- Verification ---")
    print("Check the 'output_images_default' and 'output_images_tiledb' directories.")
    print("The output images should be visually similar.")
