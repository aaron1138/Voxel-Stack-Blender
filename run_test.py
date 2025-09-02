import os
import sys
import time
import copy
import gc
import shutil

# Headless Qt Application for event processing
from PySide6.QtCore import QCoreApplication

from config import Config
from processing_pipeline import ProcessingPipelineThread

def run_processing(config):
    """Runs the processing pipeline and waits for it to complete."""
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    thread = ProcessingPipelineThread(app_config=config, max_workers=config.thread_count)

    thread.status_update.connect(lambda msg: print(f"STATUS: {msg}"))
    thread.progress_update.connect(lambda val: print(f"PROGRESS: {val}%"))
    thread.error_signal.connect(lambda err: print(f"ERROR: {err}"))

    finished = False
    def on_finish():
        nonlocal finished
        print("Processing finished signal received.")
        finished = True

    thread.finished_signal.connect(on_finish)

    print(f"--- Starting processing (Sparse mode: {config.load_sparse}, Workers: {config.thread_count}) ---")
    thread.start()

    start_time = time.time()
    while not finished:
        if time.time() - start_time > 300: # 5 minute timeout
             print("ERROR: Test run timed out.")
             thread.stop_processing()
             break
        # Process Qt events to allow signals to be received
        QCoreApplication.processEvents()
        time.sleep(0.1) # Small sleep to prevent busy-waiting

    thread.wait()
    print("--- Processing complete ---")

def setup_test_environment(test_dir, source_image):
    """Creates a test directory with a small set of images."""
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    shutil.copy(source_image, os.path.join(test_dir, "test_000.png"))
    shutil.copy(source_image, os.path.join(test_dir, "test_001.png"))
    print(f"Test environment created at '{test_dir}'")

def cleanup_test_environment(*dirs):
    """Removes test directories."""
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Cleaned up '{d}'")

if __name__ == "__main__":
    # A QCoreApplication is needed for the QThread's signals to work
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv)

    TEST_IMAGE_DIR = "temp_test_images"
    NORMAL_OUTPUT_DIR = "output_normal"
    SPARSE_OUTPUT_DIR = "output_sparse"
    SOURCE_IMAGE = "images/comparison-1920.jpg"

    if not os.path.exists(SOURCE_IMAGE):
        print(f"ERROR: Source image not found at {SOURCE_IMAGE}")
        sys.exit(1)

    cleanup_test_environment(TEST_IMAGE_DIR, NORMAL_OUTPUT_DIR, SPARSE_OUTPUT_DIR)
    setup_test_environment(TEST_IMAGE_DIR, SOURCE_IMAGE)

    base_config = Config()
    base_config.input_mode = "folder"
    base_config.input_folder = TEST_IMAGE_DIR
    base_config.receding_layers = 1
    base_config.use_numba_jit = True
    base_config.blending_mode = "enhanced_edt"
    base_config.thread_count = 2

    exit_code = 0
    try:
        print("\nPreparing for NORMAL mode run...")
        normal_config = copy.deepcopy(base_config)
        normal_config.output_folder = NORMAL_OUTPUT_DIR
        normal_config.load_sparse = False
        run_processing(normal_config)

        gc.collect()

        print("\nPreparing for SPARSE mode run...")
        sparse_config = copy.deepcopy(base_config)
        sparse_config.output_folder = SPARSE_OUTPUT_DIR
        sparse_config.load_sparse = True
        run_processing(sparse_config)

        print("\n--- Comparing outputs ---")
        normal_files = sorted(os.listdir(NORMAL_OUTPUT_DIR))
        sparse_files = sorted(os.listdir(SPARSE_OUTPUT_DIR))

        if normal_files != sparse_files:
            raise RuntimeError(f"File listings differ: {normal_files} vs {sparse_files}")

        import numpy as np
        import cv2
        all_match = True
        for filename in normal_files:
            img_normal = cv2.imread(os.path.join(NORMAL_OUTPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
            img_sparse = cv2.imread(os.path.join(SPARSE_OUTPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)

            if img_normal is None or img_sparse is None:
                raise RuntimeError(f"Could not read image {filename}")

            if not np.array_equal(img_normal, img_sparse):
                print(f"ERROR: Images for {filename} do not match!")
                all_match = False
            else:
                print(f"OK: {filename} matches.")

        if not all_match:
            raise RuntimeError("Image content did not match.")

        print("\nSUCCESS: All output images match.")

    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        cleanup_test_environment(TEST_IMAGE_DIR, NORMAL_OUTPUT_DIR, SPARSE_OUTPUT_DIR)
        # We must call exit on the app for a clean shutdown
        app.exit(exit_code)
        sys.exit(exit_code)
