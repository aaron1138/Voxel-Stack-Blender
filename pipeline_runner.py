"""
Orchestrates the Modular-Stacker image processing pipeline.
Manages image loading, buffering, multi-threaded processing,
and saving of output images.
"""

import os
import cv2
import threading
import time
import numpy as np
import math
from typing import List, Optional, Tuple, Any
from collections import deque

import config
import image_loader
import stacking_processor
import run_logger

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance
    # Also set config reference for imported modules
    image_loader.set_config_reference(config_instance)
    stacking_processor.set_config_reference(config_instance)
    run_logger.set_config_reference(config_instance)

class PipelineRunner:
    """
    Manages the end-to-end image stacking and processing pipeline.
    """
    def __init__(self):
        if _config_ref is None:
            raise RuntimeError("Config reference not set in PipelineRunner. Aborting.")
        
        self.config = _config_ref
        self.image_loader = image_loader.ImageLoader(self.config.input_dir, self.config.file_pattern)
        self.total_input_images = self.image_loader.get_total_images()
        
        self.input_start_idx = max(0, self.config.resume_from - 1)
        self.input_end_idx = min(self.total_input_images - 1, self.config.stop_at - 1)
        
        self.relevant_input_count = (self.input_end_idx - self.input_start_idx + 1)
        if self.relevant_input_count <= 0:
            self.total_output_stacks = 0
            self.output_stack_indices = []
        else:
            is_vb_substitute = not self.config.vertical_blend_pre_process and self.config.blend_mode.startswith("vertical_")
            primary = 1 if is_vb_substitute else self.config.primary
            self.total_output_stacks = math.ceil(self.relevant_input_count / primary)
            self.output_stack_indices = list(range(self.total_output_stacks))
        
        # --- DYNAMICALLY CALCULATE REQUIRED RADIUS ---
        vb_radius = 0
        is_vb_active = self.config.vertical_blend_pre_process or self.config.blend_mode.startswith("vertical_")
        if is_vb_active:
            vb_radius = max(self.config.vertical_receding_layers, self.config.vertical_overhang_layers)
        
        stacker_radius = self.config.radius
        
        if not is_vb_active:
             self.effective_radius = stacker_radius
        elif self.config.vertical_blend_pre_process:
             self.effective_radius = vb_radius + stacker_radius
        else: # VB substitute mode
             self.effective_radius = vb_radius

        primary_for_buffer = 1 if self.config.blend_mode.startswith("vertical_") else self.config.primary
        max_window_size = primary_for_buffer + (2 * self.effective_radius)
        min_buffer_capacity = max_window_size * self.config.threads * 2
        self.image_buffer = image_loader.ImageBuffer(capacity=max(min_buffer_capacity, 50))
        
        self._loader_thread: Optional[image_loader.ImageLoaderThread] = None
        self._processing_threads: List[threading.Thread] = []
        self._output_queue: deque[Tuple[int, np.ndarray]] = deque()
        self._output_queue_lock = threading.Lock()
        self._output_queue_event = threading.Event()
        self._output_writer_thread: Optional[threading.Thread] = None
        
        self._current_processed_count = 0
        self._stop_event = threading.Event()

    def _get_image_window_for_stack(self, output_stack_index: int) -> List[np.ndarray]:
        """
        Retrieves the list of image data for a given output stacking window,
        using the dynamically calculated effective_radius.
        """
        image_data_window: List[np.ndarray] = []
        is_vb_substitute = not self.config.vertical_blend_pre_process and self.config.blend_mode.startswith("vertical_")
        primary = 1 if is_vb_substitute else self.config.primary

        base_input_index_for_output = self.input_start_idx + (output_stack_index * primary)
        
        window_start_input_idx = base_input_index_for_output - self.effective_radius
        window_end_input_idx = base_input_index_for_output + primary - 1 + self.effective_radius

        for slice_idx in range(window_start_input_idx, window_end_input_idx + 1):
            if self._stop_event.is_set():
                return []

            if slice_idx < 0:
                # Original logic: Reuse the first actual input image for bottom padding
                first_relevant_image_path = self.image_loader.get_image_path(self.input_start_idx)
                if first_relevant_image_path is not None:
                    img = self.image_loader.load_single_image(first_relevant_image_path)
                else: # Fallback to blank
                    img = self.image_loader.load_single_image(None)
                image_data_window.append(img)
            elif slice_idx >= self.total_input_images:
                # Use blank images for top padding
                img = self.image_loader.load_single_image(None)
                image_data_window.append(img)
            else:
                try:
                    img = self.image_buffer.get(slice_idx, timeout=30)
                    image_data_window.append(img)
                except TimeoutError:
                    print(f"Error: Timeout fetching image {slice_idx} from buffer. Aborting.")
                    self._stop_event.set()
                    return []
        
        return image_data_window

    def _processing_worker(self, thread_id: int):
        """Worker function for each processing thread."""
        while not self._stop_event.is_set():
            try:
                with self._output_queue_lock:
                    if not self.output_stack_indices:
                        break
                    current_output_index = self.output_stack_indices.pop(0)
                
                image_window_data = self._get_image_window_for_stack(current_output_index)
                if not image_window_data:
                    break

                processed_image = stacking_processor.process_image_stack(image_window_data)
                
                with self._output_queue_lock:
                    self._output_queue.append((current_output_index, processed_image))
                    self._output_queue_event.set()
                    self._current_processed_count += 1
                    if self.config.progress_callback:
                        self.config.progress_callback(self._current_processed_count, self.total_output_stacks)

            except IndexError:
                break
            except Exception as e:
                print(f"Worker {thread_id}: Error processing stack {current_output_index}: {e}")
                self._stop_event.set()
                break

    def _output_writer(self):
        """Dedicated thread for writing processed images to disk."""
        while not (self._stop_event.is_set() and len(self._output_queue) == 0):
            self._output_queue_event.wait(0.5)
            try:
                with self._output_queue_lock:
                    if not self._output_queue:
                        if all(not t.is_alive() for t in self._processing_threads):
                            break
                        continue
                    
                    output_index, image_data = self._output_queue.popleft()
                
                output_filename = self._generate_output_filename(output_index)
                output_path = os.path.join(self.config.output_dir, output_filename)
                
                os.makedirs(self.config.output_dir, exist_ok=True)
                cv2.imwrite(output_path, image_data)
            
            except IndexError:
                continue
            except Exception as e:
                print(f"OutputWriter: Error saving image: {e}")
                self._stop_event.set()
                break

    def _generate_output_filename(self, index: int) -> str:
        """Generates the output filename with padding if configured."""
        if self.config.pad_filenames:
            return f"stack_{index:0{self.config.pad_length}d}.png"
        return f"stack_{index}.png"

    def run_pipeline(self):
        """Starts the entire processing pipeline."""
        if self.total_output_stacks == 0:
            if self.config.progress_callback:
                self.config.progress_callback(0, 0)
            return

        self._stop_event.clear()
        self._current_processed_count = 0
        self.image_buffer.clear()
        self._output_queue.clear()

        run_logger.log_run(run_logger.get_last_run_index() + 1, self.config.to_dict())

        is_vb_substitute = not self.config.vertical_blend_pre_process and self.config.blend_mode.startswith("vertical_")
        primary = 1 if is_vb_substitute else self.config.primary
        loader_start_idx = max(0, self.input_start_idx - self.effective_radius)
        loader_end_idx = min(self.total_input_images - 1, self.input_end_idx + primary - 1 + self.effective_radius)

        self._loader_thread = image_loader.ImageLoaderThread(
            image_loader=self.image_loader,
            image_buffer=self.image_buffer,
            total_images=self.total_input_images,
            start_index=loader_start_idx,
            end_index=loader_end_idx
        )
        self._loader_thread.start()

        self._output_writer_thread = threading.Thread(target=self._output_writer, name="OutputWriterThread")
        self._output_writer_thread.start()

        self._processing_threads = []
        for i in range(self.config.threads):
            thread = threading.Thread(target=self._processing_worker, args=(i,), name=f"Worker-{i}")
            self._processing_threads.append(thread)
            thread.start()

        for thread in self._processing_threads:
            thread.join()

        self._loader_thread.stop()
        self._loader_thread.join()
        
        self._output_queue_event.set()
        if self._output_writer_thread:
            self._output_writer_thread.join()

        if self.config.progress_callback and not self._stop_event.is_set():
            self.config.progress_callback(self.total_output_stacks, self.total_output_stacks)

    def stop_pipeline(self):
        """Signals all threads to stop processing."""
        self._stop_event.set()
        if self._loader_thread:
            self._loader_thread.stop()
        self.image_buffer.clear()
        self._output_queue_event.set()

if __name__ == '__main__':
    print("--- Pipeline Runner Module Test ---")

    from config import Config, LutConfig, XYBlendOperation

    class MockConfig(Config):
        def __init__(self):
            super().__init__()
            self.run_log_file = "test_pipeline_runs.log"
            self.progress_updates = []
            def mock_progress_callback(current, total):
                self.progress_updates.append((current, total))
                print(f"Progress: {current}/{total}")
            self.progress_callback = mock_progress_callback

    # --- Setup Test Environment ---
    test_input_dir = "test_input_images"
    test_output_dir = "test_output_stacks"
    os.makedirs(test_input_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    num_dummy_images = 20
    for i in range(1, num_dummy_images + 1):
        dummy_img = np.full((50, 50), i*10, dtype=np.uint8)
        cv2.imwrite(os.path.join(test_input_dir, f"image_{i:03d}.png"), dummy_img)

    # --- Test Case 1: Standard Gaussian Blend ---
    print("\n--- Test Case 1: Standard Gaussian Blend ---")
    mock_config_std = MockConfig()
    mock_config_std.input_dir = test_input_dir
    mock_config_std.output_dir = test_output_dir
    mock_config_std.file_pattern = "image_*.png"
    mock_config_std.blend_mode = "gaussian"
    mock_config_std.primary = 5
    mock_config_std.radius = 2
    
    set_config_reference(mock_config_std)
    runner_std = PipelineRunner()
    print(f"Standard Mode: Effective Radius = {runner_std.effective_radius}, Total Stacks = {runner_std.total_output_stacks}")
    runner_std.run_pipeline()
    assert runner_std.effective_radius == 2, "Standard radius calculation failed"
    assert runner_std.total_output_stacks == 4, "Standard stack count failed"

    # --- Test Case 2: Vertical Blend Substitute Mode ---
    print("\n--- Test Case 2: Vertical Blend Substitute Mode ---")
    mock_config_vb = MockConfig()
    mock_config_vb.input_dir = test_input_dir
    mock_config_vb.output_dir = test_output_dir
    mock_config_vb.file_pattern = "image_*.png"
    mock_config_vb.blend_mode = "vertical_combined" # Substitute mode
    mock_config_vb.vertical_blend_pre_process = False
    mock_config_vb.vertical_receding_layers = 4
    mock_config_vb.vertical_overhang_layers = 3
    
    set_config_reference(mock_config_vb)
    runner_vb = PipelineRunner()
    print(f"VB Substitute Mode: Effective Radius = {runner_vb.effective_radius}, Total Stacks = {runner_vb.total_output_stacks}")
    # Not running the pipeline to save time, just verifying calculations
    assert runner_vb.effective_radius == 4, "VB Substitute radius calculation failed"
    assert runner_vb.total_output_stacks == 20, "VB Substitute stack count failed"

    # --- Test Case 3: Vertical Blend Pre-processor Mode ---
    print("\n--- Test Case 3: Vertical Blend Pre-processor Mode ---")
    mock_config_pre = MockConfig()
    mock_config_pre.input_dir = test_input_dir
    mock_config_pre.output_dir = test_output_dir
    mock_config_pre.file_pattern = "image_*.png"
    mock_config_pre.blend_mode = "gaussian" # Stacking mode after pre-processing
    mock_config_pre.vertical_blend_pre_process = True
    mock_config_pre.primary = 5
    mock_config_pre.radius = 2
    mock_config_pre.vertical_receding_layers = 4
    mock_config_pre.vertical_overhang_layers = 3

    set_config_reference(mock_config_pre)
    runner_pre = PipelineRunner()
    print(f"VB Pre-process Mode: Effective Radius = {runner_pre.effective_radius}, Total Stacks = {runner_pre.total_output_stacks}")
    assert runner_pre.effective_radius == 6, "VB Pre-process radius calculation failed" # 4 (vb) + 2 (stacker)
    assert runner_pre.total_output_stacks == 4, "VB Pre-process stack count failed"

    # --- Cleanup ---
    print("\nCleaning up test directories...")
    for f in os.listdir(test_input_dir):
        os.remove(os.path.join(test_input_dir, f))
    os.rmdir(test_input_dir)
    for f in os.listdir(test_output_dir):
        os.remove(os.path.join(test_output_dir, f))
    os.rmdir(test_output_dir)
    if os.path.exists(mock_config_std.run_log_file):
        os.remove(mock_config_std.run_log_file)
    print("Cleanup complete.")
