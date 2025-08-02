# pipeline_runner.py

"""
Orchestrates the Modular-Stacker image processing pipeline.
Manages image loading, buffering, multi-threaded processing,
and saving of output images.
"""

import os
import cv2
import threading
import time
import numpy as np # Import numpy here
import math # Import math for ceil
from typing import List, Optional, Tuple, Any
from collections import deque # Import deque here

import config
import image_loader
import stacking_processor
import run_logger

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

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
        
        # Determine the effective range of images to process from the input stack
        # These are 0-based indices
        self.input_start_idx = max(0, self.config.resume_from - 1)
        self.input_end_idx = min(self.total_input_images - 1, self.config.stop_at - 1)
        
        # Calculate the number of input images relevant for processing
        self.relevant_input_count = (self.input_end_idx - self.input_start_idx + 1)
        if self.relevant_input_count <= 0:
            print("Warning: No relevant input images to process based on resume_from/stop_at settings.")
            self.total_output_stacks = 0
            self.output_stack_indices = []
        else:
            # Calculate total output stacks based on Primary
            # Each output stack corresponds to a 'primary' block of input images.
            self.total_output_stacks = math.ceil(self.relevant_input_count / self.config.primary)
            # The output_stack_indices will be 0, 1, 2, ..., total_output_stacks - 1
            self.output_stack_indices = list(range(self.total_output_stacks))
        
        # Image buffer for producer-consumer model
        # Buffer size should be large enough to hold multiple windows for multiple threads
        # Max images in any window: primary + 2*radius.
        # We need to load all images that *could* be part of any window, considering radius.
        # The loader should load from input_start_idx - radius up to input_end_idx + radius.
        # The buffer capacity should reflect this.
        max_window_size = self.config.primary + (2 * self.config.radius)
        min_buffer_capacity = max_window_size * self.config.threads * 2 # heuristic: 2 full windows per thread
        self.image_buffer = image_loader.ImageBuffer(capacity=max(min_buffer_capacity, 50)) # Min 50 images
        
        self._loader_thread: Optional[image_loader.ImageLoaderThread] = None
        self._processing_threads: List[threading.Thread] = []
        self._output_queue: deque[Tuple[int, np.ndarray]] = deque() # (output_index, image_data)
        self._output_queue_lock = threading.Lock()
        self._output_queue_event = threading.Event() # Signals new output available
        self._output_writer_thread: Optional[threading.Thread] = None
        
        self._current_processed_count = 0
        self._stop_event = threading.Event() # For external stop requests

    def _get_image_window_for_stack(self, output_stack_index: int) -> List[np.ndarray]:
        """
        Retrieves the list of image data (NumPy arrays) for a given output stacking window.
        This will fetch from the ImageBuffer, blocking if necessary.
        Handles padding/reusing images at the start/end of the stack.
        """
        image_data_window: List[np.ndarray] = []

        # Calculate the base input index for this output stack
        # This is the first input image *within the relevant range* for this output stack
        base_input_index_for_output = self.input_start_idx + (output_stack_index * self.config.primary)

        # Calculate the actual start and end indices for the full blending window
        # This window includes the primary images and the radius images on both sides.
        window_start_input_idx = base_input_index_for_output - self.config.radius
        window_end_input_idx = base_input_index_for_output + self.config.primary - 1 + self.config.radius

        for slice_idx in range(window_start_input_idx, window_end_input_idx + 1):
            if self._stop_event.is_set():
                return [] # Stop if cancellation requested during image fetching

            if slice_idx < 0:
                # Reuse the first actual input image (0-indexed) for bottom padding
                # Get the path of the first relevant input image
                first_relevant_image_path = self.image_loader.get_image_path(self.input_start_idx)
                print(f"DEBUG: For slice_idx < 0 ({slice_idx}), first_relevant_image_path: {first_relevant_image_path}, type: {type(first_relevant_image_path)}") # Debug print
                if first_relevant_image_path is not None:
                    img = self.image_loader.load_single_image(first_relevant_image_path)
                    image_data_window.append(img)
                else:
                    # Fallback to blank if even the first relevant image path is invalid (shouldn't happen if input_dir is valid)
                    img = self.image_loader.load_single_image(None)
                    image_data_window.append(img)
            elif slice_idx >= self.total_input_images:
                # Use blank images for top padding
                print(f"DEBUG: For slice_idx >= total_input_images ({slice_idx}), calling load_single_image(None)") # Debug print
                img = self.image_loader.load_single_image(None) # Returns a blank image
                image_data_window.append(img)
            else:
                # Fetch actual image data from the buffer
                try:
                    img = self.image_buffer.get(slice_idx, timeout=30) # 30 sec timeout
                    image_data_window.append(img)
                except TimeoutError:
                    print(f"Error: Timeout fetching image {slice_idx} from buffer. Aborting processing.")
                    self._stop_event.set() # Signal stop
                    return [] 
                except Exception as e:
                    print(f"Error fetching image {slice_idx}: {e}. Aborting processing.")
                    self._stop_event.set()
                    return []
        
        return image_data_window

    def _processing_worker(self, thread_id: int):
        """Worker function for each processing thread."""
        while not self._stop_event.is_set():
            try:
                # Atomically get the next output stack index to process
                with self._output_queue_lock:
                    if not self.output_stack_indices:
                        break # No more output stacks to process
                    current_output_index = self.output_stack_indices.pop(0)
                    
                print(f"Worker {thread_id}: Processing output stack {current_output_index}...")

                # Get the image window for this output stack
                image_window_data = self._get_image_window_for_stack(current_output_index)
                if not image_window_data and self._stop_event.is_set():
                    break # Aborted due to image fetching error

                # Process the stack
                processed_image = stacking_processor.process_image_stack(image_window_data)
                
                # Put the result into the output queue
                with self._output_queue_lock:
                    self._output_queue.append((current_output_index, processed_image))
                    self._output_queue_event.set() # Signal new output available
                
                with self._output_queue_lock: # Use the same lock to update count
                    self._current_processed_count += 1
                    if self.config.progress_callback:
                        self.config.progress_callback(self._current_processed_count, self.total_output_stacks)

            except IndexError: # output_stack_indices might become empty between checks
                break # No more output stacks to process
            except Exception as e:
                print(f"Worker {thread_id}: Error processing output stack {current_output_index}: {e}")
                self._stop_event.set() # Signal stop on error
                break
        print(f"Worker {thread_id}: Exiting.")

    def _output_writer(self):
        """Dedicated thread for writing processed images to disk."""
        written_count = 0
        while not (self._stop_event.is_set() and len(self._output_queue) == 0 and all(not t.is_alive() for t in self._processing_threads)):
            try:
                with self._output_queue_lock:
                    if not self._output_queue:
                        self._output_queue_event.clear() # No data, clear event
                        # Wait for new data or stop signal
                        if not self._output_queue_event.wait(0.5): # Wait with timeout
                            # If timeout and processing threads are done and queue is empty, exit
                            if self._stop_event.is_set() and all(not t.is_alive() for t in self._processing_threads):
                                break
                            continue # Continue waiting if not ready to exit

                    output_index, image_data = self._output_queue.popleft()
                
                output_filename = self._generate_output_filename(output_index)
                output_path = os.path.join(self.config.output_dir, output_filename)
                
                os.makedirs(self.config.output_dir, exist_ok=True)
                cv2.imwrite(output_path, image_data)
                written_count += 1
                print(f"OutputWriter: Saved {output_path} (Processed {written_count}/{self.total_output_stacks})")
                
            except IndexError: # Queue might become empty between checks
                continue
            except Exception as e:
                print(f"OutputWriter: Error saving image {output_index} to {output_path}: {e}")
                self._stop_event.set() # Signal stop on error
                break
        print("OutputWriter: Exiting.")

    def _generate_output_filename(self, index: int) -> str:
        """Generates the output filename with padding if configured."""
        if self.config.pad_filenames:
            return f"stack_{index:0{self.config.pad_length}d}.png"
        return f"stack_{index}.png"

    def run_pipeline(self):
        """Starts the entire processing pipeline."""
        if self.total_output_stacks == 0:
            print("No images to process based on current configuration.")
            if self.config.progress_callback:
                self.config.progress_callback(0, 0)
            return

        print(f"PipelineRunner: Starting processing for {self.total_output_stacks} stacks.")
        self._stop_event.clear() # Reset stop flag
        self._current_processed_count = 0
        self.image_buffer.clear() # Clear buffer from previous runs
        self._output_queue.clear() # Clear output queue

        # Log the run
        current_run_index = run_logger.get_last_run_index() + 1
        run_logger.log_run(current_run_index, self.config.to_dict())

        # Determine the range of input images the loader needs to load
        # This should cover all images that *could* be part of any window.
        # From the first relevant input image minus radius, to the last relevant input image plus primary plus radius.
        loader_start_idx = max(0, self.input_start_idx - self.config.radius)
        loader_end_idx = min(self.total_input_images - 1, self.input_end_idx + self.config.radius + self.config.primary -1) # Ensure enough images are loaded for the last window

        self._loader_thread = image_loader.ImageLoaderThread(
            image_loader=self.image_loader,
            image_buffer=self.image_buffer,
            total_images=self.total_input_images,
            start_index=loader_start_idx, # Pass the actual start index for loading
            end_index=loader_end_idx # Pass the actual end index for loading
        )
        self._loader_thread.start()

        # Start output writer thread
        self._output_writer_thread = threading.Thread(target=self._output_writer, name="OutputWriterThread")
        self._output_writer_thread.daemon = True
        self._output_writer_thread.start()

        # Start processing worker threads
        self._processing_threads = []
        for i in range(self.config.threads):
            thread = threading.Thread(target=self._processing_worker, args=(i,), name=f"Worker-{i}")
            thread.daemon = True
            self._processing_threads.append(thread)
            thread.start()

        # Main thread waits for all processing to complete or stop signal
        for thread in self._processing_threads:
            thread.join() # Wait for all workers to finish

        # Signal loader to stop if it hasn't already (e.g., if workers finished early due to error)
        self._loader_thread.stop()
        self._loader_thread.join()

        # Wait for output writer to finish processing any remaining items
        self._output_queue_event.set() # Ensure writer is unblocked to check for completion
        if self._output_writer_thread:
            self._output_writer_thread.join()

        if self._stop_event.is_set():
            print("PipelineRunner: Processing stopped prematurely due to an error or user request.")
        else:
            print("PipelineRunner: Processing completed successfully.")
        
        if self.config.progress_callback:
            self.config.progress_callback(self.total_output_stacks, self.total_output_stacks) # Ensure 100% progress

    def stop_pipeline(self):
        """Signals all threads to stop processing."""
        print("PipelineRunner: Stop requested.")
        self._stop_event.set() # Signal stop to all threads
        if self._loader_thread:
            self._loader_thread.stop() # Explicitly stop loader
        self.image_buffer.clear() # Clear buffer to unblock any waiting consumers
        self._output_queue_event.set() # Unblock output writer

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- Pipeline Runner Module Test ---")

    # Create dummy input/output directories and dummy PNG files
    test_input_dir = "test_input_images"
    test_output_dir = "test_output_stacks"
    os.makedirs(test_input_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    num_dummy_images = 15 # Total images
    for i in range(1, num_dummy_images + 1):
        dummy_img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        cv2.imwrite(os.path.join(test_input_dir, f"image_{i:03d}.png"), dummy_img)
    print(f"Created {num_dummy_images} dummy images in {test_input_dir}")

    # Dummy Config for testing
    class MockXYBlendOperation:
        def __init__(self, op_type: str, **kwargs):
            self.type = op_type
            # Initialize all possible parameters with their default values from config.py's XYBlendOperation
            # This ensures the mock object has all expected attributes
            self.gaussian_ksize_x: int = kwargs.get("gaussian_ksize_x", 3)
            self.gaussian_ksize_y: int = kwargs.get("gaussian_ksize_y", 3)
            self.gaussian_sigma_x: float = kwargs.get("gaussian_sigma_x", 0.0)
            self.gaussian_sigma_y: float = kwargs.get("gaussian_sigma_y", 0.0)
            self.bilateral_d: int = kwargs.get("bilateral_d", 9)
            self.bilateral_sigma_color: float = kwargs.get("bilateral_sigma_color", 75.0)
            self.bilateral_sigma_space: float = kwargs.get("bilateral_sigma_space", 75.0)
            self.median_ksize: int = kwargs.get("median_ksize", 5)
            self.unsharp_amount: float = kwargs.get("unsharp_amount", 1.0)
            self.unsharp_threshold: int = kwargs.get("unsharp_threshold", 0)
            self.unsharp_blur_ksize: int = kwargs.get("unsharp_blur_ksize", 5)
            self.unsharp_blur_sigma: float = kwargs.get("unsharp_blur_sigma", 0.0)
            self.resize_width: Optional[int] = kwargs.get("resize_width", None)
            self.resize_height: Optional[int] = kwargs.get("resize_height", None)
            self.resample_mode: str = kwargs.get("resample_mode", "LANCZOS4")

        def to_dict(self):
            data = {"type": self.type}
            params = {}
            if self.type == "gaussian_blur":
                params = {
                    "gaussian_ksize_x": self.gaussian_ksize_x,
                    "gaussian_ksize_y": self.gaussian_ksize_y,
                    "gaussian_sigma_x": self.gaussian_sigma_x,
                    "gaussian_sigma_y": self.gaussian_sigma_y,
                }
            elif self.type == "bilateral_filter":
                params = {
                    "bilateral_d": self.bilateral_d,
                    "bilateral_sigma_color": self.bilateral_sigma_color,
                    "bilateral_sigma_space": self.bilateral_sigma_space,
                }
            elif self.type == "median_blur":
                params = {
                    "median_ksize": self.median_ksize,
                }
            elif self.type == "unsharp_mask":
                params = {
                    "unsharp_amount": self.unsharp_amount,
                    "unsharp_threshold": self.unsharp_threshold,
                    "unsharp_blur_ksize": self.unsharp_blur_ksize,
                    "unsharp_blur_sigma": self.unsharp_blur_sigma,
                }
            elif self.type == "resize":
                params = {
                    "resize_width": self.resize_width,
                    "resize_height": self.resize_height,
                    "resample_mode": self.resample_mode,
                }
            data["params"] = params
            return data

        def __post_init__(self):
            pass # Simplified for mock

    class MockConfig(config.Config): # Inherit from actual Config to get all fields
        def __init__(self):
            # Manually initialize _initialized to True to prevent recursive load from super().__init__()
            object.__setattr__(self, '_initialized', True)
            super().__init__() # Call parent dataclass init to set defaults
            
            self.input_dir = test_input_dir
            self.output_dir = test_output_dir
            self.file_pattern = "image_*.png"
            self.primary = 3
            self.radius = 1
            self.threads = 2 # Use 2 worker threads for testing
            self.resume_from = 1
            self.stop_at = num_dummy_images # Process all dummy images
            self.pad_filenames = True
            self.pad_length = 4
            self.blend_mode = "gaussian"
            self.blend_param = 1.0
            self.directional_blend = False
            self.dir_sigma = 1.0
            self.scale_bits = 12
            self.binary_threshold = 128
            self.gradient_threshold = 128
            self.top_surface_smoothing = False
            self.top_surface_strength = 0.5
            self.gradient_smooth = False # Not used in this module directly now
            self.gradient_blend_strength = 0.0 # Not used in this module directly now
            
            # LUT settings
            self.lut_source = "generated"
            self.lut_generation_type = "linear"
            self.linear_min_input = 0
            self.linear_max_output = 255

            # XY pipeline: Resize down, then blur
            self.xy_blend_pipeline = [
                MockXYBlendOperation("resize", resize_width=25, resize_height=25, resample_mode="BILINEAR"),
                MockXYBlendOperation("gaussian_blur", gaussian_ksize_x=3, gaussian_ksize_y=3, gaussian_sigma_x=0.8, gaussian_sigma_y=0.8)
            ]
            self.run_log_file = "test_pipeline_runs.log"

            self.progress_updates = []
            def mock_progress_callback(current, total):
                self.progress_updates.append((current, total))
                print(f"Progress: {current}/{total}")
            self.progress_callback = mock_progress_callback

    mock_config = MockConfig()
    config.app_config = mock_config # Set the global app_config to our mock
    set_config_reference(mock_config) # Set config reference for all modules

    # Ensure LUT is updated based on mock config
    lut_manager.update_active_lut_from_config()

    runner = PipelineRunner()
    
    # Run the pipeline
    print("\nStarting pipeline run...")
    runner.run_pipeline()
    print("\nPipeline run finished.")

    # Verify output files
    output_files = os.listdir(test_output_dir)
    print(f"\nOutput files in {test_output_dir}: {sorted(output_files)}")
    expected_output_count = runner.total_output_stacks
    print(f"Expected output files: {expected_output_count}")
    print(f"Actual output files: {len(output_files)}")
    assert len(output_files) == expected_output_count, "Mismatch in number of output files!"

    # Verify progress callback
    print(f"Progress updates received: {runner.config.progress_updates}")
    assert runner.config.progress_updates[-1] == (expected_output_count, expected_output_count), "Progress callback did not reach 100%."

    # Verify log file
    if os.path.exists(mock_config.run_log_file):
        with open(mock_config.run_log_file, 'r') as f:
            log_contents = f.readlines()
        print(f"\nLog file '{mock_config.run_log_file}' content:")
        for line in log_contents:
            print(line.strip())
        assert len(log_contents) >= 1, "Log file should contain at least one entry."
    else:
        print(f"Log file '{mock_config.run_log_file}' not found.")

    # Clean up dummy files and directories
    print("\nCleaning up test directories...")
    for f in os.listdir(test_input_dir):
        os.remove(os.path.join(test_input_dir, f))
    os.rmdir(test_input_dir)

    for f in os.listdir(test_output_dir):
        os.remove(os.path.join(test_output_dir, f))
    os.rmdir(test_output_dir)

    if os.path.exists(mock_config.run_log_file):
        os.remove(mock_config.run_log_file)
    print("Cleanup complete.")
