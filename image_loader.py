# image_loader.py

"""
Implements the image loading component for Modular-Stacker.
This module acts as a producer in a producer-consumer model,
loading image slices from disk into a shared buffer (queue)
for consumption by processing threads.
It handles file discovery, natural sorting, and provides
a mechanism for buffering image data.
"""

import os
import re
import cv2
import numpy as np
import threading
import glob # Added: For glob.glob functionality
from collections import deque
from typing import List, Optional, Tuple, Deque, Dict, Any

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

class ImageLoader:
    """
    Loads and buffers image slices. Designed to be run in a separate thread
    to pre-load images, acting as a producer.
    """
    def __init__(self, input_dir: str, file_pattern: str):
        self.input_dir = input_dir
        self.file_pattern = file_pattern
        self._image_paths: List[str] = self._collect_numbered_images()
        self._total_images = len(self._image_paths)
        self._blank_image_shape: Optional[Tuple[int, int]] = None

        print(f"ImageLoader: Found {self._total_images} images in {input_dir} matching {file_pattern}")
        if self._image_paths:
            print(f"DEBUG: ImageLoader.__init__: First image path: {self._image_paths[0]}, type: {type(self._image_paths[0])}")
            print(f"DEBUG: ImageLoader.__init__: Last image path: {self._image_paths[-1]}, type: {type(self._image_paths[-1])}")


    def _natural_sort_key(self, path: str) -> int:
        """
        Extract the first integer in the filename for natural sorting.
        Files without digits sort at the end (infinite).
        """
        name = os.path.basename(path)
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else float('inf')

    def _collect_numbered_images(self) -> List[str]:
        """
        Scan `input_dir` for files matching `file_pattern`, filter to filenames that contain
        one or more digits followed by ".png", ".tif", or ".tiff", then sort naturally by that integer.
        Returns a sorted list of absolute paths.
        """
        pattern = os.path.join(self.input_dir, self.file_pattern)
        all_files = [f for f in glob_safe(pattern) if os.path.isfile(f)] # Ensure it's a file
        
        # Filter to files that contain numbers and are valid image types (png, tif, tiff)
        # This regex now allows any characters before the digits, handling patterns like "image_001.png"
        numbered = [p for p in all_files if re.search(r'.*?(\d+)\.(png|tif|tiff)$', os.path.basename(p), re.IGNORECASE)]
        
        return sorted(numbered, key=self._natural_sort_key)

    def _infer_image_shape(self) -> Optional[Tuple[int, int]]:
        """Infers the shape (H, W) from the first valid image."""
        if not self._image_paths:
            return None
        
        for path in self._image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img.shape
        return None

    def load_single_image(self, path: Optional[str]) -> np.ndarray:
        """Loads a grayscale image (uint8) or returns a blank image if path is None.
        This method is intended to be called by the ImageLoaderThread or directly for blanks.
        """
        print(f"DEBUG: load_single_image called with path: {path}, type: {type(path)}") # Debug print
        if path is None:
            if self._blank_image_shape is None:
                # Infer shape only once, if needed for blank images
                self._blank_image_shape = self._infer_image_shape()
                if self._blank_image_shape is None:
                    # Fallback if no images found to infer shape
                    print("Warning: No valid images found to infer blank image shape. Defaulting to 100x100.")
                    self._blank_image_shape = (100, 100)
            return np.zeros(self._blank_image_shape, dtype=np.uint8)
        
        # Explicitly cast path to str to ensure cv2.imread gets a string
        path_str = str(path)
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load image {path_str}. Returning a blank image.")
            if self._blank_image_shape is None:
                self._blank_image_shape = self._infer_image_shape()
                if self._blank_image_shape is None:
                    self._blank_image_shape = (100, 100)
            return np.zeros(self._blank_image_shape, dtype=np.uint8)
        return img

    def get_total_images(self) -> int:
        """Returns the total number of discoverable images."""
        return self._total_images

    def get_image_path(self, index: int) -> Optional[str]:
        """Returns the path for a given global image index, or None if out of bounds."""
        if 0 <= index < self._total_images:
            return self._image_paths[index]
        return None

class ImageBuffer:
    """
    A thread-safe buffer for image data, acting as the shared resource
    between the ImageLoaderThread (producer) and processing workers (consumers).
    It maintains a sliding window of loaded images.
    """
    def __init__(self, capacity: int):
        self._buffer: Dict[int, np.ndarray] = {} # {global_index: image_data}
        self._capacity = capacity
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._min_index_in_buffer = -1 # Smallest global index currently in buffer
        self._max_index_in_buffer = -1 # Largest global index currently in buffer

    def put(self, index: int, image_data: np.ndarray):
        """Adds an image to the buffer."""
        with self._condition:
            # Evict oldest if capacity exceeded
            while len(self._buffer) >= self._capacity:
                # Find the oldest image to remove (e.g., the one with the smallest index)
                if not self._buffer: # Should not happen if len >= capacity
                    break
                oldest_index = min(self._buffer.keys())
                del self._buffer[oldest_index]
                # Update min_index_in_buffer if the oldest was removed
                if oldest_index == self._min_index_in_buffer:
                    self._min_index_in_buffer = min(self._buffer.keys()) if self._buffer else -1

            self._buffer[index] = image_data
            if self._min_index_in_buffer == -1 or index < self._min_index_in_buffer:
                self._min_index_in_buffer = index
            if index > self._max_index_in_buffer:
                self._max_index_in_buffer = index
            
            self._condition.notify_all() # Notify consumers that data is available

    def get(self, index: int, timeout: Optional[float] = None) -> np.ndarray:
        """
        Retrieves an image from the buffer by its global index.
        Blocks until the image is available or timeout occurs.
        """
        with self._condition:
            while index not in self._buffer:
                if not self._condition.wait(timeout):
                    raise TimeoutError(f"Timeout waiting for image at index {index}")
            return self._buffer[index]

    def contains(self, index: int) -> bool:
        """Checks if an image with the given index is in the buffer."""
        with self._lock:
            return index in self._buffer

    def get_buffered_range(self) -> Tuple[int, int]:
        """Returns the current min and max global indices buffered."""
        with self._lock:
            if not self._buffer:
                return -1, -1
            return self._min_index_in_buffer, self._max_index_in_buffer

    def clear(self):
        """Clears the buffer."""
        with self._lock:
            self._buffer.clear()
            self._min_index_in_buffer = -1
            self._max_index_in_buffer = -1
            self._condition.notify_all() # Notify any waiting threads that buffer is empty

# Helper for glob that handles potential path issues
def glob_safe(pattern: str) -> List[str]:
    """A safer glob.glob that handles non-existent directories gracefully."""
    try:
        import glob
        return glob.glob(pattern)
    except Exception as e:
        print(f"Warning: glob.glob failed for pattern '{pattern}': {e}. Returning empty list.")
        return []


class ImageLoaderThread(threading.Thread):
    """
    A dedicated thread for loading images from disk and populating the ImageBuffer.
    This is the producer in the producer-consumer model.
    """
    def __init__(self, image_loader: ImageLoader, image_buffer: ImageBuffer, total_images: int,
                 start_index: int = 0, end_index: Optional[int] = None):
        super().__init__(name="ImageLoaderThread")
        self.image_loader = image_loader # Reference to the ImageLoader instance
        self.image_buffer = image_buffer
        self._running = threading.Event()
        self._running.set() # Set to true initially
        self._total_images = total_images # Total images found by ImageLoader
        self._start_index = start_index
        # If end_index is None, load up to the last image found by ImageLoader
        self._end_index = end_index if end_index is not None else (total_images - 1)

    def run(self):
        """The main loop for the image loading producer thread."""
        # Iterate from _start_index to _end_index (inclusive)
        for i in range(self._start_index, self._end_index + 1):
            if not self._running.is_set(): # Check if stop was requested
                print(f"ImageLoaderThread: Stop requested, halting loading at index {i}.")
                break
            
            # Get the path from ImageLoader
            image_path = self.image_loader.get_image_path(i)
            # Load the image using ImageLoader's method.
            # load_single_image handles None paths (for blank images) but here we expect actual paths.
            # If get_image_path returns None, it means the index is out of the *discovered* image range.
            # This should ideally not happen if loader_end_idx is correctly calculated in PipelineRunner.
            if image_path is not None:
                image_data = self.image_loader.load_single_image(image_path)
                # Put it into the buffer
                self.image_buffer.put(i, image_data)
            else:
                print(f"ImageLoaderThread: Warning: No image path found for index {i}. Skipping.")
            
            # Optional: Add a small delay to simulate work and prevent busy-waiting
            # import time
            # time.sleep(0.001) 
        
        print("ImageLoaderThread: Finished loading images in specified range.")

    def stop(self):
        """Signals the loader thread to stop."""
        self._running.clear()


# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- Image Loader Module Test ---")
    
    # Create a dummy input directory and some dummy PNG files
    test_input_dir = "test_input_images"
    os.makedirs(test_input_dir, exist_ok=True)
    for i in range(1, 11): # Create 10 dummy images
        dummy_img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        cv2.imwrite(os.path.join(test_input_dir, f"{i:03d}.png"), dummy_img)
    # Also create some with prefixes to test the new regex
    for i in range(1, 11):
        dummy_img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        cv2.imwrite(os.path.join(test_input_dir, f"prefix_{i:03d}.png"), dummy_img)
    print(f"Created dummy images in {test_input_dir}")

    # Dummy Config for testing
    class MockConfig:
        def __init__(self):
            self.input_dir = test_input_dir
            self.file_pattern = "*.png" # This pattern will pick up both types
            self.threads = 2 # Simulate multiple consumers
            self.buffer_size = 5 # Small buffer for testing

    mock_config = MockConfig()
    set_config_reference(mock_config)

    # Initialize ImageLoader and ImageBuffer
    image_loader_instance = ImageLoader(mock_config.input_dir, mock_config.file_pattern)
    image_buffer = ImageBuffer(capacity=mock_config.buffer_size)
    
    total_images_found = image_loader_instance.get_total_images()
    print(f"Total images found by ImageLoader (including prefixed): {total_images_found}")

    # Test loading a specific range
    loader_start = 5
    loader_end = 15 # Load images from index 5 to 15 (inclusive)
    print(f"\nTesting ImageLoaderThread loading from index {loader_start} to {loader_end}...")
    loader_thread = ImageLoaderThread(
        image_loader=image_loader_instance, # Pass the ImageLoader instance
        image_buffer=image_buffer,
        total_images=total_images_found, # Still need total_images for get_image_path bounds check
        start_index=loader_start,
        end_index=loader_end
    )

    # Start the loader thread
    loader_thread.start()

    # Simulate consumers fetching images from the *full* range, not just the loaded range
    print("\nSimulating consumers fetching images...")
    fetched_images = []
    # We expect 20 images now (10 simple, 10 prefixed)
    # Try to fetch slightly outside loaded range
    for i in range(loader_start - 2, loader_end + 3): 
        try:
            print(f"Consumer: Requesting image {i}...")
            if image_loader_instance.get_image_path(i) is not None: # Only try to get if it's a real image index
                img = image_buffer.get(i, timeout=5) # Wait up to 5 seconds
                fetched_images.append(img)
                print(f"Consumer: Fetched image {i}, shape: {img.shape}, dtype: {img.dtype}")
            else:
                print(f"Consumer: Index {i} is outside discovered image range. Skipping buffer fetch.")
        except TimeoutError:
            print(f"Consumer: Timed out waiting for image {i}. Loader might have stopped or no more images in range.")
            # If timeout, it means the loader didn't put it, which is expected for indices outside loader_start/end
            pass # Don't break, continue to see if other images are there
        except Exception as e:
            print(f"Consumer: An error occurred fetching image {i}: {e}")
            break

    # Stop the loader thread
    loader_thread.stop()
    loader_thread.join() # Wait for the thread to fully finish

    print(f"\nTotal images fetched: {len(fetched_images)}")
    print(f"Image buffer final state (keys): {sorted(image_buffer._buffer.keys())}")

    # Clean up dummy images and directory
    for f in os.listdir(test_input_dir):
        os.remove(os.path.join(test_input_dir, f))
    try:
        os.rmdir(test_input_dir)
        print(f"Cleaned up {test_input_dir}")
    except OSError:
        pass # Directory might not be empty if test failed mid-way
