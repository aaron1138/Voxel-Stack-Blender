"""
Manages image loading from disk and provides a thread-safe,
memory-managed buffer for the processing pipeline.
"""
import os
import cv2
import threading
import time
import numpy as np
from typing import List, Optional, Tuple, Any, Deque
from collections import deque

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

class ImageLoader:
    """Handles discovering and loading individual images."""
    def __init__(self, input_dir: str, file_pattern: str):
        self.input_dir = input_dir
        self.image_paths = self._find_images(input_dir, file_pattern)
        self.total_images = len(self.image_paths)
        print(f"ImageLoader: Found {self.total_images} images in {input_dir} matching {file_pattern}")

    def _find_images(self, directory: str, pattern: str) -> List[str]:
        """Finds and sorts image files based on numeric parts of filenames."""
        import re
        numeric_pattern = re.compile(r'(\d+)')
        
        def get_sort_key(filename):
            parts = numeric_pattern.findall(filename)
            return [int(p) for p in parts] if parts else [filename]

        try:
            import fnmatch
            matching_files = fnmatch.filter(os.listdir(directory), pattern)
            sorted_files = sorted(matching_files, key=get_sort_key)
            return [os.path.join(directory, f) for f in sorted_files]
        except FileNotFoundError:
            return []

    def get_total_images(self) -> int:
        return self.total_images

    def get_image_path(self, index: int) -> Optional[str]:
        if 0 <= index < self.total_images:
            return self.image_paths[index]
        return None

    def load_single_image(self, filepath: Optional[str]) -> np.ndarray:
        """Loads one image, returning a blank one if path is None."""
        if filepath and os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        
        if self.total_images > 0:
            sample_img = cv2.imread(self.image_paths[0], cv2.IMREAD_GRAYSCALE)
            if sample_img is not None:
                return np.zeros(sample_img.shape, dtype=np.uint8)
        return np.zeros((100, 100), dtype=np.uint8)

class ImageBuffer:
    """A thread-safe, capacity-limited buffer for holding loaded images."""
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.buffer = {}
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self._stop_event = threading.Event()

    def put(self, index: int, image: np.ndarray):
        with self.lock:
            # This is a non-blocking put. The loader thread is responsible for checking capacity.
            if self._stop_event.is_set(): return
            self.buffer[index] = image
            self.not_empty.notify_all()

    def get(self, index: int, timeout: int = 30) -> np.ndarray:
        with self.not_empty:
            start_time = time.time()
            while index not in self.buffer:
                if self._stop_event.is_set() or (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Timeout or stop requested while waiting for image {index}")
                if not self.not_empty.wait(0.1):
                    # Check for timeout again after wait
                    if (time.time() - start_time) > timeout:
                         raise TimeoutError(f"Timeout after wait for image {index}")
            return self.buffer[index]

    def prune(self, below_index: int):
        """Removes images with indices lower than the given index."""
        with self.lock:
            keys_to_remove = [k for k in self.buffer if k < below_index]
            for key in keys_to_remove:
                del self.buffer[key]
    
    def clear(self):
        with self.lock:
            self.buffer.clear()

    def stop(self):
        self._stop_event.set()
        with self.not_empty:
            self.not_empty.notify_all()

    def start(self):
        self._stop_event.clear()

class ImageLoaderThread(threading.Thread):
    """A thread that intelligently loads images into the buffer ahead of workers."""
    def __init__(self, image_loader: ImageLoader, image_buffer: ImageBuffer, start_index: int, end_index: int, 
                 job_queue: Deque[int], job_queue_lock: threading.Lock, primary_step: int, radius: int, 
                 input_start_offset: int, stop_event: threading.Event):
        super().__init__(name="ImageLoaderThread")
        self.image_loader = image_loader
        self.image_buffer = image_buffer
        self.start_index = start_index
        self.end_index = end_index
        self.job_queue = job_queue
        self.job_queue_lock = job_queue_lock
        self.primary_step = primary_step
        self.radius = radius
        self.input_start_offset = input_start_offset
        self._stop_event = stop_event
        self.daemon = True

    def run(self):
        """
        Continuously loads images needed for upcoming jobs, but only if the
        buffer has capacity, creating a true sliding window.
        """
        current_load_idx = self.start_index
        while not self._stop_event.is_set() and current_load_idx <= self.end_index:
            
            # 1. Check if the buffer is full. If so, wait.
            if len(self.image_buffer.buffer) >= self.image_buffer.capacity:
                time.sleep(0.1)
                continue

            # 2. Determine the maximum image index we need to have loaded right now.
            max_needed_idx = 0
            with self.job_queue_lock:
                if not self.job_queue:
                    if self._stop_event.is_set(): break
                    time.sleep(0.1)
                    continue
                
                # We need to ensure images are loaded for the very next job in the queue.
                next_job_index = self.job_queue[0]
                next_job_base_idx = self.input_start_offset + (next_job_index * self.primary_step)
                max_needed_idx = next_job_base_idx + self.primary_step - 1 + self.radius

            # 3. Load the next required image if it's not already in the buffer.
            if current_load_idx <= max_needed_idx:
                if current_load_idx not in self.image_buffer.buffer:
                    filepath = self.image_loader.get_image_path(current_load_idx)
                    image_data = self.image_loader.load_single_image(filepath)
                    self.image_buffer.put(current_load_idx, image_data)
                current_load_idx += 1
            else:
                # We are caught up, wait a moment before checking again.
                time.sleep(0.05)
