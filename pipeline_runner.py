"""
Orchestrates the Modular-Stacker image processing pipeline.
Manages image loading, buffering, multi-threaded processing,
and saving of output images. This version includes dynamic buffer
pruning to manage memory usage for large image sets.
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
import vertical_blend_processor

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance
    image_loader.set_config_reference(config_instance)
    stacking_processor.set_config_reference(config_instance)
    run_logger.set_config_reference(config_instance)
    vertical_blend_processor.set_config_reference(config_instance)

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
            self.output_stack_indices = deque()
        else:
            is_vb_substitute = not self.config.vertical_blend_pre_process and self.config.blend_mode.startswith("vertical_")
            self.primary_for_buffer = 1 if is_vb_substitute else self.config.primary
            self.total_output_stacks = math.ceil(self.relevant_input_count / self.primary_for_buffer)
            self.output_stack_indices = deque(range(self.total_output_stacks))
        
        vb_radius = 0
        is_vb_active = self.config.vertical_blend_pre_process or self.config.blend_mode.startswith("vertical_")
        if is_vb_active:
            vb_radius = max(self.config.vertical_receding_layers, self.config.vertical_overhang_layers)
        
        stacker_radius = self.config.radius
        
        if not is_vb_active:
            self.effective_radius = stacker_radius
        elif self.config.vertical_blend_pre_process:
            self.effective_radius = vb_radius + stacker_radius
        else:
            self.effective_radius = vb_radius

        max_window_size = self.primary_for_buffer + (2 * self.effective_radius)
        min_buffer_capacity = (max_window_size * self.config.threads) + 20 
        self.image_buffer = image_loader.ImageBuffer(capacity=min_buffer_capacity)
        
        self._loader_thread: Optional[image_loader.ImageLoaderThread] = None
        self._processing_threads: List[threading.Thread] = []
        self._output_queue: deque[Tuple[int, np.ndarray]] = deque()
        self._output_queue_lock = threading.Lock()
        self._job_queue_lock = threading.Lock()
        self._output_queue_event = threading.Event()
        self._output_writer_thread: Optional[threading.Thread] = None
        
        self._current_processed_count = 0
        self._stop_event = threading.Event()

    def _get_image_window_for_stack(self, output_stack_index: int) -> List[np.ndarray]:
        """Retrieves the list of image data for a given output stacking window."""
        image_data_window: List[np.ndarray] = []
        base_input_index_for_output = self.input_start_idx + (output_stack_index * self.primary_for_buffer)
        
        window_start_input_idx = base_input_index_for_output - self.effective_radius
        window_end_input_idx = base_input_index_for_output + self.primary_for_buffer - 1 + self.effective_radius

        for slice_idx in range(window_start_input_idx, window_end_input_idx + 1):
            if self.is_stop_requested(): return []

            if slice_idx < 0:
                img = self.image_loader.load_single_image(self.image_loader.get_image_path(self.input_start_idx))
                image_data_window.append(img)
            elif slice_idx >= self.total_input_images:
                img = self.image_loader.load_single_image(None)
                image_data_window.append(img)
            else:
                try:
                    img = self.image_buffer.get(slice_idx, timeout=30)
                    image_data_window.append(img)
                except TimeoutError:
                    print(f"Error: Timeout fetching image {slice_idx} from buffer.")
                    self.stop_pipeline()
                    return []
        return image_data_window

    def _prune_image_buffer(self):
        """Removes images from the buffer that are no longer needed."""
        min_needed_slice_idx = float('inf')
        with self._job_queue_lock:
            if not self.output_stack_indices:
                min_needed_slice_idx = float('inf')
            else:
                min_output_idx = self.output_stack_indices[0] # Check the next job
                min_needed_base_idx = self.input_start_idx + (min_output_idx * self.primary_for_buffer)
                min_needed_slice_idx = min_needed_base_idx - self.effective_radius
        
        if hasattr(self.image_buffer, 'prune'):
            self.image_buffer.prune(below_index=min_needed_slice_idx)

    def _processing_worker(self, thread_id: int):
        """Worker function for each processing thread."""
        while not self.is_stop_requested():
            try:
                with self._job_queue_lock:
                    if not self.output_stack_indices: break
                    current_output_index = self.output_stack_indices.popleft()
                
                image_window_data = self._get_image_window_for_stack(current_output_index)
                if not image_window_data: break

                processed_image = stacking_processor.process_image_stack(image_window_data)
                
                with self._output_queue_lock:
                    self._output_queue.append((current_output_index, processed_image))
                    self._output_queue_event.set()
                    self._current_processed_count += 1
                    if self.config.progress_callback:
                        self.config.progress_callback(self._current_processed_count, self.total_output_stacks)
                
                self._prune_image_buffer()

            except IndexError: break
            except Exception as e:
                print(f"Worker {thread_id}: Error on stack {current_output_index}: {e}")
                self.stop_pipeline()
                break

    def _output_writer(self):
        """Dedicated thread for writing processed images to disk."""
        written_count = 0
        while not (self.is_stop_requested() and len(self._output_queue) == 0):
            if not self._output_queue_event.wait(0.5): continue
            
            with self._output_queue_lock:
                if not self._output_queue:
                    if all(not t.is_alive() for t in self._processing_threads): break
                    continue
                
                self._output_queue = deque(sorted(self._output_queue))
                output_index, image_data = self._output_queue.popleft()
            
            try:
                output_filename = self._generate_output_filename(output_index)
                output_path = os.path.join(self.config.output_dir, output_filename)
                os.makedirs(self.config.output_dir, exist_ok=True)
                cv2.imwrite(output_path, image_data)
                written_count += 1
            except Exception as e:
                print(f"OutputWriter: Error saving image: {e}")
                self.stop_pipeline()
                break

    def _generate_output_filename(self, index: int) -> str:
        """Generates the output filename with padding."""
        if not self.config.pad_filenames:
            return f"stack_{index}.png"
        pad_total_width = len(str(self.total_output_stacks -1))
        pad_len = max(self.config.pad_length, pad_total_width)
        return f"stack_{index:0{pad_len}d}.png"

    def run_pipeline(self):
        """Starts the entire processing pipeline."""
        if self.total_output_stacks == 0:
            if self.config.progress_callback: self.config.progress_callback(0, 0)
            return

        self.reset_stop_flag()
        self._current_processed_count = 0
        self.image_buffer.clear()
        self._output_queue.clear()

        run_logger.log_run(run_logger.get_last_run_index() + 1, self.config.to_dict())

        loader_start_idx = max(0, self.input_start_idx - self.effective_radius)
        loader_end_idx = min(self.total_input_images - 1, self.input_end_idx + self.primary_for_buffer - 1 + self.effective_radius)

        self._loader_thread = image_loader.ImageLoaderThread(
            image_loader=self.image_loader,
            image_buffer=self.image_buffer,
            start_index=loader_start_idx,
            end_index=loader_end_idx,
            job_queue=self.output_stack_indices,
            job_queue_lock=self._job_queue_lock,
            primary_step=self.primary_for_buffer,
            radius=self.effective_radius,
            input_start_offset=self.input_start_idx,
            stop_event=self._stop_event
        )
        self._loader_thread.start()

        self._output_writer_thread = threading.Thread(target=self._output_writer, name="OutputWriterThread")
        self._output_writer_thread.start()

        self._processing_threads = [
            threading.Thread(target=self._processing_worker, args=(i,), name=f"Worker-{i}")
            for i in range(self.config.threads)
        ]
        for thread in self._processing_threads: thread.start()
        for thread in self._processing_threads: thread.join()

        self.stop_pipeline()
        if self._loader_thread: self._loader_thread.join()
        if self._output_writer_thread: self._output_writer_thread.join()

        if self.config.progress_callback and not self.is_stop_requested():
            self.config.progress_callback(self.total_output_stacks, self.total_output_stacks)

    def stop_pipeline(self):
        """Signals all threads to stop processing."""
        self._stop_event.set()
        self.image_buffer.stop()
        self._output_queue_event.set()

    def is_stop_requested(self) -> bool:
        """Checks if the stop event has been set."""
        return self._stop_event.is_set()

    def reset_stop_flag(self):
        """Resets the stop event for a new run."""
        self._stop_event.clear()
        self.image_buffer.start()
