# ui_components.py (Modified)

import os
import re
import cv2
import concurrent.futures # For multi-threading
import collections # For deque for prior image cache
import numpy as np
from typing import List

# PySide6 imports
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QTabWidget # Added QTabWidget for tabbed interface
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QSettings
from PySide6.QtGui import QIntValidator # For validating integer input

# Import the new configuration system
from config import app_config as config, Config, XYBlendOperation, LutParameters, DEFAULT_NUM_WORKERS

# Import the processing functions from our modularized backend
import processing_core as core
import xy_blend_processor # New import for XY pipeline
import lut_manager # Import lut_manager (used by xy_blend_processor, but good for clarity)

# The DEFAULT_NUM_WORKERS is now defined in config.py and imported.

class ImageProcessorThread(QThread):
    """
    Manages the image processing pipeline in a separate thread to keep the GUI responsive.
    Now supports multi-threading for individual image processing tasks.
    """
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, app_config: Config, max_workers: int): # Added max_workers parameter
        super().__init__()
        self.app_config = app_config # Use the globally loaded app_config
        self._is_running = True
        self.max_workers = max_workers # Use the provided thread count

    # Static method to be executed by each worker in the thread pool
    @staticmethod
    def _process_single_image_task(
        filepath: str,
        base_filename_no_ext: str,
        current_binary_image: np.ndarray,
        current_original_image: np.ndarray,
        prior_binary_masks_snapshot: collections.deque, # A snapshot for this task
        processing_core_params: dict, # Parameters for processing_core functions
        xy_blend_pipeline_ops: List[XYBlendOperation], # The XY pipeline definition
        output_folder: str,
        debug_save: bool
    ) -> str: # Returns the path of the saved output image
        """
        Processes a single image completely, including vertical blending,
        XY pipeline, and saving. This function runs in a worker thread.
        """
        debug_info = {
            'output_folder': output_folder,
            'base_filename': base_filename_no_ext
        } if debug_save else None

        # --- Stage 1: Vertical Blending (from processing_core) ---
        prior_white_combined_mask = core.find_prior_combined_white_mask(list(prior_binary_masks_snapshot))
        
        receding_gradient = core.calculate_receding_gradient_field(
            current_binary_image, 
            prior_white_combined_mask,
            processing_core_params['use_fixed_norm'],
            processing_core_params['fixed_fade_distance'],
            debug_info=debug_info
        )

        output_image_from_core = core.merge_to_output(current_original_image, receding_gradient)
        
        # Ensure output from core is 8-bit grayscale for next stage
        if output_image_from_core.dtype != np.uint8:
            output_image_from_core = np.clip(output_image_from_core, 0, 255).astype(np.uint8)

        # --- Stage 2: XY Blend Pipeline (post-processing) ---
        # Pass the 8-bit image directly in memory to the next stage
        final_processed_image = xy_blend_processor.process_xy_pipeline(output_image_from_core, xy_blend_pipeline_ops)

        # --- Stage 3: Save Output ---
        output_filename = f"processed_{base_filename_no_ext}.png"
        output_filepath = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_filepath, final_processed_image)

        return output_filepath

    def run(self):
        """The main processing loop, now orchestrating multi-threaded tasks."""
        self.status_update.emit("Processing started...")
        
        # Helper to extract numbers from filenames for sorting
        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')

        try:
            # Prepare image filenames
            all_image_filenames = sorted(
                [f for f in os.listdir(self.app_config.input_folder) if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff'))],
                key=get_numeric_part
            )

            # Filter based on start/stop indices
            image_filenames_filtered = []
            for f in all_image_filenames:
                numeric_part = get_numeric_part(f)
                if self.app_config.start_index is not None and numeric_part < self.app_config.start_index:
                    continue
                if self.app_config.stop_index is not None and numeric_part > self.app_config.stop_index:
                    continue
                image_filenames_filtered.append(f)

            total_images = len(image_filenames_filtered)
            if total_images == 0:
                self.error_signal.emit("No images found in the specified folder or index range.")
                return

            # This deque stores binary masks of prior images needed for the receding gradient.
            # It only holds `n_layers` images to manage memory.
            prior_binary_masks_cache = collections.deque(maxlen=self.app_config.n_layers)
            
            futures = []
            processed_count = 0

            # Use ThreadPoolExecutor for concurrent image processing
            # Use the max_workers value passed from the GUI config
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i, filename in enumerate(image_filenames_filtered):
                    if not self._is_running:
                        self.status_update.emit("Processing stopped by user.")
                        break

                    filepath = os.path.join(self.app_config.input_folder, filename)
                    base_filename_no_ext = os.path.splitext(filename)[0]
                    self.status_update.emit(f"Loading {filename} ({i + 1}/{total_images})")

                    # Load current image in the main thread (I/O bound)
                    current_binary_image, current_original_image = core.load_image(filepath)
                    if current_binary_image is None:
                        self.status_update.emit(f"Skipping unloadable image: {filename}")
                        continue

                    # Prepare parameters for the worker task
                    processing_core_params = {
                        'use_fixed_norm': self.app_config.use_fixed_norm,
                        'fixed_fade_distance': self.app_config.fixed_fade_distance,
                    }
                    
                    # Submit task to the thread pool
                    # Pass a copy of the current state of prior_binary_masks_cache
                    # The actual NumPy arrays are immutable in their content for the worker's read,
                    # so a shallow copy of the deque's list of references is sufficient.
                    future = executor.submit(
                        ImageProcessorThread._process_single_image_task,
                        filepath, # Not directly used by task, but useful for error reporting
                        base_filename_no_ext,
                        current_binary_image,
                        current_original_image,
                        collections.deque(prior_binary_masks_cache), # Pass a new deque copy
                        processing_core_params,
                        self.app_config.xy_blend_pipeline, # Pass the pipeline ops
                        self.app_config.output_folder,
                        self.app_config.debug_save
                    )
                    futures.append(future)

                    # Update cache for next iteration (main thread manages the deque)
                    prior_binary_masks_cache.append(current_binary_image)
                    
                    # Update progress in a non-blocking way for images submitted
                    self.progress_update.emit(int(((i + 1) / total_images) * 100))
                
                # Wait for all submitted tasks to complete and collect results/errors
                for future in concurrent.futures.as_completed(futures):
                    if not self._is_running: # Check stop flag while waiting
                        break
                    try:
                        output_path = future.result()
                        # self.status_update.emit(f"Saved: {os.path.basename(output_path)}")
                        processed_count += 1
                        # Update progress based on completed tasks, not submitted
                        # self.progress_update.emit(int((processed_count / total_images) * 100)) 
                        # The previous progress update (based on submission) is usually smoother
                    except Exception as exc:
                        # Log detailed traceback for debugging
                        import traceback
                        error_detail = f"Image processing task failed for {filepath}: {exc}\n{traceback.format_exc()}"
                        self.error_signal.emit(error_detail)
                        self._is_running = False # Stop further processing on first error

        except Exception as e:
            # Catch exceptions from the main thread loop (e.g., os.listdir)
            import traceback
            error_info = f"An error occurred in main thread: {e}\n\nTraceback:\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
            self._is_running = False

        finally:
            # Ensure thread pool is shut down if not already
            if 'executor' in locals():
                executor.shutdown(wait=True) # Ensure all tasks complete before thread exits

            if self._is_running: # Only emit complete if not stopped by error or user
                self.status_update.emit("Processing complete!")
            self.finished_signal.emit()

    def stop_processing(self):
        """Flags the thread to stop processing."""
        self._is_running = False


class ImageProcessorApp(QWidget):
    """The main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Stack Euclidean Distance Blending & XY Pipeline")
        self.settings = QSettings("YourCompany", "VoxelBlendApp") # Use QSettings for simple app-level settings (like window size)
        self.processor_thread = None
        self.init_ui()
        self.load_settings() # Load general settings (like folder paths)
        # Note: app_config handles its own load/save of complex pipeline settings

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Tab Widget for Main Settings and XY Blend Pipeline ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Main Processing Tab
        self.main_processing_tab = QWidget()
        main_processing_layout = QVBoxLayout(self.main_processing_tab)
        self.tab_widget.addTab(self.main_processing_tab, "Main Processing")

        # --- Input/Output Folders ---
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Input Folder:"))
        self.input_folder_edit = QLineEdit()
        self.input_folder_button = QPushButton("Browse...")
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit, "Select Input Image Folder"))
        folder_layout.addWidget(self.input_folder_edit)
        folder_layout.addWidget(self.input_folder_button)
        main_processing_layout.addLayout(folder_layout)

        folder_layout_out = QHBoxLayout()
        folder_layout_out.addWidget(QLabel("Output Folder:"))
        self.output_folder_edit = QLineEdit()
        self.output_folder_button = QPushButton("Browse...")
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit, "Select Output Image Folder"))
        folder_layout_out.addWidget(self.output_folder_edit)
        folder_layout_out.addWidget(self.output_folder_button)
        main_processing_layout.addLayout(folder_layout_out)

        # --- Processing Parameters ---
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("N Layers (look down):"))
        self.n_layers_edit = QLineEdit("3")
        self.n_layers_edit.setToolTip("Number of prior layers to check for overlap.")
        params_layout.addWidget(self.n_layers_edit)
        
        params_layout.addWidget(QLabel("Start Index:"))
        self.start_idx_edit = QLineEdit("0")
        self.start_idx_edit.setToolTip("Inclusive. Start processing from this image number (e.g., '1' for 001.png).")
        params_layout.addWidget(self.start_idx_edit)

        params_layout.addWidget(QLabel("Stop Index:"))
        self.stop_idx_edit = QLineEdit()
        self.stop_idx_edit.setToolTip("Inclusive. Stop at this image number. Leave blank for no limit.")
        params_layout.addWidget(self.stop_idx_edit)
        main_processing_layout.addLayout(params_layout)
        
        # --- NEW: Thread Count ---
        thread_count_layout = QHBoxLayout()
        thread_count_layout.addWidget(QLabel("Thread Count:"))
        self.thread_count_edit = QLineEdit(str(DEFAULT_NUM_WORKERS)) # Default value from config.py
        self.thread_count_edit.setFixedWidth(60)
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self)) # Min 1, max a reasonable number
        thread_count_layout.addWidget(self.thread_count_edit)
        thread_count_layout.addStretch(1)
        main_processing_layout.addLayout(thread_count_layout)


        # --- Receding Gradient Parameters (Renamed from Gradient Parameters) ---
        receding_gradient_params_layout = QHBoxLayout()
        self.fixed_normalization_checkbox = QCheckBox("Use Fixed Fade Distance")
        self.fixed_normalization_checkbox.setToolTip("If checked, gradient fades over a fixed pixel distance.")
        receding_gradient_params_layout.addWidget(self.fixed_normalization_checkbox)

        receding_gradient_params_layout.addWidget(QLabel("Fade Distance (px):"))
        self.fade_distance_edit = QLineEdit("10.0")
        self.fade_distance_edit.setToolTip("Pixel distance for the fade gradient. Used if 'Fixed Fade' is checked.")
        receding_gradient_params_layout.addWidget(self.fade_distance_edit)
        
        main_processing_layout.addLayout(receding_gradient_params_layout)

        # Debug Checkbox
        self.debug_checkbox = QCheckBox("Save Intermediate Debug Images")
        main_processing_layout.addWidget(self.debug_checkbox)

        main_processing_layout.addStretch(1) # Push content to top

        # --- XY Blend Pipeline Tab ---
        # We need to import pyside_xy_blend_tab and instantiate it here.
        # Assuming pyside_xy_blend_tab is in a separate file.
        from pyside_xy_blend_tab import XYBlendTab # Import it here to avoid circular dependencies if it imports ui_components

        self.xy_blend_tab = XYBlendTab(self) # Pass self (ImageProcessorApp) as parent_gui
        self.tab_widget.addTab(self.xy_blend_tab, "XY Blend Pipeline")

        # --- Controls and Status (moved to main_layout to be below tabs) ---
        button_layout = QHBoxLayout()
        self.start_stop_button = QPushButton("Start Processing")
        self.start_stop_button.setMinimumHeight(40)
        self.start_stop_button.clicked.connect(self.toggle_processing)
        button_layout.addWidget(self.start_stop_button)
        
        main_layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("background-color: #333; color: #00FF00; padding: 5px; border-radius: 3px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)


    def browse_folder(self, line_edit, caption):
        """Opens a dialog to select a folder and updates the line edit."""
        folder = QFileDialog.getExistingDirectory(self, caption, line_edit.text())
        if folder:
            line_edit.setText(folder)
            # Update config immediately
            if line_edit == self.input_folder_edit:
                config.input_folder = folder
            elif line_edit == self.output_folder_edit:
                config.output_folder = folder

    def load_settings(self):
        """Loads settings from QSettings and applies them to config and UI."""
        # Load simple window size/pos from QSettings
        self.resize(self.settings.value("window_size", self.size()))
        self.move(self.settings.value("window_position", self.pos()))

        # app_config is already loaded on module import (from its JSON file)
        # Now, populate UI elements from app_config
        self.input_folder_edit.setText(config.input_folder)
        self.output_folder_edit.setText(config.output_folder)
        self.n_layers_edit.setText(str(config.n_layers))
        self.start_idx_edit.setText(str(config.start_index) if config.start_index is not None else "")
        self.stop_idx_edit.setText(str(config.stop_index) if config.stop_index is not None else "")
        self.debug_checkbox.setChecked(config.debug_save)
        self.fixed_normalization_checkbox.setChecked(config.use_fixed_norm)
        self.fade_distance_edit.setText(str(config.fixed_fade_distance))
        self.thread_count_edit.setText(str(config.thread_count)) # NEW: Set thread count UI

        # Apply settings to the XY blend tab
        self.xy_blend_tab.apply_settings(config) # This will re-populate its pipeline list and details


    def save_settings(self):
        """Saves current UI settings to QSettings and app_config."""
        # Save simple window size/pos to QSettings
        self.settings.setValue("window_size", self.size())
        self.settings.setValue("window_position", self.pos())

        # Update app_config from simple UI elements first
        try:
            config.n_layers = int(self.n_layers_edit.text())
        except ValueError:
            config.n_layers = 3 # Default or last valid
        
        config.start_index = int(s) if (s := self.start_idx_edit.text()) else None
        config.stop_index = int(s) if (s := self.stop_idx_edit.text()) else None
        config.debug_save = self.debug_checkbox.isChecked()
        config.use_fixed_norm = self.fixed_normalization_checkbox.isChecked()
        
        try:
            config.fixed_fade_distance = float(self.fade_distance_edit.text())
        except ValueError:
            config.fixed_fade_distance = 10.0 # Default or last valid

        try: # NEW: Save thread count
            config.thread_count = int(self.thread_count_edit.text())
        except ValueError:
            config.thread_count = DEFAULT_NUM_WORKERS # Default or last valid
        config.thread_count = max(1, config.thread_count) # Ensure at least 1 thread

        config.input_folder = self.input_folder_edit.text()
        config.output_folder = self.output_folder_edit.text()

        # The xy_blend_tab automatically updates config.xy_blend_pipeline directly via its signals
        # No explicit call needed here for that.
        
        # Save the entire app_config to JSON
        config.save("app_config.json") # Save to default config file

    def closeEvent(self, event):
        """Saves settings on exit."""
        self.save_settings()
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop_processing()
            self.processor_thread.wait(5000) # Wait up to 5s for thread to finish gracefully
        event.accept()

    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop_processing()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
            self.status_label.setText("Status: Requesting stop...")
        else:
            self.start_processing()

    def start_processing(self):
        """Validates inputs and starts the processing thread."""
        try:
            # First, ensure config is up-to-date from UI elements
            self.save_settings() # This updates app_config and saves it

            # --- Input Validation (using updated config values) ---
            if not os.path.isdir(config.input_folder):
                raise ValueError("Input folder is not a valid directory.")
            if not os.path.isdir(config.output_folder):
                raise ValueError("Output folder is not a valid directory.")
            if config.n_layers < 0:
                raise ValueError("N Layers must be a non-negative integer.")
            if config.fixed_fade_distance <= 0:
                raise ValueError("Fade Distance must be a positive number.")
            # NEW: Validate thread count
            if config.thread_count <= 0:
                raise ValueError("Thread count must be at least 1.")

        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Input Error", str(e))
            return

        # --- Launch Thread ---
        # Pass the configured thread count to the ImageProcessorThread
        self.processor_thread = ImageProcessorThread(app_config=config, max_workers=config.thread_count)
        self.processor_thread.status_update.connect(self.update_status)
        self.processor_thread.progress_update.connect(self.progress_bar.setValue)
        self.processor_thread.error_signal.connect(self.show_error)
        self.processor_thread.finished_signal.connect(self.processing_finished)
        
        self.processor_thread.start()
        self.set_ui_enabled(False)
        self.start_stop_button.setText("Stop Processing")
        self.status_label.setText("Status: Starting...")
        self.progress_bar.setValue(0)

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot(str)
    def show_error(self, message):
        QMessageBox.critical(self, "Processing Error", message)
        self.processing_finished()

    @Slot()
    def processing_finished(self):
        self.status_label.setText("Status: Finished or Stopped.")
        self.start_stop_button.setText("Start Processing")
        self.set_ui_enabled(True)
        self.processor_thread = None

    def set_ui_enabled(self, enabled):
        """Toggles the enabled state of UI widgets."""
        # Enable/disable widgets on the Main Processing tab
        self.input_folder_edit.setEnabled(enabled)
        self.input_folder_button.setEnabled(enabled)
        self.output_folder_edit.setEnabled(enabled)
        self.output_folder_button.setEnabled(enabled)
        self.n_layers_edit.setEnabled(enabled)
        self.start_idx_edit.setEnabled(enabled)
        self.stop_idx_edit.setEnabled(enabled)
        self.fixed_normalization_checkbox.setEnabled(enabled)
        self.fade_distance_edit.setEnabled(enabled)
        self.debug_checkbox.setEnabled(enabled)
        self.thread_count_edit.setEnabled(enabled) # NEW: Enable/disable thread count edit
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.xy_blend_tab), enabled) # Enable/disable XY tab

        # The start/stop button is managed separately
        self.start_stop_button.setEnabled(True)