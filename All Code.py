# main_gui.py

import sys
import os
import threading # Added: For running the pipeline in a separate thread
from typing import Optional # Added: For type hinting

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget,
    QStatusBar, QLabel, QMessageBox, QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QIcon # For application icon

# Import application modules
from config import app_config as config
import lut_manager
import run_logger
import pipeline_runner

# Import GUI tabs
from pyside_file_io_tab import FileIOTab
from pyside_stacking_tab import StackingTab
from pyside_lut_tab import LutTab
from pyside_xy_blend_tab import XYBlendTab
from pyside_advanced_tab import AdvancedTab

class SuperStackerPysideGUI(QMainWindow):
    """
    Main GUI window for the Modular-Stacker application.
    Integrates all configuration tabs and orchestrates the backend pipeline.
    """
    # Define a signal for updating progress from non-GUI threads
    progress_update_signal = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular-Stacker")
        self.setGeometry(100, 100, 1000, 800) # Initial window size

        # Set application icon (optional)
        # self.setWindowIcon(QIcon("path/to/your/icon.png"))

        # Initialize global module references with the config instance
        lut_manager.set_config_reference(config)
        run_logger.set_config_reference(config)
        pipeline_runner.set_config_reference(config)

        # Set up the main UI components
        self._setup_ui()
        self._connect_signals()

        # Initialize the pipeline runner (will be created on demand or once here)
        self.runner: Optional[pipeline_runner.PipelineRunner] = None

        # Connect the progress update signal to the GUI slot
        self.progress_update_signal.connect(self._update_progress_bar)

        # Set the progress callback in the config for the runner
        config.progress_callback = self.progress_update_signal.emit
        config.stop_callback = self._check_stop_requested # Set stop callback for runner

        # Progress dialog for long-running operations
        self.progress_dialog: Optional[QProgressDialog] = None


    def _setup_ui(self):
        """Sets up the main window layout and tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create and add tabs
        self.file_io_tab = FileIOTab(self)
        self.tab_widget.addTab(self.file_io_tab, "File I/O")

        self.stacking_tab = StackingTab(self)
        self.tab_widget.addTab(self.stacking_tab, "Z-Stacking")

        self.lut_tab = LutTab(self)
        self.tab_widget.addTab(self.lut_tab, "Z-LUT")

        self.xy_blend_tab = XYBlendTab(self)
        self.tab_widget.addTab(self.xy_blend_tab, "XY Processing")

        self.advanced_tab = AdvancedTab(self)
        self.tab_widget.addTab(self.advanced_tab, "Advanced")

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("Ready")
        self.statusBar.addWidget(self.status_label)

    def _connect_signals(self):
        """Connects signals from tabs to main window slots."""
        self.file_io_tab.run_requested.connect(self._start_processing)
        self.file_io_tab.stop_requested.connect(self._stop_processing)
        self.file_io_tab.save_settings_requested.connect(self._save_settings)
        self.file_io_tab.load_settings_requested.connect(self._load_settings)
        
        # Connect LUT tab's lut_changed signal to update the status bar if needed
        self.lut_tab.lut_changed.connect(self._on_lut_changed)

    def _update_config_from_gui(self):
        """Collects all settings from GUI tabs and updates the global config."""
        try:
            # Collect settings from each tab
            file_io_settings = self.file_io_tab.get_config()
            stacking_settings = self.stacking_tab.get_config()
            lut_settings = self.lut_tab.get_config()
            xy_blend_settings = self.xy_blend_tab.get_config()
            advanced_settings = self.advanced_tab.get_config()

            # Update global config instance
            for key, value in file_io_settings.items():
                setattr(config, key, value)
            for key, value in stacking_settings.items():
                setattr(config, key, value)
            for key, value in lut_settings.items():
                setattr(config, key, value)
            for key, value in xy_blend_settings.items():
                setattr(config, key, value)
            for key, value in advanced_settings.items():
                setattr(config, key, value)
            
            # Ensure XYBlendOperation objects are correctly updated in config
            # (This should already be handled by XYBlendTab's internal logic)
            
            # Re-run post_init on config to apply any cross-field validations
            config.__post_init__()

            self.statusBar.showMessage("Settings updated from GUI.", 3000)
            return True
        except ValueError as e:
            QMessageBox.warning(self, "Configuration Error", f"Invalid input in GUI: {e}")
            self.statusBar.showMessage(f"Error: {e}", 5000)
            return False
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"An unexpected error occurred while updating settings: {e}")
            self.statusBar.showMessage(f"Error: {e}", 5000)
            return False

    def _apply_config_to_gui(self):
        """Applies settings from the global config to all GUI tabs."""
        self.file_io_tab.apply_settings(config)
        self.stacking_tab.apply_settings(config)
        self.lut_tab.apply_settings(config)
        self.xy_blend_tab.apply_settings(config)
        self.advanced_tab.apply_settings(config)
        self.statusBar.showMessage("Settings applied to GUI.", 3000)

    @Slot()
    def _start_processing(self):
        """Initiates the image processing pipeline."""
        if not self._update_config_from_gui():
            return # Configuration failed

        # Ensure LUT is updated in lut_manager before starting pipeline
        try:
            lut_manager.update_active_lut_from_config()
        except Exception as e:
            QMessageBox.critical(self, "LUT Error", f"Failed to prepare LUT for processing: {e}")
            self.statusBar.showMessage(f"Error preparing LUT: {e}", 5000)
            return

        # Disable run button, enable stop button
        self.file_io_tab.set_run_button_state(False)
        self.file_io_tab.set_stop_button_state(True)
        self.statusBar.showMessage("Processing started...", 0) # 0 means persistent

        # Initialize and run the pipeline in a separate thread
        # Pass the config instance to the runner
        self.runner = pipeline_runner.PipelineRunner()
        
        # Set up progress dialog
        self.progress_dialog = QProgressDialog("Processing Images...", "Cancel", 0, self.runner.total_output_stacks, self)
        self.progress_dialog.setWindowTitle("Modular-Stacker Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0) # Show immediately
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self._stop_processing)
        self.progress_dialog.show()

        # Start the pipeline in a new thread to keep GUI responsive
        self.pipeline_thread = threading.Thread(target=self.runner.run_pipeline, name="PipelineThread")
        self.pipeline_thread.daemon = True # Allow app to close if this thread is still running
        self.pipeline_thread.start()

        # Start a QTimer to periodically check if the pipeline thread is alive
        # and if processing is complete. This is safer than directly joining
        # the thread in the GUI thread.
        self.check_pipeline_timer = QTimer(self)
        self.check_pipeline_timer.timeout.connect(self._check_pipeline_status)
        self.check_pipeline_timer.start(100) # Check every 100 ms

    @Slot()
    def _stop_processing(self):
        """Signals the pipeline to stop."""
        if self.runner:
            self.runner.stop_pipeline()
            self.statusBar.showMessage("Stop requested. Waiting for pipeline to halt...", 0)
            self.file_io_tab.set_stop_button_state(False) # Disable stop button once requested
            # The _check_pipeline_status will handle re-enabling run button

    def _check_stop_requested(self) -> bool:
        """Callback for the runner to check if stop was requested."""
        return config.stop_requested # Check the flag in the global config

    @Slot(int, int)
    def _update_progress_bar(self, current: int, total: int):
        """Updates the progress dialog and status bar."""
        if self.progress_dialog:
            self.progress_dialog.setMaximum(total)
            self.progress_dialog.setValue(current)
            self.statusBar.showMessage(f"Processing: {current}/{total} stacks", 0)
            if current >= total:
                self.progress_dialog.setValue(total) # Ensure 100%
                self.progress_dialog.close() # Close dialog on completion
                self.statusBar.showMessage("Processing complete.", 5000)
                self._reset_gui_for_new_run()


    @Slot()
    def _check_pipeline_status(self):
        """Periodically checks the status of the pipeline thread."""
        if self.pipeline_thread and not self.pipeline_thread.is_alive():
            self.check_pipeline_timer.stop() # Stop checking
            print("Pipeline thread has finished.")
            if self.progress_dialog and self.progress_dialog.isVisible():
                self.progress_dialog.close() # Ensure dialog is closed

            # Check if processing completed successfully or was stopped/errored
            if config.stop_requested:
                self.statusBar.showMessage("Processing stopped by user or error.", 5000)
            else:
                # If progress_callback didn't fire 100% due to no output stacks, ensure message is consistent
                if self.runner and self.runner.total_output_stacks == 0:
                    self.statusBar.showMessage("No images to process based on current configuration.", 5000)
                else:
                    self.statusBar.showMessage("Processing complete.", 5000)
            
            self._reset_gui_for_new_run()

    def _reset_gui_for_new_run(self):
        """Resets GUI elements after a run completes or is stopped."""
        self.file_io_tab.set_run_button_state(True)
        self.file_io_tab.set_stop_button_state(False)
        config.stop_requested = False # Reset stop flag in config

    @Slot()
    def _save_settings(self):
        """Saves the current configuration to file."""
        if self._update_config_from_gui():
            config.save()
            QMessageBox.information(self, "Save Settings", "Current settings saved successfully.")
            self.statusBar.showMessage("Settings saved.", 3000)

    @Slot()
    def _load_settings(self):
        """Loads configuration from file and applies to GUI."""
        config.load()
        self._apply_config_to_gui()
        # Re-initialize lut_manager's active LUT based on newly loaded config
        try:
            lut_manager.update_active_lut_from_config()
            self.lut_tab._load_lut_to_text_edit(lut_manager.get_current_z_lut()) # Update LUT tab display
            self.lut_tab.lut_filepath_edit.setText(config.fixed_lut_path) # Update file path display
            self.lut_tab.lut_source_combo.setCurrentText(config.lut_source.capitalize()) # Update source combo
            self.lut_tab.lut_generation_type_combo.setCurrentText(config.lut_generation_type) # Update gen type combo
            # Re-apply initial state for LUT tab to ensure all its dynamic controls are correct
            self.lut_tab._apply_initial_state() 

        except Exception as e:
            QMessageBox.warning(self, "Load Settings Error", f"Failed to load LUT from configuration: {e}. Default LUT applied.")
            # Fallback to default LUT if loading fixed_lut_path failed
            lut_manager.set_current_z_lut(lut_manager.get_default_z_lut())
            self.lut_tab._load_lut_to_text_edit(lut_manager.get_default_z_lut())
            self.lut_tab.lut_filepath_edit.setText("")
            self.lut_tab.lut_source_combo.setCurrentText("Generated")
            self.lut_tab.lut_generation_type_combo.setCurrentText("linear")
            self.lut_tab._apply_initial_state() # Re-apply initial state for LUT tab


        QMessageBox.information(self, "Load Settings", "Settings loaded successfully.")
        self.statusBar.showMessage("Settings loaded.", 3000)

    @Slot()
    def _on_lut_changed(self):
        """Handle LUT changes from LutTab, e.g., update status bar."""
        self.statusBar.showMessage("Z-LUT updated.", 2000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SuperStackerPysideGUI()
    window.show()
    sys.exit(app.exec())




# config.py

"""
Defines the canonical Config data structure for the new Modular-Stacker pipeline.
All parameters are declared here with sensible defaults.
This class will be a singleton to ensure a single source of truth for settings.
"""

import os
import json
import math
from dataclasses import dataclass, field, fields
from typing import Optional, Callable, Dict, Any, List

@dataclass
class XYBlendOperation:
    """
    Represents a single XY blending/smoothing/sharpening/resizing operation in the pipeline.
    Contains the operation type and its specific parameters.
    """
    type: str = "none" # "none", "gaussian_blur", "bilateral_filter", "median_blur", "unsharp_mask", "resize"
    
    # Parameters for Gaussian Blur (relevant if type is "gaussian_blur")
    gaussian_ksize_x: int = 3
    gaussian_ksize_y: int = 3
    gaussian_sigma_x: float = 0.0
    gaussian_sigma_y: float = 0.0

    # Parameters for Bilateral Filter (relevant if type is "bilateral_filter")
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Parameters for Median Blur (relevant if type is "median_blur")
    median_ksize: int = 5

    # Parameters for Unsharp Masking (relevant if type is "unsharp_mask")
    unsharp_amount: float = 1.0
    unsharp_threshold: int = 0
    unsharp_blur_ksize: int = 5
    unsharp_blur_sigma: float = 0.0

    # Parameters for Resize operation (relevant if type is "resize")
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    resample_mode: str = "LANCZOS4" # NEAREST, BILINEAR, BICUBIC, LANCZOS4, AREA

    def __post_init__(self):
        # Ensure kernel sizes are odd and positive where applicable
        if self.type in ["gaussian_blur", "median_blur", "unsharp_mask"]:
            self.gaussian_ksize_x = self._ensure_odd_positive(self.gaussian_ksize_x)
            self.gaussian_ksize_y = self._ensure_odd_positive(self.gaussian_ksize_y)
            self.median_ksize = self._ensure_odd_positive(self.median_ksize)
            self.unsharp_blur_ksize = self._ensure_odd_positive(self.unsharp_blur_ksize)

    def _ensure_odd_positive(self, val: int) -> int:
        """Ensures a kernel size is an odd positive integer, adjusting if necessary."""
        if val <= 0:
            return 1
        return val if val % 2 != 0 else val + 1

    def to_dict(self) -> Dict[str, Any]:
        """Converts the XYBlendOperation object to a dictionary for serialization."""
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'XYBlendOperation':
        """Creates an XYBlendOperation instance from a dictionary."""
        op_type = d.get("type", "none")
        params = d.get("params", {})
        
        # Create a default instance and then override its attributes with loaded params
        instance = cls(type=op_type)
        for key, value in params.items():
            if hasattr(instance, key):
                # Special handling for resize_width/height: convert 0 to None if that's the default behavior
                if key in ["resize_width", "resize_height"] and value == 0:
                    setattr(instance, key, None)
                else:
                    setattr(instance, key, value)
        instance.__post_init__() # Ensure validation logic runs after setting attributes
        return instance


@dataclass
class Config:
    """
    Application configuration for Modular-Stacker.
    This is a dataclass defining all configuration parameters.
    The singleton pattern is managed externally at the module level.
    """
    _config_file: str = "modular_stacker_config.json"

    # --- I/O settings ---
    input_dir: str = ""  # Path to folder of numbered PNGs
    file_pattern: str = "*.png" # Pattern to match input files (e.g., "*.png", "image_*.tif")
    output_dir: str = "" # Where to write output PNGs
    run_log_file: str = "modular_stacker_runs.log" # File to log run settings and timestamps
    
    # --- Stacking (blend) parameters ---
    primary: int = 3     # Number of primary layers per output
    radius: int = 1      # Number of neighbors on each side
    blend_mode: str = "gaussian" # Weight curve: gaussian, linear, flat, exp_decay, cosine,
                                 # binary_contour, gradient_contour, z_column_lift, z_contour_interp
    blend_param: float = 1.0     # σ or exponent for blend_mode
    directional_blend: bool = False # Enable Z-bias directionality
    dir_sigma: float = 1.0       # σ for directional bias

    # --- LUT Management (replaces old remap, gamma, normalize, clamp, preserve_black, floor/ceil) ---
    # This defines how the active Z-LUT is generated or loaded.
    # The actual LUT data is managed by the lut_manager module.
    lut_source: str = "generated" # "generated" or "file"
    fixed_lut_path: str = ""       # Path to a custom Z-LUT JSON file if lut_source is "file"
    
    # Parameters for algorithmic LUT generation (if lut_source is "generated")
    lut_generation_type: str = "linear" # "linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard"
    gamma_value: float = 1.0            # For "gamma" generation type
    linear_min_input: int = 0           # For "linear" generation type
    linear_max_output: int = 255        # For "linear" generation type
    s_curve_contrast: float = 0.5       # For "s_curve" generation type (0.0 to 1.0)
    log_param: float = 10.0             # For "log" generation type (e.g., base for log, or strength)
    exp_param: float = 2.0              # For "exp" generation type (e.g., exponent)
    sqrt_param: float = 1.0             # For "sqrt" generation type (e.g., strength, currently simplified)
    rodbard_param: float = 1.0          # For "rodbard" generation type (e.g., strength, currently simplified)

    # --- XY Blending / Smoothing / Sharpening / Resizing Pipeline ---
    # This is a list of XYBlendOperation objects, allowing for a sequence of operations.
    xy_blend_pipeline: List[XYBlendOperation] = field(default_factory=lambda: [XYBlendOperation()])

    # --- Output & threading (global output resize removed, now per-slot in XY pipeline) ---
    threads: int = field(default_factory=lambda: os.cpu_count() or 4) # number of worker threads
    pad_filenames: bool = False          # zero-pad output filenames
    pad_length: int = 4                  # digits for filename padding

    # --- Advanced toggles ---
    integer_mode: bool = True            # use integer engine (always true for new app)
    cap_layers: int = 0                  # extra blank “cap” slices
    resume_from: int = 1                 # first stack index to process
    stop_at: int = 999999                # last stack index to process
    scale_bits: int = 12                 # Fixed-point scaling bits for integer mode (for internal 16-bit processing)

    # --- Zarr-Engine Settings ---
    zarr_store: str = ""                 # Path to Zarr store (future use)
    zarr_cache_chunks: int = 1           # Number of Zarr chunks to cache (future use)

    # --- Callbacks (not serialized, handled by runtime) ---
    stop_requested: bool = field(default=False, compare=False) # internal cancellation flag
    progress_callback: Optional[Callable[[int, int], None]] = field(default=None, compare=False) # receives (current, total) tuples
    stop_callback: Optional[Callable[[], bool]] = field(default=None, compare=False) # if set, can abort the run

    # --- Binary/Gradient contour thresholds ---
    binary_threshold: int = 128          # B/W cutoff for shadow/overhang/contour
    gradient_threshold: int = 128        # Cutoff for gradient-based contour

    # --- Other Advanced Toggles (now only these, clamp/preserve_black removed) ---
    top_surface_smoothing: bool = False # boost trailing-edge pixels
    top_surface_strength: float = 0.0 # 0.0–1.0 strength of surface smooth
    gradient_smooth: bool = True         # enable XY gradient smoothing (for float mode, but kept for consistency)
    gradient_blend_strength: float = 0.0 # 0.0–1.0 strength of gradient smooth

    def __post_init__(self):
        # Ensure threads is at least 1
        if self.threads < 1:
            self.threads = 1
        
        # Apply post_init logic for each XYBlendOperation in the pipeline
        # Ensure xy_blend_pipeline is a list of XYBlendOperation objects
        if not isinstance(self.xy_blend_pipeline, list):
            self.xy_blend_pipeline = [XYBlendOperation()] # Default to one 'none' operation
        else:
            # Ensure elements are XYBlendOperation instances, converting from dicts if necessary
            self.xy_blend_pipeline = [
                op if isinstance(op, XYBlendOperation) else XYBlendOperation.from_dict(op)
                for op in self.xy_blend_pipeline
            ]
        for op in self.xy_blend_pipeline:
            op.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Config object to a dictionary, suitable for JSON serialization."""
        data = {}
        for f in fields(self):
            # Skip non-serializable fields like Callables and internal singleton state
            if f.name in ["_config_file", "progress_callback", "stop_callback"]: # Removed _instance, _initialized
                continue
            value = getattr(self, f.name)
            if f.name == "xy_blend_pipeline":
                data[f.name] = [op.to_dict() for op in value]
            else:
                data[f.name] = value
        return data

    def load(self) -> None:
        """Loads configuration from the JSON file if it exists."""
        print("Config: Attempting to load config from file...")
        loaded_data = {}
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Error decoding config file '{self._config_file}': {e}. Using default settings.")
            except Exception as e:
                print(f"Warning: An unexpected error occurred while loading config '{self._config_file}': {e}. Using default settings.")
        
        # Apply loaded data, overriding current defaults set by dataclass __init__
        for key, value in loaded_data.items():
            if hasattr(self, key): # Only set attributes that are defined in the dataclass
                if key == "xy_blend_pipeline":
                    # Reconstruct XYBlendOperation objects from dicts
                    if isinstance(value, list):
                        # Use object.__setattr__ to bypass any custom setters if they existed
                        object.__setattr__(self, key, [XYBlendOperation.from_dict(op_dict) for op_dict in value])
                    else:
                        print(f"Warning: Expected list for '{key}' in config, but got {type(value)}. Retaining default.")
                elif key in ["resize_width", "resize_height"] and value == 0:
                    object.__setattr__(self, key, None) # Convert 0 to None for Optional[int] fields
                else:
                    object.__setattr__(self, key, value)
            else:
                print(f"Warning: Loaded config contains unknown key '{key}'. Skipping.")
        
        print(f"Config: After loading, xy_blend_pipeline exists: {'xy_blend_pipeline' in self.__dict__}")
        # __post_init__ is called automatically by the dataclass after super().__init__()
        # and after all default fields are set. We do NOT call it explicitly here.

    def save(self) -> None:
        """Saves the current configuration to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self._config_file) or ".", exist_ok=True)
            with open(self._config_file, 'w', encoding='utf-8') as f:
                serializable_data = self.to_dict()
                json.dump(serializable_data, f, indent=4)
        except Exception as e:
            print(f"Error saving config file '{self._config_file}': {e}")

# Global instance management for Config
_app_config_instance: Optional[Config] = None

def get_app_config() -> Config:
    """
    Returns the singleton instance of the Config.
    Initializes it if it doesn't exist.
    """
    global _app_config_instance
    if _app_config_instance is None:
        print("Config: Creating new singleton instance.")
        _app_config_instance = Config()
        _app_config_instance.load() # Load settings from file after creation
    return _app_config_instance

# Global instance of the configuration
app_config = get_app_config()




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



# image_utils.py

"""
Provides general image utilities for Modular-Stacker.
(Currently minimal, as many functions have been absorbed by other modules).
"""

import numpy as np
import cv2 # Still needed for image loading/saving in other modules, but not directly used here for common ops.

# This module is now intentionally minimal.
# Functions like apply_black_mask, normalize_image, and resize_image
# have been moved or their responsibilities absorbed by the LUT and XY pipeline.

# Example of a utility function that *might* be needed here in the future:
# def convert_to_grayscale_if_needed(image: np.ndarray) -> np.ndarray:
#     """Converts a color image to grayscale if it's not already."""
#     if len(image.shape) == 3 and image.shape[2] == 3: # Check for 3-channel BGR
#         return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return image # Already grayscale or single channel

# No specific content needed for this module based on current plan,
# but keeping the file for organizational purposes.

if __name__ == '__main__':
    print("--- Image Utilities Module Test (Minimal) ---")
    print("This module is currently minimal, its core functionalities have been moved.")
    # Example of a dummy image for potential future tests
    dummy_image = np.zeros((50, 50), dtype=np.uint8)
    print(f"Dummy image created: shape={dummy_image.shape}, dtype={dummy_image.dtype}")



# lut_manager.py

"""
Manages the active Z-axis Look-Up Table (LUT) for Modular-Stacker.
Provides functions to generate LUTs based on various algorithms (linear, gamma, S-curve, etc.)
or load them from a file. The active LUT is a global singleton.
"""

import numpy as np
import json
import os
import math
from typing import Optional, Any # Import Any here

# Default Z Remapping LUT (linear 0-255)
_DEFAULT_Z_REMAP_LUT_ARRAY = np.arange(256, dtype=np.uint8)

# Currently Active Z Remapping LUT
_current_z_remap_lut: np.ndarray = _DEFAULT_Z_REMAP_LUT_ARRAY.copy()

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

def get_default_z_lut() -> np.ndarray:
    """Returns a copy of the default Z-remapping LUT (linear)."""
    return _DEFAULT_Z_REMAP_LUT_ARRAY.copy()

def get_current_z_lut() -> np.ndarray:
    """Returns the currently active Z-remapping LUT."""
    return _current_z_remap_lut.copy()

def set_current_z_lut(new_lut: np.ndarray):
    """Sets the currently active Z-remapping LUT.
    Args:
        new_lut (np.ndarray): A 256-entry NumPy array of dtype uint8.
    """
    if not isinstance(new_lut, np.ndarray) or new_lut.dtype != np.uint8 or new_lut.shape != (256,):
        raise ValueError("New LUT must be a 256-entry NumPy array of dtype uint8.")
    global _current_z_remap_lut
    _current_z_remap_lut = new_lut.copy()

def apply_z_lut(image_array: np.ndarray) -> np.ndarray:
    """
    Applies the currently active Z-REMAP_LUT to an 8-bit grayscale image (NumPy array).
    Args:
        image_array (np.ndarray): An 8-bit grayscale NumPy array (uint8).
                                  Expected values are 0-255.
    Returns:
        np.ndarray: A new NumPy array with the LUT applied,
                    remapped to 0-255 uint8 values.
    """
    if image_array.dtype != np.uint8:
        # If input is float, convert to uint8 before applying LUT
        # This assumes the float values are in the 0-255 range or need to be scaled
        # If the float image is normalized 0-1, it should be scaled to 0-255 first
        # before calling this function.
        raise TypeError("Input image_array for apply_z_lut must be of type np.uint8.")
    
    # Apply the LUT using direct indexing (most efficient way for 0-255 range)
    remapped_array = _current_z_remap_lut[image_array]
    
    return remapped_array

def save_lut(filepath: str, lut_array: np.ndarray):
    """Saves a LUT array to a JSON file."""
    if lut_array.dtype != np.uint8 or lut_array.shape != (256,):
        raise ValueError("LUT must be a 256-entry NumPy array of dtype uint8 to save.")
    
    try:
        with open(filepath, 'w') as f:
            # Convert NumPy array to Python list for JSON serialization
            json.dump(lut_array.tolist(), f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to save LUT to '{filepath}': {e}")

def load_lut(filepath: str) -> np.ndarray:
    """Loads a LUT array from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"LUT file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            lut_list = json.load(f)
        
        # Validate and convert to NumPy array
        if not isinstance(lut_list, list) or len(lut_list) != 256:
            raise ValueError("Invalid LUT file format: Expected a list of 256 numbers.")
        
        loaded_lut = np.array(lut_list, dtype=np.uint8)
        return loaded_lut
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in LUT file '{filepath}': {e}")
    except Exception as e:
        raise IOError(f"Failed to load LUT from '{filepath}': {e}")

# --- Algorithmic LUT Generation Functions ---

def generate_linear_lut(min_input: int, max_output: int) -> np.ndarray:
    """Generates a linear LUT that maps input range [0, 255] to [min_input, max_output]."""
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        # Scale input i (0-255) to a normalized 0-1 range
        normalized_input = i / 255.0
        # Then map this normalized value to the desired output range [min_input, max_output]
        # This assumes min_input and max_output are within 0-255
        output_val = min_input + normalized_input * (max_output - min_input)
        lut[i] = np.clip(output_val, 0, 255) # Ensure output is within 0-255
    return lut.astype(np.uint8)

def generate_gamma_lut(gamma_value: float) -> np.ndarray:
    """Generates a gamma correction LUT."""
    if gamma_value <= 0:
        raise ValueError("Gamma value must be positive.")
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized_input = i / 255.0
        output_val = np.power(normalized_input, 1.0 / gamma_value) * 255.0
        lut[i] = np.clip(output_val, 0, 255)
    return lut.astype(np.uint8)

def generate_s_curve_lut(contrast: float) -> np.ndarray:
    """Generates an S-curve (contrast) LUT."""
    if not (0.0 <= contrast <= 1.0):
        raise ValueError("Contrast must be between 0.0 and 1.0.")
    
    lut = np.zeros(256, dtype=np.float32)
    x = np.linspace(0, 1, 256) # Normalized input values

    # Adjust the S-curve formula based on contrast
    # A common S-curve formula uses a logistic function or a power function around a midpoint.
    # For simplicity, we can use a power function based on contrast.
    # Higher contrast makes the curve steeper, lower makes it flatter.
    
    # Example S-curve approximation using a piecewise power function
    midpoint = 0.5
    if contrast == 0.0: # Linear
        y = x
    elif contrast == 1.0: # Max contrast (hard clip)
        y = np.where(x < midpoint, 0.0, 1.0)
    else:
        # A more flexible S-curve can be achieved with a cubic or similar function
        # For simplicity, we'll use a power-based approach for now, similar to what's often seen.
        # This one is more like a gamma curve applied to halves.
        gamma_factor = 1.0 / (1.0 + contrast * 4) # Adjust this factor for desired curve
        y = np.where(x < midpoint,
                     np.power(x / midpoint, gamma_factor) * midpoint,
                     1.0 - np.power((1.0 - x) / midpoint, gamma_factor) * midpoint)
        
    lut = np.clip(y * 255.0, 0, 255)
    return lut.astype(np.uint8)

def generate_log_lut(param: float) -> np.ndarray:
    """Generates a logarithmic LUT."""
    if param <= 0:
        raise ValueError("Log parameter must be positive.")
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized_input = i / 255.0
        # Avoid log(0)
        if normalized_input == 0:
            output_val = 0.0
        else:
            output_val = np.log1p(normalized_input * param) / np.log1p(param) * 255.0
        lut[i] = np.clip(output_val, 0, 255)
    return lut.astype(np.uint8)

def generate_exp_lut(param: float) -> np.ndarray:
    """Generates an exponential LUT."""
    if param <= 0:
        raise ValueError("Exp parameter must be positive.")
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized_input = i / 255.0
        output_val = np.power(normalized_input, param) * 255.0
        lut[i] = np.clip(output_val, 0, 255)
    return lut.astype(np.uint8)

def generate_sqrt_lut(param: float) -> np.ndarray:
    """Generates a square root LUT."""
    if param <= 0: # Param currently unused, but for future scaling
        param = 1.0
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        normalized_input = i / 255.0
        output_val = np.sqrt(normalized_input) * 255.0 # Simple sqrt
        lut[i] = np.clip(output_val, 0, 255)
    return lut.astype(np.uint8)

def generate_rodbard_lut(param: float) -> np.ndarray:
    """Generates an ACES-style Rodbard contrast LUT."""
    # Parameters are constants tuned for specific curve. 'param' can be used for scaling.
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        x = i / 255.0 # Normalized input
        num = x * (a * x + b)
        den = x * (c * x + d) + e
        y = np.clip(num / den, 0.0, 1.0)
        lut[i] = np.clip(y * 255.0, 0, 255)
    return lut.astype(np.uint8)


def update_active_lut_from_config():
    """
    Updates the global active LUT based on the current settings in the Config.
    This should be called whenever config settings related to LUT generation change.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in lut_manager. Cannot update active LUT.")
        return

    cfg = _config_ref # Use the globally set config instance

    if cfg.lut_source == "file":
        if cfg.fixed_lut_path and os.path.exists(cfg.fixed_lut_path):
            try:
                loaded_lut = load_lut(cfg.fixed_lut_path)
                set_current_z_lut(loaded_lut)
                print(f"LUT Manager: Loaded LUT from file: {cfg.fixed_lut_path}")
            except Exception as e:
                print(f"Error loading LUT from file '{cfg.fixed_lut_path}': {e}. Falling back to default linear LUT.")
                set_current_z_lut(get_default_z_lut())
        else:
            print("LUT Manager: Fixed LUT path not specified or file not found. Falling back to default linear LUT.")
            set_current_z_lut(get_default_z_lut())
    elif cfg.lut_source == "generated":
        generated_lut = None
        if cfg.lut_generation_type == "linear":
            generated_lut = generate_linear_lut(cfg.linear_min_input, cfg.linear_max_output)
        elif cfg.lut_generation_type == "gamma":
            generated_lut = generate_gamma_lut(cfg.gamma_value)
        elif cfg.lut_generation_type == "s_curve":
            generated_lut = generate_s_curve_lut(cfg.s_curve_contrast)
        elif cfg.lut_generation_type == "log":
            generated_lut = generate_log_lut(cfg.log_param)
        elif cfg.lut_generation_type == "exp":
            generated_lut = generate_exp_lut(cfg.exp_param)
        elif cfg.lut_generation_type == "sqrt":
            generated_lut = generate_sqrt_lut(cfg.sqrt_param)
        elif cfg.lut_generation_type == "rodbard":
            generated_lut = generate_rodbard_lut(cfg.rodbard_param)
        else:
            print(f"Warning: Unknown LUT generation type '{cfg.lut_generation_type}'. Falling back to default linear LUT.")
            generated_lut = get_default_z_lut()
        
        if generated_lut is not None:
            set_current_z_lut(generated_lut)
            print(f"LUT Manager: Generated '{cfg.lut_generation_type}' LUT.")
        else:
            print("LUT Manager: Failed to generate LUT. Falling back to default linear LUT.")
            set_current_z_lut(get_default_z_lut())
    else:
        print(f"Warning: Unknown LUT source '{cfg.lut_source}'. Falling back to default linear LUT.")
        set_current_z_lut(get_default_z_lut())

# Initial setup: ensure a default LUT is active
set_current_z_lut(get_default_z_lut())

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- LUT Manager Module Test ---")
    
    # Dummy Config for testing
    class MockConfig:
        def __init__(self):
            self.lut_source = "generated"
            self.lut_generation_type = "linear"
            self.gamma_value = 2.2
            self.linear_min_input = 50
            self.linear_max_output = 200
            self.s_curve_contrast = 0.7
            self.log_param = 50.0
            self.exp_param = 0.5
            self.sqrt_param = 1.0 # Not actively used in sqrt generation, but for consistency
            self.rodbard_param = 1.0 # Not actively used in rodbard generation, but for consistency
            self.fixed_lut_path = ""
    
    mock_config = MockConfig()
    set_config_reference(mock_config) # Set the mock config

    # Test linear LUT generation
    mock_config.lut_generation_type = "linear"
    mock_config.linear_min_input = 50
    mock_config.linear_max_output = 200
    update_active_lut_from_config()
    linear_lut = get_current_z_lut()
    print(f"\nLinear LUT (first 10): {linear_lut[:10]}")
    print(f"Linear LUT (last 10): {linear_lut[-10:]}")
    
    # Test gamma LUT generation
    mock_config.lut_generation_type = "gamma"
    mock_config.gamma_value = 0.5 # Brightening gamma
    update_active_lut_from_config()
    gamma_lut = get_current_z_lut()
    print(f"\nGamma LUT (gamma=0.5, first 10): {gamma_lut[:10]}")
    print(f"Gamma LUT (gamma=0.5, last 10): {gamma_lut[-10:]}")

    # Test S-curve LUT generation
    mock_config.lut_generation_type = "s_curve"
    mock_config.s_curve_contrast = 0.8
    update_active_lut_from_config()
    s_curve_lut = get_current_z_lut()
    print(f"\nS-Curve LUT (contrast=0.8, first 10): {s_curve_lut[:10]}")
    print(f"S-Curve LUT (contrast=0.8, last 10): {s_curve_lut[-10:]}")

    # Test saving and loading a LUT
    test_file = "test_generated_lut.json"
    try:
        save_lut(test_file, linear_lut)
        print(f"\nGenerated linear LUT saved to {test_file}")
        loaded_lut = load_lut(test_file)
        print(f"Loaded LUT from {test_file} (first 10): {loaded_lut[:10]}")
        if np.array_equal(linear_lut, loaded_lut):
            print("Loaded LUT matches saved LUT.")
        else:
            print("Loaded LUT DOES NOT match saved LUT.")
    except Exception as e:
        print(f"Error during LUT file operations: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up {test_file}")

    # Test applying LUT to a dummy image
    dummy_image = np.array([[0, 50, 100, 150, 200, 255],
                            [10, 60, 110, 160, 210, 240]], dtype=np.uint8)
    
    mock_config.lut_generation_type = "linear"
    mock_config.linear_min_input = 0
    mock_config.linear_max_output = 255
    update_active_lut_from_config() # Ensure linear LUT is active
    
    remapped_image = apply_z_lut(dummy_image)
    print("\nOriginal dummy image:\n", dummy_image)
    print("Remapped dummy image (linear 0-255):\n", remapped_image)

    mock_config.lut_generation_type = "gamma"
    mock_config.gamma_value = 2.2 # Darkening gamma
    update_active_lut_from_config() # Ensure gamma LUT is active
    
    remapped_image_gamma = apply_z_lut(dummy_image)
    print("\nRemapped dummy image (gamma=2.2):\n", remapped_image_gamma)


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





# pyside_advanced_tab.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QSpinBox, QPushButton, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator

from config import app_config as config, Config # Import the global config instance and Config class

class AdvancedTab(QWidget):
    """
    PySide6 tab for advanced settings, including threading, resume/stop at,
    and other miscellaneous toggles.
    Zarr-engine settings are present but disabled for this version.
    """
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config # Use the global config instance

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config
        self._apply_initial_state() # Ensure initial state is correctly set

    def _setup_ui(self):
        """Sets up the widgets and layout for the Advanced tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Performance & Output Group ---
        perf_output_group = QGroupBox("Performance & Run Control")
        perf_output_layout = QVBoxLayout(perf_output_group)
        perf_output_layout.setContentsMargins(10, 10, 10, 10)
        perf_output_layout.setSpacing(5)

        # Threads
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Threads:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, os.cpu_count() or 4) # Max threads based on CPU count
        threads_layout.addWidget(self.threads_spin)
        threads_layout.addStretch(1)
        perf_output_layout.addLayout(threads_layout)

        # Resume From / Stop At
        resume_stop_layout = QHBoxLayout()
        resume_stop_layout.addWidget(QLabel("Resume From Stack Index:"))
        self.resume_from_edit = QLineEdit()
        self.resume_from_edit.setFixedWidth(60)
        self.resume_from_edit.setValidator(QIntValidator(1, 999999, self))
        resume_stop_layout.addWidget(self.resume_from_edit)

        resume_stop_layout.addWidget(QLabel("Stop At Stack Index:"))
        self.stop_at_edit = QLineEdit()
        self.stop_at_edit.setFixedWidth(60)
        self.stop_at_edit.setValidator(QIntValidator(1, 999999, self))
        resume_stop_layout.addWidget(self.stop_at_edit)
        resume_stop_layout.addStretch(1)
        perf_output_layout.addLayout(resume_stop_layout)

        # Cap Layers
        cap_layers_layout = QHBoxLayout()
        cap_layers_layout.addWidget(QLabel("Cap Layers (Blank Slices):"))
        self.cap_layers_spin = QSpinBox()
        self.cap_layers_spin.setRange(0, 100) # Reasonable max for cap layers
        cap_layers_layout.addWidget(self.cap_layers_spin)
        cap_layers_layout.addStretch(1)
        perf_output_layout.addLayout(cap_layers_layout)

        # Scale Bits (for internal integer precision)
        scale_bits_layout = QHBoxLayout()
        scale_bits_layout.addWidget(QLabel("Integer Scale Bits:"))
        self.scale_bits_spin = QSpinBox()
        self.scale_bits_spin.setRange(8, 16) # Typically 8 to 16 bits for precision
        self.scale_bits_spin.setToolTip("Number of bits for internal fixed-point integer precision (e.g., 12 for 12-bit precision).")
        scale_bits_layout.addWidget(self.scale_bits_spin)
        scale_bits_layout.addStretch(1)
        perf_output_layout.addLayout(scale_bits_layout)

        main_layout.addWidget(perf_output_group)

        # --- Zarr-Engine Settings (Present but Disabled) ---
        zarr_group = QGroupBox("Zarr-Engine Settings (Future Use)")
        zarr_layout = QVBoxLayout(zarr_group)
        zarr_layout.setContentsMargins(10, 10, 10, 10)
        zarr_layout.setSpacing(5)

        zarr_store_layout = QHBoxLayout()
        zarr_store_layout.addWidget(QLabel("Zarr Store Path:"))
        self.zarr_store_edit = QLineEdit()
        self.zarr_store_edit.setEnabled(False) # Grayed out for now
        zarr_store_layout.addWidget(self.zarr_store_edit)
        self.zarr_store_browse_btn = QPushButton("Browse...")
        self.zarr_store_browse_btn.setEnabled(False) # Grayed out for now
        zarr_store_layout.addWidget(self.zarr_store_browse_btn)
        zarr_layout.addLayout(zarr_store_layout)

        zarr_cache_layout = QHBoxLayout()
        zarr_cache_layout.addWidget(QLabel("Cache Chunks:"))
        self.zarr_cache_spin = QSpinBox()
        self.zarr_cache_spin.setRange(1, 32)
        self.zarr_cache_spin.setEnabled(False) # Grayed out for now
        zarr_cache_layout.addWidget(self.zarr_cache_spin)
        zarr_cache_layout.addStretch(1)
        zarr_layout.addLayout(zarr_cache_layout)

        main_layout.addWidget(zarr_group)

        # --- Other Advanced Toggles ---
        other_toggles_group = QGroupBox("Other Advanced Toggles")
        other_toggles_layout = QVBoxLayout(other_toggles_group)
        other_toggles_layout.setContentsMargins(10, 10, 10, 10)
        other_toggles_layout.setSpacing(5)

        # Top Surface Smoothing
        top_surface_layout = QHBoxLayout()
        self.top_surface_smoothing_check = QCheckBox("Top Surface Smoothing")
        top_surface_layout.addWidget(self.top_surface_smoothing_check)

        top_surface_layout.addWidget(QLabel("Strength:"))
        self.top_surface_strength_edit = QLineEdit()
        self.top_surface_strength_edit.setFixedWidth(60)
        self.top_surface_strength_edit.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        top_surface_layout.addWidget(self.top_surface_strength_edit)
        top_surface_layout.addStretch(1)
        other_toggles_layout.addLayout(top_surface_layout)

        # Gradient Smooth (kept for consistency, though integer mode might not use directly)
        gradient_smooth_layout = QHBoxLayout()
        self.gradient_smooth_check = QCheckBox("Gradient Smooth")
        gradient_smooth_layout.addWidget(self.gradient_smooth_check)

        gradient_smooth_layout.addWidget(QLabel("Blend Strength:"))
        self.gradient_blend_strength_edit = QLineEdit()
        self.gradient_blend_strength_edit.setFixedWidth(60)
        self.gradient_blend_strength_edit.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        gradient_smooth_layout.addWidget(self.gradient_blend_strength_edit)
        gradient_smooth_layout.addStretch(1)
        other_toggles_layout.addLayout(gradient_smooth_layout)

        main_layout.addWidget(other_toggles_group)
        main_layout.addStretch(1)


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        # Zarr browse button (disabled for now, but connect for future)
        self.zarr_store_browse_btn.clicked.connect(self._browse_zarr_store)

        # Connect top surface smoothing checkbox to enable/disable strength edit
        self.top_surface_smoothing_check.stateChanged.connect(self._update_top_surface_controls)
        # Connect gradient smooth checkbox to enable/disable blend strength edit
        self.gradient_smooth_check.stateChanged.connect(self._update_gradient_smooth_controls)

        # Input validation for numeric fields
        self.resume_from_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.resume_from_edit, text, int))
        self.stop_at_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.stop_at_edit, text, int))
        self.top_surface_strength_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.top_surface_strength_edit, text, float))
        self.gradient_blend_strength_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.gradient_blend_strength_edit, text, float))


    def _validate_numeric_input(self, line_edit: QLineEdit, text: str, data_type: type):
        """
        Validates if the input text can be converted to the specified data_type.
        If not, applies red border.
        """
        if not text:
            line_edit.setStyleSheet("") # Clear any error styling for empty string
            return

        try:
            if data_type == int:
                val = int(text)
            elif data_type == float:
                val = float(text)
            line_edit.setStyleSheet("") # Clear any error styling
        except ValueError:
            line_edit.setStyleSheet("border: 1px solid red;")

    def _apply_initial_state(self):
        """Applies initial control states based on default config."""
        self._update_top_surface_controls(Qt.Checked if self.config.top_surface_smoothing else Qt.Unchecked)
        self._update_gradient_smooth_controls(Qt.Checked if self.config.gradient_smooth else Qt.Unchecked)

    def _browse_zarr_store(self):
        """Opens a directory dialog for Zarr store path selection."""
        # This function is currently disabled in the UI
        dir_path = QFileDialog.getExistingDirectory(self, "Select Zarr Store Folder", self.zarr_store_edit.text())
        if dir_path:
            self.zarr_store_edit.setText(dir_path)

    def _update_top_surface_controls(self, state: int):
        """Enables/disables top surface strength edit based on checkbox state."""
        is_checked = (state == Qt.Checked)
        self.top_surface_strength_edit.setEnabled(is_checked)

    def _update_gradient_smooth_controls(self, state: int):
        """Enables/disables gradient blend strength edit based on checkbox state."""
        is_checked = (state == Qt.Checked)
        self.gradient_blend_strength_edit.setEnabled(is_checked)


    def get_config(self) -> dict:
        """Collects current settings from this tab's widgets."""
        config_data = {}
        config_data["threads"] = self.threads_spin.value()
        config_data["resume_from"] = int(self.resume_from_edit.text()) if self.resume_from_edit.text() else 1
        config_data["stop_at"] = int(self.stop_at_edit.text()) if self.stop_at_edit.text() else 999999
        config_data["cap_layers"] = self.cap_layers_spin.value()
        config_data["scale_bits"] = self.scale_bits_spin.value()

        # Zarr settings (disabled in UI, but collect if values exist)
        config_data["zarr_store"] = self.zarr_store_edit.text()
        config_data["zarr_cache_chunks"] = self.zarr_cache_spin.value()

        # Other Advanced Toggles
        config_data["top_surface_smoothing"] = self.top_surface_smoothing_check.isChecked()
        config_data["top_surface_strength"] = float(self.top_surface_strength_edit.text()) if self.top_surface_strength_edit.text() else 0.0
        config_data["gradient_smooth"] = self.gradient_smooth_check.isChecked()
        config_data["gradient_blend_strength"] = float(self.gradient_blend_strength_edit.text()) if self.gradient_blend_strength_edit.text() else 0.0

        return config_data

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        self.threads_spin.setValue(cfg.threads)
        self.resume_from_edit.setText(str(cfg.resume_from))
        self.stop_at_edit.setText(str(cfg.stop_at))
        self.cap_layers_spin.setValue(cfg.cap_layers)
        self.scale_bits_spin.setValue(cfg.scale_bits)

        # Zarr settings (apply even if disabled in UI, for consistency)
        self.zarr_store_edit.setText(cfg.zarr_store)
        self.zarr_cache_spin.setValue(cfg.zarr_cache_chunks)

        # Other Advanced Toggles
        self.top_surface_smoothing_check.setChecked(cfg.top_surface_smoothing)
        self.top_surface_strength_edit.setText(str(cfg.top_surface_strength))
        self.gradient_smooth_check.setChecked(cfg.gradient_smooth)
        self.gradient_blend_strength_edit.setText(str(cfg.gradient_blend_strength))

        # Ensure dynamic controls are updated after setting values
        self._update_top_surface_controls(Qt.Checked if cfg.top_surface_smoothing else Qt.Unchecked)
        self._update_gradient_smooth_controls(Qt.Checked if cfg.gradient_smooth else Qt.Unchecked)


# pyside_file_io_tab.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QFileDialog
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIntValidator

from config import app_config as config, Config # Import Config class explicitly

class FileIOTab(QWidget):
    """
    PySide6 tab for File I/O settings, including input/output directories,
    output options (filename padding), and global controls like Run, Stop, Save, Load.
    Output size and resampling are now handled in the XY Blend tab.
    """
    # Signals to communicate with the main GUI window
    run_requested = Signal()
    stop_requested = Signal()
    save_settings_requested = Signal()
    load_settings_requested = Signal()

    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui # Reference to the main GUI window
        self.config = config # Use the global config instance

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config

    def _setup_ui(self):
        """Sets up the widgets and layout for the File I/O tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- I/O Settings Group ---
        io_group = QGroupBox("I/O Settings")
        io_layout = QVBoxLayout(io_group)
        io_layout.setContentsMargins(10, 10, 10, 10)
        io_layout.setSpacing(5)

        # Input Directory
        input_dir_layout = QHBoxLayout()
        input_dir_layout.addWidget(QLabel("Input Dir:"))
        self.input_dir_edit = QLineEdit()
        input_dir_layout.addWidget(self.input_dir_edit)
        self.input_browse_btn = QPushButton("Browse...")
        input_dir_layout.addWidget(self.input_browse_btn)
        io_layout.addLayout(input_dir_layout)

        # Output Directory
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Dir:"))
        self.output_dir_edit = QLineEdit()
        output_dir_layout.addWidget(self.output_dir_edit)
        self.output_browse_btn = QPushButton("Browse...")
        output_dir_layout.addWidget(self.output_browse_btn)
        io_layout.addLayout(output_dir_layout)

        # File Pattern
        file_pattern_layout = QHBoxLayout()
        file_pattern_layout.addWidget(QLabel("File Pattern:"))
        self.file_pattern_edit = QLineEdit()
        self.file_pattern_edit.setPlaceholderText("e.g., *.png or image_*.tif")
        file_pattern_layout.addWidget(self.file_pattern_edit)
        io_layout.addLayout(file_pattern_layout)

        main_layout.addWidget(io_group)

        # --- Output Options Group ---
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(10, 10, 10, 10)
        output_layout.setSpacing(5)

        # Filename Padding
        pad_layout = QHBoxLayout()
        self.pad_filenames_check = QCheckBox("Pad Filenames")
        pad_layout.addWidget(self.pad_filenames_check)

        pad_layout.addWidget(QLabel("Pad Length:"))
        self.pad_length_edit = QLineEdit()
        self.pad_length_edit.setFixedWidth(60)
        self.pad_length_edit.setValidator(QIntValidator(1, 10, self)) # Max 10 digits for padding
        pad_layout.addWidget(self.pad_length_edit)
        pad_layout.addStretch(1) # Push to left
        output_layout.addLayout(pad_layout)

        main_layout.addWidget(output_group)

        # --- Controls Group (Run/Stop/Save/Load) ---
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)

        self.run_btn = QPushButton("Run")
        controls_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False) # Initially disabled
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addStretch(1) # Push save/load to the right

        self.save_settings_btn = QPushButton("Save Settings")
        controls_layout.addWidget(self.save_settings_btn)

        self.load_settings_btn = QPushButton("Load Settings")
        controls_layout.addWidget(self.load_settings_btn)

        main_layout.addWidget(controls_group)

        main_layout.addStretch(1) # Pushes all content to the top


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.input_browse_btn.clicked.connect(self._browse_input_dir)
        self.output_browse_btn.clicked.connect(self._browse_output_dir)

        # Connect internal buttons to tab's signals, which are then connected
        # to the main GUI's slots in SuperStackerPysideGUI._setup_ui
        self.run_btn.clicked.connect(self.run_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.save_settings_btn.clicked.connect(self.save_settings_requested.emit)
        self.load_settings_btn.clicked.connect(self.load_settings_requested.emit)

        # Input validation for numeric fields
        self.pad_length_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.pad_length_edit, text, int))


    def _browse_input_dir(self):
        """Opens a directory dialog for input path selection."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.input_dir_edit.text())
        if dir_path:
            self.input_dir_edit.setText(dir_path)

    def _browse_output_dir(self):
        """Opens a directory dialog for output path selection."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_dir_edit.text())
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _validate_numeric_input(self, line_edit: QLineEdit, text: str, data_type: type):
        """
        Validates if the input text can be converted to the specified data_type.
        If not, applies red border.
        """
        if not text:
            line_edit.setStyleSheet("") # Clear any error styling for empty string
            return

        try:
            if data_type == int:
                val = int(text)
            elif data_type == float:
                val = float(text)
            line_edit.setStyleSheet("") # Clear any error styling
        except ValueError:
            line_edit.setStyleSheet("border: 1px solid red;")

    def get_config(self) -> dict:
        """
        Collects current settings from this tab's widgets.
        Includes basic validation and conversion.
        """
        config_data = {}
        config_data["input_dir"] = self.input_dir_edit.text()
        config_data["output_dir"] = self.output_dir_edit.text()
        config_data["file_pattern"] = self.file_pattern_edit.text()

        # Numeric fields with validation
        try:
            config_data["pad_length"] = int(self.pad_length_edit.text()) if self.pad_length_edit.text() else 4
        except ValueError as e:
            raise ValueError(f"Invalid numeric input for Pad Length: {e}")

        config_data["pad_filenames"] = self.pad_filenames_check.isChecked()

        return config_data

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        self.input_dir_edit.setText(cfg.input_dir)
        self.output_dir_edit.setText(cfg.output_dir)
        self.file_pattern_edit.setText(cfg.file_pattern)
        self.pad_filenames_check.setChecked(cfg.pad_filenames)
        self.pad_length_edit.setText(str(cfg.pad_length))

    def set_run_button_state(self, enabled: bool):
        """Sets the enabled state of the Run button."""
        self.run_btn.setEnabled(enabled)

    def set_stop_button_state(self, enabled: bool):
        """Sets the enabled state of the Stop button."""
        self.stop_btn.setEnabled(enabled)



# pyside_lut_tab.py

import sys
import numpy as np
import json
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QLabel, QLineEdit,
    QTextEdit, QComboBox, QInputDialog, QSlider, QSpinBox, QStackedWidget,
    QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator

# Matplotlib imports for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import lut_manager # Import your lut_manager module
from config import app_config as config, Config # Import the global config instance and Config class

# Define a directory for saved LUT versions
LUT_VERSIONS_DIR = "saved_luts"
if not os.path.exists(LUT_VERSIONS_DIR):
    os.makedirs(LUT_VERSIONS_DIR)


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PySide6."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout() # Adjusts plot parameters for a tight layout


class LutTab(QWidget):
    """
    PySide6 tab for editing and managing Z-Axis Look-Up Tables (LUTs).
    Includes text editor, plot visualization, file/version operations,
    and algorithmic LUT generation controls.
    """
    # Signal to notify main GUI if the active LUT was changed (e.g., by loading/applying)
    lut_changed = Signal()

    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config # Use the global config instance
        
        # Initialize lut_manager's config reference (already done in main_gui, but safe to re-set)
        lut_manager.set_config_reference(self.config)

        # Get the initial active LUT from lut_manager, which should already be set by main_gui.py
        self.current_lut = lut_manager.get_current_z_lut()

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config
        self._load_lut_to_text_edit(self.current_lut) # Populate text edit with initial LUT
        self.plot_lut(self.current_lut) # Plot initial LUT
        self._load_lut_versions() # Populate the versions dropdown
        self._apply_initial_state() # Ensure initial control states are correct

    def _setup_ui(self):
        """Sets up the widgets and layout for the LUT tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- LUT Source Selection ---
        lut_source_group = QGroupBox("LUT Source")
        lut_source_layout = QHBoxLayout(lut_source_group)
        lut_source_layout.addWidget(QLabel("Source:"))
        self.lut_source_combo = QComboBox()
        self.lut_source_combo.addItems(["Generated", "File"])
        lut_source_layout.addWidget(self.lut_source_combo)
        lut_source_layout.addStretch(1)
        main_layout.addWidget(lut_source_group)

        # --- Generated LUT Controls ---
        self.generated_lut_group = QGroupBox("Generated LUT Parameters") # Made instance variable
        generated_lut_layout = QVBoxLayout(self.generated_lut_group)
        generated_lut_layout.setContentsMargins(10, 10, 10, 10)
        generated_lut_layout.setSpacing(5)

        # LUT Generation Type
        gen_type_layout = QHBoxLayout()
        gen_type_layout.addWidget(QLabel("Type:"))
        self.lut_generation_type_combo = QComboBox()
        self.lut_generation_type_combo.addItems(["linear", "gamma", "s_curve", "log", "exp", "sqrt", "rodbard"])
        gen_type_layout.addWidget(self.lut_generation_type_combo)
        gen_type_layout.addStretch(1)
        generated_lut_layout.addLayout(gen_type_layout)

        # Stacked Widget for specific generation parameters
        self.gen_params_stacked_widget = QStackedWidget()
        generated_lut_layout.addWidget(self.gen_params_stacked_widget)

        # --- Linear LUT Params ---
        self.linear_params_widget = QWidget()
        linear_layout = QHBoxLayout(self.linear_params_widget)
        linear_layout.addWidget(QLabel("Min Input:"))
        self.linear_min_input_edit = QLineEdit()
        self.linear_min_input_edit.setFixedWidth(60)
        self.linear_min_input_edit.setValidator(QIntValidator(0, 255, self))
        linear_layout.addWidget(self.linear_min_input_edit)
        linear_layout.addWidget(QLabel("Max Output:"))
        self.linear_max_output_edit = QLineEdit()
        self.linear_max_output_edit.setFixedWidth(60)
        self.linear_max_output_edit.setValidator(QIntValidator(0, 255, self))
        linear_layout.addWidget(self.linear_max_output_edit)
        linear_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.linear_params_widget) # Index 0

        # --- Gamma LUT Params ---
        self.gamma_params_widget = QWidget()
        gamma_layout = QHBoxLayout(self.gamma_params_widget)
        gamma_layout.addWidget(QLabel("Gamma Value:"))
        self.gamma_value_edit = QLineEdit()
        self.gamma_value_edit.setFixedWidth(60)
        self.gamma_value_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        gamma_layout.addWidget(self.gamma_value_edit)
        self.gamma_value_slider = QSlider(Qt.Horizontal)
        self.gamma_value_slider.setRange(1, 1000) # 0.01 to 10.00
        self.gamma_value_slider.setSingleStep(1)
        gamma_layout.addWidget(self.gamma_value_slider)
        gamma_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.gamma_params_widget) # Index 1

        # --- S-Curve LUT Params ---
        self.s_curve_params_widget = QWidget()
        s_curve_layout = QHBoxLayout(self.s_curve_params_widget)
        s_curve_layout.addWidget(QLabel("Contrast:"))
        self.s_curve_contrast_edit = QLineEdit()
        self.s_curve_contrast_edit.setFixedWidth(60)
        self.s_curve_contrast_edit.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        s_curve_layout.addWidget(self.s_curve_contrast_edit)
        self.s_curve_contrast_slider = QSlider(Qt.Horizontal)
        self.s_curve_contrast_slider.setRange(0, 100) # 0.0 to 1.0
        self.s_curve_contrast_slider.setSingleStep(1)
        s_curve_layout.addWidget(self.s_curve_contrast_slider)
        s_curve_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.s_curve_params_widget) # Index 2

        # --- Log LUT Params ---
        self.log_params_widget = QWidget()
        log_layout = QHBoxLayout(self.log_params_widget)
        log_layout.addWidget(QLabel("Param:"))
        self.log_param_edit = QLineEdit()
        self.log_param_edit.setFixedWidth(60)
        self.log_param_edit.setValidator(QDoubleValidator(0.01, 100.0, 2, self))
        log_layout.addWidget(self.log_param_edit)
        self.log_param_slider = QSlider(Qt.Horizontal)
        self.log_param_slider.setRange(1, 1000) # 0.01 to 100.0
        self.log_param_slider.setSingleStep(1)
        log_layout.addWidget(self.log_param_slider)
        log_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.log_params_widget) # Index 3

        # --- Exp LUT Params ---
        self.exp_params_widget = QWidget()
        exp_layout = QHBoxLayout(self.exp_params_widget)
        exp_layout.addWidget(QLabel("Param:"))
        self.exp_param_edit = QLineEdit()
        self.exp_param_edit.setFixedWidth(60)
        self.exp_param_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        exp_layout.addWidget(self.exp_param_edit)
        self.exp_param_slider = QSlider(Qt.Horizontal)
        self.exp_param_slider.setRange(1, 1000) # 0.01 to 10.0
        self.exp_param_slider.setSingleStep(1)
        exp_layout.addWidget(self.exp_param_slider)
        exp_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.exp_params_widget) # Index 4

        # --- Sqrt LUT Params (minimal) ---
        self.sqrt_params_widget = QWidget()
        sqrt_layout = QHBoxLayout(self.sqrt_params_widget)
        sqrt_layout.addWidget(QLabel("Param:")) # Placeholder, currently not used in `lut_manager`
        self.sqrt_param_edit = QLineEdit("1.0")
        self.sqrt_param_edit.setFixedWidth(60)
        self.sqrt_param_edit.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        sqrt_layout.addWidget(self.sqrt_param_edit)
        sqrt_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.sqrt_params_widget) # Index 5

        # --- Rodbard LUT Params (minimal) ---
        self.rodbard_params_widget = QWidget()
        rodbard_layout = QHBoxLayout(self.rodbard_params_widget)
        rodbard_layout.addWidget(QLabel("Param:")) # Placeholder, currently not used in `lut_manager`
        self.rodbard_param_edit = QLineEdit("1.0")
        self.rodbard_param_edit.setFixedWidth(60)
        rodbard_layout.addWidget(self.rodbard_param_edit)
        rodbard_layout.addStretch(1)
        self.gen_params_stacked_widget.addWidget(self.rodbard_params_widget) # Index 6

        # Generate Button
        generate_btn_layout = QHBoxLayout()
        generate_btn_layout.addStretch(1)
        self.generate_lut_button = QPushButton("Generate LUT")
        generate_btn_layout.addWidget(self.generate_lut_button)
        generate_btn_layout.addStretch(1)
        generated_lut_layout.addLayout(generate_btn_layout)

        main_layout.addWidget(self.generated_lut_group)

        # --- File-based LUT Controls ---
        self.file_lut_group = QGroupBox("File-based LUT Controls") # Made instance variable
        file_lut_layout = QVBoxLayout(self.file_lut_group)
        file_lut_layout.setContentsMargins(10, 10, 10, 10)
        file_lut_layout.setSpacing(5)

        lut_path_layout = QHBoxLayout()
        lut_path_layout.addWidget(QLabel("Loaded File:"))
        self.lut_filepath_edit = QLineEdit() # This will display the path of loaded/saved LUT
        self.lut_filepath_edit.setReadOnly(True)
        lut_path_layout.addWidget(self.lut_filepath_edit)
        file_lut_layout.addLayout(lut_path_layout)

        file_button_layout = QHBoxLayout()
        self.load_file_button = QPushButton("Load from File...")
        file_button_layout.addWidget(self.load_file_button)
        self.save_file_button = QPushButton("Save to File...")
        file_button_layout.addWidget(self.save_file_button)
        file_button_layout.addStretch(1)
        file_lut_layout.addLayout(file_button_layout)

        main_layout.addWidget(self.file_lut_group)

        # --- LUT Editor (TextEdit) and Plot ---
        editor_plot_group = QGroupBox("Current LUT")
        editor_plot_layout_group = QHBoxLayout(editor_plot_group)
        
        # Text Editor for LUT values
        lut_editor_layout = QVBoxLayout()
        lut_editor_layout.addWidget(QLabel("LUT Values (0-255, one per line):"))
        self.lut_text_edit = QTextEdit()
        self.lut_text_edit.setPlaceholderText("Enter 256 integer values (0-255), one per line.")
        self.lut_text_edit.setMinimumWidth(150)
        self.lut_text_edit.setMaximumWidth(200) # Keep it narrow for single column
        self.lut_text_edit.setMinimumHeight(200)
        lut_editor_layout.addWidget(self.lut_text_edit)
        editor_plot_layout_group.addLayout(lut_editor_layout)

        # Matplotlib Plot
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setMinimumHeight(200)
        editor_plot_layout_group.addWidget(self.canvas)

        main_layout.addWidget(editor_plot_group)

        # --- Version Management ---
        version_group = QGroupBox("Saved Versions")
        version_layout = QHBoxLayout(version_group)
        version_layout.addWidget(QLabel("Select Version:"))
        self.version_combo = QComboBox()
        self.version_combo.setMinimumWidth(150)
        version_layout.addWidget(self.version_combo)

        self.load_version_button = QPushButton("Load Version")
        version_layout.addWidget(self.load_version_button)
        self.save_version_button = QPushButton("Save Current as Version...")
        version_layout.addWidget(self.save_version_button)
        self.delete_version_button = QPushButton("Delete Version")
        version_layout.addWidget(self.delete_version_button)
        version_layout.addStretch(1)
        main_layout.addWidget(version_group)

        # --- Global LUT Controls ---
        global_lut_controls_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Default LUT")
        global_lut_controls_layout.addWidget(self.reset_button)
        global_lut_controls_layout.addStretch(1)
        self.apply_button = QPushButton("Apply LUT to Pipeline")
        global_lut_controls_layout.addWidget(self.apply_button)
        global_lut_controls_layout.addStretch(1)
        main_layout.addLayout(global_lut_controls_layout)

        main_layout.addStretch(1) # Pushes all content to the top


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        # LUT Source and Generation Type
        self.lut_source_combo.currentTextChanged.connect(self._update_lut_source_controls)
        self.lut_generation_type_combo.currentTextChanged.connect(self._update_generated_lut_params_widget)
        self.generate_lut_button.clicked.connect(self._generate_and_display_lut)

        # Generated LUT parameter edits and sliders
        self.linear_min_input_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.linear_min_input_edit, text, int))
        self.linear_max_output_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.linear_max_output_edit, text, int))
        
        self.gamma_value_slider.valueChanged.connect(lambda val: self.gamma_value_edit.setText(f"{val / 100.0:.2f}"))
        self.gamma_value_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.gamma_value_slider, text, 100.0))
        
        self.s_curve_contrast_slider.valueChanged.connect(lambda val: self.s_curve_contrast_edit.setText(f"{val / 100.0:.2f}"))
        self.s_curve_contrast_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.s_curve_contrast_slider, text, 100.0))

        self.log_param_slider.valueChanged.connect(lambda val: self.log_param_edit.setText(f"{val / 10.0:.1f}"))
        self.log_param_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.log_param_slider, text, 10.0))

        self.exp_param_slider.valueChanged.connect(lambda val: self.exp_param_edit.setText(f"{val / 100.0:.2f}"))
        self.exp_param_edit.textChanged.connect(lambda text: self._update_slider_from_edit(self.exp_param_slider, text, 100.0))

        # Manual LUT editing and plotting
        self.lut_text_edit.textChanged.connect(self._on_text_edit_changed)
        
        # File Operations
        self.load_file_button.clicked.connect(self._load_lut_from_file)
        self.save_file_button.clicked.connect(self._save_lut_to_file)
        
        # Version Management
        self.load_version_button.clicked.connect(self._load_selected_version)
        self.save_version_button.clicked.connect(self._save_current_as_version)
        self.delete_version_button.clicked.connect(self._delete_selected_version)

        # Global LUT Controls
        self.reset_button.clicked.connect(self._reset_to_default_lut)
        self.apply_button.clicked.connect(self._apply_lut_changes)


    def _validate_numeric_input(self, line_edit: QLineEdit, text: str, data_type: type):
        """
        Validates if the input text can be converted to the specified data_type.
        If not, applies red border.
        """
        if not text:
            line_edit.setStyleSheet("") # Clear any error styling for empty string
            return

        try:
            if data_type == int:
                val = int(text)
            elif data_type == float:
                val = float(text)
            line_edit.setStyleSheet("") # Clear any error styling
        except ValueError:
            line_edit.setStyleSheet("border: 1px solid red;")

    def _update_slider_from_edit(self, slider: QSlider, text: str, scale_factor: float):
        """Updates a slider's value based on a QLineEdit's text."""
        try:
            val = float(text)
            slider.setValue(int(val * scale_factor))
        except ValueError:
            pass # Let validator handle invalid input visually

    def _apply_initial_state(self):
        """Applies initial control states based on current config values and active LUT."""
        # Set LUT source and generation type based on config
        self.lut_source_combo.setCurrentText(self.config.lut_source.capitalize())
        self.lut_generation_type_combo.setCurrentText(self.config.lut_generation_type)
        
        # Manually trigger updates for UI enabling/disabling based on source/type
        # These calls should NOT trigger LUT generation/loading themselves during initial setup.
        self.generated_lut_group.setEnabled(self.config.lut_source == "generated")
        self.file_lut_group.setEnabled(self.config.lut_source == "file")
        self._update_generated_lut_params_widget(self.config.lut_generation_type) # Just for stacking widget visibility

        # Populate generated LUT parameter edits and sliders from config
        self.linear_min_input_edit.setText(str(self.config.linear_min_input))
        self.linear_max_output_edit.setText(str(self.config.linear_max_output))
        self.gamma_value_edit.setText(str(self.config.gamma_value))
        self.gamma_value_slider.setValue(int(self.config.gamma_value * 100))
        self.s_curve_contrast_edit.setText(str(self.config.s_curve_contrast))
        self.s_curve_contrast_slider.setValue(int(self.config.s_curve_contrast * 100))
        self.log_param_edit.setText(str(self.config.log_param))
        self.log_param_slider.setValue(int(self.config.log_param * 10))
        self.exp_param_edit.setText(str(self.config.exp_param))
        self.exp_param_slider.setValue(int(self.config.exp_param * 100))
        self.sqrt_param_edit.setText(str(self.config.sqrt_param))
        self.rodbard_param_edit.setText(str(self.config.rodbard_param))

        # Set the fixed LUT path display
        self.lut_filepath_edit.setText(self.config.fixed_lut_path)

        # Load and plot the CURRENTLY ACTIVE LUT from lut_manager
        # This LUT should already be correctly set by main_gui.py's initial call to update_active_lut_from_config()
        self._load_lut_to_text_edit(lut_manager.get_current_z_lut())
        self.plot_lut(lut_manager.get_current_z_lut())


    def _update_lut_source_controls(self, source_text: str):
        """Enables/disables UI elements based on LUT source selection."""
        is_generated = (source_text.lower() == "generated")
        
        self.generated_lut_group.setEnabled(is_generated)
        self.file_lut_group.setEnabled(not is_generated)

        # When source changes, update the config's lut_source
        self.config.lut_source = source_text.lower()

        # If switching to 'generated', regenerate and display the LUT based on current GUI params
        if is_generated:
            self._generate_and_display_lut()
        # If switching to 'file', display the LUT from the currently configured fixed_lut_path
        else:
            filepath = self.lut_filepath_edit.text()
            if filepath and os.path.exists(filepath):
                try:
                    loaded_lut = lut_manager.load_lut(filepath)
                    self._load_lut_to_text_edit(loaded_lut)
                except Exception as e:
                    QMessageBox.warning(self, "LUT Load Error", f"Could not load LUT from '{filepath}': {e}. Displaying default generated LUT.")
                    self.lut_filepath_edit.setText("") # Clear invalid path
                    self._load_lut_to_text_edit(lut_manager.get_default_z_lut()) # Show default linear
            else:
                QMessageBox.information(self, "No LUT File", "No valid LUT file path specified. Displaying default generated LUT.")
                self.lut_filepath_edit.setText("")
                self._load_lut_to_text_edit(lut_manager.get_default_z_lut()) # Show default linear


    def _update_generated_lut_params_widget(self, lut_type: str):
        """Switches the stacked widget to show parameters for the selected LUT generation type."""
        if lut_type == "linear":
            self.gen_params_stacked_widget.setCurrentWidget(self.linear_params_widget)
        elif lut_type == "gamma":
            self.gen_params_stacked_widget.setCurrentWidget(self.gamma_params_widget)
        elif lut_type == "s_curve":
            self.gen_params_stacked_widget.setCurrentWidget(self.s_curve_params_widget)
        elif lut_type == "log":
            self.gen_params_stacked_widget.setCurrentWidget(self.log_params_widget)
        elif lut_type == "exp":
            self.gen_params_stacked_widget.setCurrentWidget(self.exp_params_widget)
        elif lut_type == "sqrt":
            self.gen_params_stacked_widget.setCurrentWidget(self.sqrt_params_widget)
        elif lut_type == "rodbard":
            self.gen_params_stacked_widget.setCurrentWidget(self.rodbard_params_widget)
        else:
            # Default to an empty widget or linear if unknown
            self.gen_params_stacked_widget.setCurrentWidget(self.linear_params_widget) # Fallback

        # Update config's lut_generation_type
        self.config.lut_generation_type = lut_type
        
        # Only generate and display if the source is currently "generated"
        if self.lut_source_combo.currentText().lower() == "generated":
            self._generate_and_display_lut()


    def _generate_and_display_lut(self):
        """
        Generates a LUT based on current GUI settings (not config) and displays it
        in the text editor and plot. This is for previewing.
        It does NOT set the active LUT in lut_manager.
        """
        lut_type = self.lut_generation_type_combo.currentText()
        generated_lut = None
        try:
            # Use .text() to get current values from GUI, not config
            if lut_type == "linear":
                min_in = int(self.linear_min_input_edit.text())
                max_out = int(self.linear_max_output_edit.text())
                generated_lut = lut_manager.generate_linear_lut(min_in, max_out)
            elif lut_type == "gamma":
                gamma = float(self.gamma_value_edit.text())
                generated_lut = lut_manager.generate_gamma_lut(gamma)
            elif lut_type == "s_curve":
                contrast = float(self.s_curve_contrast_edit.text())
                generated_lut = lut_manager.generate_s_curve_lut(contrast)
            elif lut_type == "log":
                param = float(self.log_param_edit.text())
                generated_lut = lut_manager.generate_log_lut(param)
            elif lut_type == "exp":
                param = float(self.exp_param_edit.text())
                generated_lut = lut_manager.generate_exp_lut(param)
            elif lut_type == "sqrt":
                param = float(self.sqrt_param_edit.text()) # Pass param, even if not used in current lut_manager
                generated_lut = lut_manager.generate_sqrt_lut(param)
            elif lut_type == "rodbard":
                param = float(self.rodbard_param_edit.text()) # Pass param, even if not used in current lut_manager
                generated_lut = lut_manager.generate_rodbard_lut(param)
            else:
                generated_lut = lut_manager.get_default_z_lut() # Fallback

            if generated_lut is not None:
                self._load_lut_to_text_edit(generated_lut)
                # self.lut_filepath_edit.setText("") # Do not clear, this is a preview
            else:
                QMessageBox.warning(self, "LUT Generation Error", "Failed to generate LUT. Check parameters.")

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid numeric input for LUT generation: {e}")
            self.lut_text_edit.setStyleSheet("border: 1px solid red;") # Indicate error in text box
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"An error occurred during LUT generation: {e}")
            self.lut_text_edit.setStyleSheet("border: 1px solid red;")

    def _on_text_edit_changed(self):
        """Called when text in the LUT editor is changed."""
        try:
            text_values = self.lut_text_edit.toPlainText().strip().split('\n')
            text_values = [v for v in text_values if v] # Filter out empty strings

            if len(text_values) != 256:
                self.lut_text_edit.setStyleSheet("border: 1px solid orange;")
                self.canvas.axes.clear()
                self.canvas.axes.text(0.5, 0.5, "Invalid LUT Length (Expected 256)",
                                      horizontalalignment='center', verticalalignment='center',
                                      transform=self.canvas.axes.transAxes, color='red')
                self.canvas.draw()
                return

            new_lut = np.zeros(256, dtype=np.uint8)
            for i, val_str in enumerate(text_values):
                val = int(val_str)
                new_lut[i] = max(0, min(255, val)) # Clamp values

            self.current_lut = new_lut
            self.plot_lut(self.current_lut)
            self.lut_text_edit.setStyleSheet("") # Clear any warning style

        except ValueError:
            self.lut_text_edit.setStyleSheet("border: 1px solid red;")
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, "Invalid Input (Non-numeric value)",
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=self.canvas.axes.transAxes, color='red')
            self.canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "LUT Editor Error", f"An unexpected error occurred: {e}")


    def _load_lut_to_text_edit(self, lut_array: np.ndarray):
        """Populates the QTextEdit with values from a NumPy array."""
        if lut_array.dtype != np.uint8 or lut_array.shape != (256,):
            QMessageBox.critical(self, "Load Error", "Invalid LUT array format.")
            return

        # Temporarily disconnect signal to prevent re-triggering during text update
        self.lut_text_edit.textChanged.disconnect(self._on_text_edit_changed)
        
        self.lut_text_edit.setText('\n'.join(map(str, lut_array)))
        self.current_lut = lut_array.copy() # Update internal copy
        self.plot_lut(self.current_lut)
        self.lut_text_edit.setStyleSheet("") # Clear any error/warning styling

        # Reconnect the signal
        self.lut_text_edit.textChanged.connect(self._on_text_edit_changed)


    def plot_lut(self, lut_array: np.ndarray):
        """Updates the matplotlib plot with the given LUT data."""
        self.canvas.axes.clear()
        x_values = np.arange(256)
        y_values = lut_array

        self.canvas.axes.plot(x_values, y_values, 'b-')
        self.canvas.axes.set_title("Current Z-LUT Remapping Curve")
        self.canvas.axes.set_xlabel("Input Value (0-255)")
        self.canvas.axes.set_ylabel("Output Value (0-255)")
        self.canvas.axes.set_xlim(0, 255)
        self.canvas.axes.set_ylim(0, 255)
        self.canvas.axes.grid(True)
        self.canvas.draw()


    def _load_lut_from_file(self):
        """Opens a file dialog to load a LUT from a JSON file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load LUT File", self.config.fixed_lut_path or "", "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            try:
                loaded_lut = lut_manager.load_lut(filepath)
                self._load_lut_to_text_edit(loaded_lut)
                self.lut_filepath_edit.setText(filepath) # Display loaded path
                self.lut_source_combo.setCurrentText("File") # Set source to File
                QMessageBox.information(self, "Load Success", f"LUT loaded from '{filepath}'. Use 'Apply LUT to Pipeline' to make it active.")
                # Do NOT emit lut_changed here; it's only a preview until applied
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load LUT: {e}")

    def _save_lut_to_file(self):
        """Opens a file dialog to save the current LUT to a JSON file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save LUT File", self.lut_filepath_edit.text() or "custom_z_lut.json", "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            try:
                lut_manager.save_lut(filepath, self.current_lut)
                self.lut_filepath_edit.setText(filepath) # Display saved path
                self.lut_source_combo.setCurrentText("File") # Set source to File
                QMessageBox.information(self, "Save Success", f"LUT saved to '{filepath}'. Use 'Apply LUT to Pipeline' to make it active.")
                # Do NOT emit lut_changed here; it's only a save operation
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save LUT: {e}")

    def _load_lut_versions(self):
        """Populates the version combo box with saved LUT files."""
        self.version_combo.clear()
        self.version_combo.addItem("--- Select a version ---")
        lut_files = [f for f in os.listdir(LUT_VERSIONS_DIR) if f.endswith(".json")]
        for f in sorted(lut_files):
            self.version_combo.addItem(f)

    def _load_selected_version(self):
        """Loads the LUT from the selected version in the combo box."""
        selected_file = self.version_combo.currentText()
        if selected_file == "--- Select a version ---" or not selected_file:
            return

        filepath = os.path.join(LUT_VERSIONS_DIR, selected_file)
        if os.path.exists(filepath):
            try:
                loaded_lut = lut_manager.load_lut(filepath)
                self._load_lut_to_text_edit(loaded_lut)
                self.lut_filepath_edit.setText(filepath) # Display loaded path
                self.lut_source_combo.setCurrentText("File") # Set source to File
                QMessageBox.information(self, "Version Loaded", f"LUT version '{selected_file}' loaded. Use 'Apply LUT to Pipeline' to make it active.")
                # Do NOT emit lut_changed here; it's only a preview until applied
            except Exception as e:
                QMessageBox.critical(self, "Load Version Error", f"Failed to load version '{selected_file}': {e}")
        else:
            QMessageBox.warning(self, "File Not Found", f"LUT version file '{selected_file}' not found.")
            self._load_lut_versions() # Refresh list

    def _save_current_as_version(self):
        """Saves the current LUT in the editor as a new version."""
        version_name, ok = QInputDialog.getText(self, "Save LUT Version", "Enter a name for this LUT version:")
        if ok and version_name:
            # Sanitize filename
            version_name = "".join(c for c in version_name if c.isalnum() or c in (' ', '.', '_')).strip()
            if not version_name:
                QMessageBox.warning(self, "Invalid Name", "Version name cannot be empty or contain only invalid characters.")
                return

            filename = f"{version_name}.json"
            filepath = os.path.join(LUT_VERSIONS_DIR, filename)
            
            if os.path.exists(filepath):
                reply = QMessageBox.question(self, "Overwrite Version?",
                                             f"Version '{version_name}' already exists. Overwrite?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return

            try:
                lut_manager.save_lut(filepath, self.current_lut)
                self._load_lut_versions() # Refresh versions dropdown
                self.version_combo.setCurrentText(filename) # Select the newly saved version
                QMessageBox.information(self, "Save Version Success", f"Current LUT saved as version '{version_name}'.")
            except Exception as e:
                QMessageBox.critical(self, "Save Version Error", f"Failed to save LUT version: {e}")

    def _delete_selected_version(self):
        """Deletes the selected LUT version from disk."""
        selected_file = self.version_combo.currentText()
        if selected_file == "--- Select a version ---" or not selected_file:
            QMessageBox.information(self, "No Version Selected", "Please select a LUT version to delete.")
            return

        reply = QMessageBox.question(self, "Delete Version",
                                     f"Are you sure you want to delete LUT version '{selected_file}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            filepath = os.path.join(LUT_VERSIONS_DIR, selected_file)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    self._load_lut_versions() # Refresh versions dropdown
                    QMessageBox.information(self, "Delete Success", f"LUT version '{selected_file}' deleted.")
                else:
                    QMessageBox.warning(self, "File Not Found", f"LUT version file '{selected_file}' not found on disk.")
                    self._load_lut_versions() # Refresh list
            except Exception as e:
                QMessageBox.critical(self, "Delete Error", f"Failed to delete LUT version: {e}")

    def _reset_to_default_lut(self):
        """Resets the LUT in the editor and plot to the module's default."""
        reply = QMessageBox.question(self, "Reset LUT",
                                     "Are you sure you want to reset the LUT to its default values (linear 0-255)? This will not affect the active pipeline LUT until 'Apply' is clicked.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            default_lut = lut_manager.get_default_z_lut()
            self._load_lut_to_text_edit(default_lut)
            self.lut_filepath_edit.setText("") # Clear file path as it's default now
            self.lut_source_combo.setCurrentText("Generated") # Set source to Generated
            self.lut_generation_type_combo.setCurrentText("linear") # Set generation type to linear
            QMessageBox.information(self, "Reset Success", "LUT editor reset to default linear values. Click 'Apply LUT to Pipeline' to make it active.")
            # Do NOT emit lut_changed here; it's only a preview until applied


    def _apply_lut_changes(self):
        """
        Applies the current LUT from the editor to the active LUT in the lut_manager module
        and updates the config based on the selected source/parameters.
        """
        # First, ensure the text edit content is valid and parsed into current_lut
        self._on_text_edit_changed() 
        if self.lut_text_edit.styleSheet() != "": # Check if there's an error style
            QMessageBox.warning(self, "Apply Error", "Cannot apply LUT with invalid input. Please correct the values.")
            return

        try:
            # Update the config based on current GUI state
            self.config.lut_source = self.lut_source_combo.currentText().lower()
            if self.config.lut_source == "generated":
                self.config.lut_generation_type = self.lut_generation_type_combo.currentText()
                # Update all generation parameters in config from GUI edits
                self.config.gamma_value = float(self.gamma_value_edit.text()) if self.gamma_value_edit.text() else 1.0
                self.config.linear_min_input = int(self.linear_min_input_edit.text()) if self.linear_min_input_edit.text() else 0
                self.config.linear_max_output = int(self.linear_max_output_edit.text()) if self.linear_max_output_edit.text() else 255
                self.config.s_curve_contrast = float(self.s_curve_contrast_edit.text()) if self.s_curve_contrast_edit.text() else 0.5
                self.config.log_param = float(self.log_param_edit.text()) if self.log_param_edit.text() else 10.0
                self.config.exp_param = float(self.exp_param_edit.text()) if self.exp_param_edit.text() else 2.0
                self.config.sqrt_param = float(self.sqrt_param_edit.text()) if self.sqrt_param_edit.text() else 1.0
                self.config.rodbard_param = float(self.rodbard_param_edit.text()) if self.rodbard_param_edit.text() else 1.0
                self.config.fixed_lut_path = "" # Clear fixed path if now generated
            elif self.config.lut_source == "file":
                self.config.fixed_lut_path = self.lut_filepath_edit.text()
                # If file source, ensure generation parameters are reset to defaults in config
                # (though they won't be used, it's good practice for clean config state)
                self.config.lut_generation_type = "linear"
                self.config.gamma_value = 1.0
                self.config.linear_min_input = 0
                self.config.linear_max_output = 255
                self.config.s_curve_contrast = 0.5
                self.config.log_param = 10.0
                self.config.exp_param = 2.0
                self.config.sqrt_param = 1.0
                self.config.rodbard_param = 1.0
            
            # Now, tell lut_manager to update its active LUT based on the (now updated) config
            lut_manager.update_active_lut_from_config()
            
            # Finally, update the GUI to reflect the *actual* active LUT from lut_manager
            # (which might have fallen back to default if fixed_lut_path was invalid)
            self._load_lut_to_text_edit(lut_manager.get_current_z_lut())
            self.lut_source_combo.setCurrentText(self.config.lut_source.capitalize()) # Re-set in case of fallback
            self.lut_filepath_edit.setText(self.config.fixed_lut_path) # Re-set in case of fallback
            self.lut_generation_type_combo.setCurrentText(self.config.lut_generation_type) # Re-set in case of fallback

            QMessageBox.information(self, "LUT Applied", "LUT settings applied to the processing pipeline.")
            self.lut_changed.emit() # Signal that the active LUT has been updated

        except ValueError as e:
            QMessageBox.warning(self, "Apply Error", f"Invalid input for LUT parameters: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"An unexpected error occurred while applying LUT: {e}")


    def get_config(self) -> dict:
        """
        Collects current settings from this tab's widgets.
        Note: The config object is updated directly by _apply_lut_changes.
        This method primarily serves to return a dictionary representation
        of the current state of config relevant to this tab.
        """
        # The config object itself is the source of truth, updated by _apply_lut_changes.
        # We just need to return the relevant parts.
        return {
            "lut_source": self.lut_source_combo.currentText().lower(),
            "fixed_lut_path": self.lut_filepath_edit.text(),
            "lut_generation_type": self.lut_generation_type_combo.currentText(),
            "gamma_value": float(self.gamma_value_edit.text()) if self.gamma_value_edit.text() else 1.0,
            "linear_min_input": int(self.linear_min_input_edit.text()) if self.linear_min_input_edit.text() else 0,
            "linear_max_output": int(self.linear_max_output_edit.text()) if self.linear_max_output_edit.text() else 255,
            "s_curve_contrast": float(self.s_curve_contrast_edit.text()) if self.s_curve_contrast_edit.text() else 0.5,
            "log_param": float(self.log_param_edit.text()) if self.log_param_edit.text() else 10.0,
            "exp_param": float(self.exp_param_edit.text()) if self.exp_param_edit.text() else 2.0,
            "sqrt_param": float(self.sqrt_param_edit.text()) if self.sqrt_param_edit.text() else 1.0,
            "rodbard_param": float(self.rodbard_param_edit.text()) if self.rodbard_param_edit.text() else 1.0,
        }

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        # This method is called during initial GUI setup and when loading settings.
        # It should populate the GUI elements with values from the provided Config object.
        
        # Disconnect signals to prevent immediate re-triggering of logic during population
        self.lut_source_combo.currentTextChanged.disconnect(self._update_lut_source_controls)
        self.lut_generation_type_combo.currentTextChanged.disconnect(self._update_generated_lut_params_widget)
        self.lut_text_edit.textChanged.disconnect(self._on_text_edit_changed)
        
        # Set values from config to GUI elements
        self.lut_source_combo.setCurrentText(cfg.lut_source.capitalize())
        self.lut_generation_type_combo.setCurrentText(cfg.lut_generation_type)

        self.linear_min_input_edit.setText(str(cfg.linear_min_input))
        self.linear_max_output_edit.setText(str(cfg.linear_max_output))
        self.gamma_value_edit.setText(str(cfg.gamma_value))
        self.gamma_value_slider.setValue(int(cfg.gamma_value * 100))
        self.s_curve_contrast_edit.setText(str(cfg.s_curve_contrast))
        self.s_curve_contrast_slider.setValue(int(cfg.s_curve_contrast * 100))
        self.log_param_edit.setText(str(cfg.log_param))
        self.log_param_slider.setValue(int(cfg.log_param * 10))
        self.exp_param_edit.setText(str(cfg.exp_param))
        self.exp_param_slider.setValue(int(cfg.exp_param * 100))
        self.sqrt_param_edit.setText(str(cfg.sqrt_param))
        self.rodbard_param_edit.setText(str(cfg.rodbard_param))

        self.lut_filepath_edit.setText(cfg.fixed_lut_path)
        
        # Reconnect signals
        self.lut_source_combo.currentTextChanged.connect(self._update_lut_source_controls)
        self.lut_generation_type_combo.currentTextChanged.connect(self._update_generated_lut_params_widget)
        self.lut_text_edit.textChanged.connect(self._on_text_edit_changed)

        # Manually trigger updates for UI enabling/disabling based on source/type
        self.generated_lut_group.setEnabled(cfg.lut_source == "generated")
        self.file_lut_group.setEnabled(cfg.lut_source == "file")
        self._update_generated_lut_params_widget(cfg.lut_generation_type) # Just for stacking widget visibility

        # Load and plot the LUT based on the *current config state*
        # This will either load from file or generate based on cfg.lut_source and cfg.fixed_lut_path/generation_type
        # This is where the initial display of the LUT in the editor and plot happens.
        # This call will also update lut_manager's active LUT if it's different.
        lut_manager.update_active_lut_from_config() # Ensure lut_manager has the correct active LUT
        self._load_lut_to_text_edit(lut_manager.get_current_z_lut()) # Display it in the UI


# pyside_stacking_tab.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QSlider, QSpinBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator

from config import app_config as config, Config # Import Config class explicitly

class StackingTab(QWidget):
    """
    PySide6 tab for Z-axis Stacking (Blend) parameters.
    """
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config # Use the global config instance

        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config
        self._apply_initial_state() # Ensure initial control states are correct

    def _setup_ui(self):
        """Sets up the widgets and layout for the Stacking tab."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Stacking (Blend) Group ---
        blend_group = QGroupBox("Z-Axis Stacking (Blend)")
        blend_layout = QVBoxLayout(blend_group)
        blend_layout.setContentsMargins(10, 10, 10, 10)
        blend_layout.setSpacing(5)

        # Primary and Radius
        primary_radius_layout = QHBoxLayout()
        primary_radius_layout.addWidget(QLabel("Primary Layers:"))
        self.primary_edit = QLineEdit()
        self.primary_edit.setFixedWidth(60)
        self.primary_edit.setValidator(QIntValidator(1, 999999, self))
        primary_radius_layout.addWidget(self.primary_edit)

        primary_radius_layout.addWidget(QLabel("Radius:"))
        self.radius_edit = QLineEdit()
        self.radius_edit.setFixedWidth(60)
        self.radius_edit.setValidator(QIntValidator(0, 999999, self))
        primary_radius_layout.addWidget(self.radius_edit)
        primary_radius_layout.addStretch(1)
        blend_layout.addLayout(primary_radius_layout)

        # Blend Mode and Parameter
        blend_mode_param_layout = QHBoxLayout()
        blend_mode_param_layout.addWidget(QLabel("Blend Mode:"))
        self.blend_mode_combo = QComboBox()
        # Ensure these match the blend_mode types in stacking_processor.py
        modes = [
            "gaussian", "linear", "cosine", "exp_decay", "flat",
            "binary_contour", "gradient_contour",
            "z_column_lift", "z_contour_interp",
            # Zarr-backed modes are not implemented in this version, so exclude for now
            # "zarr_binary_shadow", "zarr_binary_overhang", "zarr_binary_contour", "zarr_gradient_contour"
        ]
        self.blend_mode_combo.addItems(modes)
        blend_mode_param_layout.addWidget(self.blend_mode_combo)

        blend_mode_param_layout.addWidget(QLabel("σ / Param:"))
        self.blend_param_edit = QLineEdit()
        self.blend_param_edit.setFixedWidth(60)
        self.blend_param_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        blend_mode_param_layout.addWidget(self.blend_param_edit)

        self.blend_param_slider = QSlider(Qt.Horizontal)
        self.blend_param_slider.setRange(0, 1000) # Scale to allow 0.0 to 100.0 with 2 decimal places
        self.blend_param_slider.setSingleStep(1)
        self.blend_param_slider.setPageStep(10)
        self.blend_param_slider.setTickInterval(100)
        self.blend_param_slider.setTickPosition(QSlider.TicksBelow)
        blend_mode_param_layout.addWidget(self.blend_param_slider)
        blend_layout.addLayout(blend_mode_param_layout)

        # Directional Blend (Z-Bias)
        zbias_layout = QHBoxLayout()
        self.zbias_check = QCheckBox("Directional Blend (Z-Bias)")
        zbias_layout.addWidget(self.zbias_check)

        zbias_layout.addWidget(QLabel("Dir σ:"))
        self.dir_sigma_edit = QLineEdit()
        self.dir_sigma_edit.setFixedWidth(60)
        self.dir_sigma_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        zbias_layout.addWidget(self.dir_sigma_edit)

        self.dir_sigma_slider = QSlider(Qt.Horizontal)
        self.dir_sigma_slider.setRange(0, 1000) # Scale to allow 0.0 to 100.0 with 2 decimal places
        self.dir_sigma_slider.setSingleStep(1)
        self.dir_sigma_slider.setPageStep(10)
        self.dir_sigma_slider.setTickInterval(100)
        self.dir_sigma_slider.setTickPosition(QSlider.TicksBelow)
        zbias_layout.addWidget(self.dir_sigma_slider)
        zbias_layout.addStretch(1)
        blend_layout.addLayout(zbias_layout)

        # Binary/Gradient Thresholds
        thresholds_layout = QHBoxLayout()
        thresholds_layout.addWidget(QLabel("Binary Thres:"))
        self.binary_threshold_edit = QLineEdit()
        self.binary_threshold_edit.setFixedWidth(60)
        self.binary_threshold_edit.setValidator(QIntValidator(0, 255, self))
        thresholds_layout.addWidget(self.binary_threshold_edit)

        thresholds_layout.addWidget(QLabel("Grad Thres:"))
        self.gradient_threshold_edit = QLineEdit()
        self.gradient_threshold_edit.setFixedWidth(60)
        self.gradient_threshold_edit.setValidator(QIntValidator(0, 255, self))
        thresholds_layout.addWidget(self.gradient_threshold_edit)
        thresholds_layout.addStretch(1)
        blend_layout.addLayout(thresholds_layout)

        main_layout.addWidget(blend_group)
        main_layout.addStretch(1)


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.blend_mode_combo.currentTextChanged.connect(self._update_blend_mode_controls)
        
        # Blend Param slider and edit
        self.blend_param_slider.valueChanged.connect(lambda val: self.blend_param_edit.setText(f"{val / 10.0:.1f}"))
        self.blend_param_edit.textChanged.connect(self._update_blend_param_from_edit)

        # Directional Blend checkbox and Dir Sigma slider/edit
        self.zbias_check.stateChanged.connect(self._update_zbias_controls)
        self.dir_sigma_slider.valueChanged.connect(lambda val: self.dir_sigma_edit.setText(f"{val / 10.0:.1f}"))
        self.dir_sigma_edit.textChanged.connect(self._update_dir_sigma_from_edit)

        # Input validation for numeric fields (using textChanged for immediate feedback)
        self.primary_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.primary_edit, text, int))
        self.radius_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.radius_edit, text, int))
        self.binary_threshold_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.binary_threshold_edit, text, int))
        self.gradient_threshold_edit.textChanged.connect(lambda text: self._validate_numeric_input(self.gradient_threshold_edit, text, int))


    def _validate_numeric_input(self, line_edit: QLineEdit, text: str, data_type: type):
        """
        Validates if the input text can be converted to the specified data_type.
        If not, applies red border.
        """
        if not text:
            line_edit.setStyleSheet("") # Clear any error styling for empty string
            return

        try:
            if data_type == int:
                val = int(text)
            elif data_type == float:
                val = float(text)
            line_edit.setStyleSheet("") # Clear any error styling
        except ValueError:
            line_edit.setStyleSheet("border: 1px solid red;")

    def _update_blend_param_from_edit(self, text):
        try:
            val = float(text)
            # Scale float value back to slider range (0-100.0 -> 0-1000)
            self.blend_param_slider.setValue(int(val * 10))
        except ValueError:
            pass # Let validator handle invalid input visually

    def _update_dir_sigma_from_edit(self, text):
        try:
            val = float(text)
            # Scale float value back to slider range (0-100.0 -> 0-1000)
            self.dir_sigma_slider.setValue(int(val * 10))
        except ValueError:
            pass # Let validator handle invalid input visually

    def _apply_initial_state(self):
        """Applies initial control states based on current config values."""
        self._update_blend_mode_controls(self.config.blend_mode)
        self._update_zbias_controls(Qt.Checked if self.config.directional_blend else Qt.Unchecked)

    def _update_blend_mode_controls(self, mode: str):
        """
        Enables/disables blend parameter controls based on the selected blend mode.
        Also manages visibility of binary/gradient thresholds.
        """
        # Modes that use blend_param (sigma/exponent)
        param_enabled_modes = ["gaussian", "linear", "cosine", "exp_decay"] # Flat doesn't use param
        disable_param = mode not in param_enabled_modes
        
        self.blend_param_edit.setEnabled(not disable_param)
        self.blend_param_slider.setEnabled(not disable_param)

        # Modes that allow directional blend (Z-Bias)
        zbias_enabled_modes = ["gaussian", "linear", "cosine", "exp_decay"]
        enable_zbias_checkbox = mode in zbias_enabled_modes
        self.zbias_check.setEnabled(enable_zbias_checkbox)
        
        # Dir Sigma controls are enabled only if Z-Bias checkbox is checked AND blend mode allows it
        enable_dir_sigma_controls = enable_zbias_checkbox and self.zbias_check.isChecked()
        self.dir_sigma_edit.setEnabled(enable_dir_sigma_controls)
        self.dir_sigma_slider.setEnabled(enable_dir_sigma_controls)

        # Modes that use binary_threshold
        binary_threshold_modes = ["binary_contour"]
        enable_binary_threshold = mode in binary_threshold_modes
        self.binary_threshold_edit.setEnabled(enable_binary_threshold)

        # Modes that use gradient_threshold
        gradient_threshold_modes = ["gradient_contour"]
        enable_gradient_threshold = mode in gradient_threshold_modes
        self.gradient_threshold_edit.setEnabled(enable_gradient_threshold)

        # Special handling for z_column_lift and z_contour_interp: they don't use blend_param
        # but their logic in stacking_processor might implicitly use blend_param for internal kernel generation
        # or other config values. For GUI, we disable blend_param for them.
        if mode in ["z_column_lift", "z_contour_interp"]:
            self.blend_param_edit.setEnabled(False)
            self.blend_param_slider.setEnabled(False)
            self.zbias_check.setEnabled(False) # No Z-bias for these modes
            self.dir_sigma_edit.setEnabled(False)
            self.dir_sigma_slider.setEnabled(False)


    def _update_zbias_controls(self, state: int):
        """Enables/disables directional sigma controls based on checkbox state."""
        is_checked = (state == Qt.Checked)
        # Only enable if the current blend mode also allows Z-bias
        blend_mode_allows_zbias = self.blend_mode_combo.currentText() in ["gaussian", "linear", "cosine", "exp_decay"]
        
        enable_dir_sigma = is_checked and blend_mode_allows_zbias
        self.dir_sigma_edit.setEnabled(enable_dir_sigma)
        self.dir_sigma_slider.setEnabled(enable_dir_sigma)

    def get_config(self) -> dict:
        """Collects current settings from this tab's widgets."""
        config_data = {}
        config_data["primary"] = int(self.primary_edit.text()) if self.primary_edit.text() else 3
        config_data["radius"] = int(self.radius_edit.text()) if self.radius_edit.text() else 1
        config_data["blend_mode"] = self.blend_mode_combo.currentText()
        config_data["blend_param"] = float(self.blend_param_edit.text()) if self.blend_param_edit.text() else 1.0
        config_data["directional_blend"] = self.zbias_check.isChecked()
        config_data["dir_sigma"] = float(self.dir_sigma_edit.text()) if self.dir_sigma_edit.text() else 1.0
        config_data["binary_threshold"] = int(self.binary_threshold_edit.text()) if self.binary_threshold_edit.text() else 128
        config_data["gradient_threshold"] = int(self.gradient_threshold_edit.text()) if self.gradient_threshold_edit.text() else 128
        
        return config_data

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        self.primary_edit.setText(str(cfg.primary))
        self.radius_edit.setText(str(cfg.radius))
        self.blend_mode_combo.setCurrentText(cfg.blend_mode)
        self.blend_param_edit.setText(str(cfg.blend_param))
        self.blend_param_slider.setValue(int(cfg.blend_param * 10)) # Scale for slider
        self.zbias_check.setChecked(cfg.directional_blend)
        self.dir_sigma_edit.setText(str(cfg.dir_sigma))
        self.dir_sigma_slider.setValue(int(cfg.dir_sigma * 10)) # Scale for slider
        self.binary_threshold_edit.setText(str(cfg.binary_threshold))
        self.gradient_threshold_edit.setText(str(cfg.gradient_threshold))

        # Ensure dynamic controls are updated after setting values
        self._update_blend_mode_controls(cfg.blend_mode)
        self._update_zbias_controls(Qt.Checked if cfg.directional_blend else Qt.Unchecked)




from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QSlider, QSpinBox, QPushButton, QListWidget,
    QListWidgetItem, QStackedWidget, QSizePolicy, QMessageBox # Added QMessageBox here
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator

from config import app_config as config, Config, XYBlendOperation # Import Config and XYBlendOperation

class XYBlendTab(QWidget):
    """
    PySide6 tab for managing the XY Blending/Processing pipeline.
    Allows users to add, remove, reorder, and configure individual
    XYBlendOperation steps (blur, sharpen, resize, etc.).
    """
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.config = config # Use the global config instance
        
        self._setup_ui()
        self._connect_signals()
        self.apply_settings(self.config) # Apply initial settings from config
        self._update_operation_list() # Populate list with initial operations
        self._update_selected_operation_details() # Show details for first item

    def _setup_ui(self):
        """Sets up the widgets and layout for the XY Blend tab."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Left Panel: Operations List and Controls ---
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(5)

        # Operations List
        ops_list_group = QGroupBox("Processing Pipeline")
        ops_list_layout = QVBoxLayout(ops_list_group)
        ops_list_layout.setContentsMargins(10, 10, 10, 10)
        ops_list_layout.setSpacing(5)

        self.ops_list_widget = QListWidget()
        self.ops_list_widget.setDragDropMode(QListWidget.InternalMove) # Enable reordering
        self.ops_list_widget.setMinimumWidth(200)
        self.ops_list_widget.setMinimumHeight(250)
        ops_list_layout.addWidget(self.ops_list_widget)

        # Add/Remove/Move Buttons
        op_buttons_layout = QHBoxLayout()
        self.add_op_button = QPushButton("Add Op")
        op_buttons_layout.addWidget(self.add_op_button)
        self.remove_op_button = QPushButton("Remove Op")
        op_buttons_layout.addWidget(self.remove_op_button)
        op_buttons_layout.addStretch(1) # Push move buttons to right
        self.move_up_button = QPushButton("Move Up")
        op_buttons_layout.addWidget(self.move_up_button)
        self.move_down_button = QPushButton("Move Down")
        op_buttons_layout.addWidget(self.move_down_button)
        ops_list_layout.addLayout(op_buttons_layout)
        
        left_panel_layout.addWidget(ops_list_group)
        left_panel_layout.addStretch(1) # Push content to top

        main_layout.addLayout(left_panel_layout)

        # --- Right Panel: Operation Details (Stacked Widget) ---
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(5)

        self.details_group = QGroupBox("Operation Details") # Made into an instance variable
        details_layout = QVBoxLayout(self.details_group)
        details_layout.setContentsMargins(10, 10, 10, 10)
        details_layout.setSpacing(5)

        # Operation Type selection for the currently selected item
        op_type_layout = QHBoxLayout()
        op_type_layout.addWidget(QLabel("Operation Type:"))
        self.selected_op_type_combo = QComboBox()
        self.selected_op_type_combo.addItems([
            "none", "gaussian_blur", "bilateral_filter", "median_blur", "unsharp_mask", "resize"
        ])
        op_type_layout.addWidget(self.selected_op_type_combo)
        op_type_layout.addStretch(1)
        details_layout.addLayout(op_type_layout)

        # Stacked Widget for specific operation parameters
        self.op_params_stacked_widget = QStackedWidget()
        details_layout.addWidget(self.op_params_stacked_widget)

        # --- None Op Params (Placeholder) ---
        self.none_params_widget = QWidget()
        none_layout = QVBoxLayout(self.none_params_widget)
        none_layout.addWidget(QLabel("No parameters for 'none' operation."))
        none_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.none_params_widget) # Index 0

        # --- Gaussian Blur Params ---
        self.gaussian_params_widget = QWidget()
        gaussian_layout = QVBoxLayout(self.gaussian_params_widget)
        gaussian_layout.addWidget(QLabel("Kernel Size (X, Y):"))
        ksize_layout = QHBoxLayout()
        self.gaussian_ksize_x_edit = QLineEdit()
        self.gaussian_ksize_x_edit.setFixedWidth(60)
        self.gaussian_ksize_x_edit.setValidator(QIntValidator(1, 99, self))
        ksize_layout.addWidget(self.gaussian_ksize_x_edit)
        self.gaussian_ksize_y_edit = QLineEdit()
        self.gaussian_ksize_y_edit.setFixedWidth(60)
        self.gaussian_ksize_y_edit.setValidator(QIntValidator(1, 99, self))
        ksize_layout.addWidget(self.gaussian_ksize_y_edit)
        ksize_layout.addStretch(1)
        gaussian_layout.addLayout(ksize_layout)

        gaussian_layout.addWidget(QLabel("Sigma (X, Y): (0 for auto)"))
        sigma_layout = QHBoxLayout()
        self.gaussian_sigma_x_edit = QLineEdit()
        self.gaussian_sigma_x_edit.setFixedWidth(60)
        self.gaussian_sigma_x_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        sigma_layout.addWidget(self.gaussian_sigma_x_edit)
        self.gaussian_sigma_y_edit = QLineEdit()
        self.gaussian_sigma_y_edit.setFixedWidth(60)
        self.gaussian_sigma_y_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        sigma_layout.addWidget(self.gaussian_sigma_y_edit)
        sigma_layout.addStretch(1)
        gaussian_layout.addLayout(sigma_layout)
        gaussian_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.gaussian_params_widget) # Index 1

        # --- Bilateral Filter Params ---
        self.bilateral_params_widget = QWidget()
        bilateral_layout = QVBoxLayout(self.bilateral_params_widget)
        bilateral_layout.addWidget(QLabel("Diameter:"))
        self.bilateral_d_edit = QLineEdit()
        self.bilateral_d_edit.setFixedWidth(60)
        self.bilateral_d_edit.setValidator(QIntValidator(1, 99, self))
        bilateral_layout.addWidget(self.bilateral_d_edit)

        bilateral_layout.addWidget(QLabel("Sigma Color:"))
        self.bilateral_sigma_color_edit = QLineEdit()
        self.bilateral_sigma_color_edit.setFixedWidth(60)
        self.bilateral_sigma_color_edit.setValidator(QDoubleValidator(0.0, 255.0, 2, self))
        bilateral_layout.addWidget(self.bilateral_sigma_color_edit)

        bilateral_layout.addWidget(QLabel("Sigma Space:"))
        self.bilateral_sigma_space_edit = QLineEdit()
        self.bilateral_sigma_space_edit.setFixedWidth(60)
        self.bilateral_sigma_space_edit.setValidator(QDoubleValidator(0.0, 255.0, 2, self))
        bilateral_layout.addWidget(self.bilateral_sigma_space_edit)
        bilateral_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.bilateral_params_widget) # Index 2

        # --- Median Blur Params ---
        self.median_params_widget = QWidget()
        median_layout = QVBoxLayout(self.median_params_widget)
        median_layout.addWidget(QLabel("Kernel Size:"))
        self.median_ksize_edit = QLineEdit()
        self.median_ksize_edit.setFixedWidth(60)
        self.median_ksize_edit.setValidator(QIntValidator(1, 99, self))
        median_layout.addWidget(self.median_ksize_edit)
        median_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.median_params_widget) # Index 3

        # --- Unsharp Mask Params ---
        self.unsharp_params_widget = QWidget()
        unsharp_layout = QVBoxLayout(self.unsharp_params_widget)
        unsharp_layout.addWidget(QLabel("Amount:"))
        self.unsharp_amount_edit = QLineEdit()
        self.unsharp_amount_edit.setFixedWidth(60)
        self.unsharp_amount_edit.setValidator(QDoubleValidator(0.0, 5.0, 2, self))
        unsharp_layout.addWidget(self.unsharp_amount_edit)

        unsharp_layout.addWidget(QLabel("Threshold:"))
        self.unsharp_threshold_edit = QLineEdit()
        self.unsharp_threshold_edit.setFixedWidth(60)
        self.unsharp_threshold_edit.setValidator(QIntValidator(0, 255, self))
        unsharp_layout.addWidget(self.unsharp_threshold_edit)

        unsharp_layout.addWidget(QLabel("Internal Blur KSize:"))
        self.unsharp_blur_ksize_edit = QLineEdit()
        self.unsharp_blur_ksize_edit.setFixedWidth(60)
        self.unsharp_blur_ksize_edit.setValidator(QIntValidator(1, 99, self))
        unsharp_layout.addWidget(self.unsharp_blur_ksize_edit)

        unsharp_layout.addWidget(QLabel("Internal Blur Sigma:"))
        self.unsharp_blur_sigma_edit = QLineEdit()
        self.unsharp_blur_sigma_edit.setFixedWidth(60)
        self.unsharp_blur_sigma_edit.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        unsharp_layout.addWidget(self.unsharp_blur_sigma_edit)
        unsharp_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.unsharp_params_widget) # Index 4

        # --- Resize Params ---
        self.resize_params_widget = QWidget()
        resize_layout = QVBoxLayout(self.resize_params_widget)
        resize_layout.addWidget(QLabel("Width (px, 0 for auto):"))
        self.resize_width_edit = QLineEdit()
        self.resize_width_edit.setFixedWidth(80)
        self.resize_width_edit.setValidator(QIntValidator(0, 9999, self)) # Max 9999 for now
        resize_layout.addWidget(self.resize_width_edit)

        resize_layout.addWidget(QLabel("Height (px, 0 for auto):"))
        self.resize_height_edit = QLineEdit()
        self.resize_height_edit.setFixedWidth(80)
        self.resize_height_edit.setValidator(QIntValidator(0, 9999, self))
        resize_layout.addWidget(self.resize_height_edit)

        resize_layout.addWidget(QLabel("Resampling Mode:"))
        self.resample_mode_combo = QComboBox()
        self.resample_mode_combo.addItems(["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS4", "AREA"])
        resize_layout.addWidget(self.resample_mode_combo)
        resize_layout.addStretch(1)
        self.op_params_stacked_widget.addWidget(self.resize_params_widget) # Index 5

        details_layout.addStretch(1) # Push content to top of details group
        right_panel_layout.addWidget(self.details_group)
        right_panel_layout.addStretch(1) # Push content to top of right panel

        main_layout.addLayout(right_panel_layout)
        main_layout.addStretch(1) # Push all content to left


    def _connect_signals(self):
        """Connects widget signals to their respective slots."""
        self.ops_list_widget.currentRowChanged.connect(self._update_selected_operation_details)
        self.ops_list_widget.model().rowsMoved.connect(self._reorder_operations_in_config)

        self.add_op_button.clicked.connect(self._add_operation)
        self.remove_op_button.clicked.connect(self._remove_operation)
        self.move_up_button.clicked.connect(self._move_operation_up)
        self.move_down_button.clicked.connect(self._move_operation_down)

        self.selected_op_type_combo.currentTextChanged.connect(self._on_selected_op_type_changed)

        # Connect parameter edits to update config when text changes
        # Gaussian
        self.gaussian_ksize_x_edit.textChanged.connect(lambda text: self._update_param_in_config("gaussian_ksize_x", text, int))
        self.gaussian_ksize_y_edit.textChanged.connect(lambda text: self._update_param_in_config("gaussian_ksize_y", text, int))
        self.gaussian_sigma_x_edit.textChanged.connect(lambda text: self._update_param_in_config("gaussian_sigma_x", text, float))
        self.gaussian_sigma_y_edit.textChanged.connect(lambda text: self._update_param_in_config("gaussian_sigma_y", text, float))
        
        # Bilateral
        self.bilateral_d_edit.textChanged.connect(lambda text: self._update_param_in_config("bilateral_d", text, int))
        self.bilateral_sigma_color_edit.textChanged.connect(lambda text: self._update_param_in_config("bilateral_sigma_color", text, float))
        self.bilateral_sigma_space_edit.textChanged.connect(lambda text: self._update_param_in_config("bilateral_sigma_space", text, float))

        # Median
        self.median_ksize_edit.textChanged.connect(lambda text: self._update_param_in_config("median_ksize", text, int))

        # Unsharp
        self.unsharp_amount_edit.textChanged.connect(lambda text: self._update_param_in_config("unsharp_amount", text, float))
        self.unsharp_threshold_edit.textChanged.connect(lambda text: self._update_param_in_config("unsharp_threshold", text, int))
        self.unsharp_blur_ksize_edit.textChanged.connect(lambda text: self._update_param_in_config("unsharp_blur_ksize", text, int))
        self.unsharp_blur_sigma_edit.textChanged.connect(lambda text: self._update_param_in_config("unsharp_blur_sigma", text, float))

        # Resize
        self.resize_width_edit.textChanged.connect(lambda text: self._update_param_in_config("resize_width", text, int, allow_none_if_zero=True))
        self.resize_height_edit.textChanged.connect(lambda text: self._update_param_in_config("resize_height", text, int, allow_none_if_zero=True))
        self.resample_mode_combo.currentTextChanged.connect(lambda text: self._update_param_in_config("resample_mode", text, str))


    def _validate_numeric_input(self, line_edit: QLineEdit, text: str, data_type: type):
        """
        Validates if the input text can be converted to the specified data_type.
        If not, applies red border.
        """
        if not text:
            line_edit.setStyleSheet("") # Clear any error styling for empty string
            return

        try:
            if data_type == int:
                val = int(text)
            elif data_type == float:
                val = float(text)
            line_edit.setStyleSheet("") # Clear any error styling
        except ValueError:
            line_edit.setStyleSheet("border: 1px solid red;")

    def _update_operation_list(self):
        """Populates the QListWidget from the config's xy_blend_pipeline."""
        self.ops_list_widget.clear()
        for i, op in enumerate(self.config.xy_blend_pipeline):
            item = QListWidgetItem(f"{i+1}. {op.type.replace('_', ' ').title()}")
            self.ops_list_widget.addItem(item)
        
        # Select the first item if list is not empty
        if self.ops_list_widget.count() > 0:
            self.ops_list_widget.setCurrentRow(0)
        else:
            self.selected_op_type_combo.setCurrentText("none") # Reset type combo
            self.op_params_stacked_widget.setCurrentWidget(self.none_params_widget) # Show none params


    def _update_selected_operation_details(self):
        """
        Updates the details panel (combo box and stacked widget)
        based on the currently selected item in the operations list.
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.config.xy_blend_pipeline):
            selected_op = self.config.xy_blend_pipeline[current_row]
            
            # Temporarily block signals to avoid re-triggering updates during UI population
            self.selected_op_type_combo.blockSignals(True)
            self.selected_op_type_combo.setCurrentText(selected_op.type)
            self.selected_op_type_combo.blockSignals(False)

            self._show_params_for_type(selected_op.type)
            self._populate_params_widgets(selected_op)
            self.details_group.setEnabled(True) # Enable details group
        else:
            # No item selected or list is empty
            self.details_group.setEnabled(False) # Disable details group
            self.selected_op_type_combo.setCurrentText("none") # Reset type combo
            self.op_params_stacked_widget.setCurrentWidget(self.none_params_widget) # Show none params


    def _on_selected_op_type_changed(self, new_type: str):
        """
        Called when the operation type combo box for the selected item changes.
        Updates the config and the displayed parameters.
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.config.xy_blend_pipeline):
            selected_op = self.config.xy_blend_pipeline[current_row]
            
            # Update the type in the config's XYBlendOperation object
            selected_op.type = new_type
            # Clear old parameters by re-initializing the object's attributes based on new type defaults
            # This is more robust than just clearing a 'params' dict.
            # Create a new default XYBlendOperation of the new type, then copy its default parameters
            # to the existing selected_op.
            default_new_op = XYBlendOperation(type=new_type)
            # Corrected line: Iterate over values of __dataclass_fields__
            for attr_name in [f.name for f in default_new_op.__dataclass_fields__.values() if f.name != 'type']:
                setattr(selected_op, attr_name, getattr(default_new_op, attr_name))
            
            selected_op.__post_init__() # Re-run post_init for new type defaults/validation

            # Update the list item text
            self.ops_list_widget.currentItem().setText(f"{current_row+1}. {new_type.replace('_', ' ').title()}")

            # Show the correct parameters widget and populate with new defaults
            self._show_params_for_type(new_type)
            self._populate_params_widgets(selected_op) # Populate with new default params for the type


    def _show_params_for_type(self, op_type: str):
        """Switches the stacked widget to show parameters for the given operation type."""
        if op_type == "none":
            self.op_params_stacked_widget.setCurrentWidget(self.none_params_widget)
        elif op_type == "gaussian_blur":
            self.op_params_stacked_widget.setCurrentWidget(self.gaussian_params_widget)
        elif op_type == "bilateral_filter":
            self.op_params_stacked_widget.setCurrentWidget(self.bilateral_params_widget)
        elif op_type == "median_blur":
            self.op_params_stacked_widget.setCurrentWidget(self.median_params_widget)
        elif op_type == "unsharp_mask":
            self.op_params_stacked_widget.setCurrentWidget(self.unsharp_params_widget)
        elif op_type == "resize":
            self.op_params_stacked_widget.setCurrentWidget(self.resize_params_widget)
        else:
            self.op_params_stacked_widget.setCurrentWidget(self.none_params_widget) # Fallback


    def _populate_params_widgets(self, op: XYBlendOperation):
        """Populates the parameter QLineEdits with values from the given XYBlendOperation."""
        # Disconnect signals temporarily to avoid triggering _update_param_in_config during population
        # This is crucial to prevent unintended config updates.
        self._block_param_signals(True)

        # Clear all edits first to ensure fresh state
        for edit in [self.gaussian_ksize_x_edit, self.gaussian_ksize_y_edit, self.gaussian_sigma_x_edit, self.gaussian_sigma_y_edit,
                     self.bilateral_d_edit, self.bilateral_sigma_color_edit, self.bilateral_sigma_space_edit,
                     self.median_ksize_edit,
                     self.unsharp_amount_edit, self.unsharp_threshold_edit, self.unsharp_blur_ksize_edit, self.unsharp_blur_sigma_edit,
                     self.resize_width_edit, self.resize_height_edit]:
            edit.setText("") # Clear text

        # Populate based on type
        if op.type == "gaussian_blur":
            self.gaussian_ksize_x_edit.setText(str(op.gaussian_ksize_x))
            self.gaussian_ksize_y_edit.setText(str(op.gaussian_ksize_y))
            self.gaussian_sigma_x_edit.setText(str(op.gaussian_sigma_x))
            self.gaussian_sigma_y_edit.setText(str(op.gaussian_sigma_y))
        elif op.type == "bilateral_filter":
            self.bilateral_d_edit.setText(str(op.bilateral_d))
            self.bilateral_sigma_color_edit.setText(str(op.bilateral_sigma_color))
            self.bilateral_sigma_space_edit.setText(str(op.bilateral_sigma_space))
        elif op.type == "median_blur":
            self.median_ksize_edit.setText(str(op.median_ksize))
        elif op.type == "unsharp_mask":
            self.unsharp_amount_edit.setText(str(op.unsharp_amount))
            self.unsharp_threshold_edit.setText(str(op.unsharp_threshold))
            self.unsharp_blur_ksize_edit.setText(str(op.unsharp_blur_ksize))
            self.unsharp_blur_sigma_edit.setText(str(op.unsharp_blur_sigma))
        elif op.type == "resize":
            # Convert None to 0 for display in QLineEdit for resize dimensions
            self.resize_width_edit.setText(str(op.resize_width or 0))
            self.resize_height_edit.setText(str(op.resize_height or 0))
            self.resample_mode_combo.setCurrentText(op.resample_mode)
        
        self._block_param_signals(False) # Reconnect signals

    def _block_param_signals(self, block: bool):
        """Blocks/unblocks signals for all parameter QLineEdits and QComboBoxes."""
        widgets_to_block = [
            self.gaussian_ksize_x_edit, self.gaussian_ksize_y_edit, self.gaussian_sigma_x_edit, self.gaussian_sigma_y_edit,
            self.bilateral_d_edit, self.bilateral_sigma_color_edit, self.bilateral_sigma_space_edit,
            self.median_ksize_edit,
            self.unsharp_amount_edit, self.unsharp_threshold_edit, self.unsharp_blur_ksize_edit, self.unsharp_blur_sigma_edit,
            self.resize_width_edit, self.resize_height_edit, self.resample_mode_combo
        ]
        for widget in widgets_to_block:
            widget.blockSignals(block)


    def _update_param_in_config(self, param_name: str, text: str, data_type: type, allow_none_if_zero: bool = False):
        """
        Updates a parameter in the currently selected XYBlendOperation's params dictionary.
        Also performs validation and visual feedback.
        """
        current_row = self.ops_list_widget.currentRow()
        if current_row < 0 or current_row >= len(self.config.xy_blend_pipeline):
            return # No operation selected

        selected_op = self.config.xy_blend_pipeline[current_row]
        
        line_edit = self.sender() # Get the QLineEdit that emitted the signal
        if not isinstance(line_edit, QLineEdit) and not isinstance(line_edit, QComboBox):
            return # Ensure it's a widget we expect

        if not text and allow_none_if_zero: # For resize width/height, empty string or 0 means None
            value = None
            line_edit.setStyleSheet("") # Clear any error styling
        else:
            try:
                if data_type == int:
                    value = int(text)
                elif data_type == float:
                    value = float(text)
                else: # For string types like resample_mode
                    value = text
                line_edit.setStyleSheet("") # Clear any error styling
            except ValueError:
                line_edit.setStyleSheet("border: 1px solid red;")
                return # Do not update config with invalid value

        # Update the parameter directly on the selected operation object
        setattr(selected_op, param_name, value)
        selected_op.__post_init__() # Re-run post_init for validation (e.g., odd ksize)
        
        # If a kernel size was adjusted by post_init, update the UI to reflect it
        # This check is important because __post_init__ might modify the value (e.g., ensure odd ksize)
        # and the UI should reflect the actual value in the config object.
        if param_name.endswith("_ksize_x") and getattr(selected_op, param_name) != value:
            self._block_param_signals(True)
            line_edit.setText(str(getattr(selected_op, param_name)))
            self._block_param_signals(False)
        elif param_name.endswith("_ksize_y") and getattr(selected_op, param_name) != value:
            self._block_param_signals(True)
            line_edit.setText(str(getattr(selected_op, param_name)))
            self._block_param_signals(False)
        elif param_name.endswith("_ksize") and getattr(selected_op, param_name) != value:
            self._block_param_signals(True)
            line_edit.setText(str(getattr(selected_op, param_name)))
            self._block_param_signals(False)


    def _add_operation(self):
        """Adds a new 'none' operation to the pipeline."""
        new_op = XYBlendOperation(type="none")
        self.config.xy_blend_pipeline.append(new_op)
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(len(self.config.xy_blend_pipeline) - 1) # Select the new item


    def _remove_operation(self):
        """Removes the currently selected operation from the pipeline."""
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.config.xy_blend_pipeline):
            reply = QMessageBox.question(self, "Remove Operation",
                                         f"Are you sure you want to remove operation {current_row+1}?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.config.xy_blend_pipeline[current_row]
                self._update_operation_list()
                # Select the item before the removed one, or the new first item
                if self.ops_list_widget.count() > 0:
                    new_row = min(current_row, self.ops_list_widget.count() - 1)
                    self.ops_list_widget.setCurrentRow(new_row)
                else:
                    self._update_selected_operation_details() # Clear details if list is empty


    def _move_operation_up(self):
        """Moves the selected operation up in the pipeline list."""
        current_row = self.ops_list_widget.currentRow()
        if current_row > 0:
            # Swap in the config list
            self.config.xy_blend_pipeline[current_row], self.config.xy_blend_pipeline[current_row - 1] = \
                self.config.xy_blend_pipeline[current_row - 1], self.config.xy_blend_pipeline[current_row]
            self._update_operation_list()
            self.ops_list_widget.setCurrentRow(current_row - 1) # Keep selection on the moved item


    def _move_operation_down(self):
        """Moves the selected operation down in the pipeline list."""
        current_row = self.ops_list_widget.currentRow()
        if current_row < len(self.config.xy_blend_pipeline) - 1:
            # Swap in the config list
            self.config.xy_blend_pipeline[current_row], self.config.xy_blend_pipeline[current_row + 1] = \
                self.config.xy_blend_pipeline[current_row + 1], self.config.xy_blend_pipeline[current_row]
            self._update_operation_list()
            self.ops_list_widget.setCurrentRow(current_row + 1) # Keep selection on the moved item

    def _reorder_operations_in_config(self, parent, start, end, destination, row):
        """
        Slot for QListWidget.model().rowsMoved.
        Updates the config's xy_blend_pipeline to reflect the reordering.
        """
        # This signal is emitted *before* the model is actually changed.
        # So, we need to get the item being moved *before* the move,
        # and then re-insert it at the new position.
        
        # Get the item that was moved
        moved_item_index = start
        moved_op = self.config.xy_blend_pipeline.pop(moved_item_index)

        # Calculate the new insertion index
        # If moving down, and the destination is after the original position,
        # the destination index will be effectively one less due to the pop.
        new_index = row
        if destination == Qt.BottomToTop: # Moving up
            if new_index > moved_item_index:
                new_index -= 1
        else: # Moving down
            if new_index < moved_item_index:
                new_index += 1

        self.config.xy_blend_pipeline.insert(new_index, moved_op)
        
        # After reordering, re-populate the list widget to ensure numbering is correct
        # and re-select the moved item.
        self._update_operation_list()
        self.ops_list_widget.setCurrentRow(new_index)


    def get_config(self) -> dict:
        """
        Collects current settings for the XY blend pipeline.
        The XYBlendOperation objects in config.xy_blend_pipeline are already
        kept up-to-date by the _update_param_in_config and _on_selected_op_type_changed methods.
        So, we just need to return a representation of the pipeline.
        """
        # The config.xy_blend_pipeline is already the source of truth.
        # We just need to ensure it's correctly serialized by Config.to_dict().
        # No direct collection needed here, as changes are applied immediately.
        return {"xy_blend_pipeline": [op.to_dict() for op in self.config.xy_blend_pipeline]}

    def apply_settings(self, cfg: Config): # Corrected type hint here
        """Applies settings from a Config object to this tab's widgets."""
        # The config.xy_blend_pipeline is already updated by Config.load()
        # So we just need to refresh the list widget and details panel.
        self._update_operation_list()
        self._update_selected_operation_details()



# run_logger.py

"""
Manages logging of Modular-Stacker run configurations and timestamps
to a semi-flat JSON Lines file.
Each line in the log file is a JSON object representing a single run.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional # Import Optional and Any here

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref

    # Ensure the config instance has the run_log_file attribute
    if not hasattr(config_instance, 'run_log_file'):
        raise AttributeError("Config instance must have a 'run_log_file' attribute.")
    
    _config_ref = config_instance

def log_run(run_index: int, config_data: Dict[str, Any]) -> None:
    """
    Logs the details of a processing run to the configured log file.

    Args:
        run_index (int): The serial/index number for this run.
        config_data (Dict[str, Any]): A dictionary containing the configuration
                                       parameters used for this run.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in run_logger. Cannot log run.")
        return

    log_filepath = _config_ref.run_log_file
    
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_filepath) or ".", exist_ok=True)

    log_entry = {
        "run_index": run_index,
        "timestamp": datetime.now().isoformat(),
        "config": config_data
    }

    try:
        with open(log_filepath, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f)
            f.write('\n') # Add a newline to make it JSON Lines
        print(f"RunLogger: Logged run {run_index} to {log_filepath}")
    except Exception as e:
        print(f"Error logging run {run_index} to '{log_filepath}': {e}")

def get_last_run_index() -> int:
    """
    Reads the log file to determine the last used run index.
    Returns 0 if the file does not exist or is empty/invalid.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in run_logger. Cannot get last run index.")
        return 0

    log_filepath = _config_ref.run_log_file
    last_index = 0
    if os.path.exists(log_filepath):
        try:
            with open(log_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if "run_index" in entry:
                            last_index = max(last_index, int(entry["run_index"]))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed line in log file: {line.strip()}")
        except Exception as e:
            print(f"Error reading log file '{log_filepath}': {e}. Resetting run index.")
            last_index = 0
    return last_index

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- Run Logger Module Test ---")
    
    # Dummy Config for testing
    class MockConfig:
        def __init__(self):
            self.run_log_file = "test_modular_stacker_runs.log"
            self.input_dir = "/path/to/test/input"
            self.output_dir = "/path/to/test/output"
            self.blend_mode = "gaussian"
            self.threads = 4
            self.lut_generation_type = "gamma"
            self.gamma_value = 2.2
            self.xy_blend_pipeline = [
                {"type": "gaussian_blur", "params": {"gaussian_ksize_x": 5, "gaussian_sigma_x": 1.0}},
                {"type": "unsharp_mask", "params": {"unsharp_amount": 1.2}}
            ]

        def to_dict(self):
            # Simulate the to_dict method from the actual Config class
            data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            if "xy_blend_pipeline" in data:
                data["xy_blend_pipeline"] = [op if isinstance(op, dict) else op.to_dict() for op in data["xy_blend_pipeline"]]
            return data

    mock_config = MockConfig()
    set_config_reference(mock_config)

    # Clean up previous test log if it exists
    if os.path.exists(mock_config.run_log_file):
        os.remove(mock_config.run_log_file)
        print(f"Cleaned up previous test log: {mock_config.run_log_file}")

    # Test initial run index
    initial_index = get_last_run_index()
    print(f"Initial last run index: {initial_index}")

    # Log a few runs
    print("\nLogging runs...")
    log_run(initial_index + 1, mock_config.to_dict())
    log_run(initial_index + 2, mock_config.to_dict())

    # Test getting last run index after logging
    new_last_index = get_last_run_index()
    print(f"New last run index: {new_last_index}")

    # Log another run with a modified config
    mock_config.blend_mode = "linear"
    log_run(new_last_index + 1, mock_config.to_dict())
    print(f"Final last run index: {get_last_run_index()}")

    # Verify content (optional, manual inspection of test_modular_stacker_runs.log)
    print(f"\nCheck the file '{mock_config.run_log_file}' for logged data.")

    # Clean up test log
    if os.path.exists(mock_config.run_log_file):
        os.remove(mock_config.run_log_file)
        print(f"Cleaned up {mock_config.run_log_file}")




# stacking_processor.py

"""
The core image processing engine for Modular-Stacker.
It performs Z-axis blending, applies the Z-LUT, and then
processes the image through the XY blending pipeline.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Any

import weights
import lut_manager
import xy_blend_processor
from config import XYBlendOperation # Import XYBlendOperation to access its attributes

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance
    # Also set config reference for imported modules
    weights.set_config_reference(config_instance)
    lut_manager.set_config_reference(config_instance)
    xy_blend_processor.set_config_reference(config_instance)


def process_image_stack(image_window: List[np.ndarray]) -> np.ndarray:
    """
    Processes a single stack (window) of grayscale images.

    Args:
        image_window (List[np.ndarray]): A list of 8-bit grayscale NumPy arrays
                                         representing the image slices in the window.
                                         Can contain blank images (all zeros).

    Returns:
        np.ndarray: The final 8-bit grayscale NumPy array after stacking,
                    Z-LUT application, and XY pipeline processing.
    """
    if _config_ref is None:
        raise RuntimeError("Config reference not set in stacking_processor. Cannot process stack.")

    if not image_window:
        raise ValueError("Image window cannot be empty.")

    # Infer image shape from the first non-None image, or fallback to a default
    # This should ideally be consistent across all images in the window.
    # Assuming all images in the window are of the same shape.
    H, W = image_window[0].shape[:2] # Assuming grayscale (H, W) or (H, W, 1)

    # --- 1. Z-axis Blending ---
    # Convert images to 16-bit for accumulation to avoid overflow during weighted sum.
    # The `scale_bits` from config determines the precision for fixed-point arithmetic.
    scale_factor = 1 << _config_ref.scale_bits # e.g., 2^12 = 4096

    # Initialize accumulator for 32-bit to prevent overflow during intermediate sums
    accumulator = np.zeros((H, W), dtype=np.uint32)
    
    # Get blend mode and parameters from config
    blend_mode = _config_ref.blend_mode
    blend_param = _config_ref.blend_param
    directional_blend = _config_ref.directional_blend
    dir_sigma = _config_ref.dir_sigma

    if blend_mode in ["flat", "linear", "cosine", "exp_decay", "gaussian"]:
        # Generate blending weights
        weights_float, _ = weights.generate_weights(
            length=len(image_window),
            blend_mode=blend_mode,
            blend_param=blend_param,
            directional=directional_blend,
            dir_sigma=dir_sigma
        )
        # Convert float weights to fixed-point integers
        weights_int = [int(w * scale_factor) for w in weights_float]
        sum_weights_int = sum(weights_int) or 1 # Avoid division by zero

        for i, img_8bit in enumerate(image_window):
            # Convert 8-bit image to 16-bit for multiplication with scaled weights
            # This is effectively img_8bit * 256, then scaled by weight.
            # We need to ensure the image data is scaled before multiplication with fixed-point weights
            # to maintain precision.
            # A common approach for 8-bit input to 16-bit fixed-point is:
            # (image_value * scale_factor_for_image_data) * fixed_point_weight
            # Here, we're converting 8-bit (0-255) to a higher range for accumulation.
            # Let's assume input images are 0-255.
            # The weights sum to 1.0. If we multiply 0-255 by weights, sum, and divide by sum_weights,
            # we get 0-255. To use fixed-point, we scale 0-255 images up to 0-255*scale_factor.
            
            # The original blending_int.py used: acc += im.astype(np.uint32) * wi
            # where 'im' was already 16-bit (from cv2.imread(..., cv2.IMREAD_UNCHANGED))
            # and 'wi' was `int(w * (1 << scale_bits))`.
            # Our input `image_window` contains 8-bit images.
            # So, we need to scale the 8-bit image to the 16-bit range before multiplying with `weights_int`.
            # A simple way to do this is to just cast to uint16, then the final division by `scale_factor`
            # will bring it back to the 0-255 range.
            
            # Let's adjust the `weights_int` logic slightly:
            # If `weights_float` sum to 1.0, and `weights_int` sum to `scale_factor`,
            # then `(image_8bit * weights_int) / sum_weights_int` is effectively
            # `image_8bit * (weights_float)`.
            # The `accumulator` is uint32.
            accumulator += img_8bit.astype(np.uint32) * weights_int[i]

        # Normalize by sum of weights (with rounding)
        # result_16bit_fixed = ((accumulator + (sum_weights_int // 2)) // sum_weights_int).astype(np.uint16)
        # The result is still in the higher fixed-point range.
        # To get back to 0-255, we need to divide by scale_factor as well.
        
        # Simplified: weighted_sum / (sum_weights * scale_factor)
        # Since weights_int already incorporates scale_factor, we just divide by sum_weights_int.
        # And the final result needs to be scaled back down to 0-255.
        # Let's assume the accumulator is holding (image_value * scale_factor * weight_float)
        # So, accumulator / sum_weights_int should give image_value * scale_factor.
        # Then we divide by scale_factor to get 0-255.

        # The correct way for fixed-point:
        # result_accumulator = (accumulator + (sum_weights_int // 2)) # Add half sum for rounding
        # result_16bit_scaled = result_accumulator // sum_weights_int # This is still scaled by (1 << scale_bits)
        # final_8bit_z_blend = (result_16bit_scaled >> (_config_ref.scale_bits - 8)).astype(np.uint8) # Shift down to 8-bit

        # Let's re-evaluate the fixed-point math for 8-bit output.
        # Input images are 0-255 (uint8).
        # We want `sum(image_i * weight_i)`.
        # If weights are `w_i` (sum to 1.0), `image_i` (0-255).
        # Fixed point: `W_i = round(w_i * S)`, where `S = 2^scale_bits`.
        # Accumulator: `A = sum(image_i * W_i)`.
        # Result: `R = round(A / S)`.
        # So, `accumulator` already holds `image_i * W_i`.
        # We need to divide `accumulator` by `S` (scale_factor).

        # Add rounding (sum_weights_int is our S here)
        result_16bit_fixed = ((accumulator + (sum_weights_int // 2)) // sum_weights_int).astype(np.uint16)
        
        # Now, `result_16bit_fixed` is effectively `blended_value * (1 << scale_bits)`.
        # To get 8-bit 0-255, we shift down.
        # Ensure we don't shift by a negative amount if scale_bits < 8 (unlikely but defensive).
        shift_amount = max(0, _config_ref.scale_bits - 8)
        z_blended_image = (result_16bit_fixed).astype(np.uint8)
        # z_blended_image = (result_16bit_fixed >> shift_amount).astype(np.uint8)

    elif blend_mode in ["binary_contour", "gradient_contour"]:
        # These modes don't use the 'weights' array in the same way;
        # they aggregate binary or gradient information.
        # They were previously in blending_int_binary.py and took a list of files.
        # Now they need to take a list of numpy arrays.
        # The output is directly 0-255 uint8.
        
        # We need to pass the config to these functions for thresholds etc.
        # For now, let's just make a placeholder call.
        
        # Temporarily import here to avoid circular dependency if needed,
        # or ensure these functions are accessible via a common module.
        # Given they are 'contour' modes, they are distinct from weighted blending.
        # Let's put them into a new `z_contour_processor.py` or similar.
        # For now, we'll implement simplified versions here or call a placeholder.

        # Re-evaluating: The plan stated `stacking_processor.py` will handle this.
        # Let's implement the logic directly here for these two modes.

        if blend_mode == "binary_contour":
            # Convert to binary (0 or 1)
            bin_stack = [(cv2.threshold(im, _config_ref.binary_threshold, 255, cv2.THRESH_BINARY)[1] // 255) for im in image_window]
            total_presence = np.sum(np.stack(bin_stack, axis=0), axis=0)
            # Normalize to 0-255
            norm = total_presence.astype(np.float32) / (total_presence.max() or 1)
            z_blended_image = (norm * 255).astype(np.uint8)
        elif blend_mode == "gradient_contour":
            # Convert to binary (0 or 1)
            bin_stack = [(cv2.threshold(im, _config_ref.gradient_threshold, 255, cv2.THRESH_BINARY)[1] // 255) for im in image_window]
            diffs = np.abs(np.diff(np.stack(bin_stack, axis=0), axis=0)) # Absolute difference between adjacent slices
            grad = np.sum(diffs, axis=0) # Sum of absolute differences (gradient magnitude)
            # Normalize to 0-255
            norm = grad.astype(np.float32) / (grad.max() or 1)
            z_blended_image = (norm * 255).astype(np.uint8)

    elif blend_mode in ["z_column_lift", "z_contour_interp"]:
        # These modes were previously in blending_zplane.py and operated on full stacks.
        # They need to be adapted to work with the `image_window` (list of arrays).
        # Their output is directly 0-255 uint8.
        
        # For z_column_lift:
        if blend_mode == "z_column_lift":
            # Load uint8 stack directly
            arr_uint8 = np.stack(image_window, axis=0) # (Z, H, W)
            
            # Build fixed-point kernel using current blend_mode/param (from config)
            # The `generate_weights` function is already suitable for this kernel.
            # We need to pass blend_mode, blend_param, etc. from config.
            kernel_float, _ = weights.generate_weights(
                length=len(image_window),
                blend_mode=_config_ref.blend_mode, # Use the main blend mode for kernel shape
                blend_param=_config_ref.blend_param,
                directional=False, # Z-column lift typically isn't directional in this sense
                dir_sigma=0.0
            )
            kernel_int = np.round(np.array(kernel_float) * scale_factor).astype(np.uint16)
            
            # Accumulate in 32-bit
            # acc = np.zeros((H, W), dtype=np.uint32)
            # for i in range(len(image_window)):
            #     acc += arr_uint8[i].astype(np.uint32) * kernel_int[i]
            
            # Optimized tensordot for weighted sum
            # np.tensordot(kernel_int, arr_uint8.astype(np.uint32), axes=(0,0))
            # This sums along the first axis of arr_uint8 (Z-axis) weighted by kernel_int.
            accumulator = np.tensordot(kernel_int, arr_uint8.astype(np.uint32), axes=(0,0))

            # Normalize and clip to 0-255
            z_blended_image = (accumulator // scale_factor).clip(0, 255).astype(np.uint8)

            # Apply top surface smoothing if enabled
            if _config_ref.top_surface_smoothing and _config_ref.top_surface_strength > 0:
                z_blended_image = cv2.GaussianBlur(z_blended_image, (0,0), _config_ref.top_surface_strength)

        # For z_contour_interp:
        elif blend_mode == "z_contour_interp":
            # Load binary stack (0 or 1)
            stacks_binary = []
            for img_8bit in image_window:
                _, bw = cv2.threshold(img_8bit, 128, 1, cv2.THRESH_BINARY) # Use 128 as default threshold
                stacks_binary.append(bw)
            arr_binary = np.stack(stacks_binary, axis=0) # (Z, H, W)

            # Projections
            base = np.max(arr_binary, axis=0) * 255 # Max projection along Z, scaled to 0-255
            xz = np.max(arr_binary, axis=1).astype(np.uint8) # Max projection along Y
            yz = np.max(arr_binary, axis=2).astype(np.uint8) # Max projection along X

            # Sobel with int16
            sx = cv2.Sobel(xz, cv2.CV_16S, 1, 0, ksize=3)
            sy = cv2.Sobel(yz, cv2.CV_16S, 1, 0, ksize=3)
            
            mx = np.max(np.abs(sx), axis=0).astype(np.uint16)
            my = np.max(np.abs(sy), axis=0).astype(np.uint16)
            
            # Normalize 0-255
            mx = ((mx * 255) // (mx.max() or 1)).astype(np.uint8)
            my = ((my * 255) // (my.max() or 1)).astype(np.uint8)
            
            # Outer product to combine XZ and YZ contours
            contour = np.outer(my, mx) # This creates a (H, W) image from (H,) and (W,) vectors
                                         # This is likely not the intended behavior for combining 2D contour maps.
                                         # It should be an element-wise combination.
                                         # If xz is (H, Z) and yz is (W, Z), then map_x is (W,) and map_y is (H,).
                                         # np.outer(map_y, map_x) results in (H, W) which is correct.
                                         # This is a common way to combine 1D projections into a 2D map.

            # Blend base image with contour using top_surface_strength as alpha
            alpha_int = int(_config_ref.top_surface_strength * 256) # Scale 0-1 float to 0-256 int
            # mixed = (((256-alpha_int)*base + alpha_int*contour) // 256).astype(np.uint8)
            # Using cv2.addWeighted for clarity and robustness
            mixed = cv2.addWeighted(base.astype(np.float32), 1.0 - _config_ref.top_surface_strength,
                                     contour.astype(np.float32), _config_ref.top_surface_strength, 0)
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)

            if _config_ref.top_surface_smoothing and _config_ref.top_surface_strength > 0:
                mixed = cv2.GaussianBlur(mixed, (0,0), _config_ref.top_surface_strength)
            z_blended_image = mixed
    else:
        # Fallback for unknown blend modes
        print(f"Warning: Unknown blend mode '{blend_mode}'. Returning first image in window.")
        z_blended_image = image_window[0].copy()


    # --- 2. Z-LUT Application ---
    # The Z-LUT is expected to handle all intensity remapping, including
    # clamping, preserving black, and floor/ceil effects.
    final_image_after_z_lut = lut_manager.apply_z_lut(z_blended_image)

    # --- 3. XY Blending Pipeline Application ---
    # This will apply all configured XY operations (blur, sharpen, resize, etc.)
    final_processed_image = xy_blend_processor.process_xy_pipeline(final_image_after_z_lut)

    return final_processed_image

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- Stacking Processor Module Test ---")

    # Local helper for mock to ensure odd kernel sizes
    def _mock_ensure_odd_ksize(val: int) -> int:
        if val <= 0:
            return 1
        return val if val % 2 != 0 else val + 1

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

        def to_dict(self): # Required for Config.to_dict
            # This method should return a dictionary that *includes* the 'params' key
            # as the actual Config.XYBlendOperation.to_dict() does.
            # This mock's purpose is to simulate the attributes, not necessarily the serialization.
            # However, for consistency with the Config, let's make it match.
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

        def __post_init__(self): # Simulate post_init for validation
            if self.type in ["gaussian_blur", "median_blur", "unsharp_mask"]:
                self.gaussian_ksize_x = _mock_ensure_odd_ksize(self.gaussian_ksize_x)
                self.gaussian_ksize_y = _mock_ensure_odd_ksize(self.gaussian_ksize_y)
                self.median_ksize = _mock_ensure_odd_ksize(self.median_ksize)
                self.unsharp_blur_ksize = _mock_ensure_odd_ksize(self.unsharp_blur_ksize)


    class MockConfig:
        def __init__(self):
            self.primary = 3
            self.radius = 1
            self.blend_mode = "gaussian" # Test with gaussian blend
            self.blend_param = 1.0
            self.directional_blend = False
            self.dir_sigma = 1.0
            self.scale_bits = 12 # Default for integer mode
            self.binary_threshold = 128
            self.gradient_threshold = 128
            self.top_surface_smoothing = False
            self.top_surface_strength = 0.5
            self.gradient_smooth = False # Not used in this module directly now
            self.gradient_blend_strength = 0.0 # Not used in this module directly now

            # LUT settings for testing
            self.lut_source = "generated"
            self.lut_generation_type = "linear"
            self.gamma_value = 1.0
            self.linear_min_input = 0
            self.linear_max_output = 255
            self.s_curve_contrast = 0.5
            self.log_param = 10.0
            self.exp_param = 2.0
            self.sqrt_param = 1.0
            self.rodbard_param = 1.0
            self.fixed_lut_path = ""

            # XY pipeline for testing
            self.xy_blend_pipeline: List[MockXYBlendOperation] = []

        def add_xy_op(self, op_type: str, **kwargs):
            self.xy_blend_pipeline.append(MockXYBlendOperation(op_type, **kwargs))

    mock_config = MockConfig()
    set_config_reference(mock_config) # Set the mock config for all sub-modules

    # Ensure LUT is updated based on mock config
    lut_manager.update_active_lut_from_config()

    # Create dummy image window (5 slices for primary=3, radius=1)
    # Center slice (index 2) is brightest, fades out.
    dummy_images: List[np.ndarray] = []
    for i in range(5):
        img = np.zeros((50, 50), dtype=np.uint8)
        intensity = int(255 * (1 - abs(i - 2) / 2)) # Max at center, min at edges
        cv2.circle(img, (25, 25), 10 + i * 2, intensity, -1)
        dummy_images.append(img)
    
    # Add a blank image at the start and end to test None handling for radius
    # This simulates a window that extends beyond the actual image stack.
    # The image_loader would provide these as np.zeros.
    dummy_images_with_blanks = [np.zeros((50,50), dtype=np.uint8)] + dummy_images + [np.zeros((50,50), dtype=np.uint8)]
    print(f"Dummy image window length: {len(dummy_images_with_blanks)}")

    # Test Case 1: Gaussian blend with linear LUT and a simple XY blur
    print("\n--- Test Case 1: Gaussian Blend + Linear LUT + XY Gaussian Blur ---")
    mock_config.blend_mode = "gaussian"
    mock_config.blend_param = 1.0
    mock_config.directional_blend = False
    mock_config.lut_generation_type = "linear"
    mock_config.linear_min_input = 0
    mock_config.linear_max_output = 255
    mock_config.xy_blend_pipeline = []
    mock_config.add_xy_op("gaussian_blur", gaussian_ksize_x=5, gaussian_ksize_y=5, gaussian_sigma_x=1.0, gaussian_sigma_y=1.0)
    
    lut_manager.update_active_lut_from_config() # Update LUT based on config
    output_img_1 = process_image_stack(dummy_images_with_blanks)
    cv2.imwrite("test_stack_gaussian_linear_xyblur.png", output_img_1)
    print(f"Output 1 shape: {output_img_1.shape}, dtype: {output_img_1.dtype}")
    print("Output saved as test_stack_gaussian_linear_xyblur.png")

    # Test Case 2: Binary Contour blend + Gamma LUT + XY Resize
    print("\n--- Test Case 2: Binary Contour + Gamma LUT + XY Resize ---")
    mock_config.blend_mode = "binary_contour"
    mock_config.binary_threshold = 100 # Adjust threshold for binary conversion
    mock_config.lut_generation_type = "gamma"
    mock_config.gamma_value = 0.5 # Brightening gamma
    mock_config.xy_blend_pipeline = []
    mock_config.add_xy_op("resize", resize_width=100, resize_height=100, resample_mode="BICUBIC")

    lut_manager.update_active_lut_from_config() # Update LUT based on config
    output_img_2 = process_image_stack(dummy_images_with_blanks)
    cv2.imwrite("test_stack_binary_gamma_xyresize.png", output_img_2)
    print(f"Output 2 shape: {output_img_2.shape}, dtype: {output_img_2.dtype}")
    print("Output saved as test_stack_binary_gamma_xyresize.png")

    # Test Case 3: Z-Column Lift + S-Curve LUT + XY Unsharp Mask
    print("\n--- Test Case 3: Z-Column Lift + S-Curve LUT + XY Unsharp Mask ---")
    mock_config.blend_mode = "z_column_lift"
    mock_config.top_surface_smoothing = True # Enable for Z-column lift
    mock_config.top_surface_strength = 0.8 # Stronger smoothing
    mock_config.lut_generation_type = "s_curve"
    mock_config.s_curve_contrast = 0.9
    mock_config.xy_blend_pipeline = []
    mock_config.add_xy_op("unsharp_mask", unsharp_amount=1.5, unsharp_threshold=10)

    lut_manager.update_active_lut_from_config() # Update LUT based on config
    output_img_3 = process_image_stack(dummy_images_with_blanks)
    cv2.imwrite("test_stack_zcolumnlift_scurve_xyunsharp.png", output_img_3)
    print(f"Output 3 shape: {output_img_3.shape}, dtype: {output_img_3.dtype}")
    print("Output saved as test_stack_zcolumnlift_scurve_xyunsharp.png")

    # Clean up test images
    import os
    for f in ["test_stack_gaussian_linear_xyblur.png", 
              "test_stack_binary_gamma_xyresize.png", 
              "test_stack_zcolumnlift_scurve_xyunsharp.png"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")



# weights.py

"""
Generate weight kernels for blending windows of slices in Modular-Stacker.
Supports flat, linear, cosine, exponential decay, and Gaussian modes.
"""

import math
import numpy as np
from typing import List, Tuple, Any, Optional

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

def generate_weights(
    length: int,
    blend_mode: str,
    blend_param: float,
    directional: bool,
    dir_sigma: float
) -> Tuple[List[float], List[int]]:
    """
    Generates a list of normalized blending weights and their relative positions.

    Args:
        length (int): The total length of the window (number of slices).
        blend_mode (str): The type of weight curve to generate
                          ('flat', 'linear', 'cosine', 'exp_decay', 'gaussian').
        blend_param (float): Parameter for blend_mode (e.g., sigma for gaussian/exp_decay).
        directional (bool): If True, apply a directional bias.
        dir_sigma (float): Decay factor for directional bias.

    Returns:
        Tuple[List[float], List[int]]:
            - weights: Normalized list of floats summing to 1.
            - positions: Relative indices from the center of the window.
    """
    if length <= 0:
        return [], []

    center = length // 2
    weights: List[float] = []
    positions: List[int] = []

    for i in range(length):
        dist = abs(i - center)
        positions.append(i - center)

        w = 0.0 # Default weight

        if blend_mode == "flat":
            w = 1.0

        elif blend_mode == "linear":
            # Linear decay from center to edges
            # The 'span' should be the maximum distance from the center to an edge.
            # For a window of length L, the max distance from center is L/2.
            span = length / 2.0
            if span == 0: # Avoid division by zero for length 1
                w = 1.0
            else:
                w = max(0.0, 1.0 - (dist / span))

        elif blend_mode == "cosine":
            # Cosine curve from 1.0 at center to 0.0 at edges
            span = length / 2.0 or 1.0 # Avoid division by zero
            angle = (dist / span) * (math.pi / 2.0) # Map distance to 0 to pi/2
            w = max(0.0, math.cos(angle))

        elif blend_mode == "exp_decay":
            # Exponential decay from center
            if blend_param <= 0: # Avoid division by zero or non-decaying curve
                w = 1.0 if dist == 0 else 0.0 # Treat as impulse if param is zero or negative
            else:
                w = math.exp(-dist / blend_param)

        elif blend_mode == "gaussian":
            # Gaussian (normal distribution) curve
            if blend_param <= 0: # Treat as impulse if sigma is zero or negative
                w = 1.0 if dist == 0 else 0.0
            else:
                w = math.exp(-0.5 * (dist / blend_param) ** 2)
        
        # Add other modes if needed, but for now, these are the primary ones for Z-blending.
        # Binary/gradient modes are handled differently, not by these weights directly.

        weights.append(w)

    # Apply directional bias if requested
    if directional and dir_sigma > 0:
        biased_weights = []
        for (pos, w) in zip(positions, weights):
            # Bias more towards 'newer' slices (positive positions)
            # The bias factor should be 1.0 at pos=0, and increase for positive pos, decrease for negative pos
            # A simple exponential bias: bias = exp(pos / dir_sigma)
            # Or, to only boost positive: bias = 1.0 + (pos / dir_sigma) if pos > 0 else 1.0
            # Let's use a symmetric exponential decay from the center, but only apply it as a multiplier
            # for positive positions (newer slices).
            
            # The original implementation had: bias = math.exp(-abs(pos) / dir_sigma) and w = w * bias if pos > 0 else w
            # This means older slices (pos < 0) get no bias, center gets exp(0)=1, newer slices get exp(-abs(pos)/dir_sigma)
            # which is a *decreasing* bias for newer slices further from center. This seems counter-intuitive for "directional bias".
            # If "directional" means "bias towards newer slices", then positive positions should get *more* weight.

            # Let's interpret "directional bias" as making newer slices more influential.
            # A simple way: linearly increase bias for newer slices, decrease for older.
            # Or, a positive exponential for newer, negative for older.
            
            # Reverting to the original logic's *intent* (as per previous code comments):
            # "if True, bias weights in the positive (newer) direction"
            # The original code's `bias = math.exp(-abs(pos) / dir_sigma)` means smaller `abs(pos)` (closer to center)
            # gives higher `bias`. If `pos > 0`, it applies this bias. This means slices *closer* to the center
            # (on the newer side) get more weight. Slices further out on the newer side get *less* weight.
            # This is a "center-weighted bias on the newer side".

            # Let's stick to the original logic for now, assuming its intended effect.
            # The `dir_sigma` controls how sharply this bias falls off.
            bias = math.exp(-abs(pos) / dir_sigma) # Bias factor: higher for positions closer to center
            if pos > 0: # Only apply bias to slices newer than the center
                w = w * bias
            biased_weights.append(w)
        weights = biased_weights

    # Normalize sum to 1.0
    total = sum(weights)
    if total == 0:
        # If all weights are zero (e.g., length=0 or invalid params),
        # distribute evenly to avoid division by zero.
        if length > 0:
            weights = [1.0 / length] * length
        else:
            weights = []
    else:
        weights = [w / total for w in weights]

    return weights, positions

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- Weights Module Test ---")

    # Dummy Config for testing
    class MockConfig:
        def __init__(self):
            self.primary = 3
            self.radius = 2
            self.blend_mode = "gaussian"
            self.blend_param = 1.0
            self.directional_blend = False
            self.dir_sigma = 1.0

    mock_config = MockConfig()
    set_config_reference(mock_config) # Set the mock config

    window_length = mock_config.primary + 2 * mock_config.radius
    print(f"Testing window length: {window_length}")

    # Test Gaussian weights
    print("\n--- Gaussian Blend (σ=1.0, no directional) ---")
    weights, positions = generate_weights(
        length=window_length,
        blend_mode="gaussian",
        blend_param=mock_config.blend_param,
        directional=mock_config.directional_blend,
        dir_sigma=mock_config.dir_sigma
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Test Linear weights
    print("\n--- Linear Blend (no directional) ---")
    weights, positions = generate_weights(
        length=window_length,
        blend_mode="linear",
        blend_param=mock_config.blend_param, # Param not used for linear, but passed for consistency
        directional=False,
        dir_sigma=mock_config.dir_sigma
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Test Gaussian with directional bias
    print("\n--- Gaussian Blend (σ=1.0, with directional σ=1.0) ---")
    weights, positions = generate_weights(
        length=window_length,
        blend_mode="gaussian",
        blend_param=mock_config.blend_param,
        directional=True,
        dir_sigma=1.0
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Test edge case: length 1
    print("\n--- Flat Blend (length=1) ---")
    weights, positions = generate_weights(
        length=1,
        blend_mode="flat",
        blend_param=mock_config.blend_param,
        directional=False,
        dir_sigma=mock_config.dir_sigma
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Test edge case: blend_param=0 for gaussian/exp_decay
    print("\n--- Gaussian Blend (σ=0.0) ---")
    weights, positions = generate_weights(
        length=window_length,
        blend_mode="gaussian",
        blend_param=0.0,
        directional=False,
        dir_sigma=mock_config.dir_sigma
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")



# xy_blend_processor.py

"""
Processes an image through a sequence of XY blending, smoothing, sharpening,
and resizing operations as defined by the Config's xy_blend_pipeline.
"""

import cv2
import numpy as np
from typing import List, Any, Optional, Dict # Import Dict here

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

def _ensure_odd_ksize(ksize: int) -> int:
    """Ensures a kernel size is an odd integer, adjusting if necessary."""
    if ksize % 2 == 0:
        return ksize + 1 if ksize > 0 else 1
    return ksize

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_gaussian_blur(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Applies Gaussian blur to an 8-bit grayscale image.
    Uses configurable separable X and Y kernel sizes and sigmas.
    """
    # Access attributes directly from the XYBlendOperation object
    ksize_x = _ensure_odd_ksize(getattr(op, "gaussian_ksize_x", 3))
    ksize_y = _ensure_odd_ksize(getattr(op, "gaussian_ksize_y", 3))
    sigma_x = getattr(op, "gaussian_sigma_x", 0.0)
    sigma_y = getattr(op, "gaussian_sigma_y", 0.0)

    # Ensure at least a minimal kernel if both ksize and sigma are zero, to prevent error or no-op
    if ksize_x == 1 and ksize_y == 1 and sigma_x == 0.0 and sigma_y == 0.0:
        ksize_x, ksize_y = 3, 3 # Default to a small blur if no parameters given

    # cv2.GaussianBlur expects ksize as a tuple (width, height)
    return cv2.GaussianBlur(image, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_bilateral_filter(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Applies a bilateral filter to an 8-bit grayscale image.
    Effective for noise reduction while preserving edges.
    """
    # Access attributes directly from the XYBlendOperation object
    d = getattr(op, "bilateral_d", 9)
    sigma_color = getattr(op, "bilateral_sigma_color", 75.0)
    sigma_space = getattr(op, "bilateral_sigma_space", 75.0)

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_median_blur(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Applies a median blur to an 8-bit grayscale image.
    Excellent for removing salt-and-pepper noise.
    """
    # Access attributes directly from the XYBlendOperation object
    ksize = _ensure_odd_ksize(getattr(op, "median_ksize", 5))
    if ksize <= 1: # Median blur requires kernel size > 1
        return image # No-op if kernel size is 1 or invalid
    return cv2.medianBlur(image, ksize)

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_unsharp_mask(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Applies unsharp masking to an 8-bit grayscale image for sharpening.
    """
    # Access attributes directly from the XYBlendOperation object
    amount = getattr(op, "unsharp_amount", 1.0)
    threshold = getattr(op, "unsharp_threshold", 0)
    blur_ksize = _ensure_odd_ksize(getattr(op, "unsharp_blur_ksize", 5))
    blur_sigma = getattr(op, "unsharp_blur_sigma", 0.0)

    # Create a blurred version of the image
    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # Calculate the 'detail' layer (original - blurred)
    # Use float32 for intermediate calculation to avoid clipping negative values
    detail_layer = image.astype(np.float32) - blurred_image.astype(np.float32)

    # Apply threshold to the detail layer
    # Pixels where original intensity is below threshold are not sharpened
    if threshold > 0:
        # Create a mask where detail should be applied
        mask = (image > threshold).astype(np.float32)
        detail_layer *= mask # Apply mask to detail layer

    # Add the scaled detail back to the original image
    # Use addWeighted for controlled blending and automatic clipping to 0-255
    sharpened_image = cv2.addWeighted(image, 1.0 + amount, blurred_image, -amount, 0)
    
    # Ensure final output is uint8 and clipped to 0-255
    return np.clip(sharpened_image, 0, 255).astype(np.uint8)

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_resize(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Resizes an image to the specified width and height using the given resampling method.
    """
    # Access attributes directly from the XYBlendOperation object
    width = getattr(op, "resize_width", None)
    height = getattr(op, "resize_height", None)
    resample_mode = getattr(op, "resample_mode", "LANCZOS4").upper()

    if width is None and height is None:
        return image # No resize requested

    current_height, current_width = image.shape[:2]

    # Handle cases where only one dimension is specified, maintaining aspect ratio
    if width is None:
        if height is not None:
            width = int(current_width * (height / current_height))
        else: # Both are None
            return image
    elif height is None:
        if width is not None:
            height = int(current_height * (width / current_width))
        else: # Both are None
            return image
    
    # If target dimensions are the same as current, no need to resize
    if width == current_width and height == current_height:
        return image

    flags = {
        "NEAREST": cv2.INTER_NEAREST,
        "BILINEAR": cv2.INTER_LINEAR,
        "BICUBIC": cv2.INTER_CUBIC,
        "LANCZOS4": cv2.INTER_LANCZOS4,
        "AREA": cv2.INTER_AREA
    }
    interp = flags.get(resample_mode, cv2.INTER_LANCZOS4) # Default to LANCZOS4

    return cv2.resize(image, (width, height), interpolation=interp)


def process_xy_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Applies the sequence of XY blending/processing operations defined in the Config.
    The input image is expected to be an 8-bit grayscale NumPy array.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in xy_blend_processor. Cannot process XY pipeline.")
        return image

    processed_image = image.copy()

    for op in _config_ref.xy_blend_pipeline:
        op_type = op.type # Access the type directly from the XYBlendOperation object
        # op_params = op.params # This line caused the AttributeError, remove it

        if op_type == "none":
            continue # Skip no-op
        elif op_type == "gaussian_blur":
            processed_image = apply_gaussian_blur(processed_image, op) # Pass the op object directly
        elif op_type == "bilateral_filter":
            processed_image = apply_bilateral_filter(processed_image, op) # Pass the op object directly
        elif op_type == "median_blur":
            processed_image = apply_median_blur(processed_image, op) # Pass the op object directly
        elif op_type == "unsharp_mask":
            processed_image = apply_unsharp_mask(processed_image, op) # Pass the op object directly
        elif op_type == "resize":
            processed_image = apply_resize(processed_image, op) # Pass the op object directly
        else:
            print(f"Warning: Unknown XY blend operation type '{op_type}'. Skipping.")
    
    return processed_image

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- XY Blend Processor Module Test ---")
    
    # Create a dummy 8-bit grayscale image (e.g., 100x100 with some noise/features)
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[20:40, 20:40] = 150 # A square
    test_image[60:80, 60:80] = 200 # Another square
    test_image = cv2.randn(test_image, (0), (20)) # Add some random noise
    test_image = np.clip(test_image, 0, 255).astype(np.uint8)
    cv2.imwrite("test_xy_original.png", test_image)
    print("Original test image saved as test_xy_original.png")

    # Dummy Config and XYBlendOperation for testing the pipeline
    from config import Config, XYBlendOperation # Import actual classes for testing

    class MockConfig(Config): # Inherit from actual Config to get all fields
        def __init__(self):
            # Manually initialize _initialized to True to prevent recursive load from super().__init__()
            object.__setattr__(self, '_initialized', True)
            super().__init__() # Call parent dataclass init to set defaults
            self.xy_blend_pipeline: List[XYBlendOperation] = [] # Override with an empty list initially

        def add_op(self, op: XYBlendOperation):
            self.xy_blend_pipeline.append(op)

    mock_config = MockConfig()
    set_config_reference(mock_config) # Set the mock config

    # Test pipeline: Gaussian Blur -> Resize -> Unsharp Mask
    print("\nTesting pipeline: Gaussian Blur -> Resize -> Unsharp Mask")
    mock_config.add_op(XYBlendOperation("gaussian_blur", gaussian_ksize_x=5, gaussian_ksize_y=5, gaussian_sigma_x=1.0, gaussian_sigma_y=1.0))
    mock_config.add_op(XYBlendOperation("resize", resize_width=50, resize_height=50, resample_mode="BICUBIC"))
    mock_config.add_op(XYBlendOperation("unsharp_mask", unsharp_amount=1.5, unsharp_threshold=10, unsharp_blur_ksize=3))

    output_image_pipeline = process_xy_pipeline(test_image)
    cv2.imwrite("test_xy_pipeline_output.png", output_image_pipeline)
    print(f"Pipeline output shape: {output_image_pipeline.shape}, dtype: {output_image_pipeline.dtype}")
    print("Pipeline output saved as test_xy_pipeline_output.png")

    # Test individual operations
    print("\nTesting individual operations:")

    # Gaussian Blur
    # Create a dummy XYBlendOperation for individual testing
    gaussian_op = XYBlendOperation(type="gaussian_blur", gaussian_ksize_x=7, gaussian_ksize_y=7, gaussian_sigma_x=1.5, gaussian_sigma_y=1.5)
    output_gaussian = apply_gaussian_blur(test_image, gaussian_op)
    cv2.imwrite("test_xy_gaussian.png", output_gaussian)
    print("Gaussian blur output saved as test_xy_gaussian.png")

    # Bilateral Filter
    bilateral_op = XYBlendOperation(type="bilateral_filter", bilateral_d=15, bilateral_sigma_color=80.0, bilateral_sigma_space=80.0)
    output_bilateral = apply_bilateral_filter(test_image, bilateral_op)
    cv2.imwrite("test_xy_bilateral.png", output_bilateral)
    print("Bilateral filter output saved as test_xy_bilateral.png")

    # Median Blur
    median_op = XYBlendOperation(type="median_blur", median_ksize=7)
    output_median = apply_median_blur(test_image, median_op)
    cv2.imwrite("test_xy_median.png", output_median)
    print("Median blur output saved as test_xy_median.png")

    # Unsharp Mask
    unsharp_op = XYBlendOperation(type="unsharp_mask", unsharp_amount=1.2, unsharp_threshold=5, unsharp_blur_ksize=5, unsharp_blur_sigma=0.0)
    output_unsharp = apply_unsharp_mask(test_image, unsharp_op)
    cv2.imwrite("test_xy_unsharp.png", output_unsharp)
    print("Unsharp mask output saved as test_xy_unsharp.png")

    # Resize
    resize_op = XYBlendOperation(type="resize", resize_width=75, resize_height=75, resample_mode="LANCZOS4")
    output_resize = apply_resize(test_image, resize_op)
    cv2.imwrite("test_xy_resize.png", output_resize)
    print("Resize output saved as test_xy_resize.png")

    # Clean up test images
    import os
    for f in ["test_xy_original.png", "test_xy_pipeline_output.png", 
              "test_xy_gaussian.png", "test_xy_bilateral.png", 
              "test_xy_median.png", "test_xy_unsharp.png", "test_xy_resize.png"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")






