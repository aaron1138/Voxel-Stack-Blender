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
