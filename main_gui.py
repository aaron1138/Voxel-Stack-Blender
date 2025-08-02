# main_gui.py

import sys
import os
import threading 
from typing import Optional 

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget,
    QStatusBar, QLabel, QMessageBox, QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QIcon 

# Import application modules
from config import app_config as config
import lut_manager
import run_logger
import pipeline_runner
import image_loader # Import image_loader to set config ref

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
    progress_update_signal = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular-Stacker")
        self.setGeometry(100, 100, 1000, 800)

        # Initialize global module references with the config instance
        lut_manager.set_config_reference(config)
        run_logger.set_config_reference(config)
        pipeline_runner.set_config_reference(config)
        image_loader.set_config_reference(config) # Ensure image_loader gets the config

        self._setup_ui()
        self._connect_signals()

        self.runner: Optional[pipeline_runner.PipelineRunner] = None
        self.progress_update_signal.connect(self._update_progress_bar)

        config.progress_callback = self.progress_update_signal.emit
        # The stop callback is now handled by the runner's internal event
        config.stop_callback = None 

        self.progress_dialog: Optional[QProgressDialog] = None

    def _setup_ui(self):
        """Sets up the main window layout and tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

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
        self.lut_tab.lut_changed.connect(self._on_lut_changed)

    def _update_config_from_gui(self):
        """Collects all settings from GUI tabs and updates the global config."""
        try:
            # Collect settings from each tab
            config_updates = {}
            config_updates.update(self.file_io_tab.get_config())
            config_updates.update(self.stacking_tab.get_config())
            config_updates.update(self.lut_tab.get_config())
            config_updates.update(self.xy_blend_tab.get_config())
            config_updates.update(self.advanced_tab.get_config())

            for key, value in config_updates.items():
                setattr(config, key, value)
            
            config.__post_init__()

            self.statusBar.showMessage("Settings updated from GUI.", 3000)
            return True
        except (ValueError, Exception) as e:
            QMessageBox.critical(self, "Configuration Error", f"An error occurred while updating settings: {e}")
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
            return

        try:
            lut_manager.update_active_lut_from_config()
        except Exception as e:
            QMessageBox.critical(self, "LUT Error", f"Failed to prepare LUT for processing: {e}")
            return

        self.file_io_tab.set_run_button_state(False)
        self.file_io_tab.set_stop_button_state(True)
        self.statusBar.showMessage("Processing started...", 0)

        self.runner = pipeline_runner.PipelineRunner()
        
        self.progress_dialog = QProgressDialog("Processing Images...", "Cancel", 0, self.runner.total_output_stacks, self)
        self.progress_dialog.setWindowTitle("Modular-Stacker Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self._stop_processing)
        self.progress_dialog.show()

        self.pipeline_thread = threading.Thread(target=self.runner.run_pipeline, name="PipelineThread")
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()

        self.check_pipeline_timer = QTimer(self)
        self.check_pipeline_timer.timeout.connect(self._check_pipeline_status)
        self.check_pipeline_timer.start(100)

    @Slot()
    def _stop_processing(self):
        """Signals the pipeline to stop."""
        if self.runner:
            self.runner.stop_pipeline()
            self.statusBar.showMessage("Stop requested. Waiting for pipeline to halt...", 0)
            self.file_io_tab.set_stop_button_state(False)

    @Slot(int, int)
    def _update_progress_bar(self, current: int, total: int):
        """Updates the progress dialog and status bar."""
        if self.progress_dialog:
            self.progress_dialog.setMaximum(total)
            self.progress_dialog.setValue(current)
            self.statusBar.showMessage(f"Processing: {current}/{total} stacks", 0)

    @Slot()
    def _check_pipeline_status(self):
        """Periodically checks the status of the pipeline thread."""
        if self.pipeline_thread and not self.pipeline_thread.is_alive():
            self.check_pipeline_timer.stop()
            
            if self.progress_dialog:
                if self.runner and not self.runner.is_stop_requested():
                     self.progress_dialog.setValue(self.progress_dialog.maximum())
                self.progress_dialog.close()

            if self.runner and self.runner.is_stop_requested():
                self.statusBar.showMessage("Processing stopped by user.", 5000)
            elif self.runner and self.runner.total_output_stacks == 0:
                self.statusBar.showMessage("No images to process based on current configuration.", 5000)
            else:
                self.statusBar.showMessage("Processing complete.", 5000)
            
            self._reset_gui_for_new_run()

    def _reset_gui_for_new_run(self):
        """Resets GUI elements after a run completes or is stopped."""
        self.file_io_tab.set_run_button_state(True)
        self.file_io_tab.set_stop_button_state(False)
        if self.runner:
            self.runner.reset_stop_flag()

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
        try:
            lut_manager.update_active_lut_from_config()
            self.lut_tab.apply_settings(config) # Make sure LUT tab reloads its state
        except Exception as e:
            QMessageBox.warning(self, "Load Settings Error", f"Failed to load LUT from configuration: {e}.")
        
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
