"""
Copyright (c) 2025 Aaron Baca

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# app-blend-slices.py

import cv2
import numpy as np
import os
import threading
import time
import re # Import the regular expression module

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QSettings

# --- Image Processing Functions ---

def load_image(filepath):
    """
    Loads an 8-bit grayscale image and ensures it's binary (0 or 255).
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    # Ensure it's truly binary (0 for black, 255 for white)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img, img # Return both binary and original grayscale

def find_prior_combined_white_mask(prior_images_list):
    """
    Combines all white areas from a list of prior binary images into a single mask.
    """
    if not prior_images_list:
        return None # No prior images to combine

    # Start with the first prior image's white areas
    combined_mask = prior_images_list[0].copy()

    # Logically OR with subsequent prior images
    for i in range(1, len(prior_images_list)):
        combined_mask = cv2.bitwise_or(combined_mask, prior_images_list[i])

    return combined_mask

def calculate_receding_gradient_field(current_white_mask, prior_white_combined_mask, debug_base_filename, output_folder, debug_save_intermediate, use_fixed_normalization, fixed_fade_distance, gamma):
    """
    Calculates a normalized 0-255 distance field that radiates from the edges
    of the current_white_mask and extends into the areas that were white in prior layers.
    This creates a "radiating" gradient effect, fading out from the current shape.
    """
    if prior_white_combined_mask is None:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 1. Identify "receding white areas": Pixels that were white in prior layers
    #    but are now black in the current image. This is where the gradient will appear.
    _, receding_white_areas = cv2.threshold(
        cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask)),
        127, 255, cv2.THRESH_BINARY
    )

    if debug_save_intermediate:
        cv2.imwrite(os.path.join(output_folder, f"{debug_base_filename}_debug_03a_receding_white_areas.png"), receding_white_areas)

    # If there are no receding areas, return an empty gradient
    if cv2.countNonZero(receding_white_areas) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 2. Calculate the distance transform from the *current* shape's boundary.
    # The `distanceTransform` function measures distance from non-zero pixels to zero pixels.
    distance_transform_src = cv2.bitwise_not(current_white_mask)

    if debug_save_intermediate:
        cv2.imwrite(os.path.join(output_folder, f"{debug_base_filename}_debug_03b_dist_src_for_transform.png"), distance_transform_src)

    # The result will have values of 0 at the current shape's boundary, increasing outward.
    distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

    # 3. Use the receding area mask to select the relevant part of the distance map.
    receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)

    # If the resulting gradient area has no distance data, return a blank image.
    if np.max(receding_distance_map) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 4. Normalize the gradient based on the selected method.
    if use_fixed_normalization:
        # Normalize with a fixed, maximum fade distance.
        clipped_distance_map = np.clip(receding_distance_map, 0, fixed_fade_distance)
        normalized_map = (clipped_distance_map / fixed_fade_distance)
    else:
        # Original behavior: normalize based on the local max distance.
        min_val, max_val, _, _ = cv2.minMaxLoc(receding_distance_map, mask=receding_white_areas)
        if max_val == min_val:
            return np.zeros_like(current_white_mask, dtype=np.uint8)
        normalized_map = (receding_distance_map - min_val) / (max_val - min_val)

    # 5. Invert the normalized map and apply gamma to create the final gradient.
    # White (255) is at the edge of the current shape, fading to black (0) at the end of the gradient.
    inverted_normalized_map = 1 - normalized_map
    final_gradient_map = (255 * (inverted_normalized_map**gamma)).astype(np.uint8)

    # 6. Ensure the gradient only exists where it should.
    final_gradient_map = cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=receding_white_areas)
    
    return final_gradient_map

def merge_to_output(original_current_image, receding_gradient, current_binary_image):
    """
    Merges the calculated receding gradient onto the original current image using a
    blending technique to ensure a smooth, monotonic transition.
    """
    # Create a black canvas of the same shape as the images.
    output_image = np.zeros_like(current_binary_image, dtype=np.uint8)

    # Place the gradient first.
    gradient_mask = (receding_gradient > 0)
    output_image[gradient_mask] = receding_gradient[gradient_mask]

    # Overwrite the area of the current shape with the original image's values.
    # This ensures that the solid white part and its anti-aliased edge are preserved,
    # creating a smooth transition to the gradient.
    current_shape_pixels = (original_current_image > 0)
    output_image[current_shape_pixels] = original_current_image[current_shape_pixels]

    return output_image

# --- Main Processing Logic Class (runs in a separate QThread) ---

class ImageProcessorThread(QThread):
    # Define signals to communicate with the GUI
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, input_folder, output_folder, N, start_index, stop_index, debug_save_intermediate, use_fixed_normalization, fixed_fade_distance, gamma):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.N = N
        self.start_index = start_index
        self.stop_index = stop_index
        self._is_running = True # Flag to control thread execution
        self.debug_save_intermediate = debug_save_intermediate
        self.use_fixed_normalization = use_fixed_normalization
        self.fixed_fade_distance = fixed_fade_distance
        self.gamma = gamma

    def run(self):
        self.status_update.emit("Processing started...")
        
        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            if match:
                return int(match.group(1))
            return float('inf')

        image_filenames = sorted([f for f in os.listdir(self.input_folder) if f.lower().endswith(('.png', '.bmp', '.tif'))],
                                 key=get_numeric_part)

        if self.start_index is not None:
            image_filenames = [f for f in image_filenames if get_numeric_part(f) >= self.start_index]
        if self.stop_index is not None:
            image_filenames = [f for f in image_filenames if get_numeric_part(f) <= self.stop_index]
        
        total_images = len(image_filenames)
        processed_count = 0
        
        prior_images_cache = []

        for i, filename in enumerate(image_filenames):
            if not self._is_running:
                self.status_update.emit("Processing stopped by user.")
                break

            filepath = os.path.join(self.input_folder, filename)
            base_filename_no_ext = os.path.splitext(filename)[0]
            self.status_update.emit(f"Processing {filename}...")
            
            try:
                current_binary_image, current_original_image = load_image(filepath)
                if self.debug_save_intermediate:
                    cv2.imwrite(os.path.join(self.output_folder, f"{base_filename_no_ext}_debug_01_current_binary.png"), current_binary_image)

                prior_white_combined_mask = find_prior_combined_white_mask(prior_images_cache)
                
                if self.debug_save_intermediate and prior_white_combined_mask is not None:
                    cv2.imwrite(os.path.join(self.output_folder, f"{base_filename_no_ext}_debug_02_prior_combined_white.png"), prior_white_combined_mask)
                elif self.debug_save_intermediate and prior_white_combined_mask is None:
                    self.status_update.emit(f"No prior combined white mask for {filename}.")

                receding_gradient = calculate_receding_gradient_field(
                    current_binary_image, prior_white_combined_mask, 
                    base_filename_no_ext, self.output_folder, self.debug_save_intermediate,
                    self.use_fixed_normalization, self.fixed_fade_distance, self.gamma
                )

                if self.debug_save_intermediate:
                    cv2.imwrite(os.path.join(self.output_folder, f"{base_filename_no_ext}_debug_03_final_gradient.png"), receding_gradient)

                if cv2.countNonZero(receding_gradient) > 0:
                    self.status_update.emit(f"Gradient detected for {filename}.")
                else:
                    self.status_update.emit(f"No receding gradient detected for {filename}. Output will primarily reflect current image.")

                output_image_layer = merge_to_output(current_original_image, receding_gradient, current_binary_image)
                output_filename = f"processed_{base_filename_no_ext}.png"
                output_filepath = os.path.join(self.output_folder, output_filename)
                cv2.imwrite(output_filepath, output_image_layer)

                prior_images_cache.append(current_binary_image)
                if len(prior_images_cache) > self.N:
                    prior_images_cache.pop(0)

                processed_count += 1
                progress = int((processed_count / total_images) * 100)
                self.progress_update.emit(progress)

            except Exception as e:
                self.error_signal.emit(f"Error processing {filename}: {e}")
                self._is_running = False
                break

        if self._is_running:
            self.status_update.emit("Processing complete!")
        self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False

# --- GUI Application (PySide6) ---

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Stack Euclidean Distance Blending")
        self.settings = QSettings("MyCompany", "ImageProcessorApp")
        self.init_ui()
        self.load_settings()
        self.processor_thread = None

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Input/Output Folders
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Input Image Folder:"))
        self.input_folder_edit = QLineEdit()
        self.input_folder_button = QPushButton("Browse...")
        self.input_folder_button.clicked.connect(self.browse_input_folder)
        folder_layout.addWidget(self.input_folder_edit)
        folder_layout.addWidget(self.input_folder_button)
        main_layout.addLayout(folder_layout)

        folder_layout_out = QHBoxLayout()
        folder_layout_out.addWidget(QLabel("Output Image Folder:"))
        self.output_folder_edit = QLineEdit()
        self.output_folder_button = QPushButton("Browse...")
        self.output_folder_button.clicked.connect(self.browse_output_folder)
        folder_layout_out.addWidget(self.output_folder_edit)
        folder_layout_out.addWidget(self.output_folder_button)
        main_layout.addLayout(folder_layout_out)

        # N Layers, Start/Stop Index
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("N Layers (look down):"))
        self.n_layers_edit = QLineEdit("3")
        self.n_layers_edit.setToolTip("Number of prior layers to check for overlap")
        params_layout.addWidget(self.n_layers_edit)
        
        params_layout.addWidget(QLabel("Start Index (inclusive):"))
        self.start_idx_edit = QLineEdit("0")
        self.start_idx_edit.setToolTip("Start processing from this image number (e.g., 001.png -> 1)")
        params_layout.addWidget(self.start_idx_edit)

        params_layout.addWidget(QLabel("Stop Index (inclusive):"))
        self.stop_idx_edit = QLineEdit("")
        self.stop_idx_edit.setToolTip("Stop processing at this image number (leave blank for all)")
        params_layout.addWidget(self.stop_idx_edit)

        main_layout.addLayout(params_layout)
        
        # Gradient Parameters
        gradient_params_layout = QHBoxLayout()
        self.fixed_normalization_checkbox = QCheckBox("Use Fixed Fade Distance")
        self.fixed_normalization_checkbox.setChecked(False) # Default to off
        self.fixed_normalization_checkbox.setToolTip("If checked, gradient fades over a fixed distance, preventing a 'halo' on small shapes.")
        gradient_params_layout.addWidget(self.fixed_normalization_checkbox)

        gradient_params_layout.addWidget(QLabel("Fade Distance (pixels):"))
        self.fade_distance_edit = QLineEdit("10")
        self.fade_distance_edit.setToolTip("Pixel distance over which the gradient should fade. Only used if 'Use Fixed Fade Distance' is checked.")
        gradient_params_layout.addWidget(self.fade_distance_edit)
        
        gradient_params_layout.addWidget(QLabel("Gamma:"))
        self.gamma_edit = QLineEdit("0.7")
        self.gamma_edit.setToolTip("Gamma value to control the fade profile of the gradient. < 1.0 makes the fade more gradual.")
        gradient_params_layout.addWidget(self.gamma_edit)

        main_layout.addLayout(gradient_params_layout)


        # Debug Images Checkbox
        checkbox_layout = QHBoxLayout()
        self.debug_checkbox = QCheckBox("Save Debug Images")
        self.debug_checkbox.setChecked(False) # Default to off
        checkbox_layout.addWidget(self.debug_checkbox)
        main_layout.addLayout(checkbox_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_stop_button = QPushButton("Start Processing")
        self.start_stop_button.setFixedSize(150, 40)
        self.start_stop_button.clicked.connect(self.toggle_processing)
        button_layout.addWidget(self.start_stop_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.setFixedSize(100, 40)
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.exit_button)
        main_layout.addLayout(button_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Status Label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("background-color: black; color: yellow; padding: 5px; border: 1px solid #555;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def load_settings(self):
        """Loads application settings from QSettings."""
        self.input_folder_edit.setText(self.settings.value("input_folder", ""))
        self.output_folder_edit.setText(self.settings.value("output_folder", ""))
        self.n_layers_edit.setText(self.settings.value("n_layers", "3"))
        self.start_idx_edit.setText(self.settings.value("start_index", "0"))
        self.stop_idx_edit.setText(self.settings.value("stop_index", ""))
        self.debug_checkbox.setChecked(self.settings.value("debug_images", "false").lower() == "true")
        self.fixed_normalization_checkbox.setChecked(self.settings.value("fixed_normalization", "false").lower() == "true")
        self.fade_distance_edit.setText(self.settings.value("fade_distance", "10"))
        self.gamma_edit.setText(self.settings.value("gamma", "0.7"))

    def save_settings(self):
        """Saves current application settings to QSettings."""
        self.settings.setValue("input_folder", self.input_folder_edit.text())
        self.settings.setValue("output_folder", self.output_folder_edit.text())
        self.settings.setValue("n_layers", self.n_layers_edit.text())
        self.settings.setValue("start_index", self.start_idx_edit.text())
        self.settings.setValue("stop_index", self.stop_idx_edit.text())
        self.settings.setValue("debug_images", str(self.debug_checkbox.isChecked()))
        self.settings.setValue("fixed_normalization", str(self.fixed_normalization_checkbox.isChecked()))
        self.settings.setValue("fade_distance", self.fade_distance_edit.text())
        self.settings.setValue("gamma", self.gamma_edit.text())

    def closeEvent(self, event):
        """Overrides the close event to save settings before closing."""
        self.save_settings()
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop_processing()
            self.processor_thread.wait(2000)
        event.accept()

    @Slot()
    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if folder:
            self.input_folder_edit.setText(folder)

    @Slot()
    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Image Folder")
        if folder:
            self.output_folder_edit.setText(folder)

    @Slot()
    def toggle_processing(self):
        if self.processor_thread is None or not self.processor_thread.isRunning():
            input_folder = self.input_folder_edit.text()
            output_folder = self.output_folder_edit.text()
            debug_save_intermediate = self.debug_checkbox.isChecked()
            use_fixed_normalization = self.fixed_normalization_checkbox.isChecked()
            
            try:
                N = int(self.n_layers_edit.text())
                if N < 0: raise ValueError("N Layers must be non-negative.")
                fade_distance = float(self.fade_distance_edit.text())
                if fade_distance <= 0: raise ValueError("Fade Distance must be a positive number.")
                gamma = float(self.gamma_edit.text())
                if gamma <= 0: raise ValueError("Gamma must be a positive number.")
            except ValueError as e:
                QMessageBox.critical(self, "Input Error", str(e))
                return

            try:
                start_idx = int(self.start_idx_edit.text()) if self.start_idx_edit.text() else 0
                if start_idx < 0: raise ValueError("Start Index must be non-negative.")
            except ValueError:
                QMessageBox.critical(self, "Input Error", "Start Index must be an integer.")
                return

            try:
                stop_idx_str = self.stop_idx_edit.text()
                stop_idx = int(stop_idx_str) if stop_idx_str else None
                if stop_idx is not None and stop_idx < 0: raise ValueError("Stop Index must be non-negative or empty.")
            except ValueError:
                QMessageBox.critical(self, "Input Error", "Stop Index must be an integer or empty.")
                return

            if not os.path.isdir(input_folder):
                QMessageBox.critical(self, "Folder Error", "Input folder does not exist or is not a directory.")
                return
            if not os.path.isdir(output_folder):
                QMessageBox.critical(self, "Folder Error", "Output folder does not exist or is not a directory.")
                return

            self.processor_thread = ImageProcessorThread(
                input_folder, output_folder, N, start_idx, stop_idx, debug_save_intermediate,
                use_fixed_normalization, fade_distance, gamma
            )
            self.processor_thread.status_update.connect(self.update_status)
            self.processor_thread.progress_update.connect(self.update_progress)
            self.processor_thread.error_signal.connect(self.show_error)
            self.processor_thread.finished_signal.connect(self.processing_finished)
            
            self.processor_thread.start()
            self.start_stop_button.setText("Stop Processing")
            self.status_label.setText("Status: Initializing processing...")
            self.progress_bar.setValue(0)
            self.set_ui_enabled(False)
        else:
            self.processor_thread.stop_processing()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
            self.status_label.setText("Status: Stopping process...")

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(str)
    def show_error(self, message):
        QMessageBox.critical(self, "Processing Error", message)
        self.processing_finished()

    @Slot()
    def processing_finished(self):
        self.start_stop_button.setText("Start Processing")
        self.start_stop_button.setEnabled(True)
        self.set_ui_enabled(True)
        self.processor_thread = None

    def set_ui_enabled(self, enabled):
        """Helper to enable/disable UI elements during processing."""
        self.input_folder_edit.setEnabled(enabled)
        self.input_folder_button.setEnabled(enabled)
        self.output_folder_edit.setEnabled(enabled)
        self.output_folder_button.setEnabled(enabled)
        self.n_layers_edit.setEnabled(enabled)
        self.start_idx_edit.setEnabled(enabled)
        self.stop_idx_edit.setEnabled(enabled)
        self.debug_checkbox.setEnabled(enabled)
        self.fixed_normalization_checkbox.setEnabled(enabled)
        self.fade_distance_edit.setEnabled(enabled)
        self.gamma_edit.setEnabled(enabled)
        self.exit_button.setEnabled(enabled)

if __name__ == "__main__":
    app = QApplication([])
    window = ImageProcessorApp()
    window.show()
    app.exec()