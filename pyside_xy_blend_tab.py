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
