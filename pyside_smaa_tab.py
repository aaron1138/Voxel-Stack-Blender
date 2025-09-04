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

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel
)

from config import app_config as config
from lut_editor_widget import LutEditorWidget

class SmaaTab(QWidget):
    """
    A dedicated tab for settings related to the Morphological AA mode.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self._connect_signals()
        self.load_settings()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        info_label = QLabel("<i>Note: This mode requires the TileDB Backend to be enabled. It performs anti-aliasing on XY, XZ, and YZ planes.</i>")
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        self.z_correction_lut_editor = LutEditorWidget(self)
        main_layout.addWidget(self.z_correction_lut_editor)

        main_layout.addStretch(1)

    def _connect_signals(self):
        # Connect the LUT editor's signal to a method that will update the config and the plot
        self.z_correction_lut_editor.lut_params_changed.connect(self._on_z_lut_params_changed)

    def _on_z_lut_params_changed(self):
        """
        Handles updates from the LUT editor, saving changes and replotting the curve.
        """
        # Save the updated parameters to the global config object
        self.config.z_correction_lut = self.z_correction_lut_editor._lut_params
        # Trigger the editor to replot its own curve
        self.z_correction_lut_editor.plot_current_lut()

    def load_settings(self):
        """Loads settings from the global config into the widgets."""
        self.z_correction_lut_editor.set_lut_params(self.config.z_correction_lut)

    def save_settings(self):
        """Saves settings from the widgets to the global config."""
        # The lut_params_changed signal already handles this, but we can be explicit.
        self._save_z_correction_lut_params()
