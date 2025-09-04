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

# distance_lut_editor_widget.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QDoubleValidator

from config import DistanceLutPoint
from typing import List

class DistanceLutEditorWidget(QWidget):
    """
    A widget for editing a list of DistanceLutPoint objects, which define
    a Look-Up Table for distance-to-scale mapping in Enhanced EDT v2.
    """
    lut_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._populate_with_defaults()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Distance (px)", "Scale Factor"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        main_layout.addWidget(self.table_widget)

        button_layout = QHBoxLayout()
        self.add_row_button = QPushButton("Add Point")
        button_layout.addWidget(self.add_row_button)
        self.remove_row_button = QPushButton("Remove Selected Point(s)")
        button_layout.addWidget(self.remove_row_button)
        button_layout.addStretch(1)
        main_layout.addLayout(button_layout)

    def _connect_signals(self):
        self.add_row_button.clicked.connect(self._add_row)
        self.remove_row_button.clicked.connect(self._remove_selected_rows)
        self.table_widget.itemChanged.connect(self._on_item_changed)

    def _populate_with_defaults(self):
        """Adds some default points to get the user started."""
        self.set_lut([
            DistanceLutPoint(distance=10.0, scale=1.0),
            DistanceLutPoint(distance=50.0, scale=1.0)
        ])

    def _add_row(self, distance: float = 0.0, scale: float = 1.0):
        """Adds a new row to the table, optionally with initial values."""
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)

        dist_item = QTableWidgetItem(str(distance))
        scale_item = QTableWidgetItem(str(scale))

        self.table_widget.setItem(row_position, 0, dist_item)
        self.table_widget.setItem(row_position, 1, scale_item)
        self._sort_table()

    def _remove_selected_rows(self):
        selected_rows = sorted(list(set(index.row() for index in self.table_widget.selectedIndexes())), reverse=True)
        if not selected_rows:
            return

        self.table_widget.blockSignals(True)
        for row in selected_rows:
            self.table_widget.removeRow(row)
        self.table_widget.blockSignals(False)
        self.lut_changed.emit()

    def _on_item_changed(self, item: QTableWidgetItem):
        """When an item is changed, validate it and emit the lut_changed signal."""
        # A simple validation could be done here if needed, e.g., ensuring numbers are valid.
        # For now, we just emit the signal.
        self._sort_table()
        self.lut_changed.emit()

    def _sort_table(self):
        """Sorts the table based on the distance column."""
        self.table_widget.blockSignals(True)
        self.table_widget.sortItems(0, Qt.AscendingOrder)
        self.table_widget.blockSignals(False)

    def set_lut(self, lut_points: List[DistanceLutPoint]):
        """Populates the table with a list of DistanceLutPoint objects."""
        self.table_widget.blockSignals(True)
        self.table_widget.setRowCount(0) # Clear the table
        for point in lut_points:
            self._add_row(point.distance, point.scale)
        self.table_widget.blockSignals(False)

    def get_lut(self) -> List[DistanceLutPoint]:
        """Retrieves the list of DistanceLutPoint objects from the table."""
        points = []
        for row in range(self.table_widget.rowCount()):
            try:
                distance = float(self.table_widget.item(row, 0).text())
                scale = float(self.table_widget.item(row, 1).text())
                points.append(DistanceLutPoint(distance=distance, scale=scale))
            except (ValueError, AttributeError):
                # Skip rows that are invalid or empty
                continue
        return points
