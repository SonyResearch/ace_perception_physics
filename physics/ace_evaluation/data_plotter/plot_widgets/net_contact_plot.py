# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Net Contact Plot Widget

Stacked histogram showing net-contact counts grouped by player or date.
Robot net contacts (vx > 0 at crossing) are stacked on the bottom;
player net contacts (vx < 0) are stacked on top.
"""

from typing import Optional
import numpy as np
import pandas as pd
import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QMessageBox, QGroupBox,
)
from PySide6.QtCore import Qt

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter


_ROBOT_COLOR = "#4a90d9"   # blue
_PLAYER_COLOR = "#e07040"  # orange


class NetContactPlot(QWidget):
    """Stacked histogram of net contacts (robot vs player) grouped by player or date."""

    def __init__(self):
        super().__init__()
        self._df: Optional[pd.DataFrame] = None
        self.plots_folder: Optional[str] = None

        layout = QVBoxLayout()

        # ── Control bar ──────────────────────────────────────────────
        control_bar = QHBoxLayout()

        # Group-by selector
        group_box = QGroupBox("Group by")
        group_layout = QHBoxLayout()
        group_layout.setContentsMargins(5, 5, 5, 5)
        self.group_combo = QComboBox()
        self.group_combo.addItems(["Player", "Date"])
        self.group_combo.currentIndexChanged.connect(self._replot)
        group_layout.addWidget(self.group_combo)
        group_box.setLayout(group_layout)
        control_bar.addWidget(group_box)

        # Metric selector
        metric_box = QGroupBox("Y-Axis")
        metric_layout = QHBoxLayout()
        metric_layout.setContentsMargins(5, 5, 5, 5)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Total", "Average per rally"])
        self.metric_combo.currentIndexChanged.connect(self._replot)
        metric_layout.addWidget(self.metric_combo)
        metric_box.setLayout(metric_layout)
        control_bar.addWidget(metric_box)

        control_bar.addStretch()

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self._save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # ── Plot widget ──────────────────────────────────────────────
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Net Contacts")
        self.plot_widget.setLabel("bottom", "")
        self.plot_widget.showGrid(x=False, y=True)
        self.plot_widget.getAxis("left").enableAutoSIPrefix(False)
        self.plot_widget.getAxis("bottom").enableAutoSIPrefix(False)

        # Legend
        self.legend = self.plot_widget.addLegend(offset=(60, 10))

        layout.addWidget(self.plot_widget)

        # ── Info label ───────────────────────────────────────────────
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "QLabel { padding: 3px; background-color: #f0f0f0; "
            "border: 1px solid #ccc; }"
        )
        layout.addWidget(self.info_label)

        self.setLayout(layout)

    # ── Public API ───────────────────────────────────────────────────

    def plot_data(self, df: pd.DataFrame):
        """Accept a DataFrame from DataProcessor.extract_net_contact_data."""
        self._df = df
        self._replot()

    # ── Internal ─────────────────────────────────────────────────────

    def _replot(self):
        self.plot_widget.clear()
        self.legend.clear()

        if self._df is None or self._df.empty:
            self.info_label.setText("No net-contact data available.")
            return

        group_key = "player" if self.group_combo.currentText() == "Player" else "date"
        metric = self.metric_combo.currentText()

        grouped = self._df.groupby(group_key)
        if metric == "Total":
            robot_vals = grouped["net_robot"].sum()
            player_vals = grouped["net_player"].sum()
            y_label = "Total Net Contacts"
        else:
            robot_vals = grouped["net_robot"].mean()
            player_vals = grouped["net_player"].mean()
            y_label = "Avg Net Contacts per Rally"

        categories = list(robot_vals.index.astype(str))
        y_robot = robot_vals.values.astype(float)
        y_player = player_vals.values.astype(float)

        if len(categories) == 0:
            self.info_label.setText("No categories to display.")
            return

        x = np.arange(len(categories))
        bar_width = 0.6

        # Robot bars (bottom)
        bar_robot = pg.BarGraphItem(
            x=x,
            height=y_robot,
            width=bar_width,
            brush=pg.mkBrush(_ROBOT_COLOR),
            pen=pg.mkPen("#2c6fbb", width=1),
            name="Robot",
        )
        self.plot_widget.addItem(bar_robot)
        self.legend.addItem(bar_robot, "Robot (vx > 0)")

        # Player bars (stacked on top)
        bar_player = pg.BarGraphItem(
            x=x,
            height=y_player,
            y0=y_robot,
            width=bar_width,
            brush=pg.mkBrush(_PLAYER_COLOR),
            pen=pg.mkPen("#b04820", width=1),
            name="Player",
        )
        self.plot_widget.addItem(bar_player)
        self.legend.addItem(bar_player, "Player (vx < 0)")

        # Value labels on top of each stacked bar
        y_total = y_robot + y_player
        for xi, yr, yp, yt in zip(x, y_robot, y_player, y_total):
            if metric == "Total":
                label_text = f"{int(yr)}+{int(yp)}={int(yt)}"
            else:
                label_text = f"{yr:.2f}+{yp:.2f}={yt:.2f}"
            text = pg.TextItem(label_text, anchor=(0.5, 1.0), color="k")
            text.setPos(xi, yt)
            self.plot_widget.addItem(text)

        # X-axis tick labels
        axis = self.plot_widget.getAxis("bottom")
        axis.setTicks([list(zip(x, categories))])

        self.plot_widget.setLabel("left", y_label)
        group_label = "Player" if group_key == "player" else "Date"
        self.plot_widget.setLabel("bottom", group_label)

        total_robot = int(self._df["net_robot"].sum())
        total_player = int(self._df["net_player"].sum())
        total_net = total_robot + total_player
        total_rallies = len(self._df)
        avg = total_net / total_rallies if total_rallies else 0
        self.info_label.setText(
            f"Net contacts: {total_net} (robot: {total_robot}, player: {total_player})  |  "
            f"Rallies: {total_rallies}  |  "
            f"Overall avg: {avg:.2f} per rally"
        )

    def _save_plot(self):
        """Export the current plot as a PNG image."""
        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = plots_dir / f"net_contacts_histogram_{timestamp}.png"

        try:
            exporter = ImageExporter(self.plot_widget.plotItem)
            exporter.parameters()["width"] = 1600
            exporter.export(str(path))
            QMessageBox.information(self, "Saved", f"Plot saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))
