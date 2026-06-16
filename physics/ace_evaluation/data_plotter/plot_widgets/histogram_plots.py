# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Histogram Plot Widgets

Contains widgets for displaying distribution histograms.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import pathlib
from datetime import datetime

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QCheckBox, QGroupBox, QMessageBox
from PySide6.QtCore import Signal, Qt

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from ace_evaluation.utilities.data_classes import FlightSegment


class HistogramPlot(QWidget):
    """Widget for displaying speed and spin histograms"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Control bar with player selection
        control_bar = QHBoxLayout()

        # Player selection group
        player_group = QGroupBox("Players")
        player_layout = QVBoxLayout()
        player_layout.setSpacing(2)
        player_layout.setContentsMargins(5, 5, 5, 5)

        self.player_checkboxes = {}
        players = ["Robot", "Player"]

        for player in players:
            checkbox = QCheckBox(player)
            checkbox.setChecked(True)  # Both checked by default
            checkbox.stateChanged.connect(self.on_player_selection_changed)
            self.player_checkboxes[player] = checkbox
            player_layout.addWidget(checkbox)

        player_group.setLayout(player_layout)
        control_bar.addWidget(player_group)
        control_bar.addStretch()

        # Add save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Create histogram plot widgets in a 2x2 grid
        self.plot_widget = pg.GraphicsLayoutWidget()

        # Row 0 - Speed histograms
        self.plot_speed_x = self.plot_widget.addPlot(row=0, col=0, title="X Speed Distribution")
        self.plot_speed_x.setLabel("left", "Count")
        self.plot_speed_x.setLabel("bottom", "Speed (m/s)")
        self.plot_speed_x.showGrid(x=True, y=True)
        self.plot_speed_x.getAxis('left').enableAutoSIPrefix(False)
        self.plot_speed_x.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_speed_y = self.plot_widget.addPlot(row=0, col=1, title="Y Speed Distribution")
        self.plot_speed_y.setLabel("left", "Count")
        self.plot_speed_y.setLabel("bottom", "Speed (m/s)")
        self.plot_speed_y.showGrid(x=True, y=True)
        self.plot_speed_y.getAxis('left').enableAutoSIPrefix(False)
        self.plot_speed_y.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_speed_z = self.plot_widget.addPlot(row=0, col=2, title="Z Speed Distribution")
        self.plot_speed_z.setLabel("left", "Count")
        self.plot_speed_z.setLabel("bottom", "Speed (m/s)")
        self.plot_speed_z.showGrid(x=True, y=True)
        self.plot_speed_z.getAxis('left').enableAutoSIPrefix(False)
        self.plot_speed_z.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_speed_mag = self.plot_widget.addPlot(row=0, col=3, title="Speed Magnitude")
        self.plot_speed_mag.setLabel("left", "Count")
        self.plot_speed_mag.setLabel("bottom", "Speed (m/s)")
        self.plot_speed_mag.showGrid(x=True, y=True)
        self.plot_speed_mag.getAxis('left').enableAutoSIPrefix(False)
        self.plot_speed_mag.getAxis('bottom').enableAutoSIPrefix(False)

        # Row 1 - Spin histograms
        self.plot_spin_x = self.plot_widget.addPlot(row=1, col=0, title="X Spin Distribution")
        self.plot_spin_x.setLabel("left", "Count")
        self.plot_spin_x.setLabel("bottom", "Spin (rad/s)")
        self.plot_spin_x.showGrid(x=True, y=True)
        self.plot_spin_x.getAxis('left').enableAutoSIPrefix(False)
        self.plot_spin_x.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_spin_y = self.plot_widget.addPlot(row=1, col=1, title="Y Spin Distribution")
        self.plot_spin_y.setLabel("left", "Count")
        self.plot_spin_y.setLabel("bottom", "Spin (rad/s)")
        self.plot_spin_y.showGrid(x=True, y=True)
        self.plot_spin_y.getAxis('left').enableAutoSIPrefix(False)
        self.plot_spin_y.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_spin_z = self.plot_widget.addPlot(row=1, col=2, title="Z Spin Distribution")
        self.plot_spin_z.setLabel("left", "Count")
        self.plot_spin_z.setLabel("bottom", "Spin (rad/s)")
        self.plot_spin_z.showGrid(x=True, y=True)
        self.plot_spin_z.getAxis('left').enableAutoSIPrefix(False)
        self.plot_spin_z.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_spin_mag = self.plot_widget.addPlot(row=1, col=3, title="Spin Magnitude")
        self.plot_spin_mag.setLabel("left", "Count")
        self.plot_spin_mag.setLabel("bottom", "Spin (rad/s)")
        self.plot_spin_mag.showGrid(x=True, y=True)
        self.plot_spin_mag.getAxis('left').enableAutoSIPrefix(False)
        self.plot_spin_mag.getAxis('bottom').enableAutoSIPrefix(False)

        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # Store data
        self.speeds_robot = None
        self.spins_robot = None
        self.speeds_player = None
        self.spins_player = None
        self.match_file = None  # Store match file path for saving

    def on_player_selection_changed(self):
        """Handle player selection change"""
        if self.speeds_robot is not None or self.speeds_player is not None:
            self.plot_data(self.speeds_robot, self.spins_robot, self.speeds_player, self.spins_player)

    def plot_data(self, speeds_robot, spins_robot, speeds_player, spins_player):
        """Plot histogram data for speeds and spins

        Args:
            speeds_robot: List of (vx, vy, vz, vmag) arrays for robot
            spins_robot: List of (wx, wy, wz, wmag) arrays for robot
            speeds_player: List of (vx, vy, vz, vmag) arrays for player
            spins_player: List of (wx, wy, wz, wmag) arrays for player
        """
        # Store data
        self.speeds_robot = speeds_robot
        self.spins_robot = spins_robot
        self.speeds_player = speeds_player
        self.spins_player = spins_player

        # Get player selection
        show_robot = self.player_checkboxes["Robot"].isChecked()
        show_player = self.player_checkboxes["Player"].isChecked()

        # Clear all plots
        self.plot_speed_x.clear()
        self.plot_speed_y.clear()
        self.plot_speed_z.clear()
        self.plot_speed_mag.clear()
        self.plot_spin_x.clear()
        self.plot_spin_y.clear()
        self.plot_spin_z.clear()
        self.plot_spin_mag.clear()

        # Plot robot data
        if show_robot and speeds_robot is not None:
            vx, vy, vz, vmag = speeds_robot
            wx, wy, wz, wmag = spins_robot

            # Speed histograms
            y_vx, x_vx = np.histogram(vx, bins=50)
            self.plot_speed_x.plot(x_vx, y_vx, stepMode=True, fillLevel=0, brush=(100, 100, 255, 100), pen=pg.mkPen(color=(0, 0, 255), width=2))

            y_vy, x_vy = np.histogram(vy, bins=50)
            self.plot_speed_y.plot(x_vy, y_vy, stepMode=True, fillLevel=0, brush=(100, 100, 255, 100), pen=pg.mkPen(color=(0, 0, 255), width=2))

            y_vz, x_vz = np.histogram(vz, bins=50)
            self.plot_speed_z.plot(x_vz, y_vz, stepMode=True, fillLevel=0, brush=(100, 100, 255, 100), pen=pg.mkPen(color=(0, 0, 255), width=2))

            y_vmag, x_vmag = np.histogram(vmag, bins=50)
            self.plot_speed_mag.plot(x_vmag, y_vmag, stepMode=True, fillLevel=0, brush=(100, 100, 255, 100), pen=pg.mkPen(color=(0, 0, 255), width=2))

            # Spin histograms
            y_wx, x_wx = np.histogram(wx, bins=50)
            self.plot_spin_x.plot(x_wx, y_wx, stepMode=True, fillLevel=0, brush=(100, 100, 255, 100), pen=pg.mkPen(color=(0, 0, 255), width=2))

            y_wy, x_wy = np.histogram(wy, bins=50)
            self.plot_spin_y.plot(x_wy, y_wy, stepMode=True, fillLevel=0, brush=(100, 100, 255, 100), pen=pg.mkPen(color=(0, 0, 255), width=2))

            y_wz, x_wz = np.histogram(wz, bins=50)
            self.plot_spin_z.plot(x_wz, y_wz, stepMode=True, fillLevel=0, brush=(100, 100, 255, 100), pen=pg.mkPen(color=(0, 0, 255), width=2))

            y_wmag, x_wmag = np.histogram(wmag, bins=50)
            self.plot_spin_mag.plot(x_wmag, y_wmag, stepMode=True, fillLevel=0, brush=(100, 100, 255, 100), pen=pg.mkPen(color=(0, 0, 255), width=2))

        # Plot player data
        if show_player and speeds_player is not None:
            vx, vy, vz, vmag = speeds_player
            wx, wy, wz, wmag = spins_player

            # Speed histograms
            y_vx, x_vx = np.histogram(vx, bins=50)
            self.plot_speed_x.plot(x_vx, y_vx, stepMode=True, fillLevel=0, brush=(255, 100, 100, 100), pen=pg.mkPen(color=(255, 0, 0), width=2))

            y_vy, x_vy = np.histogram(vy, bins=50)
            self.plot_speed_y.plot(x_vy, y_vy, stepMode=True, fillLevel=0, brush=(255, 100, 100, 100), pen=pg.mkPen(color=(255, 0, 0), width=2))

            y_vz, x_vz = np.histogram(vz, bins=50)
            self.plot_speed_z.plot(x_vz, y_vz, stepMode=True, fillLevel=0, brush=(255, 100, 100, 100), pen=pg.mkPen(color=(255, 0, 0), width=2))

            y_vmag, x_vmag = np.histogram(vmag, bins=50)
            self.plot_speed_mag.plot(x_vmag, y_vmag, stepMode=True, fillLevel=0, brush=(255, 100, 100, 100), pen=pg.mkPen(color=(255, 0, 0), width=2))

            # Spin histograms
            y_wx, x_wx = np.histogram(wx, bins=50)
            self.plot_spin_x.plot(x_wx, y_wx, stepMode=True, fillLevel=0, brush=(255, 100, 100, 100), pen=pg.mkPen(color=(255, 0, 0), width=2))

            y_wy, x_wy = np.histogram(wy, bins=50)
            self.plot_spin_y.plot(x_wy, y_wy, stepMode=True, fillLevel=0, brush=(255, 100, 100, 100), pen=pg.mkPen(color=(255, 0, 0), width=2))

            y_wz, x_wz = np.histogram(wz, bins=50)
            self.plot_spin_z.plot(x_wz, y_wz, stepMode=True, fillLevel=0, brush=(255, 100, 100, 100), pen=pg.mkPen(color=(255, 0, 0), width=2))

            y_wmag, x_wmag = np.histogram(wmag, bins=50)
            self.plot_spin_mag.plot(x_wmag, y_wmag, stepMode=True, fillLevel=0, brush=(255, 100, 100, 100), pen=pg.mkPen(color=(255, 0, 0), width=2))

    def save_plot(self):
        """Save the current histogram plots as a PNG image"""
        if self.speeds_robot is None and self.speeds_player is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"histogram_speed_spin_{timestamp}.png"

        output_path = plots_dir / filename

        try:
            # Export the plot widget
            exporter = ImageExporter(self.plot_widget.scene())
            exporter.parameters()['width'] = 1920  # Set width in pixels
            exporter.export(str(output_path))

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")




class ConfidenceDistributionPlot(QWidget):
    """Widget for displaying confidence-based distributions of velocity and spin"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Control bar
        control_bar = QHBoxLayout()

        # Info label
        self.info_label = QLabel("Confidence Distribution Analysis")
        self.info_label.setStyleSheet("font-weight: bold;")
        control_bar.addWidget(self.info_label, stretch=1)

        # Add save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Create plot widgets in a 2x2 grid
        self.plot_widget = pg.GraphicsLayoutWidget()

        # Row 0, Col 0 - Velocity magnitude distribution by confidence
        self.plot_vel_conf = self.plot_widget.addPlot(row=0, col=0, title="Velocity Magnitude Distribution by Confidence")
        self.plot_vel_conf.setLabel("left", "Count")
        self.plot_vel_conf.setLabel("bottom", "Velocity Magnitude (m/s)")
        self.plot_vel_conf.showGrid(x=True, y=True)
        self.plot_vel_conf.addLegend()
        self.plot_vel_conf.getAxis('left').enableAutoSIPrefix(False)
        self.plot_vel_conf.getAxis('bottom').enableAutoSIPrefix(False)

        # Row 0, Col 1 - Spin magnitude distribution by confidence
        self.plot_spin_conf = self.plot_widget.addPlot(row=0, col=1, title="Spin Magnitude Distribution by Confidence")
        self.plot_spin_conf.setLabel("left", "Count")
        self.plot_spin_conf.setLabel("bottom", "Spin Magnitude (rad/s)")
        self.plot_spin_conf.showGrid(x=True, y=True)
        self.plot_spin_conf.addLegend()
        self.plot_spin_conf.getAxis('left').enableAutoSIPrefix(False)
        self.plot_spin_conf.getAxis('bottom').enableAutoSIPrefix(False)

        # Row 1, Col 0 - Segment duration distribution by confidence
        self.plot_duration_conf = self.plot_widget.addPlot(row=1, col=0, title="Segment Duration Distribution by Confidence")
        self.plot_duration_conf.setLabel("left", "Count")
        self.plot_duration_conf.setLabel("bottom", "Segment Duration (s)")
        self.plot_duration_conf.showGrid(x=True, y=True)
        self.plot_duration_conf.addLegend()
        self.plot_duration_conf.getAxis('left').enableAutoSIPrefix(False)
        self.plot_duration_conf.getAxis('bottom').enableAutoSIPrefix(False)

        # Row 1, Col 1 - Distribution by match date
        self.plot_date_conf = self.plot_widget.addPlot(row=1, col=1, title="Segment Count by Match Date")
        self.plot_date_conf.setLabel("left", "Count")
        self.plot_date_conf.setLabel("bottom", "Match Date")
        self.plot_date_conf.showGrid(x=True, y=True)
        self.plot_date_conf.addLegend()
        self.plot_date_conf.getAxis('left').enableAutoSIPrefix(False)
        self.plot_date_conf.getAxis('bottom').enableAutoSIPrefix(False)

        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # Store data
        self.data = None
        self.match_file = None

    def plot_data(self, vel_high_conf, spin_high_conf, vel_low_conf, spin_low_conf, duration_high_conf, duration_low_conf, dates_high_conf=None, dates_low_conf=None, match_file=None):
        """Plot confidence distribution data

        Args:
            vel_high_conf: Velocity magnitudes with confidence >= 0.5
            spin_high_conf: Spin magnitudes with confidence >= 0.5
            vel_low_conf: Velocity magnitudes with confidence < 0.5
            spin_low_conf: Spin magnitudes with confidence < 0.5
            duration_high_conf: Segment durations (s) with confidence >= 0.5
            duration_low_conf: Segment durations (s) with confidence < 0.5
            dates_high_conf: Match dates for segments with confidence >= 0.5
            dates_low_conf: Match dates for segments with confidence < 0.5
            match_file: Path to match file for saving
        """
        # Store data and match file
        self.data = (vel_high_conf, spin_high_conf, vel_low_conf, spin_low_conf, duration_high_conf, duration_low_conf, dates_high_conf, dates_low_conf)
        self.match_file = match_file

        # Clear previous plots
        self.plot_vel_conf.clear()
        self.plot_spin_conf.clear()
        self.plot_duration_conf.clear()
        self.plot_date_conf.clear()

        # Plot velocity distributions
        if len(vel_high_conf) > 0:
            y_high, x_high = np.histogram(vel_high_conf, bins=60, range=(0, 30))
            self.plot_vel_conf.plot(x_high, y_high, stepMode=True, fillLevel=0,
                                   brush=(100, 255, 100, 100),
                                   pen=pg.mkPen(color=(0, 200, 0), width=2),
                                   name=f"Conf ≥ 0.5 (n={len(vel_high_conf)})")

        if len(vel_low_conf) > 0:
            y_low, x_low = np.histogram(vel_low_conf, bins=60, range=(0, 30))
            self.plot_vel_conf.plot(x_low, y_low, stepMode=True, fillLevel=0,
                                   brush=(255, 100, 100, 100),
                                   pen=pg.mkPen(color=(200, 0, 0), width=2),
                                   name=f"Conf < 0.5 (n={len(vel_low_conf)})")

        # Plot spin distributions
        if len(spin_high_conf) > 0:
            y_high, x_high = np.histogram(spin_high_conf, bins=40, range=(0, 1000))
            self.plot_spin_conf.plot(x_high, y_high, stepMode=True, fillLevel=0,
                                    brush=(100, 255, 100, 100),
                                    pen=pg.mkPen(color=(0, 200, 0), width=2),
                                    name=f"Conf ≥ 0.5 (n={len(spin_high_conf)})")

        if len(spin_low_conf) > 0:
            y_low, x_low = np.histogram(spin_low_conf, bins=40, range=(0, 1000))
            self.plot_spin_conf.plot(x_low, y_low, stepMode=True, fillLevel=0,
                                    brush=(255, 100, 100, 100),
                                    pen=pg.mkPen(color=(200, 0, 0), width=2),
                                    name=f"Conf < 0.5 (n={len(spin_low_conf)})")

        # Plot segment duration distributions
        # Use fixed range 0-1s with bin size of 0.01s (100 bins)
        if len(duration_high_conf) > 0:
            y_high, x_high = np.histogram(duration_high_conf, bins=100, range=(0, 1.0))
            self.plot_duration_conf.plot(x_high, y_high, stepMode=True, fillLevel=0,
                                        brush=(100, 255, 100, 100),
                                        pen=pg.mkPen(color=(0, 200, 0), width=2),
                                        name=f"Conf ≥ 0.5 (n={len(duration_high_conf)})")

        if len(duration_low_conf) > 0:
            y_low, x_low = np.histogram(duration_low_conf, bins=100, range=(0, 1.0))
            self.plot_duration_conf.plot(x_low, y_low, stepMode=True, fillLevel=0,
                                        brush=(255, 100, 100, 100),
                                        pen=pg.mkPen(color=(200, 0, 0), width=2),
                                        name=f"Conf < 0.5 (n={len(duration_low_conf)})")

        # Plot distribution by match date (using step histogram style like other plots)
        if dates_high_conf is not None and dates_low_conf is not None:
            # Combine all dates and get unique sorted dates
            all_dates = np.concatenate([dates_high_conf, dates_low_conf]) if len(dates_high_conf) > 0 or len(dates_low_conf) > 0 else np.array([])
            if len(all_dates) > 0:
                unique_dates = sorted(set(all_dates))
                date_to_idx = {d: i for i, d in enumerate(unique_dates)}

                # Count segments per date for each confidence level
                counts_high = np.zeros(len(unique_dates))
                counts_low = np.zeros(len(unique_dates))

                for d in dates_high_conf:
                    counts_high[date_to_idx[d]] += 1
                for d in dates_low_conf:
                    counts_low[date_to_idx[d]] += 1

                # Create histogram-style x bins (edges)
                x_edges = np.arange(len(unique_dates) + 1) - 0.5

                # Plot as step histogram (same style as velocity/spin/duration)
                if np.sum(counts_high) > 0:
                    self.plot_date_conf.plot(x_edges, counts_high, stepMode=True, fillLevel=0,
                                            brush=(100, 255, 100, 100),
                                            pen=pg.mkPen(color=(0, 200, 0), width=2),
                                            name=f"Conf ≥ 0.5 (n={int(np.sum(counts_high))})")

                if np.sum(counts_low) > 0:
                    self.plot_date_conf.plot(x_edges, counts_low, stepMode=True, fillLevel=0,
                                            brush=(255, 100, 100, 100),
                                            pen=pg.mkPen(color=(200, 0, 0), width=2),
                                            name=f"Conf < 0.5 (n={int(np.sum(counts_low))})")

                # Set x-axis ticks to show dates
                # Format: show abbreviated dates if many
                if len(unique_dates) > 10:
                    # Show fewer labels
                    step = max(1, len(unique_dates) // 10)
                    ticks = [(i, unique_dates[i]) for i in range(0, len(unique_dates), step)]
                else:
                    ticks = [(i, d) for i, d in enumerate(unique_dates)]

                ax = self.plot_date_conf.getAxis('bottom')
                ax.setTicks([ticks])

        # Update info label
        total_high = len(vel_high_conf)
        total_low = len(vel_low_conf)
        total = total_high + total_low
        if total > 0:
            pct_high = 100.0 * total_high / total
            pct_low = 100.0 * total_low / total
            self.info_label.setText(
                f"Confidence Distribution: High conf (≥0.5): {total_high} ({pct_high:.1f}%), "
                f"Low conf (<0.5): {total_low} ({pct_low:.1f}%)"
            )

    def save_plot(self):
        """Save the current confidence distribution plots as a PNG image"""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"confidence_distribution_{timestamp}.png"

        output_path = plots_dir / filename

        try:
            # Export the plot widget
            exporter = ImageExporter(self.plot_widget.scene())
            exporter.parameters()['width'] = 1920  # Set width in pixels
            exporter.export(str(output_path))

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")


class GCSSpinConfidenceHistogramPlot(QWidget):
    """Widget for displaying normalized histogram of GCS spin confidence values (median per trajectory)"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Control bar
        control_bar = QHBoxLayout()

        # Info label
        self.info_label = QLabel("GCS Spin Confidence Distribution (Median per Trajectory)")
        self.info_label.setStyleSheet("font-weight: bold;")
        control_bar.addWidget(self.info_label, stretch=1)

        # Add save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Create plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()

        # Histogram plot
        self.plot_hist = self.plot_widget.addPlot(row=0, col=0, title="GCS Spin Confidence Distribution (Median per Trajectory)")
        self.plot_hist.setLabel("left", "Density")
        self.plot_hist.setLabel("bottom", "w_confidence (median)")
        self.plot_hist.showGrid(x=True, y=True)
        self.plot_hist.getAxis('left').enableAutoSIPrefix(False)
        self.plot_hist.getAxis('bottom').enableAutoSIPrefix(False)

        # Set x-axis range to 0-1
        self.plot_hist.setXRange(0, 1, padding=0.02)

        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # Store data
        self.data = None
        self.data_wx = None
        self.match_file = None

    def plot_data(self, median_confidences: np.ndarray, match_file=None, median_wx_confidences: np.ndarray = None):
        """Plot histogram of median w_confidence values per trajectory

        Args:
            median_confidences: Array of median w_confidence values (one per trajectory)
            match_file: Path to match file for saving
            median_wx_confidences: Array of median wx_confidence values (optional, for overlay)
        """
        # Store data and match file
        self.data = median_confidences
        self.data_wx = median_wx_confidences
        self.match_file = match_file

        # Clear previous plot
        self.plot_hist.clear()

        if len(median_confidences) == 0:
            self.info_label.setText("GCS Spin Confidence: No data available")
            return

        # Create normalized histogram (density) for w_confidence
        n_bins = 50
        counts, bin_edges = np.histogram(median_confidences, bins=n_bins, range=(0, 1))

        # Normalize to create density (sum of bar areas = 1)
        bin_width = bin_edges[1] - bin_edges[0]
        density = counts / (np.sum(counts) * bin_width)

        # Plot w_confidence histogram (blue)
        self.plot_hist.plot(bin_edges, density, stepMode='center', fillLevel=0,
                           brush=(100, 150, 255, 100),
                           pen=pg.mkPen(color=(50, 100, 200), width=2),
                           name="w_confidence (GCS)")

        # Plot wx_confidence histogram (orange) if provided
        if median_wx_confidences is not None and len(median_wx_confidences) > 0:
            counts_wx, bin_edges_wx = np.histogram(median_wx_confidences, bins=n_bins, range=(0, 1))
            density_wx = counts_wx / (np.sum(counts_wx) * bin_width)

            self.plot_hist.plot(bin_edges_wx, density_wx, stepMode='center', fillLevel=0,
                               brush=(255, 150, 100, 100),
                               pen=pg.mkPen(color=(200, 100, 50), width=2),
                               name="wx_confidence (GT200)")

        # Add statistics as text
        mean_conf = np.mean(median_confidences)
        std_conf = np.std(median_confidences)
        median_conf = np.median(median_confidences)
        pct_high = 100.0 * np.sum(median_confidences >= 0.9) / len(median_confidences)
        pct_low = 100.0 * np.sum(median_confidences < 0.5) / len(median_confidences)

        # Add vertical line at mean for w_confidence
        mean_line = pg.InfiniteLine(pos=mean_conf, angle=90, pen=pg.mkPen(color=(50, 100, 200), width=2, style=Qt.DashLine))
        self.plot_hist.addItem(mean_line)

        # Build info text
        info_text = (
            f"w_conf (n={len(median_confidences)}): Mean={mean_conf:.3f}, Med={median_conf:.3f}, "
            f"≥0.9: {pct_high:.1f}%"
        )

        # Add legend
        legend = pg.LegendItem(offset=(60, 30))
        legend.setParentItem(self.plot_hist.graphicsItem())

        # Add w_confidence to legend
        w_conf_item = pg.PlotDataItem(pen=pg.mkPen(color=(50, 100, 200), width=2))
        legend.addItem(w_conf_item, f"w_conf (mean={mean_conf:.3f})")

        # Add wx_confidence stats and legend if available
        if median_wx_confidences is not None and len(median_wx_confidences) > 0:
            mean_wx = np.mean(median_wx_confidences)
            pct_high_wx = 100.0 * np.sum(median_wx_confidences >= 0.9) / len(median_wx_confidences)

            # Add vertical line at mean for wx_confidence
            mean_line_wx = pg.InfiniteLine(pos=mean_wx, angle=90, pen=pg.mkPen(color=(200, 100, 50), width=2, style=Qt.DashLine))
            self.plot_hist.addItem(mean_line_wx)

            info_text += f" | wx_conf (n={len(median_wx_confidences)}): Mean={mean_wx:.3f}, ≥0.9: {pct_high_wx:.1f}%"

            wx_conf_item = pg.PlotDataItem(pen=pg.mkPen(color=(200, 100, 50), width=2))
            legend.addItem(wx_conf_item, f"wx_conf (mean={mean_wx:.3f})")

        # Update info label with statistics
        self.info_label.setText(info_text)

    def save_plot(self):
        """Save the current histogram plot as a PNG image"""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gcs_spin_confidence_histogram_{timestamp}.png"

        output_path = plots_dir / filename

        try:
            # Export the plot widget
            exporter = ImageExporter(self.plot_widget.scene())
            exporter.parameters()['width'] = 1920  # Set width in pixels
            exporter.export(str(output_path))

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
