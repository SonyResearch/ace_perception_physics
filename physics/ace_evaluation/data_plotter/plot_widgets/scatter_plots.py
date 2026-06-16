# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Scatter Plot Widgets

Contains interactive scatter plot widgets for analyzing relationships between variables.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import pathlib
from datetime import datetime

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QCheckBox, QGroupBox, QMessageBox, QSlider
from PySide6.QtCore import Signal, Qt

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from ace_evaluation.utilities.data_classes import FlightSegment
from .base_coefficient_plot import save_plot_as_image


class SpinSpeedScatterPlot(QWidget):
    """Widget for interactive spin vs speed scatter plot"""

    # Define signal for flight segment selection (includes metadata and full data)
    flight_segment_selected = Signal(
        object, object, object, object, object, object
    )  # (FlightSegment, metadata, shot_data, rally_data, shot_object, rally_object)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Top control bar with data source checkboxes
        control_bar = QHBoxLayout()

        # Data source selection group
        data_source_group = QGroupBox("Data Sources")
        data_source_layout = QVBoxLayout()
        data_source_layout.setSpacing(2)
        data_source_layout.setContentsMargins(5, 5, 5, 5)

        self.data_source_checkboxes = {}
        data_sources = [("Robot", "blue"), ("Player", "red")]

        for source, color in data_sources:
            checkbox = QCheckBox(source)
            checkbox.setChecked(True)  # Both checked by default
            checkbox.stateChanged.connect(self.on_data_source_changed)
            self.data_source_checkboxes[source] = checkbox
            data_source_layout.addWidget(checkbox)

        data_source_group.setLayout(data_source_layout)
        control_bar.addWidget(data_source_group)

        # Confidence range sliders
        self._confidence_group = QGroupBox("Confidence Filter")
        confidence_layout = QVBoxLayout()
        confidence_layout.setSpacing(5)
        confidence_layout.setContentsMargins(5, 5, 5, 5)

        # Min confidence slider
        conf_min_layout = QHBoxLayout()
        conf_min_label_text = QLabel("Min:")
        conf_min_layout.addWidget(conf_min_label_text)
        self.conf_min_slider = QSlider(Qt.Horizontal)
        self.conf_min_slider.setMinimum(0)
        self.conf_min_slider.setMaximum(100)  # 0.0 to 1.0 scaled by 100
        self.conf_min_slider.setValue(0)
        self.conf_min_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_min_slider.setTickInterval(10)
        self.conf_min_slider.valueChanged.connect(self.on_confidence_range_changed)
        conf_min_layout.addWidget(self.conf_min_slider)
        self.conf_min_label = QLabel("0.00")
        self.conf_min_label.setMinimumWidth(35)
        conf_min_layout.addWidget(self.conf_min_label)
        confidence_layout.addLayout(conf_min_layout)

        # Max confidence slider
        conf_max_layout = QHBoxLayout()
        conf_max_label_text = QLabel("Max:")
        conf_max_layout.addWidget(conf_max_label_text)
        self.conf_max_slider = QSlider(Qt.Horizontal)
        self.conf_max_slider.setMinimum(0)
        self.conf_max_slider.setMaximum(100)
        self.conf_max_slider.setValue(100)
        self.conf_max_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_max_slider.setTickInterval(10)
        self.conf_max_slider.valueChanged.connect(self.on_confidence_range_changed)
        conf_max_layout.addWidget(self.conf_max_slider)
        self.conf_max_label = QLabel("1.00")
        self.conf_max_label.setMinimumWidth(35)
        conf_max_layout.addWidget(self.conf_max_label)
        confidence_layout.addLayout(conf_max_layout)

        self._confidence_group.setLayout(confidence_layout)
        control_bar.addWidget(self._confidence_group)

        # Hidden RMSE OPT sliders (driven by global filter in app.py)
        self.rmse_min_slider = QSlider(Qt.Horizontal)
        self.rmse_min_slider.setMinimum(0)
        self.rmse_min_slider.setMaximum(100)
        self.rmse_min_slider.setValue(0)
        self.rmse_min_slider.setVisible(False)
        self.rmse_min_slider.valueChanged.connect(self.update_plot)
        self.rmse_slider = QSlider(Qt.Horizontal)
        self.rmse_slider.setMinimum(0)
        self.rmse_slider.setMaximum(100)
        self.rmse_slider.setValue(100)
        self.rmse_slider.setVisible(False)
        self.rmse_slider.valueChanged.connect(self.update_plot)

        # Color By selection
        color_group = QGroupBox("Color By")
        color_layout = QVBoxLayout()
        color_layout.setSpacing(2)
        color_layout.setContentsMargins(5, 5, 5, 5)
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["Player", "Avg X Position", "Avg Y Position", "Avg Z Position"])
        self.color_by_combo.currentIndexChanged.connect(self.on_color_by_changed)
        color_layout.addWidget(self.color_by_combo)
        color_group.setLayout(color_layout)
        control_bar.addWidget(color_group)

        # Info label
        self.info_label = QLabel("Click on a point to view the flight segment")
        control_bar.addWidget(self.info_label, stretch=1)

        # Publication mode checkbox
        self.publication_checkbox = QCheckBox("Publication")
        control_bar.addWidget(self.publication_checkbox)

        # Add save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Create plot widget
        self.plot_widget = pg.PlotWidget(title="Spin Magnitude vs Speed Magnitude Before Contact")
        self.plot_widget.setLabel("left", "Spin Magnitude (rad/s)")
        self.plot_widget.setLabel("bottom", "Speed Magnitude Before Contact (m/s)")
        self.plot_widget.showGrid(x=True, y=True)
        # Disable SI prefix for full value display
        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        # Color bar for position-based coloring (hidden by default)
        self.color_bar_widget = pg.GraphicsLayoutWidget()
        self.color_bar_widget.setMaximumHeight(40)
        self.color_bar_widget.setVisible(False)

        layout.addWidget(self.plot_widget)
        layout.addWidget(self.color_bar_widget)
        self.setLayout(layout)

        # Store data references for both robot and player
        self.robot_data = None
        self.player_data = None
        self.scatter_items = {}  # Dictionary to store scatter items by source

        # Color palette for player-based coloring (up to 10 distinct players)
        self._player_colors = [
            (31, 119, 180),    # blue
            (255, 127, 14),    # orange
            (44, 160, 44),     # green
            (214, 39, 40),     # red
            (148, 103, 189),   # purple
            (140, 86, 75),     # brown
            (227, 119, 194),   # pink
            (127, 127, 127),   # gray
            (188, 189, 34),    # olive
            (23, 190, 207),    # cyan
        ]
        self._player_color_map = {}  # player_name -> color tuple

    def on_data_source_changed(self):
        """Handle data source checkbox state change"""
        self.update_plot()

    def on_confidence_range_changed(self):
        """Handle confidence range slider changes"""
        # Get slider values (scaled 0-100)
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0

        # Ensure min <= max
        if conf_min > conf_max:
            if self.sender() == self.conf_min_slider:
                self.conf_min_slider.setValue(int(conf_max * 100))
                conf_min = conf_max
            else:
                self.conf_max_slider.setValue(int(conf_min * 100))
                conf_max = conf_min

        # Update labels
        self.conf_min_label.setText(f"{conf_min:.2f}")
        self.conf_max_label.setText(f"{conf_max:.2f}")

        # Update plot
        self.update_plot()

    def _get_player_color(self, player_name: str):
        """Get a consistent color for a player name."""
        if player_name not in self._player_color_map:
            idx = len(self._player_color_map) % len(self._player_colors)
            self._player_color_map[player_name] = self._player_colors[idx]
        return self._player_color_map[player_name]

    def _darker(self, color, factor=0.75):
        """Return a darker version of an RGB tuple for pen outlines."""
        return tuple(int(c * factor) for c in color)

    def on_color_by_changed(self):
        """Handle Color By combo box change"""
        self.update_plot()

    @staticmethod
    def _compute_avg_positions(flight_segments):
        """
        Compute average x, y, z positions for each flight segment.

        Prefers x_aps/y_aps/z_aps columns; falls back to x_opt/y_opt/z_opt.

        Args:
            flight_segments: List of FlightSegment objects

        Returns:
            Tuple of (avg_x, avg_y, avg_z) numpy arrays
        """
        n = len(flight_segments)
        avg_x = np.full(n, np.nan)
        avg_y = np.full(n, np.nan)
        avg_z = np.full(n, np.nan)

        for i, fs in enumerate(flight_segments):
            cols = fs.data.columns
            if 'x_aps' in cols:
                x_col, y_col, z_col = 'x_aps', 'y_aps', 'z_aps'
            elif 'x_opt' in cols:
                x_col, y_col, z_col = 'x_opt', 'y_opt', 'z_opt'
            else:
                continue
            avg_x[i] = np.nanmean(fs.data[x_col].values)
            avg_y[i] = np.nanmean(fs.data[y_col].values)
            avg_z[i] = np.nanmean(fs.data[z_col].values)

        return avg_x, avg_y, avg_z

    @staticmethod
    def _value_to_color(values, cmap_name='coolwarm'):
        """
        Map a numpy array of float values to RGBA colors using a matplotlib-style colormap.

        Args:
            values: 1-D array of floats (may contain NaN)
            cmap_name: Name of the colormap ('coolwarm', 'viridis', etc.)

        Returns:
            Tuple of (colors_rgba_list, vmin, vmax) where colors is a list of (r,g,b,a) tuples
        """
        valid = ~np.isnan(values)
        if not np.any(valid):
            return [(128, 128, 128, 200)] * len(values), 0.0, 1.0

        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        span = vmax - vmin if vmax != vmin else 1.0

        # Built-in colormaps (no matplotlib dependency)
        # coolwarm: blue -> white -> red
        def coolwarm(t):
            # t in [0, 1]
            if t < 0.5:
                s = t * 2  # 0..1
                r = int(59 + s * (221 - 59))
                g = int(76 + s * (221 - 76))
                b = int(192 + s * (221 - 192))
            else:
                s = (t - 0.5) * 2  # 0..1
                r = int(221 + s * (180 - 221))
                g = int(221 - s * (221 - 4))
                b = int(221 - s * (221 - 38))
            return (r, g, b)

        colors = []
        for v in values:
            if np.isnan(v):
                colors.append((128, 128, 128, 120))  # gray for NaN
            else:
                t = (v - vmin) / span
                r, g, b = coolwarm(np.clip(t, 0, 1))
                colors.append((r, g, b, 200))

        return colors, vmin, vmax

    def _update_color_bar(self, vmin, vmax, label):
        """Show a horizontal color bar below the plot."""
        self.color_bar_widget.clear()
        self.color_bar_widget.setVisible(True)

        plot = self.color_bar_widget.addPlot()
        plot.hideAxis('left')
        plot.setMouseEnabled(x=False, y=False)
        plot.setMenuEnabled(False)
        plot.setXRange(vmin, vmax, padding=0)
        plot.setYRange(0, 1, padding=0)
        plot.setLabel('bottom', label)
        plot.getAxis('bottom').enableAutoSIPrefix(False)

        # Draw gradient as thin image
        n_steps = 256
        gradient = np.zeros((1, n_steps, 4), dtype=np.ubyte)
        for i in range(n_steps):
            t = i / (n_steps - 1)
            if t < 0.5:
                s = t * 2
                gradient[0, i] = [int(59 + s * (221 - 59)), int(76 + s * (221 - 76)), int(192 + s * (221 - 192)), 255]
            else:
                s = (t - 0.5) * 2
                gradient[0, i] = [int(221 + s * (180 - 221)), int(221 - s * (221 - 4)), int(221 - s * (221 - 38)), 255]

        img = pg.ImageItem(gradient)
        img.setRect(vmin, 0, vmax - vmin, 1)
        plot.addItem(img)

    def update_plot(self):
        """Update the plot based on selected data sources, confidence range, coloring by player"""

        try:
            self.plot_widget.clear()
            self.scatter_items = {}

            filtered_points = 0

            # Get confidence range
            conf_min = self.conf_min_slider.value() / 100.0
            conf_max = self.conf_max_slider.value() / 100.0

            # Get RMSE OPT range (slider values in mm, convert to meters for comparison)
            rmse_min_mm = self.rmse_min_slider.value()
            rmse_max_mm = self.rmse_slider.value()
            rmse_min_m = rmse_min_mm / 1000.0
            rmse_max_m = rmse_max_mm / 1000.0
            rmse_filtering_active = (rmse_min_mm > 0 or rmse_max_mm < self.rmse_slider.maximum())

            # ── Use pre-merged data + vectorized filtering ──
            m = self._merged
            if m is None:
                self.info_label.setText("No data points to display")
                return

            total_points = len(m['speeds'])

            # Source filter (checkbox-driven)
            show_robot = self.data_source_checkboxes["Robot"].isChecked()
            show_player = self.data_source_checkboxes["Player"].isChecked()
            source_mask = np.zeros(total_points, dtype=bool)
            if show_robot:
                source_mask |= (m['source_labels'] == 'Robot')
            if show_player:
                source_mask |= (m['source_labels'] == 'Player')

            # Confidence mask
            conf_mask = (m['confidence'] >= conf_min) & (m['confidence'] <= conf_max)

            # RMSE OPT mask
            if rmse_filtering_active:
                rmse_valid = ~np.isnan(m['rmse_opt'])
                rmse_mask = rmse_valid & (m['rmse_opt'] >= rmse_min_m) & (m['rmse_opt'] <= rmse_max_m)
            else:
                rmse_mask = np.ones(total_points, dtype=bool)

            # Combined mask
            filt = source_mask & conf_mask & rmse_mask

            # Build groups using numpy unique on filtered data
            # Group key = "source_label\x1fplayer_name" (using Unit Separator;
            # \x00 cannot be used because numpy fixed-width strings treat it as a terminator)
            group_keys = np.char.add(np.char.add(m['source_labels'][filt].astype(str), '\x1f'), m['player_names'][filt].astype(str))
            filt_indices = np.where(filt)[0]

            groups = {}
            if len(group_keys) > 0:
                unique_keys, inv = np.unique(group_keys, return_inverse=True)
                for gidx, gk in enumerate(unique_keys):
                    sel = filt_indices[inv == gidx]
                    source_label, player_name = gk.split('\x1f', 1)
                    groups[(source_label, player_name)] = {
                        'speeds': m['speeds'][sel],
                        'spins': m['spins'][sel],
                        'confs': m['confidence'][sel],
                        'flight_segments': m['flight_segments'][sel],
                        'metadata': m['metadata'][sel],
                        'shot_data': m['shot_data'][sel],
                        'rally_data': m['rally_data'][sel],
                        'shot_objects': m['shot_objects'][sel],
                        'rally_objects': m['rally_objects'][sel],
                    }

            # Determine coloring mode
            color_mode = self.color_by_combo.currentText()
            use_position_coloring = color_mode in ("Avg X Position", "Avg Y Position", "Avg Z Position")

            if not use_position_coloring:
                # --- Player-based coloring (original behavior) ---
                self.color_bar_widget.setVisible(False)
                legend = self.plot_widget.addLegend(offset=(10, 10))

                for (source_label, player_name), g in sorted(groups.items()):
                    n = len(g['speeds'])
                    if n == 0:
                        continue

                    color = self._get_player_color(player_name)
                    pen_color = self._darker(color)

                    # Symbol: 'o' for Robot, 't' (triangle) for Player
                    symbol = 'o' if source_label == "Robot" else 't'

                    brushes = self._create_confidence_brushes(g['confs'], color=color)
                    pen = pg.mkPen(color=pen_color, width=1)

                    scatter_item = pg.ScatterPlotItem(
                        x=np.array(g['speeds']), y=np.array(g['spins']),
                        size=8, pen=pen, brush=brushes, symbol=symbol,
                        name=f"{player_name} ({source_label})",
                    )
                    scatter_item.sigClicked.connect(self.on_point_clicked)
                    self.plot_widget.addItem(scatter_item)
                    self.scatter_items[f"{source_label}_{player_name}"] = (
                        scatter_item, g['flight_segments'], g['metadata'],
                        g['shot_data'], g['rally_data'],
                        g['shot_objects'], g['rally_objects'],
                    )
                    filtered_points += n
            else:
                # --- Position-based coloring ---
                # Flatten all groups into one pool
                all_speeds = []
                all_spins = []
                all_symbols = []
                all_flight_segments = []
                all_metadata = []
                all_shot_data = []
                all_rally_data = []
                all_shot_objects = []
                all_rally_objects = []

                for (source_label, player_name), g in sorted(groups.items()):
                    n = len(g['speeds'])
                    if n == 0:
                        continue
                    all_speeds.extend(g['speeds'])
                    all_spins.extend(g['spins'])
                    all_symbols.extend(['o' if source_label == "Robot" else 't'] * n)
                    all_flight_segments.extend(g['flight_segments'])
                    all_metadata.extend(g['metadata'])
                    all_shot_data.extend(g['shot_data'])
                    all_rally_data.extend(g['rally_data'])
                    all_shot_objects.extend(g['shot_objects'])
                    all_rally_objects.extend(g['rally_objects'])

                filtered_points = len(all_speeds)
                if filtered_points > 0:
                    # Compute average positions
                    avg_x, avg_y, avg_z = self._compute_avg_positions(all_flight_segments)

                    axis_map = {
                        "Avg X Position": (avg_x, "Avg X (m)"),
                        "Avg Y Position": (avg_y, "Avg Y (m)"),
                        "Avg Z Position": (avg_z, "Avg Z (m)"),
                    }
                    pos_values, axis_label = axis_map[color_mode]
                    colors_rgba, vmin, vmax = self._value_to_color(pos_values)
                    self._update_color_bar(vmin, vmax, axis_label)

                    brushes = [pg.mkBrush(*c) for c in colors_rgba]
                    pens = [pg.mkPen(color=(max(0, c[0]-40), max(0, c[1]-40), max(0, c[2]-40)), width=1) for c in colors_rgba]

                    spots = []
                    for i in range(filtered_points):
                        spots.append({
                            'pos': (all_speeds[i], all_spins[i]),
                            'size': 8,
                            'pen': pens[i],
                            'brush': brushes[i],
                            'symbol': all_symbols[i],
                        })

                    scatter_item = pg.ScatterPlotItem()
                    scatter_item.addPoints(spots)
                    scatter_item.sigClicked.connect(self.on_point_clicked)
                    self.plot_widget.addItem(scatter_item)
                    self.scatter_items["_position_colored"] = (
                        scatter_item, all_flight_segments, all_metadata,
                        all_shot_data, all_rally_data,
                        all_shot_objects, all_rally_objects,
                    )

            if total_points > 0:
                rmse_info = f", RMSE: {rmse_min_mm}-{rmse_max_mm}mm" if rmse_filtering_active else ""
                self.info_label.setText(f"Showing {filtered_points}/{total_points} flight segments (conf: {conf_min:.2f}-{conf_max:.2f}{rmse_info}). Click to view details.")
            else:
                self.info_label.setText("No data points to display")

        except Exception as e:
            print(f"ERROR in SpinSpeedScatterPlot.update_plot: {e}")
            import traceback
            traceback.print_exc()
            self.info_label.setText(f"Error updating plot: {e}")

    def _create_confidence_brushes(self, confidence_list, color):
        """
        Create brushes with confidence-based alpha values

        Args:
            confidence_list: List of confidence values (0-1)
            color: RGB tuple (r, g, b)

        Returns:
            List of brushes or single brush
        """
        import numpy as np

        if confidence_list is None or len(confidence_list) == 0:
            # No confidence data, use default alpha
            return pg.mkBrush(*color, 120)

        # Map confidence (0-1) to alpha (51-255)
        # 51 = ~20% opacity (still visible), 255 = 100% opacity
        alpha_min = 51
        alpha_max = 255

        brushes = []
        for conf in confidence_list:
            alpha = int(conf * (alpha_max - alpha_min) + alpha_min)
            alpha = np.clip(alpha, alpha_min, alpha_max)
            brushes.append(pg.mkBrush(*color, alpha))

        return brushes

    def plot_data(self, robot_data_tuple, player_data_tuple):
        """
        Create scatter plot from extracted data for both robot and player.

        Pre-merges robot + player data into numpy arrays and extracts player
        names / source labels so that update_plot() can use vectorized masking
        and numpy-based grouping instead of a per-point Python loop.
        """
        try:
            self.robot_data = robot_data_tuple
            self.player_data = player_data_tuple

            # ── Pre-merge into numpy arrays (done ONCE) ──
            self._merged = None
            speed_parts, spin_parts, conf_parts, rmse_parts = [], [], [], []
            source_parts, player_parts = [], []
            obj_keys = ['flight_segments', 'metadata', 'shot_data', 'rally_data', 'shot_objects', 'rally_objects']
            obj_lists = {k: [] for k in obj_keys}

            for source_label, data_tuple in [('Robot', robot_data_tuple), ('Player', player_data_tuple)]:
                if data_tuple is None:
                    continue
                if len(data_tuple) >= 10:
                    speeds, spins, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list, rmse_opt_list = data_tuple[:10]
                else:
                    speeds, spins, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list = data_tuple[:9]
                    rmse_opt_list = [np.nan] * len(speeds)
                if len(speeds) == 0:
                    continue
                n = len(speeds)
                speed_parts.append(np.asarray(speeds))
                spin_parts.append(np.asarray(spins))
                conf_parts.append(np.asarray(confidence_list))
                rmse_parts.append(np.asarray(rmse_opt_list))
                source_parts.extend([source_label] * n)
                player_parts.extend([m.get('match_player', 'unknown') for m in metadata_list])
                obj_lists['flight_segments'].extend(flight_segments)
                obj_lists['metadata'].extend(metadata_list)
                obj_lists['shot_data'].extend(shot_data_list)
                obj_lists['rally_data'].extend(rally_data_list)
                obj_lists['shot_objects'].extend(shot_object_list)
                obj_lists['rally_objects'].extend(rally_object_list)

            if speed_parts:
                m = {
                    'speeds': np.concatenate(speed_parts),
                    'spins': np.concatenate(spin_parts),
                    'confidence': np.concatenate(conf_parts),
                    'rmse_opt': np.concatenate(rmse_parts),
                    'source_labels': np.array(source_parts, dtype=object),
                    'player_names': np.array(player_parts, dtype=object),
                }
                for k, lst in obj_lists.items():
                    m[k] = np.array(lst, dtype=object)
                self._merged = m

            self.update_plot()
        except Exception as e:
            print(f"ERROR in SpinSpeedScatterPlot.plot_data: {e}")
            import traceback
            traceback.print_exc()

    def on_point_clicked(self, plot_item, points):
        """Handle click on scatter plot point"""
        if len(points) > 0:
            point = points[0]
            idx = point.index()

            # Find which scatter item was clicked
            for source_name, (scatter_item, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list) in self.scatter_items.items():
                if scatter_item == plot_item:
                    # Emit signal with the corresponding flight segment, metadata, and full data
                    if idx < len(flight_segments):
                        fs = flight_segments[idx]
                        metadata = metadata_list[idx] if idx < len(metadata_list) else None
                        shot_object = shot_object_list[idx] if idx < len(shot_object_list) else None
                        rally_object = rally_object_list[idx] if idx < len(rally_object_list) else None
                        # Build shot/rally DataFrames on demand (not pre-stored, saves ~3 GB)
                        shot_data = None
                        if shot_object is not None:
                            shot_data = pd.concat([seg.data for seg in shot_object.flight_segments], ignore_index=False)
                        rally_data = None
                        if rally_object is not None:
                            all_dfs = [seg.data for s in rally_object.shots for seg in s.flight_segments]
                            if all_dfs:
                                rally_data = pd.concat(all_dfs, ignore_index=False)
                        self.flight_segment_selected.emit(fs, metadata, shot_data, rally_data, shot_object, rally_object)
                    break

    def save_plot(self):
        """Save the current scatter plot as a PNG image"""
        if self.robot_data is None and self.player_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scatter_plot_spin_vs_speed_{timestamp}.png"

        output_path = plots_dir / filename

        try:
            # Export the plot widget
            publication = self.publication_checkbox.isChecked()
            save_plot_as_image(self.plot_widget, str(output_path), publication_mode=publication)

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
