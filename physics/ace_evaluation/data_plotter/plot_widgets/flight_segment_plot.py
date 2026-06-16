# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Flight Segment Plot Widget

Widget for displaying flight segment 3D trajectory with position, velocity, and spin components.
"""

from typing import List, Dict, Any
import warnings
import numpy as np
import pandas as pd
import pathlib
from datetime import datetime

# Suppress harmless pyqtgraph warning when plotting data series that are entirely NaN
warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QCheckBox, QGroupBox, QMessageBox, QSizePolicy
from PySide6.QtCore import Signal, Qt, QSize

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from scipy.spatial.transform import Rotation as _R

from ace_evaluation.utilities.data_classes import FlightSegment, RacketContactEvent
from .aero_estimates import get_drag_estimate_v1, get_magnus_estimate_v2


class FlightSegmentPlot(QWidget):
    """Widget for displaying flight segment 3D trajectory"""

    @staticmethod
    def _set_axis_style(plot_item):
        """Configure axis to show full values without scientific notation"""
        plot_item.getAxis('left').enableAutoSIPrefix(False)
        plot_item.getAxis('bottom').enableAutoSIPrefix(False)

    def __init__(self):
        super().__init__()

        # Set minimum size to ensure the widget gets enough space in splitter
        self.setMinimumWidth(400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout()

        # Top bar with info label and view selector
        top_bar = QHBoxLayout()

        # Add info label
        self.info_label = QLabel("Click on a point in the scatter plot to view details")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc; }")
        top_bar.addWidget(self.info_label, stretch=1)

        # Add view selector
        view_selector_layout = QVBoxLayout()
        view_label = QLabel("View:")
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Segment", "Shot", "Rally"])
        self.view_combo.currentTextChanged.connect(self.on_view_changed)
        view_selector_layout.addWidget(view_label)
        view_selector_layout.addWidget(self.view_combo)
        top_bar.addLayout(view_selector_layout)

        # Add data source selector with checkboxes
        data_source_group = QGroupBox("Data Sources")
        data_source_layout = QVBoxLayout()
        data_source_layout.setSpacing(2)
        data_source_layout.setContentsMargins(5, 5, 5, 5)

        # Create checkboxes for each data source
        self.data_source_checkboxes = {}
        data_sources = ["OBS", "GT200", "OPT", "0226", "NAK", "Racket", "RCM_SIM", "NAK_REF", "CPP_NORES_REF", "RCM_LATEST"]  # Can easily add more sources here

        for source in data_sources:
            checkbox = QCheckBox(source)
            checkbox.setChecked(source not in ("Racket", "RCM_SIM", "NAK_REF", "CPP_NORES_REF", "RCM_LATEST"))  # Off by default for overlay sources
            checkbox.stateChanged.connect(self.on_data_source_changed)
            self.data_source_checkboxes[source] = checkbox
            data_source_layout.addWidget(checkbox)

        data_source_group.setLayout(data_source_layout)
        top_bar.addWidget(data_source_group)

        # Add reset button
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        self.reset_button.setMaximumWidth(100)
        top_bar.addWidget(self.reset_button)

        # Add save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        top_bar.addWidget(self.save_button)

        layout.addLayout(top_bar)

        # Create plot widgets in a 4x3 grid
        # Rows: Position, Position Difference, Velocity, Spin
        # Columns: X, Y, Z
        self.plot_widget = pg.GraphicsLayoutWidget()

        # Row 0 - Position
        self.plot_pos_x = self.plot_widget.addPlot(row=0, col=0, title="X Position")
        self.plot_pos_x.setLabel("left", "Position (m)")
        self.plot_pos_x.setLabel("bottom", "Time (s)")
        self.plot_pos_x.showGrid(x=True, y=True)
        self.plot_pos_x.addLegend()  # Only add legend to top-left plot
        self._set_axis_style(self.plot_pos_x)

        self.plot_pos_y = self.plot_widget.addPlot(row=0, col=1, title="Y Position")
        self.plot_pos_y.setLabel("left", "Position (m)")
        self.plot_pos_y.setLabel("bottom", "Time (s)")
        self.plot_pos_y.showGrid(x=True, y=True)
        self._set_axis_style(self.plot_pos_y)

        self.plot_pos_z = self.plot_widget.addPlot(row=0, col=2, title="Z Position")
        self.plot_pos_z.setLabel("left", "Position (m)")
        self.plot_pos_z.setLabel("bottom", "Time (s)")
        self.plot_pos_z.showGrid(x=True, y=True)
        self._set_axis_style(self.plot_pos_z)

        # Row 1 - Position Difference (GT200-OBS, OPT-OBS, CPP-OBS)
        self.plot_diff_x = self.plot_widget.addPlot(row=1, col=0, title="X Pos. Diff (vs OBS)")
        self.plot_diff_x.setLabel("left", "Diff (m)")
        self.plot_diff_x.setLabel("bottom", "Time (s)")
        self.plot_diff_x.showGrid(x=True, y=True)
        self.plot_diff_x.setXLink(self.plot_pos_x)
        self.plot_diff_x.addLegend()
        self._set_axis_style(self.plot_diff_x)

        self.plot_diff_y = self.plot_widget.addPlot(row=1, col=1, title="Y Pos. Diff (vs OBS)")
        self.plot_diff_y.setLabel("left", "Diff (m)")
        self.plot_diff_y.setLabel("bottom", "Time (s)")
        self.plot_diff_y.showGrid(x=True, y=True)
        self.plot_diff_y.setXLink(self.plot_pos_x)
        self._set_axis_style(self.plot_diff_y)

        self.plot_diff_z = self.plot_widget.addPlot(row=1, col=2, title="Z Pos. Diff (vs OBS)")
        self.plot_diff_z.setLabel("left", "Diff (m)")
        self.plot_diff_z.setLabel("bottom", "Time (s)")
        self.plot_diff_z.showGrid(x=True, y=True)
        self.plot_diff_z.setXLink(self.plot_pos_x)
        self._set_axis_style(self.plot_diff_z)

        # Row 2 - Velocity
        self.plot_vel_x = self.plot_widget.addPlot(row=2, col=0, title="X Velocity")
        self.plot_vel_x.setLabel("left", "Velocity (m/s)")
        self.plot_vel_x.setLabel("bottom", "Time (s)")
        self.plot_vel_x.showGrid(x=True, y=True)
        self.plot_vel_x.setXLink(self.plot_pos_x)  # Link X axis to position plot
        self._set_axis_style(self.plot_vel_x)

        self.plot_vel_y = self.plot_widget.addPlot(row=2, col=1, title="Y Velocity")
        self.plot_vel_y.setLabel("left", "Velocity (m/s)")
        self.plot_vel_y.setLabel("bottom", "Time (s)")
        self.plot_vel_y.showGrid(x=True, y=True)
        self.plot_vel_y.setXLink(self.plot_pos_x)  # Link X axis to position plot
        self._set_axis_style(self.plot_vel_y)

        self.plot_vel_z = self.plot_widget.addPlot(row=2, col=2, title="Z Velocity")
        self.plot_vel_z.setLabel("left", "Velocity (m/s)")
        self.plot_vel_z.setLabel("bottom", "Time (s)")
        self.plot_vel_z.showGrid(x=True, y=True)
        self.plot_vel_z.setXLink(self.plot_pos_x)  # Link X axis to position plot
        self._set_axis_style(self.plot_vel_z)

        # Row 3 - Spin (Angular Velocity)
        self.plot_spin_x = self.plot_widget.addPlot(row=3, col=0, title="X Spin")
        self.plot_spin_x.setLabel("left", "Angular Vel. (rad/s)")
        self.plot_spin_x.setLabel("bottom", "Time (s)")
        self.plot_spin_x.showGrid(x=True, y=True)
        self.plot_spin_x.setXLink(self.plot_pos_x)  # Link X axis to position plot
        self._set_axis_style(self.plot_spin_x)

        self.plot_spin_y = self.plot_widget.addPlot(row=3, col=1, title="Y Spin")
        self.plot_spin_y.setLabel("left", "Angular Vel. (rad/s)")
        self.plot_spin_y.setLabel("bottom", "Time (s)")
        self.plot_spin_y.showGrid(x=True, y=True)
        self.plot_spin_y.setXLink(self.plot_pos_x)  # Link X axis to position plot
        self._set_axis_style(self.plot_spin_y)

        self.plot_spin_z = self.plot_widget.addPlot(row=3, col=2, title="Z Spin")
        self.plot_spin_z.setLabel("left", "Angular Vel. (rad/s)")
        self.plot_spin_z.setLabel("bottom", "Time (s)")
        self.plot_spin_z.showGrid(x=True, y=True)
        self.plot_spin_z.setXLink(self.plot_pos_x)  # Link X axis to position plot
        self._set_axis_style(self.plot_spin_z)

        # Also link Y and Z position plots to X for consistency
        self.plot_pos_y.setXLink(self.plot_pos_x)
        self.plot_pos_z.setXLink(self.plot_pos_x)

        # Set fixed width for all left Y-axes to ensure column alignment
        # This ensures that plots in the same column have aligned time axes
        fixed_axis_width = 60
        for plot in [self.plot_pos_x, self.plot_diff_x, self.plot_vel_x, self.plot_spin_x]:
            plot.getAxis('left').setWidth(fixed_axis_width)
        for plot in [self.plot_pos_y, self.plot_diff_y, self.plot_vel_y, self.plot_spin_y]:
            plot.getAxis('left').setWidth(fixed_axis_width)
        for plot in [self.plot_pos_z, self.plot_diff_z, self.plot_vel_z, self.plot_spin_z]:
            plot.getAxis('left').setWidth(fixed_axis_width)

        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # Store current data
        self.current_flight_segment = None
        self.current_metadata = None
        self.current_shot_data = None
        self.current_rally_data = None
        self.current_shot_object = None
        self.current_rally_object = None

        # Track confidence overlay viewboxes for cleanup
        self._confidence_viewboxes = {}
        self._confidence_axes = {}

    def _plot_with_confidence(self, plot_widget, time_rel, values, confidence_data, add_legend=False, color=(255, 0, 0), label="GT200", overlay_confidence_data=None):
        """
        Plot data with confidence-based transparency and a filled area overlay

        Args:
            plot_widget: The plot widget to add points to
            time_rel: Time values (x-axis)
            values: Data values (y-axis)
            confidence_data: Confidence values (0-1) for marker transparency, or None (will show as 0)
            add_legend: If True, add label to the legend (only for first call)
            color: RGB tuple for the point color (default: red for GT200)
            label: Label for the legend (default: "GT200")
            overlay_confidence_data: Separate confidence values for the filled overlay, or None to use confidence_data
        """
        # Check for all-NaN or empty data to avoid pyqtgraph warnings
        if len(values) == 0 or np.all(np.isnan(values)):
            return  # Nothing to plot

        # Filter out NaN values
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            return

        time_valid = time_rel[valid_mask]
        values_valid = values[valid_mask]

        # Use the point color for contour (full opacity for visibility)
        contour_pen = pg.mkPen(color=(*color, 255), width=1)

        # Plot data points — always as a single batch ScatterPlotItem
        if confidence_data is None:
            # No confidence data, uniform transparency
            brushes = [pg.mkBrush(*color, 150)] * int(np.sum(valid_mask))
        else:
            # Get confidence values
            if hasattr(confidence_data, 'values'):
                confidence = confidence_data.values
            else:
                confidence = np.array(confidence_data)

            # Map confidence (0-1) to alpha (51-255)
            alpha_min, alpha_max = 51, 255
            confidence_clean = np.nan_to_num(confidence, nan=1.0, posinf=1.0, neginf=1.0)
            alpha_values = np.clip(
                (confidence_clean * (alpha_max - alpha_min) + alpha_min).astype(int),
                alpha_min, alpha_max,
            )
            alpha_valid = alpha_values[valid_mask]
            brushes = [pg.mkBrush(*color, int(a)) for a in alpha_valid]

        scatter = pg.ScatterPlotItem(
            x=time_valid, y=values_valid,
            symbol='o', size=8, pen=contour_pen, brush=brushes,
            name=label if add_legend else None,
        )
        plot_widget.addItem(scatter)

        # Use overlay_confidence_data if provided, otherwise fall back to confidence_data
        overlay_conf = overlay_confidence_data if overlay_confidence_data is not None else confidence_data
        # Always add confidence overlay - use zeros if confidence data is not available
        self._add_confidence_overlay(plot_widget, time_rel, overlay_conf, valid_mask)

    def _add_confidence_overlay(self, plot_widget, time_rel, confidence_data, valid_mask):
        """
        Add a filled confidence overlay to the plot using a secondary y-axis

        Args:
            plot_widget: The plot widget to add the overlay to
            time_rel: Time values (x-axis)
            confidence_data: Confidence values (0-1) or None (will show as 0)
            valid_mask: Boolean mask for valid data points
        """
        # Check if this plot already has a confidence overlay
        plot_id = id(plot_widget)
        if plot_id in self._confidence_viewboxes:
            # Already has overlay, skip (only show first confidence data)
            return

        time_valid = time_rel[valid_mask]

        if len(time_valid) < 2:
            return

        # Get confidence values - default to zeros if not available
        if confidence_data is None:
            confidence_clean = np.zeros(len(time_valid))
        else:
            if hasattr(confidence_data, 'values'):
                confidence = confidence_data.values
            else:
                confidence = np.array(confidence_data)

            # Apply the same valid mask and handle NaN/inf
            confidence_valid = confidence[valid_mask]

            # Handle NaN/inf values by replacing with 0 (no confidence)
            confidence_clean = np.nan_to_num(confidence_valid, nan=0.0, posinf=1.0, neginf=0.0)
            # Clip to valid range
            confidence_clean = np.clip(confidence_clean, 0.0, 1.0)

        # Sort by time for proper line plotting
        sort_idx = np.argsort(time_valid)
        time_sorted = time_valid[sort_idx]
        conf_sorted = confidence_clean[sort_idx]

        # Create a ViewBox for the confidence overlay (secondary y-axis)
        conf_viewbox = pg.ViewBox()
        plot_widget.scene().addItem(conf_viewbox)

        # Add right axis for confidence - only if not already present in the layout
        conf_axis = pg.AxisItem('right')
        conf_axis.setLabel('Confidence', color='#888888')

        # Check if cell (2, 3) is already occupied in this plot's layout
        # If so, remove the old item first
        # Note: PlotItem's internal layout is 3x3 by default, we're adding column 3
        layout = plot_widget.layout
        # Check layout bounds before accessing itemAt to avoid Qt warnings
        if layout.columnCount() > 3:
            existing_item = layout.itemAt(2, 3)
            if existing_item is not None:
                layout.removeItem(existing_item)
                if existing_item.scene() is not None:
                    existing_item.scene().removeItem(existing_item)

        layout.addItem(conf_axis, 2, 3)
        conf_axis.linkToView(conf_viewbox)
        conf_viewbox.setXLink(plot_widget)

        # Set the confidence y-range to 0-1
        conf_viewbox.setYRange(0, 1, padding=0)
        conf_viewbox.setMouseEnabled(x=True, y=False)  # Disable y-axis mouse interaction

        # Store for cleanup
        self._confidence_viewboxes[plot_id] = conf_viewbox
        self._confidence_axes[plot_id] = conf_axis

        # Update viewbox geometry when plot resizes
        def updateViews():
            conf_viewbox.setGeometry(plot_widget.vb.sceneBoundingRect())
            conf_viewbox.linkedViewChanged(plot_widget.vb, conf_viewbox.XAxis)

        plot_widget.vb.sigResized.connect(updateViews)
        updateViews()

        # Create filled polygon for confidence area (more reliable than FillBetweenItem)
        # Build a polygon: start at (time[0], 0), go along confidence line, end at (time[-1], 0)
        fill_x = np.concatenate([[time_sorted[0]], time_sorted, [time_sorted[-1]]])
        fill_y = np.concatenate([[0], conf_sorted, [0]])

        fill_brush = pg.mkBrush(color=(128, 128, 128, 80))
        fill_pen = pg.mkPen(color=(128, 128, 128, 200), width=1.5)
        fill_item = pg.PlotDataItem(fill_x, fill_y, pen=fill_pen, fillLevel=0, brush=fill_brush)
        conf_viewbox.addItem(fill_item)

    def _clear_confidence_overlays(self):
        """Remove all confidence overlay viewboxes and axes"""
        for plot_id, viewbox in self._confidence_viewboxes.items():
            # Remove items from viewbox
            viewbox.clear()
            # Remove viewbox from scene
            if viewbox.scene() is not None:
                viewbox.scene().removeItem(viewbox)

        for plot_id, axis in self._confidence_axes.items():
            # Remove axis from the grid layout properly
            # The axis was added to a PlotItem's layout, get that layout
            if axis.scene() is not None:
                axis.scene().removeItem(axis)
            # Also try to remove from parent layout if it exists
            try:
                parent_layout = axis.parentLayoutItem()
                if parent_layout is not None:
                    parent_layout.removeItem(axis)
            except Exception:
                pass

        self._confidence_viewboxes.clear()
        self._confidence_axes.clear()

    def on_view_changed(self, view_type: str):
        """Handle view type change"""
        # Re-plot with current data using new view type
        if self.current_flight_segment is not None:
            self.plot_flight_segment(
                self.current_flight_segment,
                self.current_metadata,
                self.current_shot_data,
                self.current_rally_data,
                self.current_shot_object,
                self.current_rally_object,
            )

    def on_data_source_changed(self):
        """Handle data source selection change"""
        # Re-plot with current data using new data source filter
        if self.current_flight_segment is not None:
            self.plot_flight_segment(
                self.current_flight_segment,
                self.current_metadata,
                self.current_shot_data,
                self.current_rally_data,
                self.current_shot_object,
                self.current_rally_object,
            )

    def reset_view(self):
        """Reset the view to auto-range and fit the data"""
        # Auto-range all plots to fit the current data
        all_plots = [
            self.plot_pos_x,
            self.plot_pos_y,
            self.plot_pos_z,
            self.plot_vel_x,
            self.plot_vel_y,
            self.plot_vel_z,
            self.plot_spin_x,
            self.plot_spin_y,
            self.plot_spin_z,
        ]

        for plot in all_plots:
            plot.enableAutoRange()
            plot.autoRange()

    def save_plot(self):
        """Save the current plot as a PNG image"""
        if self.current_metadata is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please select a flight segment first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on metadata and timestamp
        view_type = self.view_combo.currentText().lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"flight_segment_{self.current_metadata['match_date']}_"
            f"game{self.current_metadata['game_id']}_"
            f"rally{self.current_metadata['rally_id']}_"
            f"{view_type}_{timestamp}.png"
        )

        output_path = plots_dir / filename

        try:
            # Export the plot widget
            exporter = ImageExporter(self.plot_widget.scene())
            exporter.parameters()['width'] = 1920  # Set width in pixels
            exporter.export(str(output_path))

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def plot_flight_segment(
        self,
        flight_segment: FlightSegment,
        metadata: Dict[str, Any] = None,
        shot_data=None,
        rally_data=None,
        shot_object=None,
        rally_object=None,
    ):
        """Plot a flight segment's trajectory with position, velocity, and spin

        Args:
            flight_segment: The flight segment to plot
            metadata: Metadata dictionary
            shot_data: Full shot data (contains all flight segments for the shot)
            rally_data: Full rally DataFrame
            shot_object: Shot object containing flight segments
            rally_object: Rally object containing shots
        """
        # Store current data for view switching
        self.current_flight_segment = flight_segment
        self.current_metadata = metadata
        self.current_shot_data = shot_data
        self.current_rally_data = rally_data
        self.current_shot_object = shot_object
        self.current_rally_object = rally_object

        # Extract aerodynamic coefficients and sensitivities from flight segment data
        aero_info = ""
        fs_data = flight_segment.data
        cols = fs_data.columns

        # Get drag coefficient (first point value)
        cd_opt = None
        if "cd_opt" in cols:
            cd_vals = fs_data["cd_opt"].values
            valid_cd = cd_vals[~np.isnan(cd_vals)]
            if len(valid_cd) > 0:
                cd_opt = valid_cd[0]

        # Get magnus coefficient (first point value)
        cm_opt = None
        if "cm_opt" in cols:
            cm_vals = fs_data["cm_opt"].values
            valid_cm = cm_vals[~np.isnan(cm_vals)]
            if len(valid_cm) > 0:
                cm_opt = valid_cm[0]

        # Compute model predictions for drag and magnus coefficients
        cd_model = None
        cm_model = None
        BALL_RADIUS = 0.02  # m
        try:
            # Get velocity at first valid point
            if "vx_opt" in cols and "vy_opt" in cols and "vz_opt" in cols:
                vx = fs_data["vx_opt"].values
                vy = fs_data["vy_opt"].values
                vz = fs_data["vz_opt"].values
                vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                valid_vel_mask = ~np.isnan(vel_mag)
                if np.any(valid_vel_mask):
                    first_valid_idx = np.where(valid_vel_mask)[0][0]
                    v0 = vel_mag[first_valid_idx]

                    # Get spin at first valid point
                    if "wx" in cols and "wy" in cols and "wz" in cols:
                        wx = fs_data["wx"].values
                        wy = fs_data["wy"].values
                        wz = fs_data["wz"].values
                        spin_mag = np.sqrt(wx**2 + wy**2 + wz**2)
                        w0 = spin_mag[first_valid_idx] if not np.isnan(spin_mag[first_valid_idx]) else 0.0

                        # Compute drag model: C_D = f(v, S) where S = r*w/v
                        if v0 > 0:
                            S0 = BALL_RADIUS * w0 / v0
                            cd_model = get_drag_estimate_v1(v0, S0)

                        # Compute magnus model: C_M = f(v, w)
                        cm_model = get_magnus_estimate_v2(v0, w0)
        except Exception:
            pass

        # Compute drag force sensitivity: Sum ||v||² × (t_end - t_i) × dt
        drag_sensitivity = None
        if "vx_opt" in cols and "vy_opt" in cols and "vz_opt" in cols and "time" in cols and len(fs_data) > 1:
            try:
                vx = fs_data["vx_opt"].values
                vy = fs_data["vy_opt"].values
                vz = fs_data["vz_opt"].values
                time_vals = fs_data["time"].values
                vel_mag_sq = vx**2 + vy**2 + vz**2
                t_end = time_vals[-1]
                dt = np.diff(time_vals)
                time_to_end = t_end - time_vals[:-1]
                valid_mask = ~np.isnan(vel_mag_sq[:-1])
                if np.any(valid_mask):
                    drag_sensitivity = np.sum(vel_mag_sq[:-1][valid_mask] * time_to_end[valid_mask] * dt[valid_mask])
            except Exception:
                pass

        # Compute magnus force sensitivity: Sum ||v × ω|| × (t_end - t_i) × dt
        magnus_sensitivity = None
        if "vx_opt" in cols and "vy_opt" in cols and "vz_opt" in cols and "wx" in cols and "wy" in cols and "wz" in cols and "time" in cols and len(fs_data) > 1:
            try:
                vx = fs_data["vx_opt"].values
                vy = fs_data["vy_opt"].values
                vz = fs_data["vz_opt"].values
                wx_vals = fs_data["wx"].values
                wy_vals = fs_data["wy"].values
                wz_vals = fs_data["wz"].values
                time_vals = fs_data["time"].values

                # Fast first-valid-value lookup (spin is constant within a segment)
                def _first_valid(arr):
                    mask = ~np.isnan(arr)
                    return arr[mask][0] if np.any(mask) else arr[0]

                wx_mode = _first_valid(wx_vals)
                wy_mode = _first_valid(wy_vals)
                wz_mode = _first_valid(wz_vals)

                # Cross product magnitude
                cross_x = vy * wz_mode - vz * wy_mode
                cross_y = vz * wx_mode - vx * wz_mode
                cross_z = vx * wy_mode - vy * wx_mode
                cross_mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

                t_end = time_vals[-1]
                dt = np.diff(time_vals)
                time_to_end = t_end - time_vals[:-1]
                valid_mask = ~np.isnan(cross_mag[:-1])
                if np.any(valid_mask):
                    magnus_sensitivity = np.sum(cross_mag[:-1][valid_mask] * time_to_end[valid_mask] * dt[valid_mask])
            except Exception:
                pass

        # Build aerodynamics info string
        if cd_opt is not None or cd_model is not None:
            cd_parts = []
            if cd_opt is not None:
                cd_parts.append(f"OPT={cd_opt:.3f}")
            if cd_model is not None:
                cd_parts.append(f"model={cd_model:.3f}")
            aero_info += f"<b>C_D:</b> {', '.join(cd_parts)}"
        if drag_sensitivity is not None:
            aero_info += f" &nbsp; <b>Drag sens:</b> {drag_sensitivity:.1f}"
        if cm_opt is not None or cm_model is not None:
            if aero_info:
                aero_info += "<br>"
            cm_parts = []
            if cm_opt is not None:
                cm_parts.append(f"OPT={cm_opt:.3f}")
            if cm_model is not None:
                cm_parts.append(f"model={cm_model:.3f}")
            aero_info += f"<b>C_M:</b> {', '.join(cm_parts)}"
        if magnus_sensitivity is not None:
            aero_info += f" &nbsp; <b>Magnus sens:</b> {magnus_sensitivity:.1f}"

        # Update info label with metadata
        if metadata:
            info_text = (
                f"<b>Match Date:</b> {metadata['match_date']}<br>"
                f"<b>Match name:</b> {metadata['game_name']}<br>"
                f"<b>Player:</b> {metadata['match_player']}<br>"
                f"<b>Game ID:</b> {metadata['game_id']}<br>"
                f"<b>Rally ID:</b> {metadata['rally_id']}<br>"
                f"<b>Segment start time:</b> {metadata['segment_start_time']}s"
            )
            log_id = metadata.get('log_identifier', '')
            if log_id:
                info_text += f"<br><b>Log ID:</b> {log_id}"
            if aero_info:
                info_text += f"<br>{aero_info}"
            self.info_label.setText(info_text)

        # Get selected view type
        view_type = self.view_combo.currentText()

        # Determine which data to plot based on view type
        if view_type == "Segment":
            data = flight_segment.data
        elif view_type == "Shot" and shot_data is not None:
            # Combine all flight segments in the shot
            data = shot_data
        elif view_type == "Rally" and rally_data is not None:
            data = rally_data
        else:
            # Fallback to segment view
            data = flight_segment.data

        # ------------------------------------------------------------------
        # Suppress intermediate redraws while building the plots
        # ------------------------------------------------------------------
        self.plot_widget.setUpdatesEnabled(False)

        # Clear previous plots and confidence overlays
        self._clear_confidence_overlays()
        self.plot_pos_x.clear()
        self.plot_pos_y.clear()
        self.plot_pos_z.clear()
        self.plot_diff_x.clear()
        self.plot_diff_y.clear()
        self.plot_diff_z.clear()
        self.plot_vel_x.clear()
        self.plot_vel_y.clear()
        self.plot_vel_z.clear()
        self.plot_spin_x.clear()
        self.plot_spin_y.clear()
        self.plot_spin_z.clear()

        # Relative time (start from 0)
        time = data["time"].values
        time_rel = time - time[0]

        # Get selected data sources from checkboxes
        show_aps = self.data_source_checkboxes["OBS"].isChecked()
        show_gt200 = self.data_source_checkboxes["GT200"].isChecked()
        show_opt = self.data_source_checkboxes["OPT"].isChecked()
        show_0226 = self.data_source_checkboxes["0226"].isChecked()
        show_nak = self.data_source_checkboxes["NAK"].isChecked()
        show_racket = self.data_source_checkboxes["Racket"].isChecked()

        # Calculate segment boundaries for Shot and Rally views
        segment_boundaries = []
        if view_type == "Shot" and shot_object is not None:
            # Find the end time of each segment in the combined data
            for seg in shot_object.flight_segments[:-1]:  # Exclude last segment
                if len(seg.data) > 0:
                    seg_time = seg.data["time"].values
                    # Boundary is at the end of this segment, relative to start
                    boundary_time = seg_time[-1] - time[0]
                    segment_boundaries.append(boundary_time)
        elif view_type == "Rally" and rally_object is not None:
            # Find the end time of each segment in the combined data
            for shot in rally_object.shots:
                for seg in shot.flight_segments:
                    if len(seg.data) > 0:
                        seg_time = seg.data["time"].values
                        boundary_time = seg_time[-1] - time[0]
                        segment_boundaries.append(boundary_time)
            # Remove the last boundary (end of rally)
            if segment_boundaries:
                segment_boundaries = segment_boundaries[:-1]

        # Define color-matched contour pens for each data source
        pen_obs = pg.mkPen(color=(0, 150, 255, 255), width=1)  # Blue
        pen_gt200 = pg.mkPen(color=(255, 0, 0, 255), width=1)  # Red
        pen_opt = pg.mkPen(color=(150, 0, 255, 255), width=1)  # Purple
        pen_0226 = pg.mkPen(color=(255, 0, 255, 255), width=1) # Magenta
        pen_nak = pg.mkPen(color=(0, 180, 100, 255), width=1)  # Teal/Green
        pen_racket = pg.mkPen(color=(255, 165, 0, 255), width=2)  # Orange, thicker line

        # Debug: what data sources are available and what will be plotted
        available_sources = []
        if "x_aps" in data.columns:
            available_sources.append("OBS")
        if "x_gt200" in data.columns:
            available_sources.append("GT200")
        if "x_opt" in data.columns:
            available_sources.append("OPT")
        if "x_0226" in data.columns:
            available_sources.append("0226")
        if "x_nakashima" in data.columns:
            available_sources.append("NAK")
        if "robot_racket_x" in data.columns:
            available_sources.append("Racket")

        enabled_sources = []
        if show_aps:
            enabled_sources.append("OBS")
        if show_gt200:
            enabled_sources.append("GT200")
        if show_opt:
            enabled_sources.append("OPT")
        if show_0226:
            enabled_sources.append("0226")
        if show_nak:
            enabled_sources.append("NAK")
        if show_racket:
            enabled_sources.append("Racket")

        will_plot = [s for s in available_sources if s in enabled_sources]

        # Plot positions (row 0)
        if "x_aps" in data.columns and show_aps:
            self.plot_pos_x.plot(time_rel, data["x_aps"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_obs, symbolBrush=pg.mkBrush(0, 150, 255, 255), name="OBS")
            self.plot_pos_y.plot(time_rel, data["y_aps"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_obs, symbolBrush=pg.mkBrush(0, 150, 255, 255), name="OBS")
            self.plot_pos_z.plot(time_rel, data["z_aps"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_obs, symbolBrush=pg.mkBrush(0, 150, 255, 255), name="OBS")
        if "x_gt200" in data.columns and show_gt200:
            self._plot_with_confidence(self.plot_pos_x, time_rel, data["x_gt200"].values, data.get("x_confidence"), add_legend=True)
            self._plot_with_confidence(self.plot_pos_y, time_rel, data["y_gt200"].values, data.get("y_confidence"))
            self._plot_with_confidence(self.plot_pos_z, time_rel, data["z_gt200"].values, data.get("z_confidence"))
        if "x_opt" in data.columns and show_opt:
            self.plot_pos_x.plot(time_rel, data["x_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")
            self.plot_pos_y.plot(time_rel, data["y_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")
            self.plot_pos_z.plot(time_rel, data["z_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")
        if "x_0226" in data.columns and show_0226:
            self.plot_pos_x.plot(time_rel, data["x_0226"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255), name="0226")
            self.plot_pos_y.plot(time_rel, data["y_0226"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255), name="0226")
            self.plot_pos_z.plot(time_rel, data["z_0226"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255), name="0226")
        if "x_nakashima" in data.columns and show_nak:
            self.plot_pos_x.plot(time_rel, data["x_nakashima"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255), name="NAK")
            self.plot_pos_y.plot(time_rel, data["y_nakashima"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255), name="NAK")
            self.plot_pos_z.plot(time_rel, data["z_nakashima"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255), name="NAK")
        if "robot_racket_x" in data.columns and show_racket:
            self.plot_pos_x.plot(time_rel, data["robot_racket_x"].values, pen=pen_racket, symbol=None, name="Racket")
            self.plot_pos_y.plot(time_rel, data["robot_racket_y"].values, pen=pen_racket, symbol=None, name="Racket")
            self.plot_pos_z.plot(time_rel, data["robot_racket_z"].values, pen=pen_racket, symbol=None, name="Racket")

            # Projected racket extent band around the racket trajectory.
            # The racket face is an ellipse in its local y-z plane:
            #   semi-axis a = 0.085 m (local y, 17 cm diameter)
            #   semi-axis b = 0.075 m (local z, 15 cm diameter)
            # Projected half-extent along world axis i:
            #   sqrt((a * ey_world[i])^2 + (b * ez_world[i])^2)
            _RACKET_SEMI_Y = 0.085
            _RACKET_SEMI_Z = 0.075
            _quat_cols = ["robot_racket_qx", "robot_racket_qy", "robot_racket_qz", "robot_racket_qw"]
            if all(c in data.columns for c in _quat_cols):
                qx = data["robot_racket_qx"].values
                qy = data["robot_racket_qy"].values
                qz = data["robot_racket_qz"].values
                qw = data["robot_racket_qw"].values
                valid_q = np.isfinite(qx) & np.isfinite(qw)
                if np.any(valid_q):
                    quats = np.column_stack([qx, qy, qz, qw])
                    # Replace NaN quaternions with identity so batch conversion works
                    quats[~valid_q] = [0.0, 0.0, 0.0, 1.0]
                    rots = _R.from_quat(quats)
                    R_mats = rots.as_matrix()  # (N, 3, 3)
                    ey_w = R_mats[:, :, 1]     # (N, 3) — local y in world
                    ez_w = R_mats[:, :, 2]     # (N, 3) — local z in world
                    half_ext = np.sqrt(
                        (_RACKET_SEMI_Y * ey_w) ** 2 + (_RACKET_SEMI_Z * ez_w) ** 2
                    )  # (N, 3)
                    # NaN out invalid samples so the band is not drawn there
                    half_ext[~valid_q] = np.nan

                    _racket_pos = {
                        "x": data["robot_racket_x"].values,
                        "y": data["robot_racket_y"].values,
                        "z": data["robot_racket_z"].values,
                    }
                    _racket_plots = [(self.plot_pos_x, "x", 0),
                                     (self.plot_pos_y, "y", 1),
                                     (self.plot_pos_z, "z", 2)]
                    for plot_w, key, ax_i in _racket_plots:
                        h = half_ext[:, ax_i]
                        pos = _racket_pos[key]
                        upper = pos + h
                        lower = pos - h
                        c_upper = plot_w.plot(time_rel, upper, pen=pg.mkPen(None))
                        c_lower = plot_w.plot(time_rel, lower, pen=pg.mkPen(None))
                        fill = pg.FillBetweenItem(c_upper, c_lower,
                                                  brush=pg.mkBrush(255, 165, 0, 40))
                        plot_w.addItem(fill)

        # Plot position differences (row 1) - differences vs OBS
        # Only plot differences where OBS data is available
        if "x_aps" in data.columns:
            x_obs = data["x_aps"].values
            y_obs = data["y_aps"].values
            z_obs = data["z_aps"].values

            # GT200 - OBS (red)
            if "x_gt200" in data.columns and show_gt200:
                diff_x_gt200 = data["x_gt200"].values - x_obs
                diff_y_gt200 = data["y_gt200"].values - y_obs
                diff_z_gt200 = data["z_gt200"].values - z_obs
                self.plot_diff_x.plot(time_rel, diff_x_gt200, pen=None, symbol="o", symbolSize=8, symbolPen=pen_gt200, symbolBrush=pg.mkBrush(255, 0, 0, 255), name="GT200-OBS")
                self.plot_diff_y.plot(time_rel, diff_y_gt200, pen=None, symbol="o", symbolSize=8, symbolPen=pen_gt200, symbolBrush=pg.mkBrush(255, 0, 0, 255))
                self.plot_diff_z.plot(time_rel, diff_z_gt200, pen=None, symbol="o", symbolSize=8, symbolPen=pen_gt200, symbolBrush=pg.mkBrush(255, 0, 0, 255))
                # Add position confidence overlays to difference plots
                valid_mask = np.isfinite(diff_x_gt200)
                self._add_confidence_overlay(self.plot_diff_x, time_rel, data.get("x_confidence"), valid_mask)
                valid_mask = np.isfinite(diff_y_gt200)
                self._add_confidence_overlay(self.plot_diff_y, time_rel, data.get("y_confidence"), valid_mask)
                valid_mask = np.isfinite(diff_z_gt200)
                self._add_confidence_overlay(self.plot_diff_z, time_rel, data.get("z_confidence"), valid_mask)

            # OPT - OBS (purple)
            if "x_opt" in data.columns and show_opt:
                diff_x_opt = data["x_opt"].values - x_obs
                diff_y_opt = data["y_opt"].values - y_obs
                diff_z_opt = data["z_opt"].values - z_obs
                self.plot_diff_x.plot(time_rel, diff_x_opt, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT-OBS")
                self.plot_diff_y.plot(time_rel, diff_y_opt, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255))
                self.plot_diff_z.plot(time_rel, diff_z_opt, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255))

            # 0226 - OBS (magenta)
            if "x_0226" in data.columns and show_0226:
                diff_x_0226 = data["x_0226"].values - x_obs
                diff_y_0226 = data["y_0226"].values - y_obs
                diff_z_0226 = data["z_0226"].values - z_obs
                self.plot_diff_x.plot(time_rel, diff_x_0226, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255), name="0226-OBS")
                self.plot_diff_y.plot(time_rel, diff_y_0226, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255))
                self.plot_diff_z.plot(time_rel, diff_z_0226, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255))

            # NAK - OBS (teal/green)
            if "x_nakashima" in data.columns and show_nak:
                diff_x_nak = data["x_nakashima"].values - x_obs
                diff_y_nak = data["y_nakashima"].values - y_obs
                diff_z_nak = data["z_nakashima"].values - z_obs
                self.plot_diff_x.plot(time_rel, diff_x_nak, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255), name="NAK-OBS")
                self.plot_diff_y.plot(time_rel, diff_y_nak, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255))
                self.plot_diff_z.plot(time_rel, diff_z_nak, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255))

            # Add a zero reference line
            self.plot_diff_x.addLine(y=0, pen=pg.mkPen(color=(100, 100, 100), width=1, style=pg.QtCore.Qt.DashLine))
            self.plot_diff_y.addLine(y=0, pen=pg.mkPen(color=(100, 100, 100), width=1, style=pg.QtCore.Qt.DashLine))
            self.plot_diff_z.addLine(y=0, pen=pg.mkPen(color=(100, 100, 100), width=1, style=pg.QtCore.Qt.DashLine))

        # Plot velocities (row 2) - use position confidence for velocities
        if "vx_aps" in data.columns and show_aps:
            self.plot_vel_x.plot(time_rel, data["vx_aps"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_obs, symbolBrush=pg.mkBrush(0, 150, 255, 255), name="OBS")
            self.plot_vel_y.plot(time_rel, data["vy_aps"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_obs, symbolBrush=pg.mkBrush(0, 150, 255, 255), name="OBS")
            self.plot_vel_z.plot(time_rel, data["vz_aps"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_obs, symbolBrush=pg.mkBrush(0, 150, 255, 255), name="OBS")
        if "vx_gt200" in data.columns and show_gt200:
            self._plot_with_confidence(self.plot_vel_x, time_rel, data["vx_gt200"].values, data.get("x_confidence"))
            self._plot_with_confidence(self.plot_vel_y, time_rel, data["vy_gt200"].values, data.get("y_confidence"))
            self._plot_with_confidence(self.plot_vel_z, time_rel, data["vz_gt200"].values, data.get("z_confidence"))
        if "vx_opt" in data.columns and show_opt:
            self.plot_vel_x.plot(time_rel, data["vx_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")
            self.plot_vel_y.plot(time_rel, data["vy_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")
            self.plot_vel_z.plot(time_rel, data["vz_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")
        if "vx_0226" in data.columns and show_0226:
            self.plot_vel_x.plot(time_rel, data["vx_0226"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255), name="0226")
            self.plot_vel_y.plot(time_rel, data["vy_0226"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255), name="0226")
            self.plot_vel_z.plot(time_rel, data["vz_0226"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_0226, symbolBrush=pg.mkBrush(255, 0, 255, 255), name="0226")
        if "vx_nakashima" in data.columns and show_nak:
            self.plot_vel_x.plot(time_rel, data["vx_nakashima"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255), name="NAK")
            self.plot_vel_y.plot(time_rel, data["vy_nakashima"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255), name="NAK")
            self.plot_vel_z.plot(time_rel, data["vz_nakashima"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_nak, symbolBrush=pg.mkBrush(0, 180, 100, 255), name="NAK")
        if "robot_racket_vx" in data.columns and show_racket:
            self.plot_vel_x.plot(time_rel, data["robot_racket_vx"].values, pen=pen_racket, symbol=None, name="Racket")
            self.plot_vel_y.plot(time_rel, data["robot_racket_vy"].values, pen=pen_racket, symbol=None, name="Racket")
            self.plot_vel_z.plot(time_rel, data["robot_racket_vz"].values, pen=pen_racket, symbol=None, name="Racket")


        # Plot spin (angular velocity) (row 3)
        # OBS (GCS) uses w_confidence for marker transparency, but GT200 confidence (wx/wy/wz_confidence) for the overlay
        if "wx_gcs" in data.columns and show_aps:
            self._plot_with_confidence(self.plot_spin_x, time_rel, data["wx_gcs"].values, data.get("w_confidence"), add_legend=True, color=(0, 150, 255), label="OBS", overlay_confidence_data=data.get("wx_confidence"))
            self._plot_with_confidence(self.plot_spin_y, time_rel, data["wy_gcs"].values, data.get("w_confidence"), color=(0, 150, 255), label="OBS", overlay_confidence_data=data.get("wy_confidence"))
            self._plot_with_confidence(self.plot_spin_z, time_rel, data["wz_gcs"].values, data.get("w_confidence"), color=(0, 150, 255), label="OBS", overlay_confidence_data=data.get("wz_confidence"))
        # GT200 uses wx/wy/wz_confidence for both marker transparency and overlay
        if "wx" in data.columns and show_gt200:
            self._plot_with_confidence(self.plot_spin_x, time_rel, data["wx"].values, data.get("wx_confidence"))
            self._plot_with_confidence(self.plot_spin_y, time_rel, data["wy"].values, data.get("wy_confidence"))
            self._plot_with_confidence(self.plot_spin_z, time_rel, data["wz"].values, data.get("wz_confidence"))
        if "wx_opt" in data.columns and show_opt:
            self.plot_spin_x.plot(time_rel, data["wx_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")
            self.plot_spin_y.plot(time_rel, data["wy_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")
            self.plot_spin_z.plot(time_rel, data["wz_opt"].values, pen=None, symbol="o", symbolSize=8, symbolPen=pen_opt, symbolBrush=pg.mkBrush(150, 0, 255, 255), name="OPT")

        # Racket angular velocity (computed from quaternion finite differences)
        _quat_cols = ["robot_racket_qx", "robot_racket_qy", "robot_racket_qz", "robot_racket_qw"]
        if all(c in data.columns for c in _quat_cols) and show_racket:
            qx = data["robot_racket_qx"].values
            qy = data["robot_racket_qy"].values
            qz = data["robot_racket_qz"].values
            qw = data["robot_racket_qw"].values
            t_abs = data["time"].values
            valid_q = np.isfinite(qx) & np.isfinite(qw)
            # Need at least 2 valid consecutive samples for finite differences
            if np.sum(valid_q) >= 2:
                quats = np.column_stack([qx, qy, qz, qw])
                quats[~valid_q] = [0.0, 0.0, 0.0, 1.0]
                # Ensure consistent quaternion hemisphere (avoid sign flips)
                for i in range(1, len(quats)):
                    if np.dot(quats[i], quats[i - 1]) < 0:
                        quats[i] = -quats[i]
                # ω = 2 * q_inv(q[i]) * dq/dt   (central differences)
                omega = np.full((len(quats), 3), np.nan)
                for i in range(1, len(quats) - 1):
                    if not valid_q[i - 1] or not valid_q[i + 1]:
                        continue
                    dt = t_abs[i + 1] - t_abs[i - 1]
                    if dt < 1e-9:
                        continue
                    dq = (quats[i + 1] - quats[i - 1]) / dt  # dq/dt
                    # q_inv for unit quaternion: [-x, -y, -z, w]
                    q_inv = np.array([-quats[i][0], -quats[i][1], -quats[i][2], quats[i][3]])
                    # Quaternion multiply: 2 * dq * q_inv  (Hamilton product)
                    # Result vector part gives angular velocity
                    omega[i, 0] = 2.0 * (dq[3] * q_inv[0] + dq[0] * q_inv[3] + dq[1] * q_inv[2] - dq[2] * q_inv[1])
                    omega[i, 1] = 2.0 * (dq[3] * q_inv[1] - dq[0] * q_inv[2] + dq[1] * q_inv[3] + dq[2] * q_inv[0])
                    omega[i, 2] = 2.0 * (dq[3] * q_inv[2] + dq[0] * q_inv[1] - dq[1] * q_inv[0] + dq[2] * q_inv[3])
                omega[~valid_q] = np.nan
                self.plot_spin_x.plot(time_rel, omega[:, 0], pen=pen_racket, symbol=None, name="Racket")
                self.plot_spin_y.plot(time_rel, omega[:, 1], pen=pen_racket, symbol=None, name="Racket")
                self.plot_spin_z.plot(time_rel, omega[:, 2], pen=pen_racket, symbol=None, name="Racket")

        # ── RCM_SIM: overlay simulated post-contact trajectories ──────
        show_rcm_sim = self.data_source_checkboxes["RCM_SIM"].isChecked()
        if show_rcm_sim:
            pen_rcm_sim = pg.mkPen(color=(255, 100, 100, 220), width=2, style=pg.QtCore.Qt.DashLine)  # Dashed red-pink
            t0_abs = time[0]  # absolute time of first sample in the current view

            # Collect all flight segments visible in the current view
            sim_segments = []
            if view_type == "Segment":
                sim_segments = [flight_segment]
            elif view_type == "Shot" and shot_object is not None:
                sim_segments = list(shot_object.flight_segments)
            elif view_type == "Rally" and rally_object is not None:
                for s in rally_object.shots:
                    sim_segments.extend(s.flight_segments)

            for seg in sim_segments:
                ev = seg.trigger_event
                if not isinstance(ev, RacketContactEvent):
                    continue
                traj = getattr(ev, "simulated_trajectory", None)
                if traj is None:
                    continue
                if "t" not in traj or len(traj["t"]) < 2:
                    continue

                # Offset the sim time (relative to contact) into the view's time_rel
                sim_t_rel = traj["t"] + (ev.timestamp - t0_abs)

                # Position
                if "x" in traj:
                    self.plot_pos_x.plot(sim_t_rel, traj["x"], pen=pen_rcm_sim, name="RCM_SIM")
                if "y" in traj:
                    self.plot_pos_y.plot(sim_t_rel, traj["y"], pen=pen_rcm_sim, name="RCM_SIM")
                if "z" in traj:
                    self.plot_pos_z.plot(sim_t_rel, traj["z"], pen=pen_rcm_sim, name="RCM_SIM")

                # Velocity
                if "vx" in traj:
                    self.plot_vel_x.plot(sim_t_rel, traj["vx"], pen=pen_rcm_sim, name="RCM_SIM")
                if "vy" in traj:
                    self.plot_vel_y.plot(sim_t_rel, traj["vy"], pen=pen_rcm_sim, name="RCM_SIM")
                if "vz" in traj:
                    self.plot_vel_z.plot(sim_t_rel, traj["vz"], pen=pen_rcm_sim, name="RCM_SIM")

                # Spin
                if "wx" in traj:
                    self.plot_spin_x.plot(sim_t_rel, traj["wx"], pen=pen_rcm_sim, name="RCM_SIM")
                if "wy" in traj:
                    self.plot_spin_y.plot(sim_t_rel, traj["wy"], pen=pen_rcm_sim, name="RCM_SIM")
                if "wz" in traj:
                    self.plot_spin_z.plot(sim_t_rel, traj["wz"], pen=pen_rcm_sim, name="RCM_SIM")

        # ── NAK_REF / RCM_LATEST: overlay simulated post-contact trajectories ──
        _rcm_overlay_sources = [
            ("NAK_REF", "simulated_trajectory_nakashima_refined",
             pg.mkPen(color=(255, 140, 140, 220), width=2, style=pg.QtCore.Qt.DashLine)),  # Dashed light red
            ("CPP_NORES_REF", "simulated_trajectory_cpp_no_residual_refined",
             pg.mkPen(color=(180, 80, 40, 220), width=2, style=pg.QtCore.Qt.DashLine)),  # Dashed brown
            ("RCM_LATEST", "simulated_trajectory_latest",
             pg.mkPen(color=(180, 80, 255, 220), width=2, style=pg.QtCore.Qt.DashLine)),  # Dashed purple
        ]
        for _label, _attr, _pen in _rcm_overlay_sources:
            if not self.data_source_checkboxes[_label].isChecked():
                continue
            t0_abs = time[0]
            sim_segments = []
            if view_type == "Segment":
                sim_segments = [flight_segment]
            elif view_type == "Shot" and shot_object is not None:
                sim_segments = list(shot_object.flight_segments)
            elif view_type == "Rally" and rally_object is not None:
                for s in rally_object.shots:
                    sim_segments.extend(s.flight_segments)

            for seg in sim_segments:
                ev = seg.trigger_event
                if not isinstance(ev, RacketContactEvent):
                    continue
                traj = getattr(ev, _attr, None)
                if traj is None or "t" not in traj or len(traj["t"]) < 2:
                    continue
                sim_t_rel = traj["t"] + (ev.timestamp - t0_abs)
                for k, plot in [("x", self.plot_pos_x), ("y", self.plot_pos_y), ("z", self.plot_pos_z)]:
                    if k in traj:
                        plot.plot(sim_t_rel, traj[k], pen=_pen, name=_label)
                for k, plot in [("vx", self.plot_vel_x), ("vy", self.plot_vel_y), ("vz", self.plot_vel_z)]:
                    if k in traj:
                        plot.plot(sim_t_rel, traj[k], pen=_pen, name=_label)
                for k, plot in [("wx", self.plot_spin_x), ("wy", self.plot_spin_y), ("wz", self.plot_spin_z)]:
                    if k in traj:
                        plot.plot(sim_t_rel, traj[k], pen=_pen, name=_label)

        # Add vertical lines at segment boundaries
        if segment_boundaries:
            separator_pen = pg.mkPen(color=(150, 150, 150), width=2, style=pg.QtCore.Qt.DashLine)
            all_plots = [
                self.plot_pos_x,
                self.plot_pos_y,
                self.plot_pos_z,
                self.plot_diff_x,
                self.plot_diff_y,
                self.plot_diff_z,
                self.plot_vel_x,
                self.plot_vel_y,
                self.plot_vel_z,
                self.plot_spin_x,
                self.plot_spin_y,
                self.plot_spin_z,
            ]
            for boundary in segment_boundaries:
                for plot in all_plots:
                    plot.addLine(x=boundary, pen=separator_pen)

        # Highlight the selected flight segment in Shot/Rally views
        if view_type in ("Shot", "Rally") and flight_segment is not None:
            fs_time = flight_segment.data["time"].values
            if len(fs_time) > 0:
                t_start_sel = fs_time[0] - time[0]
                t_end_sel = fs_time[-1] - time[0]
                highlight_plots = [
                    self.plot_pos_x, self.plot_pos_y, self.plot_pos_z,
                    self.plot_diff_x, self.plot_diff_y, self.plot_diff_z,
                    self.plot_vel_x, self.plot_vel_y, self.plot_vel_z,
                    self.plot_spin_x, self.plot_spin_y, self.plot_spin_z,
                ]
                for plot in highlight_plots:
                    region = pg.LinearRegionItem(
                        values=(t_start_sel, t_end_sel),
                        brush=pg.mkBrush(255, 255, 100, 35),
                        pen=pg.mkPen(255, 255, 100, 120, width=1),
                        movable=False,
                    )
                    plot.addItem(region)

        # ------------------------------------------------------------------
        # Re-enable redraws and show everything at once
        # ------------------------------------------------------------------
        self.plot_widget.setUpdatesEnabled(True)

        # Reset view to show all data
        self.reset_view()
