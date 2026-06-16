# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Drag Coefficient Plot Widget

Widget for visualizing drag coefficients vs velocity and spin ratio.
"""

from typing import List, Dict, Any
import numpy as np
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QMessageBox, QSlider, QCheckBox, QGroupBox
from PySide6.QtCore import Signal, Qt, QTimer

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.exporters import ImageExporter

from ace_evaluation.utilities.data_classes import FlightSegment
from .aero_estimates import get_drag_estimate, get_drag_estimate_v1, get_drag_estimate_v2, get_drag_estimate_smoothstep, get_drag_estimate_spline
from .base_coefficient_plot import create_range_slider, create_combo_selector, save_plot_as_image


class DragCoefficientPlot(QWidget):
    """Widget for 3D plot of drag coefficient vs velocity and spin ratio"""

    # Define signal for flight segment selection
    flight_segment_selected = Signal(
        object, object, object, object, object, object
    )  # (FlightSegment, metadata, shot_data, rally_data, shot_object, rally_object)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Debounce timer for slider updates (prevents lag with large datasets)
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)  # 150ms debounce
        self._update_timer.timeout.connect(self.update_plot)

        # Top control bar
        control_bar = QHBoxLayout()

        # Add combo selectors using helper function
        view_layout, self.view_combo = create_combo_selector("View:", ["2D", "3D"], self.on_view_changed)
        control_bar.addLayout(view_layout)

        velocity_layout, self.velocity_combo = create_combo_selector("Velocity:", ["v", "v_eff_drag"], self.update_plot, default_index=1)
        control_bar.addLayout(velocity_layout)

        color_layout, self.color_combo = create_combo_selector("Color By:", ["Velocity", "Spin Ratio", "Drag Coefficient", "Drag force sensitivity", "Confidence", "Duration", "X Travel", "Player"], self.update_plot)
        control_bar.addLayout(color_layout)

        xaxis_layout, self.xaxis_combo = create_combo_selector("X-Axis:", ["C_D vs Spin Ratio", "C_D vs Velocity", "Velocity vs Spin Ratio", "Spin vs Velocity"], self.update_plot)
        control_bar.addLayout(xaxis_layout)

        # Add range sliders using helper function
        vel_layout, self.vel_min_slider, self.vel_max_slider, self.vel_min_label, self.vel_max_label = \
            create_range_slider("Velocity Range (m/s):", 0, 30, 0, 30, 5, self.on_velocity_range_changed, "{}", 1.0, 25)
        control_bar.addLayout(vel_layout)

        self._conf_layout, self.conf_min_slider, self.conf_max_slider, self.conf_min_label, self.conf_max_label = \
            create_range_slider("Confidence Range:", 0, 100, 50, 100, 10, self.on_confidence_range_changed, "{:.2f}", 100.0, 35)
        control_bar.addLayout(self._conf_layout)

        dur_layout, self.dur_min_slider, self.dur_max_slider, self.dur_min_label, self.dur_max_label = \
            create_range_slider("Duration Range (s):", 0, 200, 30, 200, 20, self.on_duration_range_changed, "{:.3f}", 200.0, 35)
        control_bar.addLayout(dur_layout)

        spin_layout, self.spin_min_slider, self.spin_max_slider, self.spin_min_label, self.spin_max_label = \
            create_range_slider("Spin Ratio Range:", 0, 20, 0, 20, 1, self.on_spin_range_changed, "{:.1f}", 10.0, 35)
        control_bar.addLayout(spin_layout)

        xtravel_layout, self.xtravel_min_slider, self.xtravel_max_slider, self.xtravel_min_label, self.xtravel_max_label = \
            create_range_slider("X Travel (m):", 0, 50, 0, 50, 10, self.on_xtravel_range_changed, "{:.1f}", 10.0, 35)
        control_bar.addLayout(xtravel_layout)

        # Error filter (RMSE vs OPT in mm)
        self._error_layout, self.error_min_slider, self.error_max_slider, self.error_min_label, self.error_max_label = \
            create_range_slider("Error vs OPT (mm):", 0, 100, 0, 100, 10, self.on_error_range_changed, "{:.0f}", 1.0, 35)
        control_bar.addLayout(self._error_layout)

        # Add checkbox to hide samples with inconsistent confidence
        self.hide_inconsistent_checkbox = QCheckBox("Hide Inconsistent Conf.")
        self.hide_inconsistent_checkbox.setChecked(True)
        self.hide_inconsistent_checkbox.setToolTip("Hide samples where confidence varies within the flight segment")
        self.hide_inconsistent_checkbox.stateChanged.connect(self.update_plot)
        control_bar.addWidget(self.hide_inconsistent_checkbox)

        # Add model variant checkboxes in a group
        model_variants_layout = QVBoxLayout()
        model_variants_label = QLabel("Model Variants:")
        model_variants_label.setStyleSheet("font-size: 9pt;")
        model_variants_layout.addWidget(model_variants_label)

        self.model_v1_checkbox = QCheckBox("V1 Piecewise")
        self.model_v1_checkbox.setChecked(True)
        self.model_v1_checkbox.stateChanged.connect(self.update_plot)
        model_variants_layout.addWidget(self.model_v1_checkbox)

        self.model_smoothstep_checkbox = QCheckBox("Smoothstep")
        self.model_smoothstep_checkbox.setChecked(False)
        self.model_smoothstep_checkbox.stateChanged.connect(self.update_plot)
        model_variants_layout.addWidget(self.model_smoothstep_checkbox)

        self.model_spline_checkbox = QCheckBox("Cubic Spline")
        self.model_spline_checkbox.setChecked(False)
        self.model_spline_checkbox.stateChanged.connect(self.update_plot)
        model_variants_layout.addWidget(self.model_spline_checkbox)

        control_bar.addLayout(model_variants_layout)

        # Point size slider
        size_layout = QVBoxLayout()
        self.point_size_label = QLabel("Point Size: 8")
        self.point_size_label.setStyleSheet("font-size: 9pt;")
        size_layout.addWidget(self.point_size_label)
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 30)
        self.point_size_slider.setValue(8)
        self.point_size_slider.setMaximumWidth(100)
        self.point_size_slider.valueChanged.connect(lambda v: self.point_size_label.setText(f"Point Size: {v}"))
        self.point_size_slider.valueChanged.connect(self.update_plot)
        size_layout.addWidget(self.point_size_slider)
        control_bar.addLayout(size_layout)

        # Publication mode checkbox
        self.publication_checkbox = QCheckBox("Publication")
        control_bar.addWidget(self.publication_checkbox)

        # Add save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Info label in a new row below controls
        self.info_label = QLabel("Click on a point to view the flight segment")
        layout.addWidget(self.info_label)

        # Create 3D plot widget using GLViewWidget
        try:
            import pyqtgraph.opengl as gl
            self.plot_widget_3d = gl.GLViewWidget()
            self.plot_widget_3d.opts['distance'] = 40
            self.plot_widget_3d.setWindowTitle('Drag Coefficient (cd_opt) vs Velocity and Spin Ratio')

            # Add grid
            gx = gl.GLGridItem()
            gx.rotate(90, 0, 1, 0)
            gx.translate(-10, 0, 0)
            self.plot_widget_3d.addItem(gx)
            gy = gl.GLGridItem()
            gy.rotate(90, 1, 0, 0)
            gy.translate(0, -10, 0)
            self.plot_widget_3d.addItem(gy)
            gz = gl.GLGridItem()
            gz.translate(0, 0, -10)
            self.plot_widget_3d.addItem(gz)

            self.has_3d = True
        except Exception as e:
            # Catch all exceptions - OpenGL init can fail in various ways
            print(f"3D plotting unavailable: {e}")
            self.plot_widget_3d = None
            self.has_3d = False
            self.view_combo.setCurrentText("2D")
            self.view_combo.setEnabled(False)

        # Create 2D plot widget with colorbar
        plot_2d_container = QHBoxLayout()

        self.plot_widget_2d = pg.PlotWidget(title="Drag Coefficient vs Spin Ratio (colored by Velocity)")
        self.plot_widget_2d.setLabel("left", "Drag Coefficient")
        self.plot_widget_2d.setLabel("bottom", "Spin Ratio (ωr/v)")
        self.plot_widget_2d.showGrid(x=True, y=True)
        # Disable SI prefix for full value display
        self.plot_widget_2d.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget_2d.getAxis('bottom').enableAutoSIPrefix(False)

        plot_2d_container.addWidget(self.plot_widget_2d)

        # Create colorbar widget using GraphicsLayoutWidget
        self.colorbar_widget_drag = pg.GraphicsLayoutWidget()
        self.colorbar_widget_drag.setMaximumWidth(100)
        self.colorbar_widget_drag.setMinimumWidth(80)
        plot_2d_container.addWidget(self.colorbar_widget_drag)

        # Create container widget for 2D plot + colorbar
        plot_2d_widget = QWidget()
        plot_2d_widget.setLayout(plot_2d_container)

        # Add both widgets to layout (will show/hide based on selection)
        if self.has_3d:
            layout.addWidget(self.plot_widget_3d, stretch=1)
        layout.addWidget(plot_2d_widget, stretch=1)

        self.plot_2d_widget = plot_2d_widget  # Store reference for show/hide

        # Initially show 2D by default
        if self.has_3d:
            self.plot_widget_3d.hide()

        # Initialize colorbar reference
        self.colorbar_drag = None

        self.setLayout(layout)

        # Store data references
        self.robot_data = None
        self.player_data = None
        self.scatter_items = {}
        self.radius_ball = 0.02  # 2cm radius

        # Pre-merged data (computed once in plot_data, reused on each slider update)
        self._merged = None

    def on_view_changed(self, view_type: str):
        """Handle view type change between 2D and 3D"""
        if view_type == "3D" and self.has_3d:
            self.plot_widget_3d.show()
            self.plot_2d_widget.hide()
        else:
            if self.has_3d:
                self.plot_widget_3d.hide()
            self.plot_2d_widget.show()

        # Re-plot with current data using new view
        self.update_plot()

    def on_velocity_range_changed(self):
        """Handle velocity range slider changes"""
        # Update labels
        vel_min = self.vel_min_slider.value()
        vel_max = self.vel_max_slider.value()

        # Ensure min <= max
        if vel_min > vel_max:
            if self.sender() == self.vel_min_slider:
                self.vel_min_slider.setValue(vel_max)
                vel_min = vel_max
            else:
                self.vel_max_slider.setValue(vel_min)
                vel_max = vel_min

        self.vel_min_label.setText(str(vel_min))
        self.vel_max_label.setText(str(vel_max))

        # Debounced update
        self._update_timer.start()

    def on_duration_range_changed(self):
        """Handle duration range slider changes"""
        # Get slider values (scaled 0.005 s per unit)
        dur_min = self.dur_min_slider.value() / 200.0
        dur_max = self.dur_max_slider.value() / 200.0

        # Ensure min <= max
        if dur_min > dur_max:
            if self.sender() == self.dur_min_slider:
                self.dur_min_slider.setValue(int(dur_max * 200))
                dur_min = dur_max
            else:
                self.dur_max_slider.setValue(int(dur_min * 200))
                dur_max = dur_min

        # Update labels
        self.dur_min_label.setText(f"{dur_min:.3f}")
        self.dur_max_label.setText(f"{dur_max:.3f}")

        # Debounced update
        self._update_timer.start()

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

        # Debounced update
        self._update_timer.start()

    def on_spin_range_changed(self):
        """Handle spin ratio range slider changes"""
        # Get slider values (scaled 0-20 -> 0.0-2.0)
        spin_min = self.spin_min_slider.value() / 10.0
        spin_max = self.spin_max_slider.value() / 10.0

        # Ensure min <= max
        if spin_min > spin_max:
            if self.sender() == self.spin_min_slider:
                self.spin_min_slider.setValue(int(spin_max * 10))
                spin_min = spin_max
            else:
                self.spin_max_slider.setValue(int(spin_min * 10))
                spin_max = spin_min

        # Update labels
        self.spin_min_label.setText(f"{spin_min:.1f}")
        self.spin_max_label.setText(f"{spin_max:.1f}")

        # Debounced update
        self._update_timer.start()

    def on_xtravel_range_changed(self):
        """Handle X travel range slider changes"""
        # Get slider values (scaled 0-50 -> 0.0-5.0m)
        xtravel_min = self.xtravel_min_slider.value() / 10.0
        xtravel_max = self.xtravel_max_slider.value() / 10.0

        # Ensure min <= max
        if xtravel_min > xtravel_max:
            if self.sender() == self.xtravel_min_slider:
                self.xtravel_min_slider.setValue(int(xtravel_max * 10))
                xtravel_min = xtravel_max
            else:
                self.xtravel_max_slider.setValue(int(xtravel_min * 10))
                xtravel_max = xtravel_min

        # Update labels
        self.xtravel_min_label.setText(f"{xtravel_min:.1f}")
        self.xtravel_max_label.setText(f"{xtravel_max:.1f}")

        # Debounced update
        self._update_timer.start()

    def on_error_range_changed(self):
        """Handle error (RMSE vs OPT) range slider changes"""
        # Get slider values (in mm)
        error_min = self.error_min_slider.value()
        error_max = self.error_max_slider.value()

        # Ensure min <= max
        if error_min > error_max:
            if self.sender() == self.error_min_slider:
                self.error_min_slider.setValue(error_max)
                error_min = error_max
            else:
                self.error_max_slider.setValue(error_min)
                error_max = error_min

        # Update labels
        self.error_min_label.setText(f"{error_min:.0f}")
        self.error_max_label.setText(f"{error_max:.0f}")

        # Debounced update
        self._update_timer.start()

    def update_plot(self):
        """Update the plot based on selected data sources"""
        view_type = self.view_combo.currentText()
        if view_type == "3D" and self.has_3d:
            self.update_3d_plot()
        else:
            self.update_2d_plot()

    def update_3d_plot(self):
        """Update 3D scatter plot"""
        import pyqtgraph.opengl as gl

        # Clear existing scatter plots safely
        for item in list(self.scatter_items.values()):
            if isinstance(item, tuple):
                try:
                    self.plot_widget_3d.removeItem(item[0])
                except (ValueError, AttributeError):
                    pass  # Item already removed or not in list
        self.scatter_items = {}

        # Use pre-merged data from plot_data() (avoids re-merging on every slider change)
        m = self._merged
        if m is None:
            self.info_label.setText("No data points to display")
            return

        # Select velocity and spin ratio based on user choice
        velocity_type = self.velocity_combo.currentText()
        if velocity_type == "v_eff_drag":
            selected_velocity = m['v_eff_drag'].copy()
            all_spin_ratio = m['spin_ratio_veff'].copy()
            vel_label = "v_eff_drag"
        else:
            selected_velocity = m['vel_mag'].copy()
            all_spin_ratio = m['spin_ratio'].copy()
            vel_label = "v"

        all_vel_mag = m['vel_mag']
        all_v_eff_drag = m['v_eff_drag']
        all_spin_mag = m['spin_mag']
        all_cd_opt = m['cd_opt']
        all_drag_impulse_proxy = m['drag_impulse_proxy']
        all_durations = m['durations']
        all_x_travel = m['x_travel']

        # Apply velocity range filter
        vel_min = self.vel_min_slider.value()
        vel_max = self.vel_max_slider.value()
        # When max is 30, show all data >= 30
        if vel_max >= 30:
            vel_mask = (selected_velocity >= vel_min)
        else:
            vel_mask = (selected_velocity >= vel_min) & (selected_velocity <= vel_max)

        # Apply confidence range filter
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0
        conf_mask = (m['confidence'] >= conf_min) & (m['confidence'] <= conf_max)

        # Apply duration range filter
        dur_min = self.dur_min_slider.value() / 200.0
        dur_max = self.dur_max_slider.value() / 200.0
        # When max is 1.0, show all data >= 1.0
        if dur_max >= 1.0:
            dur_mask = (all_durations >= dur_min)
        else:
            dur_mask = (all_durations >= dur_min) & (all_durations <= dur_max)

        # Apply spin ratio range filter
        spin_min = self.spin_min_slider.value() / 10.0
        spin_max = self.spin_max_slider.value() / 10.0
        # When max is 2.0, show all data >= 2.0
        if spin_max >= 2.0:
            spin_mask = (all_spin_ratio >= spin_min)
        else:
            spin_mask = (all_spin_ratio >= spin_min) & (all_spin_ratio <= spin_max)

        # Apply X travel range filter
        xtravel_min = self.xtravel_min_slider.value() / 10.0
        xtravel_max = self.xtravel_max_slider.value() / 10.0
        # When max is 5.0, show all data >= 5.0; also allow NaN (missing x_aps data)
        if xtravel_max >= 5.0:
            xtravel_mask = (all_x_travel >= xtravel_min) | np.isnan(all_x_travel)
        else:
            xtravel_mask = ((all_x_travel >= xtravel_min) & (all_x_travel <= xtravel_max)) | np.isnan(all_x_travel)

        # Apply error (RMSE vs OPT) range filter - values are in mm
        error_min = self.error_min_slider.value() / 1000.0  # Convert mm to meters
        error_max = self.error_max_slider.value() / 1000.0
        # When max is at slider maximum, show all data >= min; also allow NaN (missing error data)
        if self.error_max_slider.value() >= self.error_max_slider.maximum():
            error_mask = (m['rmse_opt'] >= error_min) | np.isnan(m['rmse_opt'])
        else:
            error_mask = ((m['rmse_opt'] >= error_min) & (m['rmse_opt'] <= error_max)) | np.isnan(m['rmse_opt'])

        # Apply inconsistent confidence filter
        if self.hide_inconsistent_checkbox.isChecked():
            consistent_mask = ~m['inconsistent']
        else:
            consistent_mask = np.ones(len(m['inconsistent']), dtype=bool)

        # NaN validity check (combine with other masks to avoid double-filtering)
        nan_mask = ~(np.isnan(selected_velocity) | np.isnan(all_spin_ratio) | np.isnan(all_cd_opt))

        # Combine all masks into one final mask
        combined_mask = vel_mask & conf_mask & dur_mask & spin_mask & xtravel_mask & error_mask & consistent_mask & nan_mask

        # Filter all arrays using numpy boolean indexing (no Python list-comps)
        selected_velocity = selected_velocity[combined_mask]
        all_vel_mag = all_vel_mag[combined_mask]
        all_v_eff_drag = all_v_eff_drag[combined_mask]
        all_spin_mag = all_spin_mag[combined_mask]
        all_spin_ratio = all_spin_ratio[combined_mask]
        all_cd_opt = all_cd_opt[combined_mask]
        all_durations = all_durations[combined_mask]
        all_drag_impulse_proxy = all_drag_impulse_proxy[combined_mask]

        all_flight_segments = m['flight_segments'][combined_mask]
        all_metadata_list = m['metadata_list'][combined_mask]
        all_shot_data_list = m['shot_data_list'][combined_mask]
        all_rally_data_list = m['rally_data_list'][combined_mask]
        all_shot_object_list = m['shot_object_list'][combined_mask]
        all_rally_object_list = m['rally_object_list'][combined_mask]
        all_confidence_list = m['confidence'][combined_mask]

        if len(selected_velocity) == 0:
            self.info_label.setText(f"No data points in velocity range [{vel_min}, {vel_max}] m/s and confidence range [{conf_min:.2f}, {conf_max:.2f}]")
            return

        # Normalization function to map data ranges to 0-25 scale
        def normalize_to_range(data, min_val, max_val, target_range=25.0):
            """Normalize data from [min_val, max_val] to [0, target_range]"""
            return (data - min_val) / (max_val - min_val) * target_range

        # Normalize each dimension to 0-25 range
        vel_norm = normalize_to_range(selected_velocity, 0, 25)
        spin_ratio_norm = normalize_to_range(all_spin_ratio, 0, 5)
        cd_norm = normalize_to_range(all_cd_opt, 0.3, 0.8)

        pos = np.column_stack([vel_norm, spin_ratio_norm, cd_norm])

        # Get color-by option
        color_by = self.color_combo.currentText()

        # Color points by selected quantity using a colormap
        import matplotlib.cm as cm
        import matplotlib.colors as colors

        # Handle player coloring separately (categorical)
        if color_by == "Player":
            # Extract player names from metadata
            all_players = [m.get("match_player", "Unknown") for m in all_metadata_list]
            unique_players = sorted(set(all_players))
            n_players = len(unique_players)

            # Use tab10 or tab20 colormap for categorical data
            cmap = cm.get_cmap('tab10' if n_players <= 10 else 'tab20')
            player_to_idx = {p: i for i, p in enumerate(unique_players)}

            # Vectorized color generation for player coloring
            player_indices = np.array([player_to_idx[p] for p in all_players])
            norm_vals = player_indices / max(n_players, 1)
            point_colors = np.array(cmap(norm_vals))
        else:
            # Determine what to color by
            # Use fixed ranges based on slider maximums for consistent color scaling
            if color_by == "Velocity":
                color_data = selected_velocity
                color_min = 0.0
                color_max = 30.0  # Slider max
            elif color_by == "Spin Ratio":
                color_data = all_spin_ratio
                color_min = 0.0
                color_max = 2.0  # Slider max
            elif color_by == "Drag Coefficient":
                color_data = all_cd_opt
                color_min = 0.3
                color_max = 0.8
            elif color_by == "Drag force sensitivity":
                # Drag force sensitivity: Sum ||v||² × (t_end - t_i) × dt (cached)
                color_data = all_drag_impulse_proxy
                color_min = 0.0
                color_max = 50.0  # Adjusted for time-weighted integral
            elif color_by == "Duration":
                color_data = all_durations
                color_min = 0.0
                color_max = 1.0  # Slider max
            else:  # Confidence
                color_data = np.array(all_confidence_list)
                color_min = 0.0
                color_max = 1.0  # Slider max

            # Ensure we have a valid range
            if color_max <= color_min:
                color_max = color_min + 0.001

            # Normalize color data for coloring
            norm = colors.Normalize(vmin=color_min, vmax=color_max)
            cmap = cm.get_cmap('turbo')  # Wide color range: blue -> cyan -> green -> yellow -> red

            # Vectorized color generation for continuous coloring
            norm_vals = norm(color_data)
            point_colors = np.array(cmap(norm_vals))

        scatter_item = gl.GLScatterPlotItem(pos=pos, color=point_colors, size=self.point_size_slider.value(), pxMode=True)
        self.plot_widget_3d.addItem(scatter_item)
        self.scatter_items["Combined"] = (scatter_item, all_flight_segments, all_metadata_list, all_shot_data_list, all_rally_data_list, all_shot_object_list, all_rally_object_list)

        self.info_label.setText(f"Showing {len(selected_velocity)} data points (vel: {vel_min}-{vel_max} m/s, conf: {conf_min:.2f}-{conf_max:.2f}). 3D normalized.")
        # Center camera at middle of normalized range (12.5, 12.5, 12.5)
        self.plot_widget_3d.opts['center'] = pg.Vector(12.5, 12.5, 12.5)
        self.plot_widget_3d.opts['distance'] = 40

    def update_2d_plot(self):
        """Update 2D scatter plot: Drag Coefficient vs Spin Ratio, colored by Velocity"""

        self.plot_widget_2d.clear()
        self.scatter_items = {}

        # Clear existing colorbar if present
        self.colorbar_widget_drag.clear()
        self.colorbar_drag = None

        # Use pre-merged data from plot_data() (avoids re-merging on every slider change)
        m = self._merged
        if m is None:
            self.info_label.setText("No data points to display")
            return

        # Select velocity and spin ratio based on user choice
        velocity_type = self.velocity_combo.currentText()
        if velocity_type == "v_eff_drag":
            selected_velocity = m['v_eff_drag'].copy()
            all_spin_ratio = m['spin_ratio_veff'].copy()
            vel_label = "v_eff_drag"
        else:
            selected_velocity = m['vel_mag'].copy()
            all_spin_ratio = m['spin_ratio'].copy()
            vel_label = "v"

        all_vel_mag = m['vel_mag']
        all_v_eff_drag = m['v_eff_drag']
        all_spin_mag = m['spin_mag']
        all_cd_opt = m['cd_opt']
        all_drag_impulse_proxy = m['drag_impulse_proxy']
        all_durations = m['durations']
        all_x_travel = m['x_travel']

        # Apply velocity range filter
        vel_min_filter = self.vel_min_slider.value()
        vel_max_filter = self.vel_max_slider.value()
        # When max is 30, show all data >= 30
        if vel_max_filter >= 30:
            vel_mask = (selected_velocity >= vel_min_filter)
        else:
            vel_mask = (selected_velocity >= vel_min_filter) & (selected_velocity <= vel_max_filter)

        # Apply confidence range filter
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0
        conf_mask = (m['confidence'] >= conf_min) & (m['confidence'] <= conf_max)

        # Apply duration range filter
        dur_min = self.dur_min_slider.value() / 200.0
        dur_max = self.dur_max_slider.value() / 200.0
        # When max is 1.0, show all data >= 1.0
        if dur_max >= 1.0:
            dur_mask = (all_durations >= dur_min)
        else:
            dur_mask = (all_durations >= dur_min) & (all_durations <= dur_max)

        # Apply spin ratio range filter
        spin_min = self.spin_min_slider.value() / 10.0
        spin_max = self.spin_max_slider.value() / 10.0
        # When max is 2.0, show all data >= 2.0
        if spin_max >= 2.0:
            spin_mask = (all_spin_ratio >= spin_min)
        else:
            spin_mask = (all_spin_ratio >= spin_min) & (all_spin_ratio <= spin_max)

        # Apply X travel range filter
        xtravel_min = self.xtravel_min_slider.value() / 10.0
        xtravel_max = self.xtravel_max_slider.value() / 10.0
        # When max is 5.0, show all data >= 5.0; also allow NaN (missing x_aps data)
        if xtravel_max >= 5.0:
            xtravel_mask = (all_x_travel >= xtravel_min) | np.isnan(all_x_travel)
        else:
            xtravel_mask = ((all_x_travel >= xtravel_min) & (all_x_travel <= xtravel_max)) | np.isnan(all_x_travel)

        # Apply error (RMSE vs OPT) range filter - values are in mm
        error_min = self.error_min_slider.value() / 1000.0  # Convert mm to meters
        error_max = self.error_max_slider.value() / 1000.0
        # When max is at slider maximum, show all data >= min; also allow NaN (missing error data)
        if self.error_max_slider.value() >= self.error_max_slider.maximum():
            error_mask = (m['rmse_opt'] >= error_min) | np.isnan(m['rmse_opt'])
        else:
            error_mask = ((m['rmse_opt'] >= error_min) & (m['rmse_opt'] <= error_max)) | np.isnan(m['rmse_opt'])

        # Apply inconsistent confidence filter
        if self.hide_inconsistent_checkbox.isChecked():
            consistent_mask = ~m['inconsistent']
        else:
            consistent_mask = np.ones(len(m['inconsistent']), dtype=bool)

        # NaN validity check (combine with other masks to avoid double-filtering)
        nan_mask = ~(np.isnan(selected_velocity) | np.isnan(all_spin_ratio) | np.isnan(all_cd_opt))

        # Combine all masks into one final mask
        combined_mask = vel_mask & conf_mask & dur_mask & spin_mask & xtravel_mask & error_mask & consistent_mask & nan_mask

        # Filter all arrays using numpy boolean indexing (no Python list-comps)
        selected_velocity = selected_velocity[combined_mask]
        all_vel_mag = all_vel_mag[combined_mask]
        all_v_eff_drag = all_v_eff_drag[combined_mask]
        all_spin_mag = all_spin_mag[combined_mask]
        all_spin_ratio = all_spin_ratio[combined_mask]
        all_cd_opt = all_cd_opt[combined_mask]
        all_durations = all_durations[combined_mask]
        all_x_travel = all_x_travel[combined_mask]
        all_drag_impulse_proxy = all_drag_impulse_proxy[combined_mask]

        all_flight_segments = m['flight_segments'][combined_mask]
        all_metadata_list = m['metadata_list'][combined_mask]
        all_shot_data_list = m['shot_data_list'][combined_mask]
        all_rally_data_list = m['rally_data_list'][combined_mask]
        all_shot_object_list = m['shot_object_list'][combined_mask]
        all_rally_object_list = m['rally_object_list'][combined_mask]
        all_confidence_list = m['confidence'][combined_mask]

        if len(selected_velocity) == 0:
            self.info_label.setText(f"No data points in velocity range [{vel_min_filter}, {vel_max_filter}] m/s and confidence range [{conf_min:.2f}, {conf_max:.2f}]")
            return

        # Get color-by option
        color_by = self.color_combo.currentText()

        # Handle player coloring separately (categorical)
        is_player_coloring = (color_by == "Player")

        if is_player_coloring:
            # Extract player names from metadata
            all_players = [m.get("match_player", "Unknown") for m in all_metadata_list]
            unique_players = sorted(set(all_players))
            n_players = len(unique_players)

            # Use tab10 or tab20 colormap for categorical data
            cmap = plt.get_cmap('tab10' if n_players <= 10 else 'tab20')
            player_to_idx = {p: i for i, p in enumerate(unique_players)}
            color_label = "Player"
            color_min = 0
            color_max = max(n_players - 1, 1)
        else:
            # Determine what to color by
            # Use fixed ranges based on slider maximums for consistent color scaling
            if color_by == "Velocity":
                color_data = selected_velocity
                color_label = f"{vel_label} (m/s)"
                color_min = float(vel_min_filter)
                color_max = float(vel_max_filter)
            elif color_by == "Spin Ratio":
                color_data = all_spin_ratio
                color_label = "Spin Ratio"
                color_min = self.spin_min_slider.value() / 10.0
                color_max = self.spin_max_slider.value() / 10.0
            elif color_by == "Drag Coefficient":
                color_data = all_cd_opt
                color_label = "Drag Coefficient"
                color_min = 0.3
                color_max = 0.8
            elif color_by == "Drag force sensitivity":
                # Drag force sensitivity: Sum ||v||² × (t_end - t_i) × dt (cached)
                color_data = all_drag_impulse_proxy
                color_label = "Drag sensitivity (m²)"
                color_min = 0.0
                color_max = 50.0  # Adjusted for time-weighted integral
            elif color_by == "Duration":
                color_data = all_durations
                color_label = "Duration (s)"
                color_min = 0.0
                color_max = 1.0  # Slider max
            elif color_by == "X Travel":
                color_data = all_x_travel
                color_label = "X Travel (m)"
                color_min = 0.0
                color_max = 5.0  # Slider max
            else:  # Confidence
                color_data = np.array(all_confidence_list)
                color_label = "Confidence"
                color_min = 0.0
                color_max = 1.0  # Slider max

            # Ensure we have a valid range
            if color_max <= color_min:
                color_max = color_min + 0.001

            # Create a colormap for the selected quantity (turbo: blue -> cyan -> green -> yellow -> red)
            cmap = plt.get_cmap('turbo')

        # Vectorized brush creation (replaces per-point Python loop)
        n_pts = len(selected_velocity)
        if is_player_coloring:
            player_indices = np.array([player_to_idx[p] for p in all_players])
            norm_vals = player_indices / max(n_players - 1, 1)
        else:
            norm_vals = np.clip(color_data, color_min, color_max)
            if color_max > color_min:
                norm_vals = (norm_vals - color_min) / (color_max - color_min)
            else:
                norm_vals = np.full(n_pts, 0.5)

        # Vectorized colormap call returns (N, 4) float array
        rgba_array = cmap(norm_vals)

        # Build (N, 4) uint8 array: RGB from colormap, alpha from confidence
        brush_colors = np.empty((n_pts, 4), dtype=np.uint8)
        brush_colors[:, :3] = (rgba_array[:, :3] * 255).astype(np.uint8)

        # Confidence-based alpha: map confidence (0-1) to alpha (51-255)
        conf_arr = np.array(all_confidence_list, dtype=np.float64)
        alpha_vals = np.clip((conf_arr * (255 - 51) + 51), 51, 255).astype(np.uint8)
        brush_colors[:, 3] = alpha_vals

        brushes = [pg.mkBrush(int(c[0]), int(c[1]), int(c[2]), int(c[3])) for c in brush_colors]

        # Get X-axis mode
        xaxis_mode = self.xaxis_combo.currentText()  # "C_D vs Spin Ratio", "C_D vs Velocity", "Velocity vs Spin Ratio", or "Spin vs Velocity"

        # Determine X and Y data based on mode
        if xaxis_mode == "Velocity vs Spin Ratio":
            x_data = selected_velocity
            y_data = all_spin_ratio
            x_label = f"{vel_label} (m/s)"
            y_label = "Spin Ratio (ωr/v)"
            x_min, x_max = 0, 30
            y_min, y_max = 0, 2
        elif xaxis_mode == "Spin vs Velocity":
            x_data = selected_velocity
            y_data = all_spin_mag
            x_label = f"{vel_label} (m/s)"
            y_label = "Spin Magnitude (rad/s)"
            x_min, x_max = 0, 30
            y_min, y_max = 0, 800
        elif xaxis_mode == "C_D vs Velocity":
            x_data = selected_velocity
            y_data = all_cd_opt
            x_label = f"{vel_label} (m/s)"
            y_label = "Drag Coefficient"
            x_min, x_max = 0, 30
            y_min, y_max = 0.3, 0.6
        else:  # "C_D vs Spin Ratio"
            x_data = all_spin_ratio
            y_data = all_cd_opt
            x_label = "Spin Ratio (ωr/v)"
            y_label = "Drag Coefficient"
            x_min, x_max = 0, 2
            y_min, y_max = 0.3, 0.6

        # Vectorized scatter plot creation (replaces per-point spot dict loop)
        scatter_item = pg.ScatterPlotItem(
            x=x_data, y=y_data,
            size=self.point_size_slider.value(),
            pen=None,
            brush=brushes,
            data=np.arange(len(x_data)),
            hoverable=True, tip=None
        )
        scatter_item.sigClicked.connect(self.on_point_clicked)
        self.plot_widget_2d.addItem(scatter_item)
        self.scatter_items["Combined"] = (scatter_item, all_flight_segments, all_metadata_list, all_shot_data_list, all_rally_data_list, all_shot_object_list, all_rally_object_list)

        # Add theoretical drag coefficient lines
        show_any_model = self.model_v1_checkbox.isChecked() or self.model_smoothstep_checkbox.isChecked() or self.model_spline_checkbox.isChecked()
        if show_any_model:
            if xaxis_mode == "C_D vs Spin Ratio":
                # Lines for each 1 m/s velocity in the filtered range
                vel_min_round = np.ceil(vel_min_filter)
                vel_max_round = np.floor(vel_max_filter)

                # Create spin ratio array for plotting theoretical curves
                spin_ratio_theory = np.linspace(0, 2, 200)

                # Generate a colormap for velocity (same as data points)
                vel_cmap = plt.get_cmap('turbo')  # Use turbo (same as data) for theory lines

                _vel_step = 2.0 if self.publication_checkbox.isChecked() else 1.0
                for v_val in np.arange(max(0, vel_min_round), vel_max_round + 1, _vel_step):
                    # Color based on velocity (normalized to filtered range)
                    vel_norm = (v_val - vel_min_filter) / max(vel_max_filter - vel_min_filter, 1) if vel_max_filter > vel_min_filter else 0.5
                    rgba = vel_cmap(vel_norm)
                    base_color = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

                    # V1 Piecewise Linear (solid line)
                    if self.model_v1_checkbox.isChecked():
                        cd_theory_v1 = get_drag_estimate_v1(v_val, spin_ratio_theory)
                        color_v1 = (*base_color, 220)
                        # Black border (plot wider black line first)
                        pen_v1_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(spin_ratio_theory, cd_theory_v1, pen=pen_v1_border)
                        # Colored line on top
                        pen_v1 = pg.mkPen(color=color_v1, width=4, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(spin_ratio_theory, cd_theory_v1, pen=pen_v1)

                    # Smoothstep (dashed line)
                    if self.model_smoothstep_checkbox.isChecked():
                        cd_theory_smooth = get_drag_estimate_smoothstep(v_val, spin_ratio_theory)
                        color_smooth = (*base_color, 220)
                        # Black border
                        pen_smooth_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.DashLine)
                        self.plot_widget_2d.plot(spin_ratio_theory, cd_theory_smooth, pen=pen_smooth_border)
                        # Colored line on top
                        pen_smooth = pg.mkPen(color=color_smooth, width=4, style=Qt.DashLine)
                        self.plot_widget_2d.plot(spin_ratio_theory, cd_theory_smooth, pen=pen_smooth)

                    # Cubic Spline (dot-dash line)
                    if self.model_spline_checkbox.isChecked():
                        cd_theory_spline = get_drag_estimate_spline(v_val, spin_ratio_theory)
                        color_spline = (*base_color, 220)
                        # Black border
                        pen_spline_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.DashDotLine)
                        self.plot_widget_2d.plot(spin_ratio_theory, cd_theory_spline, pen=pen_spline_border)
                        # Colored line on top
                        pen_spline = pg.mkPen(color=color_spline, width=4, style=Qt.DashDotLine)
                        self.plot_widget_2d.plot(spin_ratio_theory, cd_theory_spline, pen=pen_spline)

            elif xaxis_mode == "C_D vs Velocity":
                # Lines for different spin ratios
                velocity_theory = np.linspace(1, 30, 200)

                # Spin ratio values for isolines
                spin_ratio_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

                # Generate a colormap for spin ratio (same as data points)
                sr_cmap = plt.get_cmap('turbo')

                for sr_val in spin_ratio_values:
                    # Color based on spin ratio (normalized to 0-2 range)
                    sr_norm = sr_val / 2.0
                    rgba = sr_cmap(sr_norm)
                    base_color = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

                    # V1 Piecewise Linear (solid line)
                    if self.model_v1_checkbox.isChecked():
                        cd_theory_v1 = get_drag_estimate_v1(velocity_theory, sr_val)
                        color_v1 = (*base_color, 220)
                        # Black border
                        pen_v1_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(velocity_theory, cd_theory_v1, pen=pen_v1_border)
                        # Colored line on top
                        pen_v1 = pg.mkPen(color=color_v1, width=4, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(velocity_theory, cd_theory_v1, pen=pen_v1)

                    # Smoothstep (dashed line)
                    if self.model_smoothstep_checkbox.isChecked():
                        cd_theory_smooth = get_drag_estimate_smoothstep(velocity_theory, sr_val)
                        color_smooth = (*base_color, 220)
                        # Black border
                        pen_smooth_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.DashLine)
                        self.plot_widget_2d.plot(velocity_theory, cd_theory_smooth, pen=pen_smooth_border)
                        # Colored line on top
                        pen_smooth = pg.mkPen(color=color_smooth, width=4, style=Qt.DashLine)
                        self.plot_widget_2d.plot(velocity_theory, cd_theory_smooth, pen=pen_smooth)

                    # Cubic Spline (dot-dash line)
                    if self.model_spline_checkbox.isChecked():
                        cd_theory_spline = get_drag_estimate_spline(velocity_theory, sr_val)
                        color_spline = (*base_color, 220)
                        # Black border
                        pen_spline_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.DashDotLine)
                        self.plot_widget_2d.plot(velocity_theory, cd_theory_spline, pen=pen_spline_border)
                        # Colored line on top
                        pen_spline = pg.mkPen(color=color_spline, width=4, style=Qt.DashDotLine)
                        self.plot_widget_2d.plot(velocity_theory, cd_theory_spline, pen=pen_spline)

            elif xaxis_mode == "Velocity vs Spin Ratio":
                # Add isolines of constant C_D values using skimage's find_contours
                # This properly handles non-monotonic C_D and separates disconnected contour segments
                from skimage import measure

                # Create a grid of (velocity, spin_ratio) values
                n_vel, n_sr = 150, 150
                vel_grid = np.linspace(1, 30, n_vel)
                sr_grid = np.linspace(0, 2, n_sr)

                # C_D isoline values to plot
                cd_values = [0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]

                # Generate a colormap for C_D values (same as data points)
                cd_cmap = plt.get_cmap('turbo')

                # For V1 Piecewise Linear model
                if self.model_v1_checkbox.isChecked():
                    # Compute C_D on the grid
                    VV, SR = np.meshgrid(vel_grid, sr_grid)
                    CD_grid = get_drag_estimate_v1(VV, SR)

                    for cd_target in cd_values:
                        # Normalize C_D to 0.3-0.8 range for coloring
                        cd_norm = (cd_target - 0.3) / 0.5
                        rgba = cd_cmap(min(max(cd_norm, 0), 1.0))
                        color_v1 = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 220)

                        # Find contours at the target C_D level
                        contours = measure.find_contours(CD_grid, cd_target)

                        for seg_idx, contour in enumerate(contours):
                            if len(contour) > 1:
                                # Convert from grid indices to actual values
                                sr_seg = np.interp(contour[:, 0], np.arange(n_sr), sr_grid)
                                vel_seg = np.interp(contour[:, 1], np.arange(n_vel), vel_grid)
                                name = f"C_D={cd_target:.2f} (V1)" if seg_idx == 0 else None
                                # Black border
                                pen_v1_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.SolidLine)
                                self.plot_widget_2d.plot(vel_seg, sr_seg, pen=pen_v1_border)
                                # Colored line on top
                                pen_v1 = pg.mkPen(color=color_v1, width=4, style=Qt.SolidLine)
                                self.plot_widget_2d.plot(vel_seg, sr_seg, pen=pen_v1, name=name)

                # For Smoothstep model
                if self.model_smoothstep_checkbox.isChecked():
                    # Compute C_D on the grid
                    VV, SR = np.meshgrid(vel_grid, sr_grid)
                    CD_grid = get_drag_estimate_smoothstep(VV, SR)

                    for cd_target in cd_values:
                        # Normalize C_D to 0.3-0.8 range for coloring
                        cd_norm = (cd_target - 0.3) / 0.5
                        rgba = cd_cmap(min(max(cd_norm, 0), 1.0))
                        color_smooth = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 220)

                        # Find contours at the target C_D level
                        contours = measure.find_contours(CD_grid, cd_target)

                        for seg_idx, contour in enumerate(contours):
                            if len(contour) > 1:
                                # Convert from grid indices to actual values
                                sr_seg = np.interp(contour[:, 0], np.arange(n_sr), sr_grid)
                                vel_seg = np.interp(contour[:, 1], np.arange(n_vel), vel_grid)
                                name = f"C_D={cd_target:.2f} (Smooth)" if seg_idx == 0 else None
                                # Black border
                                pen_smooth_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.DashLine)
                                self.plot_widget_2d.plot(vel_seg, sr_seg, pen=pen_smooth_border)
                                # Colored line on top
                                pen_smooth = pg.mkPen(color=color_smooth, width=4, style=Qt.DashLine)
                                self.plot_widget_2d.plot(vel_seg, sr_seg, pen=pen_smooth, name=name)

                # For Cubic Spline model
                if self.model_spline_checkbox.isChecked():
                    # Compute C_D on the grid
                    VV, SR = np.meshgrid(vel_grid, sr_grid)
                    CD_grid = get_drag_estimate_spline(VV, SR)

                    for cd_target in cd_values:
                        # Normalize C_D to 0.3-0.8 range for coloring
                        cd_norm = (cd_target - 0.3) / 0.5
                        rgba = cd_cmap(min(max(cd_norm, 0), 1.0))
                        color_spline = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 220)

                        # Find contours at the target C_D level
                        contours = measure.find_contours(CD_grid, cd_target)

                        for seg_idx, contour in enumerate(contours):
                            if len(contour) > 1:
                                # Convert from grid indices to actual values
                                sr_seg = np.interp(contour[:, 0], np.arange(n_sr), sr_grid)
                                vel_seg = np.interp(contour[:, 1], np.arange(n_vel), vel_grid)
                                name = f"C_D={cd_target:.2f} (Spline)" if seg_idx == 0 else None
                                # Black border
                                pen_spline_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.DashDotLine)
                                self.plot_widget_2d.plot(vel_seg, sr_seg, pen=pen_spline_border)
                                # Colored line on top
                                pen_spline = pg.mkPen(color=color_spline, width=4, style=Qt.DashDotLine)
                                self.plot_widget_2d.plot(vel_seg, sr_seg, pen=pen_spline, name=name)

        # Add colorbar/legend for the selected quantity using GraphicsLayoutWidget
        if is_player_coloring:
            # For player coloring, add a legend with player names and colors
            legend = pg.LegendItem(offset=(0, 0))
            for player in unique_players:
                idx = player_to_idx[player]
                norm_val = idx / max(n_players - 1, 1)
                rgba = cmap(norm_val)
                color = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                # Create a small scatter item for the legend
                scatter = pg.ScatterPlotItem([0], [0], pen=pg.mkPen(color=(50, 50, 50)), brush=pg.mkBrush(*color), size=10)
                legend.addItem(scatter, player)
            self.plot_widget_2d.addItem(legend)
        else:
            colormap = pg.colormap.get('turbo')
            self.colorbar_widget_drag.addItem(pg.ColorBarItem(
                values=(color_min, color_max),
                colorMap=colormap,
                label=color_label,
                limits=(color_min, color_max)
            ))

        # Update plot title and axis labels to reflect mode and color selection
        if xaxis_mode == "Velocity vs Spin Ratio":
            self.plot_widget_2d.setTitle(f"Velocity vs Spin Ratio (colored by {color_by})")
            self.plot_widget_2d.setLabel('bottom', f'{vel_label} (m/s)')
            self.plot_widget_2d.setLabel('left', 'Spin Ratio (ωr/v)')
        elif xaxis_mode == "Spin vs Velocity":
            self.plot_widget_2d.setTitle(f"Spin vs Velocity (colored by {color_by})")
            self.plot_widget_2d.setLabel('bottom', f'{vel_label} (m/s)')
            self.plot_widget_2d.setLabel('left', 'Spin Magnitude (rad/s)')
        elif xaxis_mode == "C_D vs Velocity":
            self.plot_widget_2d.setTitle(f"Drag Coefficient vs Velocity (colored by {color_by})")
            self.plot_widget_2d.setLabel('bottom', f'{vel_label} (m/s)')
            self.plot_widget_2d.setLabel('left', 'Drag Coefficient')
        else:
            self.plot_widget_2d.setTitle(f"Drag Coefficient vs Spin Ratio (colored by {color_by})")
            self.plot_widget_2d.setLabel('bottom', 'Spin Ratio (ωr/v)')
            self.plot_widget_2d.setLabel('left', 'Drag Coefficient')

        # Set axis ranges
        self.plot_widget_2d.setXRange(x_min, x_max, padding=0)
        self.plot_widget_2d.setYRange(y_min, y_max, padding=0)

        # Calculate actual data range for info
        actual_vel_min = np.min(selected_velocity)
        actual_vel_max = np.max(selected_velocity)
        self.info_label.setText(f"Showing {len(selected_velocity)} data points (vel: {vel_min_filter}-{vel_max_filter} m/s, conf: {conf_min:.2f}-{conf_max:.2f})")

    @staticmethod
    def _compute_durations(flight_segments):
        """Compute duration array for a list of flight segments."""
        return np.array([fs.data["time"].values[-1] - fs.data["time"].values[0]
                         for fs in flight_segments]) if len(flight_segments) > 0 else np.array([])

    @staticmethod
    def _compute_x_travel(flight_segments):
        """Compute x-travel distance array for a list of flight segments."""
        result = np.empty(len(flight_segments))
        for i, fs in enumerate(flight_segments):
            if "x_aps" in fs.data.columns:
                x = fs.data["x_aps"].values
                valid_x = x[~np.isnan(x)]
                if len(valid_x) >= 2:
                    result[i] = abs(valid_x[-1] - valid_x[0])
                    continue
            result[i] = np.nan
        return result

    def plot_data(self, robot_data_tuple, player_data_tuple):
        """
        Create plot from extracted data for both robot and player.

        Pre-merges robot + player data into numpy arrays once so that
        update_3d_plot() and update_2d_plot() only need to filter and render.
        """
        self.robot_data = robot_data_tuple
        self.player_data = player_data_tuple
        self._merged = None

        # ── Pre-merge robot + player data into numpy arrays (done ONCE) ──
        num_lists = {k: [] for k in [
            'vel_mag', 'v_eff_drag', 'spin_mag', 'spin_ratio', 'cd_opt',
            'drag_impulse_proxy', 'confidence', 'rmse_opt', 'inconsistent',
        ]}
        obj_lists = {k: [] for k in [
            'flight_segments', 'metadata_list', 'shot_data_list',
            'rally_data_list', 'shot_object_list', 'rally_object_list',
        ]}
        dur_parts, xtr_parts = [], []

        for data_tuple in (robot_data_tuple, player_data_tuple):
            if data_tuple is None:
                continue
            (vel_mag, v_eff_drag, spin_mag, spin_ratio, cd_opt,
             flight_segments, metadata_list, shot_data_list, rally_data_list,
             shot_object_list, rally_object_list, confidence_list,
             rmse_opt_list, inconsistent_list, drag_impulse_proxy_list) = data_tuple
            if len(vel_mag) == 0:
                continue
            num_lists['vel_mag'].append(np.asarray(vel_mag))
            num_lists['v_eff_drag'].append(np.asarray(v_eff_drag))
            num_lists['spin_mag'].append(np.asarray(spin_mag))
            num_lists['spin_ratio'].append(np.asarray(spin_ratio))
            num_lists['cd_opt'].append(np.asarray(cd_opt))
            num_lists['drag_impulse_proxy'].append(np.asarray(drag_impulse_proxy_list))
            num_lists['confidence'].append(np.asarray(confidence_list))
            num_lists['rmse_opt'].append(np.asarray(rmse_opt_list))
            num_lists['inconsistent'].append(np.asarray(inconsistent_list))
            obj_lists['flight_segments'].extend(flight_segments)
            obj_lists['metadata_list'].extend(metadata_list)
            obj_lists['shot_data_list'].extend(shot_data_list)
            obj_lists['rally_data_list'].extend(rally_data_list)
            obj_lists['shot_object_list'].extend(shot_object_list)
            obj_lists['rally_object_list'].extend(rally_object_list)
            dur_parts.append(self._compute_durations(flight_segments))
            xtr_parts.append(self._compute_x_travel(flight_segments))

        if not num_lists['vel_mag']:
            self.update_plot()
            return

        m = {}
        for k, arrs in num_lists.items():
            m[k] = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
        m['durations'] = np.concatenate(dur_parts) if dur_parts else np.array([])
        m['x_travel'] = np.concatenate(xtr_parts) if xtr_parts else np.array([])
        # Object arrays support numpy boolean indexing (no list-comp filtering needed)
        for k, lst in obj_lists.items():
            m[k] = np.array(lst, dtype=object)
        # Pre-compute spin_ratio variant for v_eff_drag velocity selection
        m['spin_ratio_veff'] = (self.radius_ball * m['spin_mag']) / np.maximum(m['v_eff_drag'], 1e-9)

        self._merged = m
        self.update_plot()

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
                        shot_data = shot_data_list[idx] if idx < len(shot_data_list) else None
                        rally_data = rally_data_list[idx] if idx < len(rally_data_list) else None
                        shot_object = shot_object_list[idx] if idx < len(shot_object_list) else None
                        rally_object = rally_object_list[idx] if idx < len(rally_object_list) else None
                        self.flight_segment_selected.emit(fs, metadata, shot_data, rally_data, shot_object, rally_object)
                    break

    def save_plot(self):
        """Save the current plot as a PNG image"""
        if self.robot_data is None and self.player_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drag_coefficient_{timestamp}.png"

        output_path = plots_dir / filename

        try:
            # Determine which plot to export based on current view
            view_type = self.view_combo.currentText()
            if view_type == "3D" and self.has_3d:
                # For 3D view, we need to grab the frame buffer
                from PySide6.QtGui import QImage
                img = self.plot_widget_3d.grabFramebuffer()
                img.save(str(output_path))
            else:
                # Export the 2D plot widget
                publication = self.publication_checkbox.isChecked()
                save_plot_as_image(self.plot_widget_2d, str(output_path), publication_mode=publication)

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
