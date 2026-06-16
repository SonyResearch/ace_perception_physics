# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Magnus Coefficient Plot Widget

Widget for visualizing magnus coefficients vs velocity and spin magnitude.
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
from .aero_estimates import get_magnus_estimate_v2
from .base_coefficient_plot import create_range_slider, create_combo_selector, save_plot_as_image


class MagnusCoefficientPlot(QWidget):
    """Widget for 3D plot of magnus coefficient vs velocity and spin magnitude"""

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

        velocity_layout, self.velocity_combo = create_combo_selector("Velocity:", ["v", "v_eff_magnus"], self.update_plot, default_index=1)
        control_bar.addLayout(velocity_layout)

        spin_layout, self.spin_combo = create_combo_selector("Spin:", ["ω", "ω_eff_perp"], self.update_plot, default_index=0)
        control_bar.addLayout(spin_layout)

        color_layout, self.color_combo = create_combo_selector("Color By:", ["Spin Magnitude", "Spin Ratio", "Velocity", "Magnus Coefficient", "Magnus force sensitivity", "Confidence", "Duration", "X Travel", "Player"], self.update_plot)
        control_bar.addLayout(color_layout)

        xaxis_layout, self.xaxis_combo = create_combo_selector("X-Axis:", ["C_M vs Velocity", "C_M vs Spin", "Velocity vs Spin", "C_L vs Spin Ratio"], self.update_plot)
        control_bar.addLayout(xaxis_layout)

        # Add range sliders using helper function
        vel_layout, self.vel_min_slider, self.vel_max_slider, self.vel_min_label, self.vel_max_label = \
            create_range_slider("Velocity Range (m/s):", 0, 30, 0, 30, 5, self.on_velocity_range_changed, "{}", 1.0, 25)
        control_bar.addLayout(vel_layout)

        spin_range_layout, self.spin_min_slider, self.spin_max_slider, self.spin_min_label, self.spin_max_label = \
            create_range_slider("Spin Range (rad/s):", 0, 1000, 0, 1000, 100, self.on_spin_range_changed, "{}", 1.0, 35)
        control_bar.addLayout(spin_range_layout)

        self._conf_layout, self.conf_min_slider, self.conf_max_slider, self.conf_min_label, self.conf_max_label = \
            create_range_slider("Confidence Range:", 0, 100, 50, 100, 10, self.on_confidence_range_changed, "{:.2f}", 100.0, 35)
        control_bar.addLayout(self._conf_layout)

        dur_layout, self.dur_min_slider, self.dur_max_slider, self.dur_min_label, self.dur_max_label = \
            create_range_slider("Duration Range (s):", 0, 200, 30, 200, 20, self.on_duration_range_changed, "{:.3f}", 200.0, 35)
        control_bar.addLayout(dur_layout)

        xtravel_layout, self.xtravel_min_slider, self.xtravel_max_slider, self.xtravel_min_label, self.xtravel_max_label = \
            create_range_slider("X Travel (m):", 0, 50, 0, 50, 10, self.on_xtravel_range_changed, "{:.1f}", 10.0, 35)
        control_bar.addLayout(xtravel_layout)

        # Error filter (RMSE vs OPT in mm)
        self._error_layout, self.error_min_slider, self.error_max_slider, self.error_min_label, self.error_max_label = \
            create_range_slider("Error vs OPT (mm):", 0, 100, 0, 100, 10, self.on_error_range_changed, "{:.0f}", 1.0, 35)
        control_bar.addLayout(self._error_layout)

        # Magnus force sensitivity filter
        impulse_layout, self.impulse_min_slider, self.impulse_max_slider, self.impulse_min_label, self.impulse_max_label = \
            create_range_slider("Magnus sensitivity:", 0, 5000, 0, 5000, 500, self.on_impulse_range_changed, "{:.0f}", 1.0, 35)
        control_bar.addLayout(impulse_layout)

        # Add checkbox to hide samples with inconsistent confidence
        self.hide_inconsistent_checkbox = QCheckBox("Hide Inconsistent Conf.")
        self.hide_inconsistent_checkbox.setChecked(True)
        self.hide_inconsistent_checkbox.setToolTip("Hide samples where confidence varies within the flight segment")
        self.hide_inconsistent_checkbox.stateChanged.connect(self.update_plot)
        control_bar.addWidget(self.hide_inconsistent_checkbox)

        # Add checkboxes to show/hide theoretical model lines
        models_layout = QVBoxLayout()
        models_label = QLabel("Models:")
        models_layout.addWidget(models_label)

        self.show_v2_checkbox = QCheckBox("Model")
        self.show_v2_checkbox.setChecked(True)
        self.show_v2_checkbox.stateChanged.connect(self.update_plot)
        models_layout.addWidget(self.show_v2_checkbox)

        control_bar.addLayout(models_layout)

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
            self.plot_widget_3d.setWindowTitle('Magnus Coefficient (cm_opt) vs Velocity and Spin Magnitude')

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

        self.plot_widget_2d = pg.PlotWidget(title="Magnus Coefficient vs Velocity (colored by Spin)")
        self.plot_widget_2d.setLabel("left", "Magnus Coefficient")
        self.plot_widget_2d.setLabel("bottom", "Velocity Magnitude (m/s)")
        self.plot_widget_2d.showGrid(x=True, y=True)
        # Disable SI prefix for full value display
        self.plot_widget_2d.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget_2d.getAxis('bottom').enableAutoSIPrefix(False)

        plot_2d_container.addWidget(self.plot_widget_2d)

        # Create colorbar widget using GraphicsLayoutWidget
        self.colorbar_widget_magnus = pg.GraphicsLayoutWidget()
        self.colorbar_widget_magnus.setMaximumWidth(100)
        self.colorbar_widget_magnus.setMinimumWidth(80)
        plot_2d_container.addWidget(self.colorbar_widget_magnus)

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
        self.colorbar_magnus = None

        self.setLayout(layout)

        # Store data references
        self.robot_data = None
        self.player_data = None
        self.scatter_items = {}

        # Cached pre-computed arrays (computed once in plot_data, reused on each slider update)
        self._robot_durations = None
        self._robot_x_travel = None
        self._player_durations = None
        self._player_x_travel = None

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

    def on_spin_range_changed(self):
        """Handle spin range slider changes"""
        # Get slider values and round to nearest 10
        spin_min = round(self.spin_min_slider.value() / 10) * 10
        spin_max = round(self.spin_max_slider.value() / 10) * 10

        # Update slider positions to snapped values (without triggering another signal)
        self.spin_min_slider.blockSignals(True)
        self.spin_max_slider.blockSignals(True)
        self.spin_min_slider.setValue(spin_min)
        self.spin_max_slider.setValue(spin_max)
        self.spin_min_slider.blockSignals(False)
        self.spin_max_slider.blockSignals(False)

        # Ensure min <= max
        if spin_min > spin_max:
            if self.sender() == self.spin_min_slider:
                self.spin_min_slider.setValue(spin_max)
                spin_min = spin_max
            else:
                self.spin_max_slider.setValue(spin_min)
                spin_max = spin_min

        self.spin_min_label.setText(str(spin_min))
        self.spin_max_label.setText(str(spin_max))

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

    def on_impulse_range_changed(self):
        """Handle Magnus force sensitivity range slider changes"""
        # Get slider values
        impulse_min = self.impulse_min_slider.value()
        impulse_max = self.impulse_max_slider.value()

        # Ensure min <= max
        if impulse_min > impulse_max:
            if self.sender() == self.impulse_min_slider:
                self.impulse_min_slider.setValue(impulse_max)
                impulse_min = impulse_max
            else:
                self.impulse_max_slider.setValue(impulse_min)
                impulse_max = impulse_min

        # Update labels
        self.impulse_min_label.setText(f"{impulse_min:.0f}")
        self.impulse_max_label.setText(f"{impulse_max:.0f}")

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

        # Combine robot and player data (always show both)
        all_vel_mag = []
        all_v_eff_magnus = []
        all_spin_mag = []
        all_spin_ratio = []
        all_w_eff_perp_magnus = []
        all_cm_opt = []
        all_flight_segments = []
        all_metadata_list = []
        all_shot_data_list = []
        all_rally_data_list = []
        all_shot_object_list = []
        all_rally_object_list = []
        all_confidence_list = []
        all_rmse_opt_list = []
        all_inconsistent_list = []
        all_magnus_impulse_proxy_list = []

        if self.robot_data is not None:
            vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp_magnus, cm_opt, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list, rmse_opt_list, inconsistent_list, magnus_impulse_proxy_list = self.robot_data
            if len(vel_mag) > 0:
                all_vel_mag.extend(vel_mag)
                all_v_eff_magnus.extend(v_eff_magnus)
                all_spin_mag.extend(spin_mag)
                all_spin_ratio.extend(spin_ratio)
                all_w_eff_perp_magnus.extend(w_eff_perp_magnus)
                all_cm_opt.extend(cm_opt)
                all_flight_segments.extend(flight_segments)
                all_metadata_list.extend(metadata_list)
                all_shot_data_list.extend(shot_data_list)
                all_rally_data_list.extend(rally_data_list)
                all_shot_object_list.extend(shot_object_list)
                all_rally_object_list.extend(rally_object_list)
                all_confidence_list.extend(confidence_list)
                all_rmse_opt_list.extend(rmse_opt_list)
                all_inconsistent_list.extend(inconsistent_list)
                all_magnus_impulse_proxy_list.extend(magnus_impulse_proxy_list)

        if self.player_data is not None:
            vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp_magnus, cm_opt, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list, rmse_opt_list, inconsistent_list, magnus_impulse_proxy_list = self.player_data
            if len(vel_mag) > 0:
                all_vel_mag.extend(vel_mag)
                all_v_eff_magnus.extend(v_eff_magnus)
                all_spin_mag.extend(spin_mag)
                all_spin_ratio.extend(spin_ratio)
                all_w_eff_perp_magnus.extend(w_eff_perp_magnus)
                all_cm_opt.extend(cm_opt)
                all_flight_segments.extend(flight_segments)
                all_metadata_list.extend(metadata_list)
                all_shot_data_list.extend(shot_data_list)
                all_rally_data_list.extend(rally_data_list)
                all_shot_object_list.extend(shot_object_list)
                all_rally_object_list.extend(rally_object_list)
                all_confidence_list.extend(confidence_list)
                all_rmse_opt_list.extend(rmse_opt_list)
                all_inconsistent_list.extend(inconsistent_list)
                all_magnus_impulse_proxy_list.extend(magnus_impulse_proxy_list)

        if len(all_vel_mag) == 0:
            self.info_label.setText("No data points to display")
            return

        # Select velocity based on user choice
        velocity_type = self.velocity_combo.currentText()
        if velocity_type == "v_eff_magnus":
            selected_velocity = np.array(all_v_eff_magnus)
            vel_label = "v_eff_magnus"
        else:  # "v"
            selected_velocity = np.array(all_vel_mag)
            vel_label = "v"

        # Select spin based on user choice
        spin_type = self.spin_combo.currentText()
        if spin_type == "ω_eff_perp":
            selected_spin = np.array(all_w_eff_perp_magnus)
            spin_label = "ω_eff_perp"
        else:  # "ω"
            selected_spin = np.array(all_spin_mag)
            spin_label = "ω"

        # Convert to numpy arrays
        all_vel_mag = np.array(all_vel_mag)
        all_v_eff_magnus = np.array(all_v_eff_magnus)
        all_spin_mag = np.array(all_spin_mag)
        all_w_eff_perp_magnus = np.array(all_w_eff_perp_magnus)
        all_spin_ratio = np.array(all_spin_ratio)
        all_cm_opt = np.array(all_cm_opt)
        all_magnus_impulse_proxy = np.array(all_magnus_impulse_proxy_list)

        # Use pre-computed duration and x_travel arrays (computed once in plot_data)
        dur_parts = []
        xtr_parts = []
        if self.robot_data is not None and self._robot_durations is not None and len(self.robot_data[0]) > 0:
            dur_parts.append(self._robot_durations)
            xtr_parts.append(self._robot_x_travel)
        if self.player_data is not None and self._player_durations is not None and len(self.player_data[0]) > 0:
            dur_parts.append(self._player_durations)
            xtr_parts.append(self._player_x_travel)
        all_durations = np.concatenate(dur_parts) if dur_parts else np.array([])
        all_x_travel = np.concatenate(xtr_parts) if xtr_parts else np.array([])

        # Apply spin magnitude range filter
        spin_min_filter = self.spin_min_slider.value()
        spin_max_filter = self.spin_max_slider.value()
        # When max is 1000, show all data >= 1000
        if spin_max_filter >= 1000:
            spin_mask = (all_spin_mag >= spin_min_filter)
        else:
            spin_mask = (all_spin_mag >= spin_min_filter) & (all_spin_mag <= spin_max_filter)

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
        conf_array = np.array(all_confidence_list)
        conf_mask = (conf_array >= conf_min) & (conf_array <= conf_max)

        # Apply duration range filter
        dur_min = self.dur_min_slider.value() / 200.0
        dur_max = self.dur_max_slider.value() / 200.0
        # When max is 1.0, show all data >= 1.0
        if dur_max >= 1.0:
            dur_mask = (all_durations >= dur_min)
        else:
            dur_mask = (all_durations >= dur_min) & (all_durations <= dur_max)

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
        all_rmse_opt = np.array(all_rmse_opt_list)
        # When max is at slider maximum, show all data >= min; also allow NaN (missing error data)
        if self.error_max_slider.value() >= self.error_max_slider.maximum():
            error_mask = (all_rmse_opt >= error_min) | np.isnan(all_rmse_opt)
        else:
            error_mask = ((all_rmse_opt >= error_min) & (all_rmse_opt <= error_max)) | np.isnan(all_rmse_opt)

        # Apply inconsistent confidence filter
        all_inconsistent = np.array(all_inconsistent_list)
        if self.hide_inconsistent_checkbox.isChecked():
            consistent_mask = ~all_inconsistent
        else:
            consistent_mask = np.ones(len(all_inconsistent), dtype=bool)

        # Apply Magnus force sensitivity filter (uses cached discrete time integral)
        impulse_min = self.impulse_min_slider.value()
        impulse_max = self.impulse_max_slider.value()
        # When max is 5000, show all data >= 5000
        if impulse_max >= 5000:
            impulse_mask = (all_magnus_impulse_proxy >= impulse_min) | np.isnan(all_magnus_impulse_proxy)
        else:
            impulse_mask = ((all_magnus_impulse_proxy >= impulse_min) & (all_magnus_impulse_proxy <= impulse_max)) | np.isnan(all_magnus_impulse_proxy)

        # Combine masks
        combined_mask = spin_mask & vel_mask & conf_mask & dur_mask & xtravel_mask & error_mask & consistent_mask & impulse_mask

        # Filter all arrays
        selected_velocity = selected_velocity[combined_mask]
        all_vel_mag = all_vel_mag[combined_mask]
        all_v_eff_magnus = all_v_eff_magnus[combined_mask]
        all_spin_mag = all_spin_mag[combined_mask]
        all_spin_ratio = all_spin_ratio[combined_mask]
        all_cm_opt = all_cm_opt[combined_mask]
        all_durations = all_durations[combined_mask]
        all_magnus_impulse_proxy = all_magnus_impulse_proxy[combined_mask]

        # Filter lists using the boolean mask
        all_flight_segments = [fs for fs, mask in zip(all_flight_segments, combined_mask) if mask]
        all_metadata_list = [m for m, mask in zip(all_metadata_list, combined_mask) if mask]
        all_shot_data_list = [s for s, mask in zip(all_shot_data_list, combined_mask) if mask]
        all_rally_data_list = [r for r, mask in zip(all_rally_data_list, combined_mask) if mask]
        all_shot_object_list = [s for s, mask in zip(all_shot_object_list, combined_mask) if mask]
        all_rally_object_list = [r for r, mask in zip(all_rally_object_list, combined_mask) if mask]
        all_confidence_list = [c for c, mask in zip(all_confidence_list, combined_mask) if mask]

        if len(selected_velocity) == 0:
            self.info_label.setText(f"No data points in spin range [{spin_min_filter}, {spin_max_filter}] rad/s and confidence range [{conf_min:.2f}, {conf_max:.2f}]")
            return

        # Check for valid (non-NaN) data in 3D plot coordinates and filter out NaN entries
        valid_mask = ~(np.isnan(selected_velocity) | np.isnan(selected_spin) | np.isnan(all_spin_mag) | np.isnan(all_cm_opt))
        if not np.any(valid_mask):
            self.info_label.setText("No valid data points after filtering (all NaN values)")
            return

        # Apply valid mask to filter out NaN entries
        selected_velocity = selected_velocity[valid_mask]
        selected_spin = selected_spin[valid_mask]
        all_vel_mag = all_vel_mag[valid_mask]
        all_v_eff_magnus = all_v_eff_magnus[valid_mask]
        all_spin_mag = all_spin_mag[valid_mask]
        all_spin_ratio = all_spin_ratio[valid_mask]
        all_cm_opt = all_cm_opt[valid_mask]
        all_durations = all_durations[valid_mask]
        all_magnus_impulse_proxy = all_magnus_impulse_proxy[valid_mask]

        # Filter lists using the valid mask
        all_flight_segments = [fs for fs, mask in zip(all_flight_segments, valid_mask) if mask]
        all_metadata_list = [m for m, mask in zip(all_metadata_list, valid_mask) if mask]
        all_shot_data_list = [s for s, mask in zip(all_shot_data_list, valid_mask) if mask]
        all_rally_data_list = [r for r, mask in zip(all_rally_data_list, valid_mask) if mask]
        all_shot_object_list = [s for s, mask in zip(all_shot_object_list, valid_mask) if mask]
        all_rally_object_list = [r for r, mask in zip(all_rally_object_list, valid_mask) if mask]
        all_confidence_list = [c for c, mask in zip(all_confidence_list, valid_mask) if mask]

        if len(selected_velocity) == 0:
            self.info_label.setText("No valid data points after NaN filtering")
            return

        # Normalization function to map data ranges to 0-25 scale
        def normalize_to_range(data, min_val, max_val, target_range=25.0):
            """Normalize data from [min_val, max_val] to [0, target_range]"""
            return (data - min_val) / (max_val - min_val) * target_range

        # Normalize each dimension to 0-25 range (use selected_spin for Y axis)
        vel_norm = normalize_to_range(selected_velocity, 0, 30)
        spin_norm = normalize_to_range(selected_spin, 0, 1000)
        cm_norm = normalize_to_range(all_cm_opt, 0.0, 0.5)

        pos = np.column_stack([vel_norm, spin_norm, cm_norm])

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
            if color_by == "Spin Magnitude":
                color_data = all_spin_mag
                color_min = self.spin_min_slider.value()
                color_max = self.spin_max_slider.value()
            elif color_by == "Spin Ratio":
                color_data = all_spin_ratio
                color_min = self.spin_ratio_min_slider.value() / 10.0 if hasattr(self, 'spin_ratio_min_slider') else 0.0
                color_max = self.spin_ratio_max_slider.value() / 10.0 if hasattr(self, 'spin_ratio_max_slider') else 2.0
            elif color_by == "Velocity":
                color_data = selected_velocity
                color_min = float(vel_min_filter)
                color_max = float(vel_max_filter)
            elif color_by == "Magnus Coefficient":
                color_data = all_cm_opt
                color_min = 0.0
                color_max = 0.5
            elif color_by == "Magnus force sensitivity":
                # Magnus force sensitivity: discrete time integral from cache
                # Use filtered data range for color scaling
                color_data = all_magnus_impulse_proxy
                valid_data = color_data[~np.isnan(color_data)]
                color_min = np.min(valid_data) if len(valid_data) > 0 else 0.0
                color_max = np.max(valid_data) if len(valid_data) > 0 else 1.0
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

        self.info_label.setText(f"Showing {len(selected_velocity)} data points (spin: {spin_min_filter}-{spin_max_filter} rad/s, conf: {conf_min:.2f}-{conf_max:.2f}). 3D normalized.")
        # Center camera at middle of normalized range (12.5, 12.5, 12.5)
        self.plot_widget_3d.opts['center'] = pg.Vector(12.5, 12.5, 12.5)
        self.plot_widget_3d.opts['distance'] = 40

    def update_2d_plot(self):
        """Update 2D scatter plot: Magnus Coefficient vs Velocity or vs Spin"""
        import matplotlib.pyplot as plt

        self.plot_widget_2d.clear()
        self.scatter_items = {}

        # Clear existing colorbar if present
        self.colorbar_widget_magnus.clear()
        self.colorbar_magnus = None

        # Get X-axis mode
        xaxis_mode = self.xaxis_combo.currentText()  # "C_M vs Velocity", "C_M vs Spin", "C_M vs ω_eff_perp", or "Velocity vs Spin"

        # Combine robot and player data (always show both)
        all_vel_mag = []
        all_v_eff_magnus = []
        all_spin_mag = []
        all_spin_ratio = []
        all_w_eff_perp_magnus = []
        all_cm_opt = []
        all_flight_segments = []
        all_metadata_list = []
        all_shot_data_list = []
        all_rally_data_list = []
        all_shot_object_list = []
        all_rally_object_list = []
        all_confidence_list = []
        all_rmse_opt_list = []
        all_inconsistent_list = []
        all_magnus_impulse_proxy_list = []

        if self.robot_data is not None:
            vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp_magnus, cm_opt, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list, rmse_opt_list, inconsistent_list, magnus_impulse_proxy_list = self.robot_data
            if len(vel_mag) > 0:
                all_vel_mag.extend(vel_mag)
                all_v_eff_magnus.extend(v_eff_magnus)
                all_spin_mag.extend(spin_mag)
                all_spin_ratio.extend(spin_ratio)
                all_w_eff_perp_magnus.extend(w_eff_perp_magnus)
                all_cm_opt.extend(cm_opt)
                all_flight_segments.extend(flight_segments)
                all_metadata_list.extend(metadata_list)
                all_shot_data_list.extend(shot_data_list)
                all_rally_data_list.extend(rally_data_list)
                all_shot_object_list.extend(shot_object_list)
                all_rally_object_list.extend(rally_object_list)
                all_confidence_list.extend(confidence_list)
                all_rmse_opt_list.extend(rmse_opt_list)
                all_inconsistent_list.extend(inconsistent_list)
                all_magnus_impulse_proxy_list.extend(magnus_impulse_proxy_list)

        if self.player_data is not None:
            vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp_magnus, cm_opt, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list, rmse_opt_list, inconsistent_list, magnus_impulse_proxy_list = self.player_data
            if len(vel_mag) > 0:
                all_vel_mag.extend(vel_mag)
                all_v_eff_magnus.extend(v_eff_magnus)
                all_spin_mag.extend(spin_mag)
                all_spin_ratio.extend(spin_ratio)
                all_w_eff_perp_magnus.extend(w_eff_perp_magnus)
                all_cm_opt.extend(cm_opt)
                all_flight_segments.extend(flight_segments)
                all_metadata_list.extend(metadata_list)
                all_shot_data_list.extend(shot_data_list)
                all_rally_data_list.extend(rally_data_list)
                all_shot_object_list.extend(shot_object_list)
                all_rally_object_list.extend(rally_object_list)
                all_confidence_list.extend(confidence_list)
                all_rmse_opt_list.extend(rmse_opt_list)
                all_inconsistent_list.extend(inconsistent_list)
                all_magnus_impulse_proxy_list.extend(magnus_impulse_proxy_list)

        if len(all_vel_mag) == 0:
            self.info_label.setText("No data points to display")
            return

        # Select velocity based on user choice
        velocity_type = self.velocity_combo.currentText()
        if velocity_type == "v_eff_magnus":
            selected_velocity = np.array(all_v_eff_magnus)
            vel_label = "v_eff_magnus"
        else:  # "v"
            selected_velocity = np.array(all_vel_mag)
            vel_label = "v"

        # Select spin based on user choice
        spin_type = self.spin_combo.currentText()
        if spin_type == "ω_eff_perp":
            selected_spin = np.array(all_w_eff_perp_magnus)
            spin_label = "ω_eff_perp"
        else:  # "ω"
            selected_spin = np.array(all_spin_mag)
            spin_label = "ω"

        # Convert to numpy arrays
        all_vel_mag = np.array(all_vel_mag)
        all_v_eff_magnus = np.array(all_v_eff_magnus)
        all_spin_mag = np.array(all_spin_mag)
        all_spin_ratio = np.array(all_spin_ratio)
        all_w_eff_perp_magnus = np.array(all_w_eff_perp_magnus)
        all_cm_opt = np.array(all_cm_opt)
        all_magnus_impulse_proxy = np.array(all_magnus_impulse_proxy_list)

        # Use pre-computed duration and x_travel arrays (computed once in plot_data)
        dur_parts = []
        xtr_parts = []
        if self.robot_data is not None and self._robot_durations is not None and len(self.robot_data[0]) > 0:
            dur_parts.append(self._robot_durations)
            xtr_parts.append(self._robot_x_travel)
        if self.player_data is not None and self._player_durations is not None and len(self.player_data[0]) > 0:
            dur_parts.append(self._player_durations)
            xtr_parts.append(self._player_x_travel)
        all_durations = np.concatenate(dur_parts) if dur_parts else np.array([])
        all_x_travel = np.concatenate(xtr_parts) if xtr_parts else np.array([])

        # Apply spin magnitude range filter
        spin_min_filter = self.spin_min_slider.value()
        spin_max_filter = self.spin_max_slider.value()
        # When max is 1000, show all data >= 1000
        if spin_max_filter >= 1000:
            spin_mask = (all_spin_mag >= spin_min_filter)
        else:
            spin_mask = (all_spin_mag >= spin_min_filter) & (all_spin_mag <= spin_max_filter)

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
        conf_array = np.array(all_confidence_list)
        conf_mask = (conf_array >= conf_min) & (conf_array <= conf_max)

        # Apply duration range filter
        dur_min = self.dur_min_slider.value() / 200.0
        dur_max = self.dur_max_slider.value() / 200.0
        # When max is 1.0, show all data >= 1.0
        if dur_max >= 1.0:
            dur_mask = (all_durations >= dur_min)
        else:
            dur_mask = (all_durations >= dur_min) & (all_durations <= dur_max)

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
        all_rmse_opt = np.array(all_rmse_opt_list)
        # When max is 500mm, show all data >= 500mm; also allow NaN (missing error data)
        if error_max >= 0.5:
            error_mask = (all_rmse_opt >= error_min) | np.isnan(all_rmse_opt)
        else:
            error_mask = ((all_rmse_opt >= error_min) & (all_rmse_opt <= error_max)) | np.isnan(all_rmse_opt)

        # Apply inconsistent confidence filter
        all_inconsistent = np.array(all_inconsistent_list)
        if self.hide_inconsistent_checkbox.isChecked():
            consistent_mask = ~all_inconsistent
        else:
            consistent_mask = np.ones(len(all_inconsistent), dtype=bool)

        # Apply Magnus force sensitivity filter (uses cached discrete time integral)
        impulse_min = self.impulse_min_slider.value()
        impulse_max = self.impulse_max_slider.value()
        # When max is 5000, show all data >= 5000
        if impulse_max >= 5000:
            impulse_mask = (all_magnus_impulse_proxy >= impulse_min) | np.isnan(all_magnus_impulse_proxy)
        else:
            impulse_mask = ((all_magnus_impulse_proxy >= impulse_min) & (all_magnus_impulse_proxy <= impulse_max)) | np.isnan(all_magnus_impulse_proxy)

        # Combine masks
        combined_mask = spin_mask & vel_mask & conf_mask & dur_mask & xtravel_mask & error_mask & consistent_mask & impulse_mask

        # Filter all arrays
        selected_velocity = selected_velocity[combined_mask]
        selected_spin = selected_spin[combined_mask]
        all_vel_mag = all_vel_mag[combined_mask]
        all_v_eff_magnus = all_v_eff_magnus[combined_mask]
        all_spin_mag = all_spin_mag[combined_mask]
        all_w_eff_perp_magnus = all_w_eff_perp_magnus[combined_mask]
        all_spin_ratio = all_spin_ratio[combined_mask]
        all_cm_opt = all_cm_opt[combined_mask]
        all_durations = all_durations[combined_mask]
        all_x_travel = all_x_travel[combined_mask]
        all_magnus_impulse_proxy = all_magnus_impulse_proxy[combined_mask]

        # Filter lists using the boolean mask
        all_flight_segments = [fs for fs, mask in zip(all_flight_segments, combined_mask) if mask]
        all_metadata_list = [m for m, mask in zip(all_metadata_list, combined_mask) if mask]
        all_shot_data_list = [s for s, mask in zip(all_shot_data_list, combined_mask) if mask]
        all_rally_data_list = [r for r, mask in zip(all_rally_data_list, combined_mask) if mask]
        all_shot_object_list = [s for s, mask in zip(all_shot_object_list, combined_mask) if mask]
        all_rally_object_list = [r for r, mask in zip(all_rally_object_list, combined_mask) if mask]
        all_confidence_list = [c for c, mask in zip(all_confidence_list, combined_mask) if mask]

        if len(selected_velocity) == 0:
            self.info_label.setText(f"No data points in spin range [{spin_min_filter}, {spin_max_filter}] rad/s, vel range [{vel_min_filter}, {vel_max_filter}] m/s, and conf range [{conf_min:.2f}, {conf_max:.2f}]")
            return

        # Check for valid (non-NaN) data and filter out NaN entries
        valid_mask = ~(np.isnan(all_spin_mag) | np.isnan(selected_spin) | np.isnan(all_cm_opt) | np.isnan(selected_velocity))
        if not np.any(valid_mask):
            self.info_label.setText("No valid data points after filtering (all NaN values)")
            return

        # Apply valid mask to filter out NaN entries
        selected_velocity = selected_velocity[valid_mask]
        selected_spin = selected_spin[valid_mask]
        all_vel_mag = all_vel_mag[valid_mask]
        all_v_eff_magnus = all_v_eff_magnus[valid_mask]
        all_spin_mag = all_spin_mag[valid_mask]
        all_w_eff_perp_magnus = all_w_eff_perp_magnus[valid_mask]
        all_spin_ratio = all_spin_ratio[valid_mask]
        all_cm_opt = all_cm_opt[valid_mask]
        all_durations = all_durations[valid_mask]
        all_x_travel = all_x_travel[valid_mask]
        all_magnus_impulse_proxy = all_magnus_impulse_proxy[valid_mask]

        # Filter lists using the valid mask
        all_flight_segments = [fs for fs, mask in zip(all_flight_segments, valid_mask) if mask]
        all_metadata_list = [m for m, mask in zip(all_metadata_list, valid_mask) if mask]
        all_shot_data_list = [s for s, mask in zip(all_shot_data_list, valid_mask) if mask]
        all_rally_data_list = [r for r, mask in zip(all_rally_data_list, valid_mask) if mask]
        all_shot_object_list = [s for s, mask in zip(all_shot_object_list, valid_mask) if mask]
        all_rally_object_list = [r for r, mask in zip(all_rally_object_list, valid_mask) if mask]
        all_confidence_list = [c for c, mask in zip(all_confidence_list, valid_mask) if mask]

        if len(selected_velocity) == 0:
            self.info_label.setText("No valid data points after NaN filtering")
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
            if color_by == "Spin Magnitude":
                color_data = all_spin_mag
                color_label = "Spin Magnitude (rad/s)"
                color_min = float(self.spin_min_slider.value())
                color_max = float(self.spin_max_slider.value())
            elif color_by == "Spin Ratio":
                color_data = all_spin_ratio
                color_label = "Spin Ratio"
                color_min = self.spin_ratio_min_slider.value() / 10.0 if hasattr(self, 'spin_ratio_min_slider') else 0.0
                color_max = self.spin_ratio_max_slider.value() / 10.0 if hasattr(self, 'spin_ratio_max_slider') else 2.0
            elif color_by == "Velocity":
                color_data = selected_velocity
                color_label = f"{vel_label} (m/s)"
                color_min = float(vel_min_filter)
                color_max = float(vel_max_filter)
            elif color_by == "Magnus Coefficient":
                color_data = all_cm_opt
                color_label = "Magnus Coefficient"
                color_min = 0.0
                color_max = 0.5
            elif color_by == "Magnus force sensitivity":
                # Magnus force sensitivity: discrete time integral from cache
                # Use filtered data range for color scaling
                color_data = all_magnus_impulse_proxy
                color_label = "Σ||v×ω||(t_end-t)dt"
                valid_data = color_data[~np.isnan(color_data)]
                color_min = np.min(valid_data) if len(valid_data) > 0 else 0.0
                color_max = np.max(valid_data) if len(valid_data) > 0 else 1.0
            elif color_by == "Duration":
                color_data = all_durations
                color_label = "Duration"
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

        # Create spots data for clickable scatter plot
        # X-axis depends on mode
        if xaxis_mode == "C_M vs Spin":
            x_data = selected_spin
            y_data = all_cm_opt
            x_label = f"{spin_label} (rad/s)"
            y_label = "Magnus Coefficient"
            x_min, x_max = 0, 1000
            y_min, y_max = 0, 0.5
        elif xaxis_mode == "Velocity vs Spin":
            x_data = selected_velocity
            y_data = selected_spin
            x_label = f"{vel_label} (m/s)"
            y_label = f"{spin_label} (rad/s)"
            x_min, x_max = 0, 30
            y_min, y_max = 0, 1000
        elif xaxis_mode == "C_L vs Spin Ratio":
            x_data = all_spin_ratio
            y_data = 2.0 * all_spin_ratio * all_cm_opt
            x_label = "Spin Ratio (Sp = r·ω/v)"
            y_label = "C_L = 2·Sp·C_M"
            x_min, x_max = 0, 2.0
            y_min, y_max = 0, 0.5
        else:  # "C_M vs Velocity"
            x_data = selected_velocity
            y_data = all_cm_opt
            x_label = f"{vel_label} (m/s)"
            y_label = "Magnus Coefficient"
            x_min, x_max = 0, 30
            y_min, y_max = 0, 0.5

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

        # Add theoretical magnus coefficient lines (only for C_M plots, not for Velocity vs Spin)
        show_v2 = self.show_v2_checkbox.isChecked()

        # Show model lines for all plot modes
        show_models = show_v2

        if show_models:
            if xaxis_mode == "C_M vs Spin":
                # Add lines for each velocity in range at 1 m/s intervals
                vel_min_round = np.ceil(vel_min_filter)
                vel_max_round = np.floor(vel_max_filter)

                # Create spin array for plotting theoretical curves
                spin_theory = np.linspace(0, 1000, 200)

                # Generate a colormap for velocity (same as data points)
                vel_cmap = plt.get_cmap('turbo')  # Use turbo (same as data) for theory lines

                _vel_step = 2.0 if self.publication_checkbox.isChecked() else 1.0
                for v_val in np.arange(max(0, vel_min_round), vel_max_round + 1, _vel_step):
                    # Color based on velocity (normalized to filtered range)
                    vel_norm = (v_val - vel_min_filter) / max(vel_max_filter - vel_min_filter, 1) if vel_max_filter > vel_min_filter else 0.5
                    rgba = vel_cmap(vel_norm)
                    base_color = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

                    # V2: Linear + quadratic model - extrapolates outside [7.5, 13.5]
                    if show_v2:
                        cm_theory_v2 = get_magnus_estimate_v2(v_val, spin_theory)
                        color_v2 = (*base_color, 220)
                        # Black border
                        pen_v2_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(spin_theory, cm_theory_v2, pen=pen_v2_border)
                        # Colored line on top
                        pen_v2 = pg.mkPen(color=color_v2, width=4, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(spin_theory, cm_theory_v2, pen=pen_v2, name=f"v={v_val} m/s")
            elif xaxis_mode == "Velocity vs Spin":
                # Add isolines of constant C_M values using skimage's find_contours
                # This properly handles non-monotonic C_M and separates disconnected contour segments
                from skimage import measure

                # Create a grid of (velocity, spin) values
                n_vel, n_spin = 150, 150
                vel_grid = np.linspace(1, 30, n_vel)
                spin_grid = np.linspace(0, 1000, n_spin)

                # C_M isoline values to plot
                cm_values = [0.025, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875, 0.2, 0.225, 0.25, 0.275, 0.3]

                # Generate a colormap for C_M values (same as data points)
                cm_cmap = plt.get_cmap('turbo')

                # For V2 model
                if show_v2:
                    # Compute C_M on the grid
                    VV, WW = np.meshgrid(vel_grid, spin_grid)
                    CM_grid = get_magnus_estimate_v2(VV, WW)

                    for cm_target in cm_values:
                        # Normalize C_M to 0-0.3 range for coloring
                        cm_norm = cm_target / 0.3
                        rgba = cm_cmap(min(cm_norm, 1.0))
                        color_v2 = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 220)

                        # Find contours at the target C_M level
                        # find_contours returns contours in (row, col) format (i.e., (spin_idx, vel_idx))
                        contours = measure.find_contours(CM_grid, cm_target)

                        for seg_idx, contour in enumerate(contours):
                            if len(contour) > 1:
                                # Convert from grid indices to actual values
                                # contour[:, 0] is row (spin index), contour[:, 1] is col (velocity index)
                                spin_seg = np.interp(contour[:, 0], np.arange(n_spin), spin_grid)
                                vel_seg = np.interp(contour[:, 1], np.arange(n_vel), vel_grid)
                                name = f"C_M={cm_target:.3f}" if seg_idx == 0 else None
                                # Black border
                                pen_v2_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.SolidLine)
                                self.plot_widget_2d.plot(vel_seg, spin_seg, pen=pen_v2_border)
                                # Colored line on top
                                pen_v2 = pg.mkPen(color=color_v2, width=4, style=Qt.SolidLine)
                                self.plot_widget_2d.plot(vel_seg, spin_seg, pen=pen_v2, name=name)
            elif xaxis_mode == "C_L vs Spin Ratio":
                # Add model curves for C_L = 2*Sp*C_M at fixed velocities
                ball_radius = 0.02
                sp_theory = np.linspace(0.01, 2.0, 200)

                vel_cmap = plt.get_cmap('turbo')
                _vel_step_cl = 2.0 if self.publication_checkbox.isChecked() else 1.0
                for v_val in np.arange(max(1, np.ceil(vel_min_filter)), np.floor(vel_max_filter) + 1, _vel_step_cl):
                    vel_norm = (v_val - vel_min_filter) / max(vel_max_filter - vel_min_filter, 1) if vel_max_filter > vel_min_filter else 0.5
                    rgba = vel_cmap(vel_norm)
                    base_color = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

                    if show_v2:
                        # w = Sp * v / r
                        w_theory = sp_theory * v_val / ball_radius
                        cm_theory = get_magnus_estimate_v2(v_val, w_theory)
                        cl_theory = 2.0 * sp_theory * cm_theory
                        color_v2 = (*base_color, 220)
                        pen_v2_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(sp_theory, cl_theory, pen=pen_v2_border)
                        pen_v2 = pg.mkPen(color=color_v2, width=4, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(sp_theory, cl_theory, pen=pen_v2, name=f"v={v_val:.0f} m/s")
            else:
                # C_M vs Velocity mode: Add lines for each spin value at 50 rad/s intervals
                # Add lines for each spin value at 50 rad/s intervals
                spin_min_int = int(np.ceil(spin_min_filter / 50.0)) * 50
                spin_max_int = int(np.floor(spin_max_filter / 50.0)) * 50

                # Create velocity array for plotting theoretical curves
                velocity_theory = np.linspace(0, 30, 200)

                # Generate a colormap for spin (same as data points)
                spin_cmap = plt.get_cmap('turbo')  # Use turbo (same as data) for theory lines

                _spin_step = 100 if self.publication_checkbox.isChecked() else 50
                for w_val in range(max(0, spin_min_int), min(1001, spin_max_int + 50), _spin_step):
                    # Color based on spin (normalized to 0-1000 rad/s range)
                    spin_norm = w_val / 1000.0
                    rgba = spin_cmap(spin_norm)
                    base_color = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

                    # Model lines - extrapolates across full range
                    if show_v2:
                        cm_theory_v2 = get_magnus_estimate_v2(velocity_theory, w_val)
                        color_v2 = (*base_color, 220)
                        # Black border
                        pen_v2_border = pg.mkPen(color=(0, 0, 0, 200), width=6, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(velocity_theory, cm_theory_v2, pen=pen_v2_border)
                        # Colored line on top
                        pen_v2 = pg.mkPen(color=color_v2, width=4, style=Qt.SolidLine)
                        self.plot_widget_2d.plot(velocity_theory, cm_theory_v2, pen=pen_v2, name=f"ω={w_val} rad/s")

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
            self.colorbar_magnus = pg.ColorBarItem(
                values=(color_min, color_max),
                colorMap=colormap,
                label=color_label,
                limits=(color_min, color_max)
            )
            self.colorbar_widget_magnus.addItem(self.colorbar_magnus)

        # Update plot title to reflect color selection and axis mode
        if xaxis_mode == "C_M vs Spin":
            self.plot_widget_2d.setTitle(f"Magnus Coefficient vs {spin_label} (colored by {color_by})")
            self.plot_widget_2d.setLabel('bottom', f'{spin_label} (rad/s)')
            self.plot_widget_2d.setLabel('left', 'Magnus Coefficient')
        elif xaxis_mode == "Velocity vs Spin":
            self.plot_widget_2d.setTitle(f"{vel_label} vs {spin_label} (colored by {color_by})")
            self.plot_widget_2d.setLabel('bottom', f'{vel_label} (m/s)')
            self.plot_widget_2d.setLabel('left', f'{spin_label} (rad/s)')
        elif xaxis_mode == "C_L vs Spin Ratio":
            self.plot_widget_2d.setTitle(f"Lift Coefficient C_L vs Spin Ratio (colored by {color_by})")
            self.plot_widget_2d.setLabel('bottom', 'Spin Ratio (Sp = r·ω/v)')
            self.plot_widget_2d.setLabel('left', 'C_L = 2·Sp·C_M')
        else:
            self.plot_widget_2d.setTitle(f"Magnus Coefficient vs Velocity (colored by {color_by})")
            self.plot_widget_2d.setLabel('bottom', f'{vel_label} (m/s)')
            self.plot_widget_2d.setLabel('left', 'Magnus Coefficient')

        # Set axis ranges
        self.plot_widget_2d.setXRange(x_min, x_max, padding=0)
        self.plot_widget_2d.setYRange(y_min, y_max, padding=0)

        self.info_label.setText(f"Showing {len(x_data)} data points (spin: {spin_min_filter}-{spin_max_filter} rad/s, vel: {vel_min_filter}-{vel_max_filter} m/s, conf: {conf_min:.2f}-{conf_max:.2f})")

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
        Create plot from extracted data for both robot and player

        Args:
            robot_data_tuple: Tuple of (vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp_magnus, cm_opt, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list) for robot
            player_data_tuple: Tuple of (vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp_magnus, cm_opt, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list) for player
        """
        self.robot_data = robot_data_tuple
        self.player_data = player_data_tuple

        # Pre-compute duration and x_travel arrays once (avoids recomputing on every slider change)
        self._robot_durations = None
        self._robot_x_travel = None
        self._player_durations = None
        self._player_x_travel = None

        if robot_data_tuple is not None:
            # flight_segments is at index 6 for magnus data tuple
            flight_segments = robot_data_tuple[6]
            if len(flight_segments) > 0:
                self._robot_durations = self._compute_durations(flight_segments)
                self._robot_x_travel = self._compute_x_travel(flight_segments)

        if player_data_tuple is not None:
            flight_segments = player_data_tuple[6]
            if len(flight_segments) > 0:
                self._player_durations = self._compute_durations(flight_segments)
                self._player_x_travel = self._compute_x_travel(flight_segments)

        self.update_plot()

    def save_plot(self):
        """Save the current plot as a PNG image"""
        if self.robot_data is None and self.player_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"magnus_coefficient_{timestamp}.png"

        output_path = plots_dir / filename

        try:
            if self.view_combo.currentText() == "3D" and self.has_3d:
                # Export 3D plot
                img = self.plot_widget_3d.grabFrameBuffer()
                img.save(str(output_path))
            else:
                # Export 2D plot
                publication = self.publication_checkbox.isChecked()
                save_plot_as_image(self.plot_widget_2d, str(output_path), publication_mode=publication)

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

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
