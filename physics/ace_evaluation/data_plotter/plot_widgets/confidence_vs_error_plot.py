# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Aerodynamics Error Analysis Plot Widget

Contains widget for displaying scatter plot of position RMSE vs various X-axis options:
- Confidence score
- Magnus force sensitivity (time-weighted v×ω integral)
- Drag force sensitivity (time-weighted v³ integral)

Also supports table contact residual errors (velocity and spin magnitude errors).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton,
    QMessageBox, QSlider, QCheckBox
)
from PySide6.QtCore import Signal, Qt, QTimer

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from .base_coefficient_plot import create_range_slider, create_combo_selector, save_plot_as_image


class ConfidenceVsErrorPlot(QWidget):
    """Widget for displaying aerodynamics error analysis scatter plot

    Shows position RMSE (y-axis) vs configurable X-axis (confidence, Magnus sensitivity, or Drag sensitivity).
    Supports multiple error types: Position RMSE, Contact Velocity Error, Contact Spin Error.
    Different colors/symbols for different trajectory sources (GT200, CPP, OPT).
    """

    # Signal emitted when a flight segment point is clicked
    flight_segment_selected = Signal(object, object, object, object, object, object)  # (fs, metadata, shot_data, rally_data, shot, rally)

    # Signal emitted when a contact error point is clicked (for contact error modes)
    contact_selected = Signal(object, object, object, object)  # (contact_data, metadata, shot, rally)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Debounce timer for slider updates
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self.update_plot)

        # Top control bar
        control_bar = QHBoxLayout()

        # Error type selector (Position RMSE vs Table Contact residuals)
        error_type_layout, self.error_type_combo = create_combo_selector(
            "Error Type:",
            ["Position RMSE", "Contact Velocity Error", "Contact Spin Error"],
            self.on_error_type_changed
        )
        control_bar.addLayout(error_type_layout)

        # Error source selector
        source_layout, self.source_combo = create_combo_selector(
            "Error Source:",
            ["GT200", "OPT", "All"],
            self.update_plot
        )
        control_bar.addLayout(source_layout)

        # X-axis selector (what to plot on X-axis)
        xaxis_layout, self.xaxis_combo = create_combo_selector(
            "X-Axis:",
            ["Confidence", "Magnus Sensitivity", "Drag Sensitivity", "Spin Density", "Spin Variance"],
            self.update_plot
        )
        control_bar.addLayout(xaxis_layout)

        # Color by selector
        color_layout, self.color_combo = create_combo_selector(
            "Color By:",
            ["Duration", "Error", "Confidence", "GCS Source"],
            self.update_plot
        )
        control_bar.addLayout(color_layout)

        # Confidence range slider
        self._conf_layout, self.conf_min_slider, self.conf_max_slider, self.conf_min_label, self.conf_max_label = \
            create_range_slider("Confidence (%):", 0, 100, 50, 100, 10, self.on_confidence_range_changed, "{}", 1.0, 25)
        control_bar.addLayout(self._conf_layout)

        # Duration range slider (0ms min to show all data by default)
        dur_layout, self.dur_min_slider, self.dur_max_slider, self.dur_min_label, self.dur_max_label = \
            create_range_slider("Duration (ms):", 0, 500, 0, 500, 50, self.on_duration_range_changed, "{}", 1.0, 25)
        control_bar.addLayout(dur_layout)

        # Error range slider (in mm) - show all data up to 200mm by default
        err_layout, self.err_min_slider, self.err_max_slider, self.err_min_label, self.err_max_label = \
            create_range_slider("Error (mm):", 0, 200, 0, 200, 20, self.on_error_range_changed, "{}", 1.0, 25)
        control_bar.addLayout(err_layout)

        # OPT RMSE quality filter (mm) – min/max range (hidden, synced from global)
        self._rmse_layout = QVBoxLayout()
        self._rmse_layout.addWidget(QLabel("OPT RMSE (mm):"))
        rmse_min_row = QHBoxLayout()
        rmse_min_row.addWidget(QLabel("Min:"))
        self.rmse_min_slider = QSlider(Qt.Horizontal)
        self.rmse_min_slider.setMinimum(0)
        self.rmse_min_slider.setMaximum(100)
        self.rmse_min_slider.setValue(0)
        self.rmse_min_slider.setTickPosition(QSlider.TicksBelow)
        self.rmse_min_slider.setTickInterval(10)
        self.rmse_min_slider.valueChanged.connect(self.update_plot)
        rmse_min_row.addWidget(self.rmse_min_slider)
        self.rmse_min_label = QLabel("0")
        self.rmse_min_label.setMinimumWidth(25)
        rmse_min_row.addWidget(self.rmse_min_label)
        self._rmse_layout.addLayout(rmse_min_row)
        rmse_max_row = QHBoxLayout()
        rmse_max_row.addWidget(QLabel("Max:"))
        self.rmse_slider = QSlider(Qt.Horizontal)
        self.rmse_slider.setMinimum(1)
        self.rmse_slider.setMaximum(100)
        self.rmse_slider.setValue(100)
        self.rmse_slider.setTickPosition(QSlider.TicksBelow)
        self.rmse_slider.setTickInterval(10)
        self.rmse_slider.valueChanged.connect(self.update_plot)
        rmse_max_row.addWidget(self.rmse_slider)
        self.rmse_label = QLabel("100")
        self.rmse_label.setMinimumWidth(25)
        rmse_max_row.addWidget(self.rmse_label)
        self._rmse_layout.addLayout(rmse_max_row)
        control_bar.addLayout(self._rmse_layout)

        # Point size slider
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Point Size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(2)
        self.size_slider.setMaximum(15)
        self.size_slider.setValue(8)
        self.size_slider.valueChanged.connect(self._update_timer.start)
        size_layout.addWidget(self.size_slider)
        self.size_label = QLabel("8")
        self.size_label.setMinimumWidth(20)
        size_layout.addWidget(self.size_label)
        control_bar.addLayout(size_layout)

        # Publication mode checkbox
        self.publication_checkbox = QCheckBox("Publication")
        control_bar.addWidget(self.publication_checkbox)

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Info label
        self.info_label = QLabel("Aerodynamics Error Analysis")
        layout.addWidget(self.info_label)

        # Create 2D plot widget with colorbar (matching drag/magnus coefficient plot style)
        plot_2d_container = QHBoxLayout()

        self.plot_widget = pg.PlotWidget(title="Aerodynamics Error Analysis")
        self.plot_widget.setLabel('left', 'Position RMSE (mm)')
        self.plot_widget.setLabel('bottom', 'Confidence Score')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)
        plot_2d_container.addWidget(self.plot_widget)

        # Create colorbar widget using GraphicsLayoutWidget
        self.colorbar_widget = pg.GraphicsLayoutWidget()
        self.colorbar_widget.setMaximumWidth(100)
        self.colorbar_widget.setMinimumWidth(80)
        plot_2d_container.addWidget(self.colorbar_widget)

        # Create container widget for plot + colorbar
        plot_container = QWidget()
        plot_container.setLayout(plot_2d_container)
        layout.addWidget(plot_container, stretch=1)

        # Initialize colorbar reference
        self.colorbar = None

        # Statistics label
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

        # Store data
        self.error_data = None  # Position RMSE data (dict with 'gt200', 'opt')
        self.confidence_data = None
        self.duration_data = None
        self.magnus_sensitivity_data = None  # Magnus force sensitivity (v x w integral)
        self.drag_sensitivity_data = None  # Drag force sensitivity (v^3 integral)
        self.spin_density_data = None  # Spin density (% of samples with spin conf > 0.5)
        self.spin_variance_data = None  # Spin variance (RMSE of GCS vs GT200 spin)
        self.gcs_source_data = None  # Per-segment GCS source tag (list of strings)
        self.metadata_list = None
        self.flight_segments = None
        self.shots = None
        self.rallies = None
        self.base_folder = None

        # Table contact residual data
        self.contact_velocity_error = None  # Velocity magnitude error from table contact
        self.contact_spin_error = None  # Spin magnitude error from table contact
        self.contact_confidence = None  # Confidence for table contact data
        self.contact_metadata_list = None
        self.contact_shots = None
        self.contact_rallies = None
        self.contact_fs_pre_list = None  # Pre-contact flight segments
        self.contact_fs_post_list = None  # Post-contact flight segments
        self.contact_max_rmse_opt = None  # OPT RMSE for contact error filtering

        # Scatter items for click handling
        self.scatter_items = {}

    def on_confidence_range_changed(self):
        """Handle confidence range slider changes"""
        conf_min = self.conf_min_slider.value()
        conf_max = self.conf_max_slider.value()

        if conf_min > conf_max:
            if self.sender() == self.conf_min_slider:
                self.conf_min_slider.setValue(conf_max)
                conf_min = conf_max
            else:
                self.conf_max_slider.setValue(conf_min)
                conf_max = conf_min

        self.conf_min_label.setText(str(conf_min))
        self.conf_max_label.setText(str(conf_max))
        self._update_timer.start()

    def on_duration_range_changed(self):
        """Handle duration range slider changes"""
        dur_min = self.dur_min_slider.value()
        dur_max = self.dur_max_slider.value()

        if dur_min > dur_max:
            if self.sender() == self.dur_min_slider:
                self.dur_min_slider.setValue(dur_max)
                dur_min = dur_max
            else:
                self.dur_max_slider.setValue(dur_min)
                dur_max = dur_min

        self.dur_min_label.setText(str(dur_min))
        self.dur_max_label.setText(str(dur_max))
        self._update_timer.start()

    def on_error_range_changed(self):
        """Handle error range slider changes"""
        err_min = self.err_min_slider.value()
        err_max = self.err_max_slider.value()

        if err_min > err_max:
            if self.sender() == self.err_min_slider:
                self.err_min_slider.setValue(err_max)
                err_min = err_max
            else:
                self.err_max_slider.setValue(err_min)
                err_max = err_min

        self.err_min_label.setText(str(err_min))
        self.err_max_label.setText(str(err_max))
        self._update_timer.start()

    def on_error_type_changed(self):
        """Handle error type selector changes"""
        error_type = self.error_type_combo.currentText()

        # Enable/disable source selector based on error type
        # Source selector only applies to Position RMSE (which has gt200, cpp, opt sources)
        if error_type == "Position RMSE":
            self.source_combo.setEnabled(True)
        else:
            self.source_combo.setEnabled(False)

        self.update_plot()

    def plot_data(
        self,
        error_dict: Dict[str, np.ndarray],
        confidence: np.ndarray,
        duration: np.ndarray,
        metadata_list: List[Dict] = None,
        flight_segments: List = None,
        shots: List = None,
        rallies: List = None,
        base_folder: str = None,
        magnus_sensitivity: np.ndarray = None,
        drag_sensitivity: np.ndarray = None,
        spin_density: np.ndarray = None,
        spin_variance: np.ndarray = None,
        gcs_source: list = None,
    ):
        """Plot aerodynamics error analysis data

        Args:
            error_dict: Dictionary with keys 'gt200', 'opt' containing RMSE arrays (in meters)
            confidence: Array of confidence scores (0-1)
            duration: Array of segment durations (in seconds)
            metadata_list: Optional list of metadata dicts for each segment
            flight_segments: Optional list of FlightSegment objects
            shots: Optional list of Shot objects
            rallies: Optional list of Rally objects
            base_folder: Optional base folder path
            magnus_sensitivity: Optional array of Magnus force sensitivity values
            drag_sensitivity: Optional array of Drag force sensitivity values
            spin_density: Optional array of spin density values (0-100%)
            spin_variance: Optional array of spin RMSE vs GT200 values (rad/s)
        """
        self.error_data = error_dict
        self.confidence_data = confidence
        self.duration_data = duration
        self.magnus_sensitivity_data = magnus_sensitivity
        self.drag_sensitivity_data = drag_sensitivity
        self.spin_density_data = spin_density
        self.spin_variance_data = spin_variance
        self.gcs_source_data = gcs_source if gcs_source is not None else []
        self.metadata_list = metadata_list
        self.flight_segments = flight_segments
        self.shots = shots
        self.rallies = rallies
        self.base_folder = base_folder
        self.update_plot()

    def set_contact_error_data(
        self,
        velocity_error: np.ndarray,
        spin_error: np.ndarray,
        confidence: np.ndarray,
        metadata_list: List[Dict] = None,
        shots: List = None,
        rallies: List = None,
        fs_pre_list: List = None,
        fs_post_list: List = None,
        max_rmse_opt: np.ndarray = None,
    ):
        """Set table contact residual error data

        Args:
            velocity_error: Array of velocity magnitude errors (m/s)
            spin_error: Array of spin magnitude errors (rad/s)
            confidence: Array of confidence scores (0-1) for contacts
            metadata_list: Optional list of metadata dicts for each contact
            shots: Optional list of Shot objects
            rallies: Optional list of Rally objects
            fs_pre_list: Optional list of pre-contact FlightSegments
            fs_post_list: Optional list of post-contact FlightSegments
            max_rmse_opt: Optional array of OPT RMSE values (meters) for quality filtering
        """
        self.contact_velocity_error = velocity_error
        self.contact_spin_error = spin_error
        self.contact_confidence = confidence
        self.contact_metadata_list = metadata_list
        self.contact_shots = shots
        self.contact_rallies = rallies
        self.contact_fs_pre_list = fs_pre_list
        self.contact_fs_post_list = fs_post_list
        self.contact_max_rmse_opt = max_rmse_opt

    def update_plot(self):
        """Update the plot based on current settings"""

        self.plot_widget.clear()
        self.scatter_items.clear()

        # Remove any existing legend
        if hasattr(self.plot_widget.plotItem, 'legend') and self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.scene().removeItem(self.plot_widget.plotItem.legend)
            self.plot_widget.plotItem.legend = None

        # Clear existing colorbar if present
        self.colorbar_widget.clear()
        self.colorbar = None

        # Update size label
        self.size_label.setText(str(self.size_slider.value()))

        error_type = self.error_type_combo.currentText()

        # Dispatch to appropriate plot method based on error type
        if error_type == "Position RMSE":
            self._update_position_rmse_plot()
        elif error_type == "Contact Velocity Error":
            self._update_contact_error_plot("velocity")
        elif error_type == "Contact Spin Error":
            self._update_contact_error_plot("spin")

    def _update_position_rmse_plot(self):
        """Update plot for position RMSE mode (original behavior)"""
        if self.error_data is None or self.confidence_data is None:
            self.info_label.setText("No data available")
            return

        source = self.source_combo.currentText()
        color_by = self.color_combo.currentText()
        x_axis = self.xaxis_combo.currentText()
        conf_min = self.conf_min_slider.value() / 100.0  # Convert from percentage
        conf_max = self.conf_max_slider.value() / 100.0
        dur_min = self.dur_min_slider.value() / 1000.0  # Convert from ms to seconds
        dur_max = self.dur_max_slider.value() / 1000.0
        err_min = self.err_min_slider.value() / 1000.0  # Convert from mm to meters
        err_max = self.err_max_slider.value() / 1000.0
        point_size = self.size_slider.value()

        # Determine X-axis data and label
        if x_axis == "Confidence":
            x_data_all = self.confidence_data
            x_label = "Confidence Score"
        elif x_axis == "Magnus Sensitivity":
            if self.magnus_sensitivity_data is None:
                self.info_label.setText("No Magnus sensitivity data available")
                return
            x_data_all = self.magnus_sensitivity_data
            x_label = "Magnus Sensitivity (Σ||v×ω||(t_end-t)dt)"
        elif x_axis == "Spin Density":
            if self.spin_density_data is None:
                self.info_label.setText("No spin density data available")
                return
            x_data_all = self.spin_density_data
            x_label = "Spin Density (%)"
        elif x_axis == "Spin Variance":
            if self.spin_variance_data is None:
                self.info_label.setText("No spin variance data available")
                return
            x_data_all = self.spin_variance_data
            x_label = "Spin Variance σ(|ω|) (rad/s)"
        else:  # Drag Sensitivity
            if self.drag_sensitivity_data is None:
                self.info_label.setText("No Drag sensitivity data available")
                return
            x_data_all = self.drag_sensitivity_data
            x_label = "Drag Sensitivity (Σ||v||³(t_end-t)dt)"

        # Define symbols for each source
        symbols = {
            'gt200': 'o',
            'opt': 's',  # Square
        }
        names = {
            'gt200': 'GT200',
            'opt': 'OPT',
        }

        total_points = 0
        stats_parts = []
        all_x_valid = []
        all_err_valid = []

        # Determine which sources to plot
        if source == "All":
            sources_to_plot = ['gt200', 'opt']
        else:
            sources_to_plot = [source.lower()]

        # Get colormap (turbo: blue -> cyan -> green -> yellow -> red)
        cmap = plt.get_cmap('turbo')

        # Determine color mode
        use_categorical = (color_by == "GCS Source")

        # Determine color label and range based on selection
        if color_by == "Duration":
            color_label = "Duration (ms)"
            colorbar_min, colorbar_max = dur_min * 1000, dur_max * 1000
        elif color_by == "Error":
            color_label = "Error (mm)"
            colorbar_min, colorbar_max = err_min * 1000, err_max * 1000
        elif color_by == "GCS Source":
            color_label = ""
            colorbar_min, colorbar_max = 0, 1
        else:  # Confidence
            color_label = "Confidence"
            colorbar_min, colorbar_max = conf_min, conf_max

        # OPT RMSE quality filter – uses error_data['opt'] regardless of displayed source
        opt_rmse_ref = self.error_data.get('opt')
        rmse_max_val = self.rmse_slider.value() / 1000.0   # mm -> meters
        rmse_min_val = self.rmse_min_slider.value() / 1000.0
        self.rmse_min_label.setText(str(self.rmse_min_slider.value()))
        self.rmse_label.setText(str(self.rmse_slider.value()))

        for src in sources_to_plot:
            errors = self.error_data.get(src, np.array([]))
            if len(errors) == 0:
                continue

            # Apply filters
            # When max slider is at maximum, show all data >= min (no upper bound)
            dur_filter = (self.duration_data >= dur_min)
            if dur_max < 0.5:  # 500ms = 0.5s is the max slider value
                dur_filter = dur_filter & (self.duration_data <= dur_max)

            err_filter = (errors >= err_min)
            if err_max < 0.2:  # 200mm = 0.2m is the max slider value
                err_filter = err_filter & (errors <= err_max)

            # OPT RMSE quality filter
            if opt_rmse_ref is not None and len(opt_rmse_ref) == len(errors):
                opt_valid = ~np.isnan(opt_rmse_ref)
                opt_rmse_filter = opt_valid.copy()
                if self.rmse_slider.value() < self.rmse_slider.maximum():
                    opt_rmse_filter = opt_rmse_filter & (opt_rmse_ref <= rmse_max_val)
                if self.rmse_min_slider.value() > 0:
                    opt_rmse_filter = opt_rmse_filter & (opt_rmse_ref >= rmse_min_val)
            else:
                opt_rmse_filter = np.ones(len(errors), dtype=bool)

            valid_mask = (
                ~np.isnan(errors) &
                ~np.isnan(self.confidence_data) &
                ~np.isnan(x_data_all) &
                (self.confidence_data >= conf_min) &
                (self.confidence_data <= conf_max) &
                dur_filter &
                err_filter &
                opt_rmse_filter
            )

            if not np.any(valid_mask):
                continue

            x_valid = x_data_all[valid_mask]
            conf_valid = self.confidence_data[valid_mask]
            err_valid = errors[valid_mask] * 1000  # Convert to mm
            dur_valid = self.duration_data[valid_mask] * 1000  # Convert to ms
            indices_valid = np.where(valid_mask)[0]

            n_points = len(x_valid)
            total_points += n_points

            # Collect for auto-scaling
            all_x_valid.extend(x_valid)
            all_err_valid.extend(err_valid)

            # Determine color data based on selection
            if use_categorical:
                # GCS Source categorical coloring
                _gcs_colors = {
                    'gcs_offline': (50, 200, 50),     # green   = preferred
                    'gcs_filtered': (255, 160, 50),   # orange  = fallback
                    'gcs': (255, 80, 80),             # red     = raw / legacy
                }
                _gcs_default = (150, 150, 150)        # gray    = unknown/none
                brushes = []
                for i in range(n_points):
                    idx = indices_valid[i]
                    gcs_src = self.gcs_source_data[idx] if idx < len(self.gcs_source_data) else 'unknown'
                    rgb = _gcs_colors.get(gcs_src, _gcs_default)
                    # Apply confidence-based alpha
                    conf = conf_valid[i]
                    alpha = int(conf * 204 + 51)
                    alpha = np.clip(alpha, 51, 255)
                    brushes.append(pg.mkBrush(*rgb, alpha))
            else:
                if color_by == "Duration":
                    color_data = dur_valid
                    color_min, color_max = colorbar_min, colorbar_max
                elif color_by == "Error":
                    color_data = err_valid
                    color_min, color_max = colorbar_min, colorbar_max
                else:  # Confidence
                    color_data = conf_valid
                    color_min, color_max = colorbar_min, colorbar_max

                # Compute colors with confidence-based alpha
                brushes = []
                for i in range(n_points):
                    val = color_data[i]
                    val_clipped = np.clip(val, color_min, color_max)
                    norm_val = (val_clipped - color_min) / (color_max - color_min) if color_max > color_min else 0.5
                    rgba = cmap(norm_val)
                    r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)

                    # Apply confidence-based alpha (map 0-1 to 51-255)
                    conf = conf_valid[i]
                    alpha = int(conf * 204 + 51)
                    alpha = np.clip(alpha, 51, 255)

                    brushes.append(pg.mkBrush(r, g, b, alpha))

            # Create spots data for clickable scatter plot
            spots = []
            for i in range(n_points):
                spots.append({
                    'pos': (x_valid[i], err_valid[i]),
                    'size': point_size,
                    'pen': pg.mkPen(color=(50, 50, 50), width=1),
                    'brush': brushes[i],
                    'symbol': symbols[src],
                    'data': indices_valid[i]
                })

            scatter = pg.ScatterPlotItem(spots=spots, hoverable=True, tip=None)
            scatter.sigClicked.connect(lambda plot, points, s=src: self._on_point_clicked(plot, points, s))
            self.plot_widget.addItem(scatter)
            self.scatter_items[src] = scatter

            # Compute statistics
            mean_err = np.mean(err_valid)
            median_err = np.median(err_valid)
            corr = np.corrcoef(x_valid, err_valid)[0, 1] if len(x_valid) > 1 else 0

            stats_parts.append(f"{names[src]}: n={n_points}, mean={mean_err:.1f}mm, median={median_err:.1f}mm, corr={corr:.3f}")

        # Auto-rescale axes based on filtered data (use 1st-99th percentile to ignore outliers)
        if len(all_x_valid) > 0 and len(all_err_valid) > 0:
            x_arr = np.array(all_x_valid)
            y_arr = np.array(all_err_valid)
            x_min, x_max = float(np.percentile(x_arr, 1)), float(np.percentile(x_arr, 99))
            y_min, y_max = float(np.percentile(y_arr, 1)), float(np.percentile(y_arr, 99))
            # Add 5% padding
            x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.05
            y_padding = (y_max - y_min) * 0.05 if y_max > y_min else 5
            self.plot_widget.setXRange(x_min - x_padding, x_max + x_padding, padding=0)
            self.plot_widget.setYRange(y_min - y_padding, y_max + y_padding, padding=0)

        # Add colorbar or legend
        if use_categorical:
            # Remove any existing legend
            if hasattr(self.plot_widget.plotItem, 'legend') and self.plot_widget.plotItem.legend is not None:
                self.plot_widget.plotItem.legend.scene().removeItem(self.plot_widget.plotItem.legend)
                self.plot_widget.plotItem.legend = None
            legend = self.plot_widget.addLegend(offset=(10, 10))
            legend.addItem(
                pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(50,200,50,200), size=8, symbol='o'),
                "gcs_offline (preferred)"
            )
            legend.addItem(
                pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(255,160,50,200), size=8, symbol='o'),
                "gcs_filtered (fallback)"
            )
            legend.addItem(
                pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(255,80,80,200), size=8, symbol='o'),
                "gcs (raw)"
            )
            legend.addItem(
                pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(150,150,150,200), size=8, symbol='o'),
                "unknown / none"
            )
        else:
            # Add colorbar using GraphicsLayoutWidget (matching drag/magnus coefficient style)
            colormap = pg.colormap.get('turbo')
            self.colorbar_widget.addItem(pg.ColorBarItem(
                values=(colorbar_min, colorbar_max),
                colorMap=colormap,
                label=color_label,
                limits=(colorbar_min, colorbar_max)
            ))

        # Update title to reflect X-axis and color selection
        self.plot_widget.setTitle(f"{x_axis} vs Position RMSE (colored by {color_by})")
        self.plot_widget.setLabel('left', 'Position RMSE (mm)')
        self.plot_widget.setLabel('bottom', x_label)

        self.info_label.setText(
            f"Showing {total_points} segments | "
            f"Conf: [{int(conf_min*100)}, {int(conf_max*100)}]% | "
            f"Dur: [{int(dur_min*1000)}, {int(dur_max*1000)}]ms | "
            f"Err: [{int(err_min*1000)}, {int(err_max*1000)}]mm | "
            f"OPT RMSE: [{self.rmse_min_slider.value()}, {self.rmse_slider.value()}]mm"
        )
        self.stats_label.setText(" | ".join(stats_parts) if stats_parts else "No valid data")

    def _update_contact_error_plot(self, error_mode: str):
        """Update plot for table contact residual error modes

        Args:
            error_mode: 'velocity' for velocity magnitude error, 'spin' for spin magnitude error
        """
        # Select appropriate error data based on mode
        if error_mode == "velocity":
            errors = self.contact_velocity_error
            error_label = "Velocity Error (m/s)"
            title = "Confidence vs Contact Velocity Residual"
            unit_scale = 1.0  # Already in m/s
            unit_str = "m/s"
        else:  # spin
            errors = self.contact_spin_error
            error_label = "Spin Error (rad/s)"
            title = "Confidence vs Contact Spin Residual"
            unit_scale = 1.0  # Already in rad/s
            unit_str = "rad/s"

        if errors is None or self.contact_confidence is None:
            self.info_label.setText(f"No {error_mode} contact error data available")
            self.plot_widget.setTitle(title)
            self.plot_widget.setLabel('left', error_label)
            return

        color_by = self.color_combo.currentText()
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0
        err_min = self.err_min_slider.value() / 1000.0  # Will use directly as raw value for contact errors
        err_max = self.err_max_slider.value() / 1000.0
        point_size = self.size_slider.value()

        # For contact errors, scale the slider values appropriately
        # Velocity errors are typically 0-2 m/s, spin errors 0-200 rad/s
        if error_mode == "velocity":
            # Scale from 0-200 mm slider to 0-2 m/s
            err_filter_min = err_min * 10  # 0-200mm -> 0-2 m/s
            err_filter_max = err_max * 10
        else:  # spin
            # Scale from 0-200 mm slider to 0-200 rad/s
            err_filter_min = err_min * 1000  # 0-200mm -> 0-200 rad/s
            err_filter_max = err_max * 1000

        # Get colormap
        cmap = plt.get_cmap('turbo')

        # Apply filters
        valid_mask = (
            ~np.isnan(errors) &
            ~np.isnan(self.contact_confidence) &
            (self.contact_confidence >= conf_min) &
            (self.contact_confidence <= conf_max) &
            (np.abs(errors) >= err_filter_min)
        )
        # Apply max filter only if not at maximum
        if err_filter_max < (2.0 if error_mode == "velocity" else 200.0):
            valid_mask = valid_mask & (np.abs(errors) <= err_filter_max)

        # OPT RMSE quality filter for contact data
        if self.contact_max_rmse_opt is not None and len(self.contact_max_rmse_opt) == len(errors):
            rmse_max_m = self.rmse_slider.value() / 1000.0
            rmse_min_m = self.rmse_min_slider.value() / 1000.0
            opt_valid = ~np.isnan(self.contact_max_rmse_opt)
            if self.rmse_slider.value() < self.rmse_slider.maximum():
                valid_mask = valid_mask & opt_valid & (self.contact_max_rmse_opt <= rmse_max_m)
            if self.rmse_min_slider.value() > 0:
                valid_mask = valid_mask & opt_valid & (self.contact_max_rmse_opt >= rmse_min_m)

        if not np.any(valid_mask):
            self.info_label.setText(f"No {error_mode} contact data matching filters")
            self.plot_widget.setTitle(title)
            self.plot_widget.setLabel('left', error_label)
            return

        conf_valid = self.contact_confidence[valid_mask]
        err_valid = np.abs(errors[valid_mask])  # Use absolute error for display
        indices_valid = np.where(valid_mask)[0]

        n_points = len(conf_valid)

        # Determine color range
        if color_by == "Error":
            color_label = f"Error ({unit_str})"
            colorbar_min, colorbar_max = np.min(err_valid), np.max(err_valid)
        else:  # Confidence (Duration not applicable for contact data)
            color_label = "Confidence"
            colorbar_min, colorbar_max = conf_min, conf_max

        # Determine color data
        if color_by == "Error":
            color_data = err_valid
        else:  # Confidence
            color_data = conf_valid

        color_min, color_max = colorbar_min, colorbar_max

        # Compute colors with confidence-based alpha
        brushes = []
        for i in range(n_points):
            val = color_data[i]
            val_clipped = np.clip(val, color_min, color_max)
            norm_val = (val_clipped - color_min) / (color_max - color_min) if color_max > color_min else 0.5
            rgba = cmap(norm_val)
            r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)

            # Apply confidence-based alpha
            conf = conf_valid[i]
            alpha = int(conf * 204 + 51)
            alpha = np.clip(alpha, 51, 255)

            brushes.append(pg.mkBrush(r, g, b, alpha))

        # Create spots data for clickable scatter plot
        spots = []
        for i in range(n_points):
            spots.append({
                'pos': (conf_valid[i], err_valid[i]),
                'size': point_size,
                'pen': pg.mkPen(color=(50, 50, 50), width=1),
                'brush': brushes[i],
                'symbol': 'o',
                'data': indices_valid[i]
            })

        scatter = pg.ScatterPlotItem(spots=spots, hoverable=True, tip=None)
        scatter.sigClicked.connect(lambda plot, points: self._on_contact_point_clicked(plot, points, error_mode))
        self.plot_widget.addItem(scatter)
        self.scatter_items['contact'] = scatter

        # Compute statistics
        mean_err = np.mean(err_valid)
        median_err = np.median(err_valid)
        corr = np.corrcoef(conf_valid, err_valid)[0, 1] if len(conf_valid) > 1 else 0

        # Auto-rescale axes (use 1st-99th percentile to ignore outliers)
        x_min, x_max = float(np.percentile(conf_valid, 1)), float(np.percentile(conf_valid, 99))
        y_min, y_max = float(np.percentile(err_valid, 1)), float(np.percentile(err_valid, 99))
        x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.05
        y_padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        self.plot_widget.setXRange(x_min - x_padding, x_max + x_padding, padding=0)
        self.plot_widget.setYRange(y_min - y_padding, y_max + y_padding, padding=0)

        # Add colorbar
        colormap = pg.colormap.get('turbo')
        self.colorbar_widget.addItem(pg.ColorBarItem(
            values=(colorbar_min, colorbar_max),
            colorMap=colormap,
            label=color_label,
            limits=(colorbar_min, colorbar_max)
        ))

        # Update title and labels
        self.plot_widget.setTitle(f"{title} (colored by {color_by})")
        self.plot_widget.setLabel('left', error_label)
        self.plot_widget.setLabel('bottom', 'Confidence Score')

        self.info_label.setText(
            f"Showing {n_points} contacts | "
            f"Conf: [{int(conf_min*100)}, {int(conf_max*100)}]%"
        )
        self.stats_label.setText(
            f"n={n_points}, mean={mean_err:.3f}{unit_str}, median={median_err:.3f}{unit_str}, corr={corr:.3f}"
        )

    def _on_contact_point_clicked(self, plot_item, points, error_mode: str):
        """Handle click on contact error scatter plot point"""
        if len(points) == 0:
            return

        point = points[0]
        idx = int(point.data())

        # Get metadata and objects for this contact
        metadata = dict(self.contact_metadata_list[idx]) if self.contact_metadata_list and idx < len(self.contact_metadata_list) else {}
        shot = self.contact_shots[idx] if self.contact_shots and idx < len(self.contact_shots) else None
        rally = self.contact_rallies[idx] if self.contact_rallies and idx < len(self.contact_rallies) else None

        # Get the pre and post flight segments for this contact
        fs_pre = self.contact_fs_pre_list[idx] if self.contact_fs_pre_list and idx < len(self.contact_fs_pre_list) else None
        fs_post = self.contact_fs_post_list[idx] if self.contact_fs_post_list and idx < len(self.contact_fs_post_list) else None

        # Add error and confidence to metadata
        if error_mode == "velocity" and self.contact_velocity_error is not None and idx < len(self.contact_velocity_error):
            metadata['velocity_error'] = self.contact_velocity_error[idx]
        if error_mode == "spin" and self.contact_spin_error is not None and idx < len(self.contact_spin_error):
            metadata['spin_error'] = self.contact_spin_error[idx]
        if self.contact_confidence is not None and idx < len(self.contact_confidence):
            metadata['confidence'] = self.contact_confidence[idx]

        # Build contact_data dict matching table contacts plot format
        contact_data = {
            'fs_pre': fs_pre,
            'fs_post': fs_post,
        }

        # Emit contact_selected signal (same format as table contacts plot)
        self.contact_selected.emit(contact_data, metadata, shot, rally)

    def _on_point_clicked(self, plot_item, points, source: str):
        """Handle click on scatter plot point"""
        if len(points) == 0:
            return

        point = points[0]
        idx = int(point.data())

        if self.flight_segments is None or idx >= len(self.flight_segments):
            return

        fs = self.flight_segments[idx]
        metadata = self.metadata_list[idx] if self.metadata_list and idx < len(self.metadata_list) else {}
        shot = self.shots[idx] if self.shots and idx < len(self.shots) else None
        rally = self.rallies[idx] if self.rallies and idx < len(self.rallies) else None

        # Build shot data if available
        shot_data = None
        if shot is not None:
            import pandas as pd
            try:
                shot_data = pd.concat([seg.data for seg in shot.flight_segments], ignore_index=False)
            except Exception:
                pass

        # Build rally data if available
        rally_data = None
        if rally is not None:
            import pandas as pd
            try:
                all_dfs = []
                for s in rally.shots:
                    for seg in s.flight_segments:
                        all_dfs.append(seg.data)
                if all_dfs:
                    rally_data = pd.concat(all_dfs, ignore_index=False)
            except Exception:
                pass

        # Add error and confidence to metadata
        if self.error_data:
            for src_name in ['gt200', 'opt']:
                if src_name in self.error_data and idx < len(self.error_data[src_name]):
                    metadata[f'rmse_{src_name}'] = self.error_data[src_name][idx] * 1000  # mm
        if self.confidence_data is not None and idx < len(self.confidence_data):
            metadata['confidence'] = self.confidence_data[idx]

        self.flight_segment_selected.emit(fs, metadata, shot_data, rally_data, shot, rally)

    def save_plot(self):
        """Save the current plot to a file"""
        try:
            plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
            plots_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = plots_dir / f"confidence_vs_error_{timestamp}.png"

            publication = self.publication_checkbox.isChecked()
            save_plot_as_image(self.plot_widget, str(filename), publication_mode=publication)
            QMessageBox.information(self, "Save Successful", f"Plot saved to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Failed to save plot:\n{str(e)}")
