# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Aero Summary Error Plot Widget

Contains widget for displaying aerodynamic error summary comparing different trajectory sources.
Shows violin plots of position errors between observations (APS) and different models (GT200, CPP, OPT).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QMessageBox, QComboBox, QSlider, QCheckBox
from PySide6.QtCore import Signal, Qt

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from .base_coefficient_plot import save_plot_as_image


class AeroSummaryErrorPlot(QWidget):
    """Widget for displaying aerodynamic error summary as box plots

    Shows position errors comparing APS observations to:
    - GT200: Ground truth at 200Hz
    - CPP: C++ forward simulation
    - OPT: Optimized trajectory
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Top control bar
        control_bar = QHBoxLayout()

        # Error metric selector
        metric_selector_layout = QVBoxLayout()
        metric_label = QLabel("Error Metric:")
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Position RMSE", "Mean Position Error", "Max Position Error", "Velocity RMSE"])
        self.metric_combo.currentTextChanged.connect(self.update_plot)
        metric_selector_layout.addWidget(metric_label)
        metric_selector_layout.addWidget(self.metric_combo)
        control_bar.addLayout(metric_selector_layout)

        # Group By selector
        group_selector_layout = QVBoxLayout()
        group_label = QLabel("Group By:")
        self.group_combo = QComboBox()
        self.group_combo.addItems(["Source", "Date", "Shot Type"])
        self.group_combo.currentTextChanged.connect(self._on_group_changed)
        group_selector_layout.addWidget(group_label)
        group_selector_layout.addWidget(self.group_combo)
        control_bar.addLayout(group_selector_layout)

        # Source selector (visible when grouping by Date or Shot Type)
        self._source_layout = QVBoxLayout()
        source_label = QLabel("Source:")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["GT200", "OPT", "0226", "0426", "NAK"])
        self.source_combo.setCurrentText("OPT")
        self.source_combo.currentTextChanged.connect(self.update_plot)
        self._source_layout.addWidget(source_label)
        self._source_layout.addWidget(self.source_combo)
        control_bar.addLayout(self._source_layout)
        # Hide source combo initially (only used for Date/Shot Type grouping)
        source_label.setVisible(False)
        self.source_combo.setVisible(False)
        self._source_label = source_label

        # Confidence range sliders
        self._conf_layout = QVBoxLayout()
        confidence_range_label = QLabel("Confidence Range:")
        self._conf_layout.addWidget(confidence_range_label)

        # Min confidence slider
        conf_min_layout = QHBoxLayout()
        conf_min_layout.addWidget(QLabel("Min:"))
        self.conf_min_slider = QSlider(Qt.Horizontal)
        self.conf_min_slider.setMinimum(0)
        self.conf_min_slider.setMaximum(100)
        self.conf_min_slider.setValue(50)
        self.conf_min_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_min_slider.setTickInterval(10)
        self.conf_min_slider.valueChanged.connect(self.on_confidence_range_changed)
        conf_min_layout.addWidget(self.conf_min_slider)
        self.conf_min_label = QLabel("0.50")
        self.conf_min_label.setMinimumWidth(35)
        conf_min_layout.addWidget(self.conf_min_label)
        self._conf_layout.addLayout(conf_min_layout)

        # Max confidence slider
        conf_max_layout = QHBoxLayout()
        conf_max_layout.addWidget(QLabel("Max:"))
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
        self._conf_layout.addLayout(conf_max_layout)

        control_bar.addLayout(self._conf_layout)

        # Source selection checkboxes
        source_select_layout = QVBoxLayout()
        source_select_label = QLabel("Sources:")
        source_select_label.setStyleSheet("font-size: 9pt;")
        source_select_layout.addWidget(source_select_label)

        self.source_checkboxes = {}
        for src_name in ['GT200', 'OPT', '0226', '0426', 'NAK']:
            cb = QCheckBox(src_name)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_plot)
            source_select_layout.addWidget(cb)
            self.source_checkboxes[src_name] = cb

        control_bar.addLayout(source_select_layout)

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

        # Duration range sliders
        duration_range_layout = QVBoxLayout()
        duration_range_label = QLabel("Duration Range (s):")
        duration_range_layout.addWidget(duration_range_label)

        # Min duration slider
        dur_min_layout = QHBoxLayout()
        dur_min_layout.addWidget(QLabel("Min:"))
        self.dur_min_slider = QSlider(Qt.Horizontal)
        self.dur_min_slider.setMinimum(0)
        self.dur_min_slider.setMaximum(200)
        self.dur_min_slider.setValue(30)
        self.dur_min_slider.setTickPosition(QSlider.TicksBelow)
        self.dur_min_slider.setTickInterval(20)
        self.dur_min_slider.valueChanged.connect(self.on_duration_range_changed)
        dur_min_layout.addWidget(self.dur_min_slider)
        self.dur_min_label = QLabel("0.150")
        self.dur_min_label.setMinimumWidth(35)
        dur_min_layout.addWidget(self.dur_min_label)
        duration_range_layout.addLayout(dur_min_layout)

        # Max duration slider
        dur_max_layout = QHBoxLayout()
        dur_max_layout.addWidget(QLabel("Max:"))
        self.dur_max_slider = QSlider(Qt.Horizontal)
        self.dur_max_slider.setMinimum(0)
        self.dur_max_slider.setMaximum(200)
        self.dur_max_slider.setValue(200)
        self.dur_max_slider.setTickPosition(QSlider.TicksBelow)
        self.dur_max_slider.setTickInterval(20)
        self.dur_max_slider.valueChanged.connect(self.on_duration_range_changed)
        dur_max_layout.addWidget(self.dur_max_slider)
        self.dur_max_label = QLabel("1.000")
        self.dur_max_label.setMinimumWidth(35)
        dur_max_layout.addWidget(self.dur_max_label)
        duration_range_layout.addLayout(dur_max_layout)

        control_bar.addLayout(duration_range_layout)

        # Info label
        self.info_label = QLabel("Error comparison between trajectory sources")
        control_bar.addWidget(self.info_label, stretch=1)

        # Publication mode checkbox
        self.publication_checkbox = QCheckBox("Publication")
        self.publication_checkbox.stateChanged.connect(self.update_plot)
        control_bar.addWidget(self.publication_checkbox)

        # Add save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Create plot widget
        self.plot_widget = pg.PlotWidget(title="Aerodynamic Error Summary")
        self.plot_widget.setLabel("left", "Position Error (m)")
        self.plot_widget.setLabel("bottom", "Trajectory Source")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

        # Store data
        self.error_data = None  # Will be dict with 'gt200', 'opt' errors
        self.confidence_data = None
        self.duration_data = None
        self.base_folder = None
        self.match_dates = None
        self.shot_types = None

    def _on_group_changed(self):
        """Handle group by combo change - show/hide source selector"""
        group_by = self.group_combo.currentText()
        show_source = group_by in ("Date", "Shot Type")
        self._source_label.setVisible(show_source)
        self.source_combo.setVisible(show_source)
        self.update_plot()

    def on_confidence_range_changed(self):
        """Handle confidence range slider changes"""
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0

        if conf_min > conf_max:
            if self.sender() == self.conf_min_slider:
                self.conf_min_slider.setValue(int(conf_max * 100))
                conf_min = conf_max
            else:
                self.conf_max_slider.setValue(int(conf_min * 100))
                conf_max = conf_min

        self.conf_min_label.setText(f"{conf_min:.2f}")
        self.conf_max_label.setText(f"{conf_max:.2f}")
        self.update_plot()

    def on_duration_range_changed(self):
        """Handle duration range slider changes"""
        dur_min = self.dur_min_slider.value() / 200.0
        dur_max = self.dur_max_slider.value() / 200.0

        if dur_min > dur_max:
            if self.sender() == self.dur_min_slider:
                self.dur_min_slider.setValue(int(dur_max * 200))
                dur_min = dur_max
            else:
                self.dur_max_slider.setValue(int(dur_min * 200))
                dur_max = dur_min

        self.dur_min_label.setText(f"{dur_min:.3f}")
        self.dur_max_label.setText(f"{dur_max:.3f}")
        self.update_plot()

    def plot_data(self, error_data: Dict[str, np.ndarray], confidence_data: np.ndarray,
                  duration_data: np.ndarray, base_folder: Optional[str] = None,
                  match_dates: Optional[List[str]] = None, shot_types: Optional[np.ndarray] = None):
        """
        Plot error data as box plots

        Args:
            error_data: Dictionary with keys 'gt200', 'opt' containing error arrays
            confidence_data: Array of confidence values for filtering
            duration_data: Array of duration values for filtering
            base_folder: Base folder path for saving plots
            match_dates: List of date strings per segment (for date grouping)
            shot_types: Array of shot type ints per segment (0=serve, 1=rally, 2=last shot)
        """
        self.error_data = error_data
        self.confidence_data = confidence_data
        self.duration_data = duration_data
        self.base_folder = base_folder
        self.match_dates = match_dates
        self.shot_types = shot_types
        self.update_plot()

    def update_plot(self):
        """Update the plot based on current filters and group by selection"""
        self.plot_widget.clear()

        if self.error_data is None:
            self.info_label.setText("No data to display")
            return

        mask, suffix, metric_title, y_label, is_velocity_metric = self._build_filter_and_metric()

        group_by = self.group_combo.currentText()
        if group_by == "Date":
            self._do_update_by_date(mask, suffix, metric_title, y_label, is_velocity_metric)
        elif group_by == "Shot Type":
            self._do_update_by_shot_type(mask, suffix, metric_title, y_label, is_velocity_metric)
        else:
            self._do_update_by_source(mask, suffix, metric_title, y_label, is_velocity_metric)

    # ── Source-key mapping ──
    _SOURCE_KEYS = {'GT200': 'gt200', 'OPT': 'opt', '0226': '0226', '0426': '0426', 'NAK': 'nakashima'}
    _SHOT_TYPE_LABELS = {0: 'Serve', 1: 'Rally', 2: 'Last Shot'}

    def _build_filter_and_metric(self):
        """Build the common mask and determine which metric/suffix to use."""
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0
        dur_min = self.dur_min_slider.value() / 200.0
        dur_max = self.dur_max_slider.value() / 200.0

        mask = np.ones(len(self.confidence_data), dtype=bool)
        if self.confidence_data is not None:
            mask &= (self.confidence_data >= conf_min) & (self.confidence_data <= conf_max)
        if self.duration_data is not None:
            if dur_max >= 1.0:
                mask &= (self.duration_data >= dur_min)
            else:
                mask &= (self.duration_data >= dur_min) & (self.duration_data <= dur_max)

        # Apply OPT RMSE filter
        rmse_min_mm = self.rmse_min_slider.value()
        rmse_max_mm = self.rmse_slider.value()
        if rmse_min_mm > 0 or rmse_max_mm < self.rmse_slider.maximum():
            opt_rmse = self.error_data.get('opt', np.array([]))
            if len(opt_rmse) == len(mask):
                rmse_min_m = rmse_min_mm / 1000.0
                rmse_max_m = rmse_max_mm / 1000.0
                valid_rmse = ~np.isnan(opt_rmse)
                mask &= valid_rmse & (opt_rmse >= rmse_min_m) & (opt_rmse <= rmse_max_m)

        selected_metric = self.metric_combo.currentText()
        if selected_metric == "Max Position Error":
            suffix, metric_title, y_label = '_max', "Max Position Error", "Position Error (m)"
        elif selected_metric == "Mean Position Error":
            suffix, metric_title, y_label = '_mean', "Mean Position Error", "Position Error (m)"
        elif selected_metric == "Velocity RMSE":
            suffix, metric_title, y_label = '_vel', "Velocity RMSE", "Velocity Error (m/s)"
        else:
            suffix, metric_title, y_label = '', "Position RMSE", "Position Error (m)"

        is_velocity_metric = (suffix == '_vel')
        return mask, suffix, metric_title, y_label, is_velocity_metric

    def _get_source_errors(self, source_key, suffix, mask):
        """Get masked + NaN-filtered error array for a given source key and metric suffix."""
        arr = self.error_data.get(f'{source_key}{suffix}', self.error_data.get(source_key, np.array([])))
        if len(arr) == len(mask):
            arr = arr[mask]
        arr = arr[~np.isnan(arr)] if len(arr) > 0 else arr
        return arr

    def _draw_violin(self, x_center, errors, color, bar_width, is_velocity_metric, show_labels=True, label_text=None):
        """Draw a single violin plot at the given x position. Returns whisker_high or 0 if empty."""
        from scipy.stats import gaussian_kde

        if len(errors) == 0:
            return 0
        q1 = np.percentile(errors, 25)
        median = np.percentile(errors, 50)
        q3 = np.percentile(errors, 75)
        mean = np.mean(errors)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        within = errors[(errors >= fence_low) & (errors <= fence_high)]
        whisker_low = np.min(within) if len(within) > 0 else q1
        whisker_high = np.max(within) if len(within) > 0 else q3

        # Clip to 2nd–98th percentile for KDE (exclude outliers from shape)
        lo, hi = np.percentile(errors, [2, 98])
        clipped = errors[(errors >= lo) & (errors <= hi)]
        if len(clipped) < 2:
            clipped = errors

        hw = bar_width / 2.0

        # KDE on inlier data only
        try:
            kde = gaussian_kde(clipped, bw_method='scott')
        except (np.linalg.LinAlgError, ValueError):
            # Constant data — draw a simple line
            self.plot_widget.plot([x_center], [median], symbol='o',
                                  pen=None, symbolBrush=pg.mkBrush(*color))
            return whisker_high

        y_min, y_max = np.min(clipped), np.max(clipped)
        y_pad = (y_max - y_min) * 0.05 or 0.1
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 200)
        density = kde(y_grid)
        max_d = np.max(density) or 1.0
        density_scaled = density / max_d * hw

        # Mirrored violin body
        right_x = x_center + density_scaled
        left_x = x_center - density_scaled
        fill_item = pg.FillBetweenItem(
            pg.PlotCurveItem(left_x, y_grid),
            pg.PlotCurveItem(right_x, y_grid),
            brush=pg.mkBrush(*color, 80),
        )
        self.plot_widget.addItem(fill_item)
        # Outline
        self.plot_widget.plot(right_x, y_grid, pen=pg.mkPen(color=color, width=1.5))
        self.plot_widget.plot(left_x, y_grid, pen=pg.mkPen(color=color, width=1.5))

        # Inner quartile lines
        for val, w in [(q1, 1), (median, 2.5), (q3, 1)]:
            d_at = float(kde([val])[0]) / max_d * hw
            self.plot_widget.plot([x_center - d_at, x_center + d_at], [val, val],
                                  pen=pg.mkPen(color=color, width=w))

        # Mean diamond
        scatter = pg.ScatterPlotItem([x_center], [mean], symbol='d', size=10,
                                     pen=pg.mkPen(color='white', width=1),
                                     brush=pg.mkBrush(*color))
        self.plot_widget.addItem(scatter)

        # Quartile labels
        if show_labels:
            text_x = x_center + hw + 0.05
            if is_velocity_metric:
                q3_text = pg.TextItem(f'{q3:.2f}', color=color, anchor=(0, 0.5))
                median_text = pg.TextItem(f'{median:.2f}', color=color, anchor=(0, 0.5))
                q1_text = pg.TextItem(f'{q1:.2f}', color=color, anchor=(0, 0.5))
            else:
                q3_text = pg.TextItem(f'{q3*1000:.1f}', color=color, anchor=(0, 0.5))
                median_text = pg.TextItem(f'{median*1000:.1f}', color=color, anchor=(0, 0.5))
                q1_text = pg.TextItem(f'{q1*1000:.1f}', color=color, anchor=(0, 0.5))
            q3_text.setPos(text_x, q3)
            self.plot_widget.addItem(q3_text)
            median_text.setPos(text_x, median)
            self.plot_widget.addItem(median_text)
            q1_text.setPos(text_x, q1)
            self.plot_widget.addItem(q1_text)

        return whisker_high

    # ── Group By: Source (default) ──
    def _do_update_by_source(self, mask, suffix, metric_title, y_label, is_velocity_metric):
        self.plot_widget.setTitle(f"Aerodynamic Error Summary - {metric_title}")
        self.plot_widget.setLabel("left", y_label)

        all_colors = [(66, 133, 244), (52, 168, 83), (255, 0, 255), (255, 140, 0), (255, 127, 14)]
        all_labels = ['GT200', 'OPT', '0226', 'NAK', '0426']
        all_source_keys = ['gt200', 'opt', '0226', 'nakashima', '0426']

        # Filter to only selected sources
        selected = [(key, label, color) for key, label, color
                     in zip(all_source_keys, all_labels, all_colors)
                     if self.source_checkboxes.get(label, None) is None or self.source_checkboxes[label].isChecked()]

        n_total = int(np.sum(mask))
        bar_width = 0.6
        sources_info = []
        tick_labels = []
        max_whisker = 0
        _pub = self.publication_checkbox.isChecked()

        for i, (key, label, color) in enumerate(selected):
            errors = self._get_source_errors(key, suffix, mask)
            wh = self._draw_violin(i, errors, color, bar_width, is_velocity_metric,
                                   show_labels=not _pub)
            max_whisker = max(max_whisker, wh)
            sources_info.append(f"{label}: {len(errors)}")
            tick_labels.append((i, label))

        axis = self.plot_widget.getAxis('bottom')
        axis.setTicks([tick_labels])

        if max_whisker > 0:
            self.plot_widget.setYRange(0, max_whisker * 1.1)

        self.info_label.setText(f"Showing {n_total} trajectories. {', '.join(sources_info)}")

    # ── Group By: Date ──
    def _do_update_by_date(self, mask, suffix, metric_title, y_label, is_velocity_metric):
        if self.match_dates is None:
            self.info_label.setText("No date metadata available")
            return

        source_name = self.source_combo.currentText()
        source_key = self._SOURCE_KEYS.get(source_name, 'opt')

        # Get per-segment errors (before NaN removal — need indices aligned with mask)
        raw_errors = self.error_data.get(f'{source_key}{suffix}', self.error_data.get(source_key, np.array([])))
        if len(raw_errors) != len(mask):
            self.info_label.setText("Data length mismatch")
            return

        dates_arr = np.array(self.match_dates)
        masked_errors = raw_errors[mask]
        masked_dates = dates_arr[mask]

        # Get unique dates sorted
        unique_dates = sorted(set(masked_dates))
        if len(unique_dates) == 0:
            self.info_label.setText("No data after filtering")
            return

        # Format date labels: YYYYMMDD → YYYY-MM-DD
        def _fmt_date(d):
            s = str(d)
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else s

        self.plot_widget.setTitle(f"Aero Error by Date - {source_name} - {metric_title}")
        self.plot_widget.setLabel("left", y_label)

        n_groups = len(unique_dates)
        bar_width = max(0.3, min(0.6, 8.0 / n_groups))
        tick_labels = []
        max_whisker = 0
        _pub = self.publication_checkbox.isChecked()

        # Generate colors using HSV
        for i, date_val in enumerate(unique_dates):
            hue = int(255 * i / max(n_groups, 1))
            color = pg.hsvColor(hue / 360.0, 0.7, 0.9).getRgb()[:3]
            group_mask = masked_dates == date_val
            errors = masked_errors[group_mask]
            errors = errors[~np.isnan(errors)]
            wh = self._draw_violin(i, errors, color, bar_width, is_velocity_metric,
                                   show_labels=(not _pub and n_groups <= 10))
            max_whisker = max(max_whisker, wh)
            tick_labels.append((i, f"{_fmt_date(date_val)}\nn={len(errors)}"))

        axis = self.plot_widget.getAxis('bottom')
        axis.setTicks([tick_labels])

        if max_whisker > 0:
            self.plot_widget.setYRange(0, max_whisker * 1.1)

        n_total = int(np.sum(mask))
        self.info_label.setText(f"Showing {n_total} trajectories across {n_groups} dates ({source_name})")

    # ── Group By: Shot Type ──
    def _do_update_by_shot_type(self, mask, suffix, metric_title, y_label, is_velocity_metric):
        if self.shot_types is None:
            self.info_label.setText("No shot type metadata available")
            return

        source_name = self.source_combo.currentText()
        source_key = self._SOURCE_KEYS.get(source_name, 'opt')

        raw_errors = self.error_data.get(f'{source_key}{suffix}', self.error_data.get(source_key, np.array([])))
        if len(raw_errors) != len(mask):
            self.info_label.setText("Data length mismatch")
            return

        masked_errors = raw_errors[mask]
        masked_types = self.shot_types[mask] if len(self.shot_types) == len(mask) else self.shot_types

        self.plot_widget.setTitle(f"Aero Error by Shot Type - {source_name} - {metric_title}")
        self.plot_widget.setLabel("left", y_label)

        colors = [(66, 133, 244), (52, 168, 83), (234, 67, 53)]  # Blue, Green, Red
        type_order = [0, 1, 2]  # Serve, Rally, Last Shot
        bar_width = 0.6
        tick_labels = []
        max_whisker = 0
        _pub = self.publication_checkbox.isChecked()

        for i, st in enumerate(type_order):
            label = self._SHOT_TYPE_LABELS[st]
            color = colors[i]
            group_mask = masked_types == st
            errors = masked_errors[group_mask]
            errors = errors[~np.isnan(errors)]
            wh = self._draw_violin(i, errors, color, bar_width, is_velocity_metric,
                                   show_labels=not _pub)
            max_whisker = max(max_whisker, wh)
            tick_labels.append((i, f"{label}\nn={len(errors)}"))

        axis = self.plot_widget.getAxis('bottom')
        axis.setTicks([tick_labels])

        if max_whisker > 0:
            self.plot_widget.setYRange(0, max_whisker * 1.1)

        n_total = int(np.sum(mask))
        self.info_label.setText(f"Showing {n_total} trajectories by shot type ({source_name})")

    def save_plot(self):
        """Save the current plot as a PNG image"""
        if self.error_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aero_summary_error_{timestamp}.png"

        output_path = plots_dir / filename

        try:
            publication = self.publication_checkbox.isChecked()
            save_plot_as_image(self.plot_widget, str(output_path),
                               publication_mode=publication,
                               hide_x_labels=publication)
            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
