# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Epsilon vs Vz Plot Widget

Shows the relationship between measured epsilon (coefficient of restitution)
and pre-contact vertical velocity (vz_pre).

epsilon_opt = |vz_post / vz_pre| (measured from data)
epsilon_model = vz_pre * 0.02 + 0.98 (linear model)
"""

from typing import Dict, Any, Optional
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton,
    QMessageBox, QSlider, QCheckBox
)
from PySide6.QtCore import Signal, Qt, QTimer

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from .base_coefficient_plot import create_range_slider, create_combo_selector, save_plot_as_image
from .table_contact_plot import contact_model_residual_vectorized


class EpsilonVsVzPlot(QWidget):
    """Widget for displaying epsilon (coefficient of restitution) vs vz_pre scatter plot.

    Shows:
    - epsilon_opt (measured): |vz_post / vz_pre|
    - epsilon_model (linear): vz_pre * 0.02 + 0.98

    Points can be colored by spin magnitude or other properties.
    """

    # Signal emitted when a contact point is clicked
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

        # Color by selector
        color_layout, self.color_combo = create_combo_selector(
            "Color By:",
            ["Spin Magnitude", "vx_pre", "vy_pre", "None"],
            self.update_plot
        )
        control_bar.addLayout(color_layout)

        # vz_pre range slider (typical range: 0 to 6 m/s, using absolute value)
        vz_layout = QVBoxLayout()
        vz_label = QLabel("|vz_pre| Range (m/s):")
        vz_layout.addWidget(vz_label)

        vz_min_layout = QHBoxLayout()
        vz_min_layout.addWidget(QLabel("Min:"))
        self.vz_min_slider = QSlider(Qt.Horizontal)
        self.vz_min_slider.setMinimum(0)
        self.vz_min_slider.setMaximum(80)  # 0 to 8.0 m/s
        self.vz_min_slider.setValue(0)  # 0.0 m/s
        self.vz_min_slider.setTickPosition(QSlider.TicksBelow)
        self.vz_min_slider.setTickInterval(10)
        self.vz_min_slider.valueChanged.connect(self._schedule_update)
        vz_min_layout.addWidget(self.vz_min_slider)
        self.vz_min_label = QLabel("0.0")
        self.vz_min_label.setMinimumWidth(35)
        vz_min_layout.addWidget(self.vz_min_label)
        vz_layout.addLayout(vz_min_layout)

        vz_max_layout = QHBoxLayout()
        vz_max_layout.addWidget(QLabel("Max:"))
        self.vz_max_slider = QSlider(Qt.Horizontal)
        self.vz_max_slider.setMinimum(0)
        self.vz_max_slider.setMaximum(80)
        self.vz_max_slider.setValue(60)  # 6.0 m/s
        self.vz_max_slider.setTickPosition(QSlider.TicksBelow)
        self.vz_max_slider.setTickInterval(10)
        self.vz_max_slider.valueChanged.connect(self._schedule_update)
        vz_max_layout.addWidget(self.vz_max_slider)
        self.vz_max_label = QLabel("6.0")
        self.vz_max_label.setMinimumWidth(35)
        vz_max_layout.addWidget(self.vz_max_label)
        vz_layout.addLayout(vz_max_layout)

        control_bar.addLayout(vz_layout)

        # Epsilon range slider (typical range: 0.5 to 1.2)
        eps_layout = QVBoxLayout()
        eps_label = QLabel("Epsilon Range:")
        eps_layout.addWidget(eps_label)

        eps_min_layout = QHBoxLayout()
        eps_min_layout.addWidget(QLabel("Min:"))
        self.eps_min_slider = QSlider(Qt.Horizontal)
        self.eps_min_slider.setMinimum(0)
        self.eps_min_slider.setMaximum(200)  # 0.0 to 2.0
        self.eps_min_slider.setValue(50)  # 0.5
        self.eps_min_slider.setTickPosition(QSlider.TicksBelow)
        self.eps_min_slider.setTickInterval(20)
        self.eps_min_slider.valueChanged.connect(self._schedule_update)
        eps_min_layout.addWidget(self.eps_min_slider)
        self.eps_min_label = QLabel("0.50")
        self.eps_min_label.setMinimumWidth(35)
        eps_min_layout.addWidget(self.eps_min_label)
        eps_layout.addLayout(eps_min_layout)

        eps_max_layout = QHBoxLayout()
        eps_max_layout.addWidget(QLabel("Max:"))
        self.eps_max_slider = QSlider(Qt.Horizontal)
        self.eps_max_slider.setMinimum(0)
        self.eps_max_slider.setMaximum(200)
        self.eps_max_slider.setValue(120)  # 1.2
        self.eps_max_slider.setTickPosition(QSlider.TicksBelow)
        self.eps_max_slider.setTickInterval(20)
        self.eps_max_slider.valueChanged.connect(self._schedule_update)
        eps_max_layout.addWidget(self.eps_max_slider)
        self.eps_max_label = QLabel("1.20")
        self.eps_max_label.setMinimumWidth(35)
        eps_max_layout.addWidget(self.eps_max_label)
        eps_layout.addLayout(eps_max_layout)

        control_bar.addLayout(eps_layout)

        # Spin magnitude filter (0 to 300 rad/s)
        spin_layout = QVBoxLayout()
        spin_label = QLabel("Spin Range (rad/s):")
        spin_layout.addWidget(spin_label)

        spin_min_layout = QHBoxLayout()
        spin_min_layout.addWidget(QLabel("Min:"))
        self.spin_min_slider = QSlider(Qt.Horizontal)
        self.spin_min_slider.setMinimum(0)
        self.spin_min_slider.setMaximum(300)
        self.spin_min_slider.setValue(0)
        self.spin_min_slider.setTickPosition(QSlider.TicksBelow)
        self.spin_min_slider.setTickInterval(50)
        self.spin_min_slider.valueChanged.connect(self._schedule_update)
        spin_min_layout.addWidget(self.spin_min_slider)
        self.spin_min_label = QLabel("0")
        self.spin_min_label.setMinimumWidth(30)
        spin_min_layout.addWidget(self.spin_min_label)
        spin_layout.addLayout(spin_min_layout)

        spin_max_layout = QHBoxLayout()
        spin_max_layout.addWidget(QLabel("Max:"))
        self.spin_max_slider = QSlider(Qt.Horizontal)
        self.spin_max_slider.setMinimum(0)
        self.spin_max_slider.setMaximum(300)
        self.spin_max_slider.setValue(300)
        self.spin_max_slider.setTickPosition(QSlider.TicksBelow)
        self.spin_max_slider.setTickInterval(50)
        self.spin_max_slider.valueChanged.connect(self._schedule_update)
        spin_max_layout.addWidget(self.spin_max_slider)
        self.spin_max_label = QLabel("300")
        self.spin_max_label.setMinimumWidth(30)
        spin_max_layout.addWidget(self.spin_max_label)
        spin_layout.addLayout(spin_max_layout)

        control_bar.addLayout(spin_layout)

        # Confidence filter (0 to 100%)
        self._conf_layout = QVBoxLayout()
        conf_label = QLabel("Confidence (%):")
        self._conf_layout.addWidget(conf_label)

        conf_min_layout = QHBoxLayout()
        conf_min_layout.addWidget(QLabel("Min:"))
        self.conf_min_slider = QSlider(Qt.Horizontal)
        self.conf_min_slider.setMinimum(0)
        self.conf_min_slider.setMaximum(100)
        self.conf_min_slider.setValue(50)
        self.conf_min_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_min_slider.setTickInterval(10)
        self.conf_min_slider.valueChanged.connect(self._schedule_update)
        conf_min_layout.addWidget(self.conf_min_slider)
        self.conf_min_label = QLabel("50")
        self.conf_min_label.setMinimumWidth(25)
        conf_min_layout.addWidget(self.conf_min_label)
        self._conf_layout.addLayout(conf_min_layout)

        conf_max_layout = QHBoxLayout()
        conf_max_layout.addWidget(QLabel("Max:"))
        self.conf_max_slider = QSlider(Qt.Horizontal)
        self.conf_max_slider.setMinimum(0)
        self.conf_max_slider.setMaximum(100)
        self.conf_max_slider.setValue(100)
        self.conf_max_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_max_slider.setTickInterval(10)
        self.conf_max_slider.valueChanged.connect(self._schedule_update)
        conf_max_layout.addWidget(self.conf_max_slider)
        self.conf_max_label = QLabel("100")
        self.conf_max_label.setMinimumWidth(25)
        conf_max_layout.addWidget(self.conf_max_label)
        self._conf_layout.addLayout(conf_max_layout)

        control_bar.addLayout(self._conf_layout)

        # OPT Error filter (0 to 100 mm)
        self._error_layout = QVBoxLayout()
        opt_err_label = QLabel("OPT Error (mm):")
        self._error_layout.addWidget(opt_err_label)

        opt_err_min_layout = QHBoxLayout()
        opt_err_min_layout.addWidget(QLabel("Min:"))
        self.opt_err_min_slider = QSlider(Qt.Horizontal)
        self.opt_err_min_slider.setMinimum(0)
        self.opt_err_min_slider.setMaximum(100)
        self.opt_err_min_slider.setValue(0)
        self.opt_err_min_slider.setTickPosition(QSlider.TicksBelow)
        self.opt_err_min_slider.setTickInterval(10)
        self.opt_err_min_slider.valueChanged.connect(self._schedule_update)
        opt_err_min_layout.addWidget(self.opt_err_min_slider)
        self.opt_err_min_label = QLabel("0")
        self.opt_err_min_label.setMinimumWidth(25)
        opt_err_min_layout.addWidget(self.opt_err_min_label)
        self._error_layout.addLayout(opt_err_min_layout)

        opt_err_max_layout = QHBoxLayout()
        opt_err_max_layout.addWidget(QLabel("Max:"))
        self.opt_err_max_slider = QSlider(Qt.Horizontal)
        self.opt_err_max_slider.setMinimum(0)
        self.opt_err_max_slider.setMaximum(100)
        self.opt_err_max_slider.setValue(100)
        self.opt_err_max_slider.setTickPosition(QSlider.TicksBelow)
        self.opt_err_max_slider.setTickInterval(10)
        self.opt_err_max_slider.valueChanged.connect(self._schedule_update)
        opt_err_max_layout.addWidget(self.opt_err_max_slider)
        self.opt_err_max_label = QLabel("100")
        self.opt_err_max_label.setMinimumWidth(25)
        opt_err_max_layout.addWidget(self.opt_err_max_label)
        self._error_layout.addLayout(opt_err_max_layout)

        control_bar.addLayout(self._error_layout)

        # Show model line checkboxes
        model_cb_layout = QVBoxLayout()
        self.show_model_checkbox = QCheckBox("Linear Model")
        self.show_model_checkbox.setChecked(True)
        self.show_model_checkbox.stateChanged.connect(self.update_plot)
        model_cb_layout.addWidget(self.show_model_checkbox)

        self.show_ittf_checkbox = QCheckBox("NakashimaITTF Model")
        self.show_ittf_checkbox.setChecked(False)
        self.show_ittf_checkbox.stateChanged.connect(self.update_plot)
        model_cb_layout.addWidget(self.show_ittf_checkbox)

        self.show_paper_checkbox = QCheckBox("NakashimaPaper Model")
        self.show_paper_checkbox.setChecked(False)
        self.show_paper_checkbox.stateChanged.connect(self.update_plot)
        model_cb_layout.addWidget(self.show_paper_checkbox)

        self.show_residual_checkbox = QCheckBox("Residual Model")
        self.show_residual_checkbox.setChecked(False)
        self.show_residual_checkbox.stateChanged.connect(self.update_plot)
        model_cb_layout.addWidget(self.show_residual_checkbox)

        self.show_0426_checkbox = QCheckBox("Nature (0426)")
        self.show_0426_checkbox.setChecked(False)
        self.show_0426_checkbox.stateChanged.connect(self.update_plot)
        model_cb_layout.addWidget(self.show_0426_checkbox)
        control_bar.addLayout(model_cb_layout)

        # Point size slider
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(2)
        self.size_slider.setMaximum(15)
        self.size_slider.setValue(6)
        self.size_slider.valueChanged.connect(self._schedule_update)
        size_layout.addWidget(self.size_slider)
        self.size_label = QLabel("6")
        self.size_label.setMinimumWidth(20)
        size_layout.addWidget(self.size_label)
        control_bar.addLayout(size_layout)

        # Info label (stretch to fill space)
        self.info_label = QLabel("Epsilon vs Vz Analysis")
        control_bar.addWidget(self.info_label, stretch=1)

        # Publication mode checkbox
        self.publication_checkbox = QCheckBox("Publication")
        control_bar.addWidget(self.publication_checkbox)

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Create plot widget with colorbar
        plot_container = QHBoxLayout()

        self.plot_widget = pg.PlotWidget(title="Coefficient of Restitution vs Pre-Contact Velocity")
        self.plot_widget.setLabel('left', 'ε (Coefficient of Restitution)')
        self.plot_widget.setLabel('bottom', '|vz_pre| (m/s)')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)
        plot_container.addWidget(self.plot_widget)

        # Create colorbar widget
        self.colorbar_widget = pg.GraphicsLayoutWidget()
        self.colorbar_widget.setMaximumWidth(100)
        self.colorbar_widget.setMinimumWidth(80)
        plot_container.addWidget(self.colorbar_widget)

        # Container for plot + colorbar
        plot_widget_container = QWidget()
        plot_widget_container.setLayout(plot_container)
        layout.addWidget(plot_widget_container, stretch=1)

        # Statistics label
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

        # Initialize colorbar reference
        self.colorbar = None

        # Store data
        self.contact_data = None  # Dict with vx_pre, vy_pre, vz_pre, etc.
        self.base_folder = None

        # Scatter items for click handling
        self.scatter_item = None
        self.model_line = None

    def _schedule_update(self):
        """Schedule a debounced update"""
        self._update_timer.start()

    def plot_data(self, contact_data: Dict[str, np.ndarray], base_folder: Optional[str] = None):
        """
        Plot epsilon vs vz data.

        Pre-computes derived quantities (spin_magnitude, epsilon_opt, model
        predictions) once so that update_plot() only needs to filter and render.
        """
        self.contact_data = contact_data
        self.base_folder = base_folder

        # ── Pre-compute all data-dependent quantities once ──
        self._precomputed = None
        if contact_data is not None:
            vz_pre_raw = np.asarray(contact_data.get('vz_pre', []))
            if len(vz_pre_raw) > 0:
                vz_post = np.asarray(contact_data.get('vz_post', []))
                vx_pre = np.asarray(contact_data.get('vx_pre', []))
                vy_pre = np.asarray(contact_data.get('vy_pre', []))
                wx_pre = np.asarray(contact_data.get('wx_pre', []))
                wy_pre = np.asarray(contact_data.get('wy_pre', []))
                wz_pre = np.asarray(contact_data.get('wz_pre', []))

                vz_abs = np.abs(vz_pre_raw)
                spin_magnitude = np.sqrt(wx_pre**2 + wy_pre**2 + wz_pre**2)

                valid = vz_abs > 0.01
                epsilon_opt = np.full_like(vz_abs, np.nan)
                epsilon_opt[valid] = np.abs(vz_post[valid] / vz_pre_raw[valid])

                # Pre-compute model epsilon values (avoids re-running models on each slider change)
                epsilon_model = vz_pre_raw * 0.02 + 0.98

                # NakashimaITTF and NakashimaPaper from HDF5 data
                eps_ittf = np.full_like(vz_abs, np.nan)
                vz_post_ittf = np.asarray(contact_data.get('vz_post_ittf', []))
                if len(vz_post_ittf) == len(vz_pre_raw):
                    eps_ittf[valid] = np.abs(vz_post_ittf[valid] / vz_pre_raw[valid])

                eps_paper = np.full_like(vz_abs, np.nan)
                vz_post_paper = np.asarray(contact_data.get('vz_post_paper', []))
                if len(vz_post_paper) == len(vz_pre_raw):
                    eps_paper[valid] = np.abs(vz_post_paper[valid] / vz_pre_raw[valid])

                # Residual model (expensive O(n) Python loop — compute ONCE here)
                _, _, vz_post_res, _, _, _ = contact_model_residual_vectorized(
                    vx_pre, vy_pre, vz_pre_raw, wx_pre, wy_pre, wz_pre, epsilon_model
                )
                eps_residual = np.full_like(vz_abs, np.nan)
                eps_residual[valid] = np.abs(vz_post_res[valid] / vz_pre_raw[valid])

                # 0426 model (from HDF5)
                eps_0426 = np.full_like(vz_abs, np.nan)
                vz_post_0426 = np.asarray(contact_data.get('vz_post_0426', []))
                if len(vz_post_0426) == len(vz_pre_raw):
                    eps_0426[valid] = np.abs(vz_post_0426[valid] / vz_pre_raw[valid])

                self._precomputed = {
                    'vz_abs': vz_abs,
                    'spin_magnitude': spin_magnitude,
                    'epsilon_opt': epsilon_opt,
                    'valid': valid,
                    'eps_ittf': eps_ittf,
                    'eps_paper': eps_paper,
                    'eps_residual': eps_residual,
                    'eps_0426': eps_0426,
                }

        self.update_plot()

    def update_plot(self):
        """Update the plot based on current filters"""
        # Clear existing legend first (before plot_widget.clear())
        try:
            legend = self.plot_widget.plotItem.legend
            if legend is not None:
                legend.clear()
                legend.setParentItem(None)
                self.plot_widget.plotItem.legend = None
        except Exception:
            pass

        self.plot_widget.clear()
        self.colorbar_widget.clear()
        self.colorbar = None

        # Clear filtered indices
        self._filtered_indices = None

        if self.contact_data is None:
            self.info_label.setText("No data to display")
            return

        # Update slider labels
        vz_min = self.vz_min_slider.value() / 10.0
        vz_max = self.vz_max_slider.value() / 10.0
        eps_min = self.eps_min_slider.value() / 100.0
        eps_max = self.eps_max_slider.value() / 100.0
        spin_min = self.spin_min_slider.value()
        spin_max = self.spin_max_slider.value()
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0
        opt_err_min = self.opt_err_min_slider.value() / 1000.0  # Convert mm to m
        opt_err_max = self.opt_err_max_slider.value() / 1000.0  # Convert mm to m
        point_size = self.size_slider.value()

        self.vz_min_label.setText(f"{vz_min:.1f}")
        self.vz_max_label.setText(f"{vz_max:.1f}")
        self.eps_min_label.setText(f"{eps_min:.2f}")
        self.eps_max_label.setText(f"{eps_max:.2f}")
        self.spin_min_label.setText(str(spin_min))
        self.spin_max_label.setText(str(spin_max))
        self.conf_min_label.setText(str(int(conf_min * 100)))
        self.conf_max_label.setText(str(int(conf_max * 100)))
        self.opt_err_min_label.setText(str(self.opt_err_min_slider.value()))
        self.opt_err_max_label.setText(str(self.opt_err_max_slider.value()))
        self.size_label.setText(str(point_size))

        # Use pre-computed data (computed once in plot_data)
        pc = self._precomputed
        if pc is None:
            self.info_label.setText("No contact data available")
            return

        vz_pre = pc['vz_abs']
        spin_magnitude = pc['spin_magnitude']
        epsilon_opt = pc['epsilon_opt']
        valid = pc['valid']
        confidence = np.asarray(self.contact_data.get('confidence', np.ones(len(vz_pre))))
        max_rmse_opt = np.asarray(self.contact_data.get('max_rmse_opt', np.zeros(len(vz_pre))))

        # Apply filters (now using positive vz values)
        mask = valid.copy()
        mask &= (vz_pre >= vz_min) & (vz_pre <= vz_max)
        mask &= (epsilon_opt >= eps_min) & (epsilon_opt <= eps_max)
        mask &= (spin_magnitude >= spin_min) & (spin_magnitude <= spin_max)
        mask &= (confidence >= conf_min) & (confidence <= conf_max)
        mask &= (max_rmse_opt >= opt_err_min) & (max_rmse_opt <= opt_err_max)

        if not np.any(mask):
            self.info_label.setText("No data points in selected range")
            return

        # Store filtered indices for click handling
        self._filtered_indices = np.where(mask)[0]

        # Filtered data
        vz_filtered = vz_pre[mask]
        eps_filtered = epsilon_opt[mask]

        # Compute color values
        color_by = self.color_combo.currentText()
        if color_by == "Spin Magnitude":
            color_values = spin_magnitude[mask]
            color_label = "Spin (rad/s)"
        elif color_by == "vx_pre":
            color_values = np.asarray(self.contact_data['vx_pre'])[mask]
            color_label = "vx_pre (m/s)"
        elif color_by == "vy_pre":
            color_values = np.asarray(self.contact_data['vy_pre'])[mask]
            color_label = "vy_pre (m/s)"
        else:  # None
            color_values = None
            color_label = None

        # Create scatter plot
        if color_values is not None:
            # Normalize color values for colormap
            c_min, c_max = np.nanmin(color_values), np.nanmax(color_values)
            if c_max - c_min < 1e-9:
                c_max = c_min + 1
            c_norm = (color_values - c_min) / (c_max - c_min)

            # Create colormap (viridis-like)
            cmap = pg.colormap.get('viridis')
            colors = cmap.map(c_norm, mode='qcolor')

            # Create brushes
            brushes = [pg.mkBrush(c) for c in colors]

            self.scatter_item = pg.ScatterPlotItem(
                x=vz_filtered,
                y=eps_filtered,
                size=point_size,
                brush=brushes,
                pen=pg.mkPen(None),
                symbol='o'
            )

            # Add colorbar
            self._add_colorbar(cmap, c_min, c_max, color_label)
        else:
            # Single color (blue)
            self.scatter_item = pg.ScatterPlotItem(
                x=vz_filtered,
                y=eps_filtered,
                size=point_size,
                brush=pg.mkBrush(66, 133, 244, 180),
                pen=pg.mkPen(None),
                symbol='o'
            )

        self.plot_widget.addItem(self.scatter_item)

        # Connect click handler
        self.scatter_item.sigClicked.connect(self._on_point_clicked)

        # Add legend (before model lines so all entries are captured)
        has_any_model = (self.show_model_checkbox.isChecked() or
                         self.show_ittf_checkbox.isChecked() or
                         self.show_paper_checkbox.isChecked() or
                         self.show_residual_checkbox.isChecked() or
                         self.show_0426_checkbox.isChecked())
        if has_any_model:
            self.plot_widget.addLegend(offset=(10, 10))

        # Show linear model line if checkbox is checked
        if self.show_model_checkbox.isChecked():
            # Original model: epsilon = vz_pre * 0.02 + 0.98 (with negative vz_pre)
            # With |vz_pre|: epsilon = -|vz_pre| * 0.02 + 0.98 = 0.98 - 0.02*|vz|
            vz_range = np.linspace(vz_min, vz_max, 100)
            eps_model = 0.98 - 0.02 * vz_range

            self.model_line = self.plot_widget.plot(
                vz_range, eps_model,
                pen=pg.mkPen(color=(255, 100, 100), width=2, style=Qt.DashLine),
                name='Linear: ε = 0.98 - 0.02·|vz|'
            )

        # Show NakashimaITTF model predictions as scatter overlay (from HDF5)
        if self.show_ittf_checkbox.isChecked():
            eps_ittf = pc['eps_ittf'][mask]
            self.plot_widget.plot(
                vz_filtered, eps_ittf,
                pen=None, symbol='o', symbolSize=max(point_size - 1, 2),
                symbolBrush=pg.mkBrush(52, 168, 83, 140),
                symbolPen=pg.mkPen(None),
                name='NakashimaITTF (ε=0.876)'
            )

        # Show NakashimaPaper model predictions as scatter overlay (from HDF5)
        if self.show_paper_checkbox.isChecked():
            eps_paper = pc['eps_paper'][mask]
            self.plot_widget.plot(
                vz_filtered, eps_paper,
                pen=None, symbol='o', symbolSize=max(point_size - 1, 2),
                symbolBrush=pg.mkBrush(255, 165, 0, 140),
                symbolPen=pg.mkPen(None),
                name='NakashimaPaper (ε=0.93)'
            )

        # Show Residual model predictions as scatter overlay (pre-computed)
        if self.show_residual_checkbox.isChecked():
            eps_res = pc['eps_residual'][mask]
            self.plot_widget.plot(
                vz_filtered, eps_res,
                pen=None, symbol='o', symbolSize=max(point_size - 1, 2),
                symbolBrush=pg.mkBrush(0, 128, 255, 140),
                symbolPen=pg.mkPen(None),
                name='Residual Model'
            )

        # Show 0426 (Nature) model predictions as scatter overlay (from HDF5)
        if self.show_0426_checkbox.isChecked():
            eps_0426 = pc['eps_0426'][mask]
            self.plot_widget.plot(
                vz_filtered, eps_0426,
                pen=None, symbol='o', symbolSize=max(point_size - 1, 2),
                symbolBrush=pg.mkBrush(255, 127, 14, 140),
                symbolPen=pg.mkPen(None),
                name='Nature (0426)'
            )

        # Set axis ranges
        self.plot_widget.setXRange(vz_min, vz_max)
        self.plot_widget.setYRange(eps_min, eps_max)

        # Compute statistics
        n_points = len(eps_filtered)
        eps_mean = np.nanmean(eps_filtered)
        eps_std = np.nanstd(eps_filtered)
        eps_median = np.nanmedian(eps_filtered)

        # Compute model prediction error
        # Model: epsilon = 0.98 - 0.02*|vz|
        eps_model_pred = 0.98 - 0.02 * vz_filtered
        model_rmse = np.sqrt(np.nanmean((eps_filtered - eps_model_pred)**2))

        self.info_label.setText(f"Showing {n_points} contacts")
        self.stats_label.setText(
            f"ε_opt: mean={eps_mean:.3f}, std={eps_std:.3f}, median={eps_median:.3f} | "
            f"Linear Model RMSE: {model_rmse:.3f}"
        )

    def _add_colorbar(self, cmap, c_min, c_max, label):
        """Add a colorbar to the plot"""
        # Create colorbar image
        colorbar_data = np.linspace(0, 1, 256).reshape(256, 1)

        # Add view box and image item
        cb_viewbox = self.colorbar_widget.addViewBox(row=0, col=0)
        cb_viewbox.setMouseEnabled(x=False, y=False)

        cb_image = pg.ImageItem(colorbar_data.T)
        cb_image.setLookupTable(cmap.getLookupTable(nPts=256))
        cb_viewbox.addItem(cb_image)
        cb_viewbox.setAspectLocked(False)

        # Add axis with labels
        axis = pg.AxisItem('right')
        axis.setRange(c_min, c_max)
        self.colorbar_widget.addItem(axis, row=0, col=1)

        # Add label
        label_item = pg.LabelItem(label, angle=-90)
        self.colorbar_widget.addItem(label_item, row=0, col=2)

    def _on_point_clicked(self, scatter_item, points):
        """Handle click on scatter point"""
        if len(points) == 0:
            return

        # Get index of clicked point (this is the index in the filtered data)
        point = points[0]
        filtered_idx = point.index()

        # Map back to original data index
        if self._filtered_indices is None or filtered_idx >= len(self._filtered_indices):
            return
        idx = self._filtered_indices[filtered_idx]

        # Get metadata if available
        metadata_list = self.contact_data.get('metadata_list', [])
        shot_list = self.contact_data.get('shot_list', [])
        rally_list = self.contact_data.get('rally_list', [])

        if idx < len(metadata_list):
            metadata = metadata_list[idx]
            shot = shot_list[idx] if idx < len(shot_list) else None
            rally = rally_list[idx] if idx < len(rally_list) else None

            # Get flight segments for trajectory display
            fs_pre_list = self.contact_data.get('fs_pre_list', [])
            fs_post_list = self.contact_data.get('fs_post_list', [])
            fs_pre = fs_pre_list[idx] if idx < len(fs_pre_list) else None
            fs_post = fs_post_list[idx] if idx < len(fs_post_list) else None

            # Compute spin magnitude for info display
            wx = self.contact_data.get('wx_pre', np.zeros(1))[idx]
            wy = self.contact_data.get('wy_pre', np.zeros(1))[idx]
            wz = self.contact_data.get('wz_pre', np.zeros(1))[idx]
            spin_mag = np.sqrt(wx**2 + wy**2 + wz**2)

            # Create contact data dict for this point (must include fs_pre/fs_post for trajectory display)
            contact_info = {
                'vz_pre': self.contact_data['vz_pre'][idx],
                'vz_post': self.contact_data['vz_post'][idx],
                'vx_pre': self.contact_data.get('vx_pre', np.zeros(1))[idx],
                'vy_pre': self.contact_data.get('vy_pre', np.zeros(1))[idx],
                'spin_magnitude': spin_mag,
                'epsilon_opt': np.abs(self.contact_data['vz_post'][idx] / self.contact_data['vz_pre'][idx]),
                'fs_pre': fs_pre,
                'fs_post': fs_post,
            }

            self.contact_selected.emit(contact_info, metadata, shot, rally)

    def save_plot(self):
        """Save the current plot as a PNG image"""
        if self.contact_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epsilon_vs_vz_{timestamp}.png"
        output_path = plots_dir / filename

        try:
            publication = self.publication_checkbox.isChecked()
            save_plot_as_image(self.plot_widget, str(output_path), publication_mode=publication)
            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
