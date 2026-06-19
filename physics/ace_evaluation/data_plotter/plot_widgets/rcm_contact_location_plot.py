# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
RCM Contact Location Plot Widget

Corner-plot (lower-left triangle + diagonal marginals) of dx, dy, dz
(pre and post) from the racket contact model data.

* Lower-left cells: scatter plots (pre = circle, post = triangle).
* Diagonal cells: marginal distribution histograms per shot type class.
* Upper-right cells: hidden.

Points / histograms are coloured by shot type:
  serve (green), rally (blue), last shot (red).
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QSlider, QGridLayout, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer, Signal

import pyqtgraph as pg
import matplotlib.pyplot as plt

from .base_coefficient_plot import create_range_slider, create_combo_selector


# ── Constants ────────────────────────────────────────────────────────────

_AXES = ['dx', 'dy', 'dz']
_AXES_DISPLAY = ['\u0394x', '\u0394y', '\u0394z']
_N = len(_AXES)  # 3

_SHOT_TYPE_COLORS = {
    0: (50, 180, 50, 180),    # serve - green
    1: (74, 144, 226, 180),   # rally - blue
    2: (220, 60, 60, 180),    # last shot - red
}
_SHOT_TYPE_LABELS = {0: "Serve", 1: "Rally", 2: "Last Shot"}
_SHOT_TYPE_PEN = {
    0: pg.mkPen(color=(50, 180, 50), width=2),
    1: pg.mkPen(color=(74, 144, 226), width=2),
    2: pg.mkPen(color=(220, 60, 60), width=2),
}
_SHOT_TYPE_BRUSH = {
    0: pg.mkBrush(50, 180, 50, 50),
    1: pg.mkBrush(74, 144, 226, 50),
    2: pg.mkBrush(220, 60, 60, 50),
}

# pyqtgraph symbol strings
_SYMBOL_PRE = 'o'   # circle  for pre
_SYMBOL_POST = 't'  # triangle for post

_N_BINS = 20  # histogram bins for marginals

# Model suffixes for velocity error computation
_MODEL_SUFFIXES = {
    'None': None,
    'Default (constant COR)': 'default',
    'RCM Tangential': 'rcm_tangential',
    'C++ tangential': 'cpp_tangential',
    'Nakashima (refined)': 'nakashima_refined',
    'C++ no-residual (refined)': 'cpp_refined',
    'RCM Tangential (refined)': 'rcm_tangential_refined',
    'RCM Tangential (polyfit)': 'rcm_tangential_polyfit',
    'ONNX alex': 'onnx_alex',
    'ONNX alex (refined)': 'onnx_alex_refined',
    'ONNX alex (polyfit)': 'onnx_alex_polyfit',
}

# Offset bands (in cm) for the error-by-offset histogram
_OFFSET_BANDS_CM = [(0, 3), (3, 5), (5, 8), (8, 100)]
_OFFSET_BAND_LABELS = ['0-3 cm', '3-5 cm', '5-8 cm', '8+ cm']
_OFFSET_BAND_COLORS = {
    0: pg.mkPen(color=(50, 180, 50), width=2),    # green
    1: pg.mkPen(color=(74, 144, 226), width=2),    # blue
    2: pg.mkPen(color=(255, 140, 0), width=2),     # orange
    3: pg.mkPen(color=(220, 60, 60), width=2),     # red
}
_OFFSET_BAND_BRUSHES = {
    0: pg.mkBrush(50, 180, 50, 50),
    1: pg.mkBrush(74, 144, 226, 50),
    2: pg.mkBrush(255, 140, 0, 50),
    3: pg.mkBrush(220, 60, 60, 50),
}


class RcmContactLocationPlot(QWidget):
    """Corner-plot of (dx, dy, dz) with marginal distributions.

    Lower-left triangle: scatter plots (pre=circle, post=triangle).
    Diagonal: marginal histograms per shot-type class.
    Upper-right: empty (hidden).
    """

    contact_selected = Signal(object, object, object, object)

    def __init__(self):
        super().__init__()
        self.plots_folder: Optional[str] = None
        self._data: Optional[Dict[str, np.ndarray]] = None

        layout = QVBoxLayout()

        # ── Debounce timer ───────────────────────────────────────────
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self._do_update)

        # ── Control bar ──────────────────────────────────────────────
        control_bar = QHBoxLayout()

        # Point size slider
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Point Size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(2)
        self.size_slider.setMaximum(15)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(self._schedule_update)
        size_layout.addWidget(self.size_slider)
        self.size_label = QLabel("5")
        self.size_label.setMinimumWidth(20)
        size_layout.addWidget(self.size_label)
        control_bar.addLayout(size_layout)

        # Pre / Post toggle
        self.show_post_checkbox = QCheckBox("Show Post")
        self.show_post_checkbox.setChecked(False)
        self.show_post_checkbox.stateChanged.connect(self._schedule_update)
        control_bar.addWidget(self.show_post_checkbox)

        # Model selector for error coloring
        model_layout, self.model_combo = create_combo_selector(
            "Error Color:",
            list(_MODEL_SUFFIXES.keys()),
            self._schedule_update,
        )
        control_bar.addLayout(model_layout)

        control_bar.addStretch()

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self._save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # ── Info label ───────────────────────────────────────────────
        self.info_label = QLabel("RCM Contact Location")
        layout.addWidget(self.info_label)

        # ── Grid container ───────────────────────────────────────────
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.grid_container, stretch=1)

        self.setLayout(layout)

        # Refs to plot widgets
        # Scatter cells: keyed by (row, col) for lower-left triangle only
        self.scatter_plots: Dict[Tuple[int, int], pg.PlotWidget] = {}
        self.scatter_pre: Dict[Tuple[int, int], pg.ScatterPlotItem] = {}
        self.scatter_post: Dict[Tuple[int, int], pg.ScatterPlotItem] = {}
        # Marginal (diagonal) cells: keyed by axis index
        self.marginal_plots: Dict[int, pg.PlotWidget] = {}
        self._header_labels: List[QLabel] = []
        self._legends: list = []
        # Track dynamically added histogram items for cleanup
        self._hist_items: list = []
        # Upper-right cell: error-by-offset-band histogram
        self._offset_hist_pw: Optional[pg.PlotWidget] = None
        self._offset_hist_items: list = []
        self._offset_legends: list = []
        # Upper-right cell (1,2): vel error vs racket angular velocity scatter
        self._err_angvel_pw: Optional[pg.PlotWidget] = None
        self._err_angvel_scatter: Optional[pg.ScatterPlotItem] = None

        # Highlight state: overlay items added on click
        self._highlight_scatters: List[pg.ScatterPlotItem] = []  # (pw, item)
        self._highlight_lines: List[Tuple[pg.PlotWidget, pg.InfiniteLine]] = []
        # Cached per-update arrays for highlighting
        self._cached_active_arrays: Optional[Dict[str, np.ndarray]] = None
        self._cached_vel_error: Optional[np.ndarray] = None
        self._cached_angvel_mag: Optional[np.ndarray] = None
        self._cached_filtered_indices: Optional[np.ndarray] = None

        self._build_grid()

    # ── Grid construction ────────────────────────────────────────────

    def _build_grid(self):
        """Build the corner-plot grid.

        Grid layout (1-indexed rows/cols; row 0 reserved, col 0 = row headers):
            Row 1, Col 1  -> diagonal  (dx marginal)
            Row 2, Col 1  -> scatter   (dy vs dx)
            Row 2, Col 2  -> diagonal  (dy marginal)
            Row 3, Col 1  -> scatter   (dz vs dx)
            Row 3, Col 2  -> scatter   (dz vs dy)
            Row 3, Col 3  -> diagonal  (dz marginal)
        Upper-right cells (col > row) are left empty.
        """
        for row in range(_N):
            # Row header on the left
            lbl = QLabel(_AXES_DISPLAY[row])
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
            self.grid_layout.addWidget(lbl, row + 1, 0)
            self._header_labels.append(lbl)

            for col in range(_N):
                if col > row:
                    # Upper-right triangle: skip
                    continue

                pw = pg.PlotWidget()
                pw.setDefaultPadding(0.02)
                pw.showGrid(x=True, y=True, alpha=0.3)
                pw.getAxis('left').enableAutoSIPrefix(False)
                pw.getAxis('bottom').enableAutoSIPrefix(False)

                if col == row:
                    # ── Diagonal: marginal histogram ─────────────────
                    if row == 2:  # dz: rotated histogram (dz on Y, count on X)
                        pw.setLabel('left', _AXES_DISPLAY[col])
                        pw.setLabel('bottom', 'Count')
                    else:
                        pw.setLabel('left', 'Count')
                        # Bottom label only on the last row
                        if row == _N - 1:
                            pw.setLabel('bottom', _AXES_DISPLAY[col])
                    self.marginal_plots[row] = pw
                    self.grid_layout.addWidget(pw, row + 1, col + 1)
                else:
                    # ── Lower-left: scatter ──────────────────────────
                    pw.enableAutoRange(axis='x', enable=True)
                    pw.enableAutoRange(axis='y', enable=True)
                    # Y-axis label on first column only
                    if col == 0:
                        pw.setLabel('left', _AXES_DISPLAY[row])
                    # X-axis label on bottom row only
                    if row == _N - 1:
                        pw.setLabel('bottom', _AXES_DISPLAY[col])

                    sc_pre = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), symbol=_SYMBOL_PRE)
                    sc_post = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), symbol=_SYMBOL_POST)
                    sc_pre.sigClicked.connect(self._on_point_clicked)
                    sc_post.sigClicked.connect(self._on_point_clicked)
                    pw.addItem(sc_pre)
                    pw.addItem(sc_post)

                    self.scatter_plots[(row, col)] = pw
                    self.scatter_pre[(row, col)] = sc_pre
                    self.scatter_post[(row, col)] = sc_post
                    self.grid_layout.addWidget(pw, row + 1, col + 1)

        # Column headers at the bottom
        for col in range(_N):
            lbl = QLabel(_AXES_DISPLAY[col])
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
            self.grid_layout.addWidget(lbl, _N + 1, col + 1)
            self._header_labels.append(lbl)

        # ── Upper-right cell: error-by-offset-band histogram ─────────
        # Place in grid position (row=1, col=3) — the (dx row, dz col)
        # corner which is otherwise empty.
        pw_off = pg.PlotWidget()
        pw_off.setDefaultPadding(0.02)
        pw_off.showGrid(x=True, y=True, alpha=0.3)
        pw_off.getAxis('left').enableAutoSIPrefix(False)
        pw_off.getAxis('bottom').enableAutoSIPrefix(False)
        pw_off.setLabel('bottom', 'Velocity Error (m/s)')
        pw_off.setLabel('left', 'Proportion')
        # Row 1, spanning columns 2 and 3
        self.grid_layout.addWidget(pw_off, 1, 2, 1, 2)
        self._offset_hist_pw = pw_off

        # ── Upper-right cell (2,3): vel error vs racket |ω| ─────────
        pw_ea = pg.PlotWidget()
        pw_ea.setDefaultPadding(0.02)
        pw_ea.showGrid(x=True, y=True, alpha=0.3)
        pw_ea.getAxis('left').enableAutoSIPrefix(False)
        pw_ea.getAxis('bottom').enableAutoSIPrefix(False)
        pw_ea.setLabel('bottom', 'Racket |ω| (rad/s)')
        pw_ea.setLabel('left', 'Vel Error (m/s)')
        pw_ea.enableAutoRange(axis='x', enable=True)
        pw_ea.enableAutoRange(axis='y', enable=True)
        sc_ea = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None))
        sc_ea.sigClicked.connect(self._on_point_clicked)
        pw_ea.addItem(sc_ea)
        self.grid_layout.addWidget(pw_ea, 2, 3)
        self._err_angvel_pw = pw_ea
        self._err_angvel_scatter = sc_ea

        # ── Link x-axes: scatter cells in the same column share x-axis.
        for col in range(_N):
            col_cells = [(r, col) for r in range(col + 1, _N) if (r, col) in self.scatter_plots]
            if len(col_cells) > 1:
                first_pw = self.scatter_plots[col_cells[0]]
                for rc in col_cells[1:]:
                    self.scatter_plots[rc].setXLink(first_pw)

        # ── Link y-axes: scatter cells in the same row share y-axis.
        for row in range(1, _N):
            row_cells = [(row, c) for c in range(row) if (row, c) in self.scatter_plots]
            if len(row_cells) > 1:
                first_pw = self.scatter_plots[row_cells[0]]
                for rc in row_cells[1:]:
                    self.scatter_plots[rc].setYLink(first_pw)

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_filter_changed(self, _=None):
        self.fit_min_label.setText(str(self.fit_min_slider.value()))
        self.fit_max_label.setText(str(self.fit_max_slider.value()))
        self._update_timer.start()

    def _schedule_update(self, _=None):
        self.size_label.setText(str(self.size_slider.value()))
        self._update_timer.start()

    # ── Public API ───────────────────────────────────────────────────

    def plot_data(self, rcm_data: Dict[str, np.ndarray]):
        """Load RCM data and render."""
        self._data = rcm_data
        self._do_update()

    # ── Update logic ─────────────────────────────────────────────────

    def _do_update(self):
        # ── Clear scatter items ──────────────────────────────────────
        for sc in self.scatter_pre.values():
            sc.clear()
        for sc in self.scatter_post.values():
            sc.clear()

        # ── Clear error vs angular velocity scatter ──────────────────
        if self._err_angvel_scatter is not None:
            self._err_angvel_scatter.clear()

        # ── Clear highlights ─────────────────────────────────────────
        self._clear_highlights()

        # ── Clear histogram items ────────────────────────────────────
        for pw, item in self._hist_items:
            try:
                pw.removeItem(item)
            except Exception:
                pass
        self._hist_items.clear()

        # ── Clear legends ────────────────────────────────────────────
        for pw, leg in self._legends:
            try:
                leg.scene().removeItem(leg)
            except Exception:
                pass
        self._legends.clear()

        # ── Clear offset histogram ───────────────────────────────────
        for pw, item in self._offset_hist_items:
            try:
                pw.removeItem(item)
            except Exception:
                pass
        self._offset_hist_items.clear()
        for pw, leg in self._offset_legends:
            try:
                leg.scene().removeItem(leg)
            except Exception:
                pass
        self._offset_legends.clear()

        data = self._data
        if data is None or len(data.get('vx_pre', [])) == 0:
            self.info_label.setText("No RCM data available")
            return

        n_total = len(data['vx_pre'])

        # ── Build mask ───────────────────────────────────────────────
        mask = np.ones(n_total, dtype=bool)

        # Confidence filter (from global sliders, attached by app.py)
        conf_min_slider = getattr(self, 'conf_min_slider', None)
        conf_max_slider = getattr(self, 'conf_max_slider', None)
        if conf_min_slider is not None or conf_max_slider is not None:
            confidence = data.get('confidence', np.full(n_total, np.nan))
            has_conf = ~np.isnan(confidence)
            if conf_min_slider is not None and conf_min_slider.value() > 0:
                c_lo = conf_min_slider.value() / 100.0
                mask &= (~has_conf) | (confidence >= c_lo)
            if conf_max_slider is not None and conf_max_slider.value() < 100:
                c_hi = conf_max_slider.value() / 100.0
                mask &= (~has_conf) | (confidence <= c_hi)

        # RMSE filter
        rmse_min_slider = getattr(self, 'rmse_min_slider', None)
        rmse_max_slider = getattr(self, 'rmse_slider', None)
        if rmse_min_slider is not None or rmse_max_slider is not None:
            max_rmse = data.get('max_rmse_opt', np.full(n_total, np.nan))
            has_rmse = ~np.isnan(max_rmse)
            rmse_scale = 1000.0
            if rmse_min_slider is not None and rmse_min_slider.value() > 0:
                r_lo = rmse_min_slider.value() / rmse_scale
                mask &= (~has_rmse) | (max_rmse >= r_lo)
            if rmse_max_slider is not None and rmse_max_slider.value() < 100:
                r_hi = rmse_max_slider.value() / rmse_scale
                mask &= (~has_rmse) | (max_rmse <= r_hi)

        # Fitness post filter (shared via app.py from scatter plot)
        _fit_max_spin = getattr(self, 'fitness_max_spin', None)
        if _fit_max_spin is not None:
            _fit_max = _fit_max_spin.value()
            if _fit_max > 0:
                _fp = data.get('fitness_post', np.full(n_total, np.nan))
                _has_fp = ~np.isnan(_fp)
                mask &= (~_has_fp) | (_fp <= _fit_max)

        if not np.any(mask):
            self.info_label.setText("No data in selected filter range")
            return

        n_points = int(np.sum(mask))
        filtered_indices = np.where(mask)[0]

        # ── Extract filtered arrays ──────────────────────────────────
        show_post = self.show_post_checkbox.isChecked()
        suffix = 'post' if show_post else 'pre'
        active_arrays = {ax: data.get(f'{ax}_{suffix}', np.full(n_total, np.nan))[mask] for ax in _AXES}
        shot_types = data.get('shot_type', np.ones(n_total, dtype=int))[mask]

        # ── Compute velocity error for scatter coloring ──────────────
        model_label = self.model_combo.currentText()
        model_suffix = _MODEL_SUFFIXES.get(model_label)
        vel_error = None  # None means no model → use shot-type coloring
        if model_suffix is not None:
            _mk = [f'vx_post_{model_suffix}', f'vy_post_{model_suffix}', f'vz_post_{model_suffix}']
            if all(k in data for k in _mk) and not np.all(np.isnan(data[_mk[0]])):
                _dvx = data[_mk[0]][mask] - data['vx_post'][mask]
                _dvy = data[_mk[1]][mask] - data['vy_post'][mask]
                _dvz = data[_mk[2]][mask] - data['vz_post'][mask]
                vel_error = np.sqrt(_dvx**2 + _dvy**2 + _dvz**2)

        # ── Build colour / brush arrays for scatter ──────────────────
        if vel_error is not None:
            # Continuous colormap by velocity error magnitude
            cmap = plt.get_cmap('turbo')
            _valid_err = vel_error[np.isfinite(vel_error)]
            c_max = float(np.percentile(_valid_err, 95)) if len(_valid_err) > 0 else 1.0
            c_min = 0.0
            clipped = np.clip(np.nan_to_num(vel_error, nan=0.0), c_min, c_max)
            denom = (c_max - c_min) if c_max > c_min else 1.0
            norm = (clipped - c_min) / denom
            rgba = cmap(norm)
            brushes = [pg.mkBrush(int(r * 255), int(g * 255), int(b * 255), 180)
                       for r, g, b, _ in rgba]
        else:
            brushes = [
                pg.mkBrush(*_SHOT_TYPE_COLORS.get(int(st), (128, 128, 128, 180)))
                for st in shot_types
            ]

        point_size = self.size_slider.value()
        pen_sc = pg.mkPen(color=(50, 50, 50), width=0.5)
        active_symbol = _SYMBOL_POST if show_post else _SYMBOL_PRE

        # ── Fill lower-left scatter cells ────────────────────────────
        for (row, col), pw in self.scatter_plots.items():
            row_ax = _AXES[row]
            col_ax = _AXES[col]

            self.scatter_pre[(row, col)].setData(
                x=active_arrays[col_ax], y=active_arrays[row_ax],
                size=point_size, pen=pen_sc,
                brush=brushes, symbol=active_symbol,
                data=filtered_indices,
            )
            # Post scatter items unused in single-mode; keep cleared
            self.scatter_post[(row, col)].clear()

        # ── Diagonal marginal histograms ─────────────────────────────
        for ax_idx, ax_name in enumerate(_AXES):
            pw = self.marginal_plots[ax_idx]
            vals_all = active_arrays[ax_name]
            valid = vals_all[~np.isnan(vals_all)]
            if len(valid) == 0:
                continue
            if ax_idx in (1, 2):  # dy, dz: fixed range
                bin_edges = np.linspace(-0.1, 0.1, _N_BINS + 1)
            else:
                bin_edges = np.linspace(np.nanmin(valid), np.nanmax(valid), _N_BINS + 1)

            for st in sorted(_SHOT_TYPE_LABELS.keys()):
                st_mask = shot_types == st
                vals = vals_all[st_mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue

                counts, edges = np.histogram(vals, bins=bin_edges)
                if ax_idx == 2:  # dz: rotated histogram
                    # Draw horizontal bars: y = bin centers, x = counts
                    bin_centers = 0.5 * (edges[:-1] + edges[1:])
                    bin_h = edges[1] - edges[0]
                    bar = pg.BarGraphItem(
                        x0=0, width=counts.astype(float),
                        y=bin_centers, height=bin_h,
                        pen=_SHOT_TYPE_PEN[st],
                        brush=_SHOT_TYPE_BRUSH[st],
                    )
                    pw.addItem(bar)
                    self._hist_items.append((pw, bar))
                else:
                    curve = pw.plot(
                        edges, counts,
                        stepMode=True,
                        fillLevel=0,
                        pen=_SHOT_TYPE_PEN[st],
                        brush=_SHOT_TYPE_BRUSH[st],
                    )
                    self._hist_items.append((pw, curve))

        # ── Auto-range all plots ─────────────────────────────────
        for midx, pw in self.marginal_plots.items():
            if midx == 2:  # dz: rotated — fixed y-range, auto x
                pw.setYRange(-0.1, 0.1, padding=0)
                pw.enableAutoRange(axis='x')
            elif midx == 1:  # dy: fixed x-range
                pw.setXRange(-0.1, 0.1, padding=0)
                pw.enableAutoRange(axis='y')
            else:
                pw.enableAutoRange()
                pw.autoRange()
        for (row, col), pw in self.scatter_plots.items():
            # Fix dy/dz axes to [-0.1, 0.1]
            fixed_x = col in (1, 2)
            fixed_y = row in (1, 2)
            if fixed_x:
                pw.setXRange(-0.1, 0.1, padding=0)
            if fixed_y:
                pw.setYRange(-0.1, 0.1, padding=0)
            if not fixed_x and not fixed_y:
                pw.enableAutoRange()
                pw.autoRange()
            elif not fixed_x:
                pw.enableAutoRange(axis='x')
            elif not fixed_y:
                pw.enableAutoRange(axis='y')

        # ── Legend in the first diagonal cell ─────────────────────────
        pw_legend = self.marginal_plots.get(0)
        if pw_legend is not None:
            legend = pw_legend.addLegend(offset=(5, 5), labelTextSize='8pt')
            self._legends.append((pw_legend, legend))
            for st, label in _SHOT_TYPE_LABELS.items():
                c = _SHOT_TYPE_COLORS[st]
                _d = pg.ScatterPlotItem(x=[0], y=[0], size=8,
                                        brush=pg.mkBrush(*c),
                                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                                        symbol=active_symbol)
                legend.addItem(_d, label)
                _d.clear()

        # ── Error-by-offset-band histogram (upper-right cell) ────────
        if self._offset_hist_pw is not None and vel_error is not None:
            pw_off = self._offset_hist_pw
            pw_off.setTitle('')  # clear placeholder title
            # Compute contact offset = sqrt(dy^2 + dz^2)
            _dy_off = active_arrays['dy']
            _dz_off = active_arrays['dz']
            contact_offset = np.sqrt(_dy_off**2 + _dz_off**2) * 100.0  # m → cm
            # Histogram error per offset band
            _err_valid = vel_error[np.isfinite(vel_error) & np.isfinite(contact_offset)]
            _off_valid = contact_offset[np.isfinite(vel_error) & np.isfinite(contact_offset)]
            if len(_err_valid) > 0:
                # Fixed 0.1 m/s bins up to 5 m/s, plus one overflow bin
                _bin_w = 0.1
                _err_max = 5.0
                bin_edges = np.arange(0.0, _err_max + _bin_w * 0.5, _bin_w)
                # Append overflow bin edge (1.5× normal width beyond 5 m/s)
                bin_edges = np.append(bin_edges, _err_max + 1.5 * _bin_w)
                _clamped_err = np.minimum(_err_valid, bin_edges[-1] - 1e-9)
                _clamped_off = _off_valid
                for bi, (lo, hi) in enumerate(_OFFSET_BANDS_CM):
                    band_mask = (_clamped_off >= lo) & (_clamped_off < hi)
                    band_err = _clamped_err[band_mask]
                    if len(band_err) == 0:
                        continue
                    counts, edges = np.histogram(band_err, bins=bin_edges)
                    # Normalize to proportion (bar heights sum to 1)
                    proportion = counts / len(band_err) if len(band_err) > 0 else counts.astype(float)
                    curve = pw_off.plot(
                        edges, proportion,
                        stepMode=True,
                        fillLevel=0,
                        pen=_OFFSET_BAND_COLORS[bi],
                        brush=_OFFSET_BAND_BRUSHES[bi],
                    )
                    self._offset_hist_items.append((pw_off, curve))
                pw_off.enableAutoRange()
                pw_off.autoRange()
                # Legend for offset bands
                off_legend = pw_off.addLegend(offset=(5, 5), labelTextSize='8pt')
                self._offset_legends.append((pw_off, off_legend))
                for bi, lbl_text in enumerate(_OFFSET_BAND_LABELS):
                    _d = pg.PlotCurveItem(x=[0], y=[0],
                                          pen=_OFFSET_BAND_COLORS[bi])
                    off_legend.addItem(_d, lbl_text)
        elif self._offset_hist_pw is not None:
            self._offset_hist_pw.setTitle("Select model for error histogram", size='8pt')

        # ── Error vs racket |ω| scatter (upper-right cell) ──────────
        if self._err_angvel_scatter is not None and vel_error is not None:
            self._err_angvel_pw.setTitle('')  # clear placeholder title
            # Compute racket angular velocity magnitude
            _wgx = data.get('wgx_racket', np.full(n_total, np.nan))[mask]
            _wgy = data.get('wgy_racket', np.full(n_total, np.nan))[mask]
            _wgz = data.get('wgz_racket', np.full(n_total, np.nan))[mask]
            _angvel_mag = np.sqrt(_wgx**2 + _wgy**2 + _wgz**2)
            self._err_angvel_scatter.setData(
                x=_angvel_mag, y=vel_error,
                size=point_size, pen=pen_sc,
                brush=brushes, symbol=active_symbol,
                data=filtered_indices,
            )
            self._err_angvel_pw.enableAutoRange()
            self._err_angvel_pw.autoRange()
        elif self._err_angvel_pw is not None:
            self._err_angvel_pw.setTitle("Select model for error scatter", size='8pt')

        # ── Cache arrays for highlighting on click ─────────────────
        self._cached_active_arrays = active_arrays
        self._cached_vel_error = vel_error
        # Compute and cache racket |ω| for the error-angvel scatter
        _wgx_c = data.get('wgx_racket', np.full(n_total, np.nan))[mask]
        _wgy_c = data.get('wgy_racket', np.full(n_total, np.nan))[mask]
        _wgz_c = data.get('wgz_racket', np.full(n_total, np.nan))[mask]
        self._cached_angvel_mag = np.sqrt(_wgx_c**2 + _wgy_c**2 + _wgz_c**2)
        self._cached_filtered_indices = filtered_indices

        # ── Info label ───────────────────────────────────────────────
        n_serve = int(np.sum(shot_types == 0))
        n_rally = int(np.sum(shot_types == 1))
        n_last = int(np.sum(shot_types == 2))
        mode_str = 'post' if show_post else 'pre'
        color_str = f"  |  Color: {model_label} vel error" if vel_error is not None else "  |  Color: Shot Type"
        self.info_label.setText(
            f"Showing {n_points}/{n_total} contacts ({mode_str})  "
            f"[Serve: {n_serve}, Rally: {n_rally}, Last: {n_last}]"
            f"{color_str}"
        )

    # ── Click handler ────────────────────────────────────────────────

    def _clear_highlights(self):
        """Remove all highlight overlays from plots."""
        for pw, item in self._highlight_scatters:
            try:
                pw.removeItem(item)
            except Exception:
                pass
        self._highlight_scatters.clear()
        for pw, line in self._highlight_lines:
            try:
                pw.removeItem(line)
            except Exception:
                pass
        self._highlight_lines.clear()

    def highlight_point(self, global_idx: int):
        """Highlight the sample with global index `global_idx` across all plots."""
        self._clear_highlights()
        if self._cached_filtered_indices is None or self._cached_active_arrays is None:
            return
        # Find position of this index in the filtered arrays
        positions = np.where(self._cached_filtered_indices == global_idx)[0]
        if len(positions) == 0:
            return
        pos = positions[0]
        active = self._cached_active_arrays
        highlight_size = self.size_slider.value() * 3
        highlight_pen = pg.mkPen(color=(255, 255, 255), width=2)
        highlight_brush = pg.mkBrush(255, 255, 0, 220)

        # ── Highlight in lower-left scatter plots ────────────────────
        for (row, col), pw in self.scatter_plots.items():
            row_ax = _AXES[row]
            col_ax = _AXES[col]
            x_val = active[col_ax][pos]
            y_val = active[row_ax][pos]
            if np.isnan(x_val) or np.isnan(y_val):
                continue
            sc = pg.ScatterPlotItem(
                x=[x_val], y=[y_val],
                size=highlight_size, pen=highlight_pen,
                brush=highlight_brush, symbol='o',
            )
            pw.addItem(sc)
            self._highlight_scatters.append((pw, sc))

        # ── Highlight in marginal histograms ─────────────────────
        for ax_idx, ax_name in enumerate(_AXES):
            pw = self.marginal_plots.get(ax_idx)
            if pw is None:
                continue
            val = active[ax_name][pos]
            if np.isnan(val):
                continue
            # dz histogram is rotated: values on y-axis → horizontal line
            angle = 0 if ax_idx == 2 else 90
            line = pg.InfiniteLine(pos=val, angle=angle,
                                   pen=pg.mkPen(color=(255, 255, 0), width=2, style=Qt.DashLine))
            pw.addItem(line)
            self._highlight_lines.append((pw, line))

        # ── Highlight in error vs angvel scatter ─────────────────────
        if (self._err_angvel_pw is not None
                and self._cached_vel_error is not None
                and self._cached_angvel_mag is not None):
            ve = self._cached_vel_error[pos]
            am = self._cached_angvel_mag[pos]
            if np.isfinite(ve) and np.isfinite(am):
                sc = pg.ScatterPlotItem(
                    x=[am], y=[ve],
                    size=highlight_size, pen=highlight_pen,
                    brush=highlight_brush, symbol='o',
                )
                self._err_angvel_pw.addItem(sc)
                self._highlight_scatters.append((self._err_angvel_pw, sc))

        # ── Highlight in offset-band histogram (vertical line) ───────
        if self._offset_hist_pw is not None and self._cached_vel_error is not None:
            ve = self._cached_vel_error[pos]
            if np.isfinite(ve):
                line = pg.InfiniteLine(pos=ve, angle=90,
                                       pen=pg.mkPen(color=(255, 255, 0), width=2, style=Qt.DashLine))
                self._offset_hist_pw.addItem(line)
                self._highlight_lines.append((self._offset_hist_pw, line))

    def _on_point_clicked(self, _plot_item, points):
        if len(points) == 0:
            return
        pt = points[0]
        idx = pt.data()
        if idx is None or self._data is None:
            return
        idx = int(idx)

        # Highlight across all plots
        self.highlight_point(idx)

        data = self._data
        metadata = data.get('metadata_list', [])
        shot_list = data.get('shot_list', [])
        rally_list = data.get('rally_list', [])
        fs_pre_list = data.get('fs_pre_list', [])
        fs_post_list = data.get('fs_post_list', [])
        md = metadata[idx] if idx < len(metadata) else None
        shot = shot_list[idx] if idx < len(shot_list) else None
        rally = rally_list[idx] if idx < len(rally_list) else None

        contact_point_data = {'_rcm_global_idx': idx}
        for k in ['vx_pre', 'vy_pre', 'vz_pre', 'wx_pre', 'wy_pre', 'wz_pre',
                   'vx_post', 'vy_post', 'vz_post', 'wx_post', 'wy_post', 'wz_post']:
            if k in data:
                contact_point_data[k] = float(data[k][idx])
        if idx < len(fs_pre_list):
            contact_point_data['fs_pre'] = fs_pre_list[idx]
        if idx < len(fs_post_list):
            contact_point_data['fs_post'] = fs_post_list[idx]
        self.contact_selected.emit(contact_point_data, md, shot, rally)

    # ── Save ─────────────────────────────────────────────────────────

    def _save_plot(self):
        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"rcm_contact_location_{ts}.png"
        out = plots_dir / fname
        try:
            px = self.grid_container.grab()
            px.save(str(out), "PNG")
            QMessageBox.information(self, "Success", f"Plot saved to:\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")
