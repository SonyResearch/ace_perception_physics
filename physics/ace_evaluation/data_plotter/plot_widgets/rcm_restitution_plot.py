# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
RCM Restitution Plot Widget

Single scatter plot of user-chosen local-frame quantities (X / Y axes
selectable via drop-downs), with observed data and togglable model
prediction overlays.
"""

from typing import Dict, Any, Optional
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QSlider, QCheckBox, QComboBox,
)
from PySide6.QtCore import Qt, QTimer

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from .base_coefficient_plot import save_plot_as_image


# ── Model definitions ────────────────────────────────────────────────────
# label → (suffix, colour_rgba)
_MODELS = {
    'Default (constant COR)': ('default',        (255, 100, 100, 120)),
    'C++ residual':           ('cpp',            (255, 165,   0, 120)),
    'Parametric 7p':          ('parametric_7p',  (200,  50, 200, 120)),
    'RCM Tangential':         ('rcm_tangential', (  0, 200, 200, 120)),
    'C++ tangential':         ('cpp_tangential', (  0, 128, 128, 120)),
    'Nakashima (refined)':   ('nakashima_refined',     (255, 140, 140, 120)),
    'C++ no-residual (refined)': ('cpp_refined',         (180,  80,  40, 120)),
    'RCM Tangential (refined)':('rcm_tangential_refined', (  0, 220, 180, 120)),
    'RCM Tangential (polyfit)':('rcm_tangential_polyfit', (100, 180, 255, 120)),
    'ONNX alex':                ('onnx_alex',             (180, 255,   0, 120)),
    'ONNX alex (refined)':      ('onnx_alex_refined',     (140, 220,  40, 120)),
    'ONNX alex (polyfit)':      ('onnx_alex_polyfit',     (100, 190,  80, 120)),
}

# X-axis: pre-contact quantities only
_PRE_QUANTITIES = [
    ('vrx_pre', 'vrx pre (m/s)'),
    ('vry_pre', 'vry pre (m/s)'),
    ('vrz_pre', 'vrz pre (m/s)'),
    ('wrx_pre', 'wrx pre (rad/s)'),
    ('wry_pre', 'wry pre (rad/s)'),
    ('wrz_pre', 'wrz pre (rad/s)'),
]
# Y-axis: post-contact quantities only
_POST_QUANTITIES = [
    ('vrx_post', 'vrx post (m/s)'),
    ('vry_post', 'vry post (m/s)'),
    ('vrz_post', 'vrz post (m/s)'),
    ('wrx_post', 'wrx post (rad/s)'),
    ('wry_post', 'wry post (rad/s)'),
    ('wrz_post', 'wrz post (rad/s)'),
]
_PRE_LABELS  = [lbl for _, lbl in _PRE_QUANTITIES]
_POST_LABELS = [lbl for _, lbl in _POST_QUANTITIES]
_LABEL_TO_KEY = {lbl: key for key, lbl in _PRE_QUANTITIES + _POST_QUANTITIES}


class RcmRestitutionPlot(QWidget):
    """Single scatter with selectable X/Y axes and model overlays."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Title
        title = QLabel("RCM – Restitution (Local Frame)")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ── Control bar ──────────────────────────────────────────────
        control_bar = QHBoxLayout()

        # X-axis selector (pre-contact quantities)
        control_bar.addWidget(QLabel("X:"))
        self.x_combo = QComboBox()
        self.x_combo.addItems(_PRE_LABELS)
        self.x_combo.setCurrentText('vrx pre (m/s)')
        self.x_combo.currentTextChanged.connect(self._refresh)
        control_bar.addWidget(self.x_combo)

        # Y-axis selector (post-contact quantities)
        control_bar.addWidget(QLabel("Y:"))
        self.y_combo = QComboBox()
        self.y_combo.addItems(_POST_LABELS)
        self.y_combo.setCurrentText('vrx post (m/s)')
        self.y_combo.currentTextChanged.connect(self._refresh)
        control_bar.addWidget(self.y_combo)

        control_bar.addWidget(QLabel("  "))

        # Model check-boxes
        control_bar.addWidget(QLabel("Models:"))
        self._model_cbs: Dict[str, QCheckBox] = {}
        for label in _MODELS:
            cb = QCheckBox(label)
            cb.setChecked(False)
            cb.stateChanged.connect(self._refresh)
            self._model_cbs[label] = cb
            control_bar.addWidget(cb)

        # Point size slider
        control_bar.addWidget(QLabel("  Size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(2)
        self.size_slider.setMaximum(15)
        self.size_slider.setValue(4)
        self.size_slider.setFixedWidth(80)
        self.size_slider.valueChanged.connect(self._refresh)
        control_bar.addWidget(self.size_slider)
        self.size_label = QLabel("4")
        self.size_label.setFixedWidth(18)
        control_bar.addWidget(self.size_label)

        control_bar.addStretch(1)

        # Info label
        self.info_label = QLabel("")
        control_bar.addWidget(self.info_label)

        # Save button
        save_btn = QPushButton("Save Plot")
        save_btn.setMaximumWidth(100)
        save_btn.clicked.connect(self._save_plot)
        control_bar.addWidget(save_btn)

        layout.addLayout(control_bar)

        # ── Single plot widget ───────────────────────────────────────
        self.pw = pg.PlotWidget()
        self.pw.showGrid(x=True, y=True)
        layout.addWidget(self.pw, stretch=1)

        self.setLayout(layout)

        # Data
        self.rcm_data: Optional[Dict[str, np.ndarray]] = None

        # Debounce
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._do_update)

    # ─── callbacks ───────────────────────────────────────────────────

    def _refresh(self, *_args):
        self.size_label.setText(str(self.size_slider.value()))
        self._timer.start()

    # ─── public API ──────────────────────────────────────────────────

    def plot_data(self, rcm_data: Dict[str, np.ndarray], base_folder: Optional[str] = None):
        self.rcm_data = rcm_data
        self.base_folder = base_folder
        self._do_update()

    # ─── update logic ────────────────────────────────────────────────

    def _do_update(self):
        self.pw.clear()

        if self.rcm_data is None:
            self.info_label.setText("No data")
            return

        data = self.rcm_data
        n_total = len(data.get('vrx_pre', []))
        if n_total == 0:
            self.info_label.setText("No RCM data")
            return

        # ── Filter (reuse global filter sliders if attached) ─────────
        mask = np.ones(n_total, dtype=bool)

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
            self.info_label.setText("No data in selected range")
            return

        n = int(np.sum(mask))

        # ── Resolve axis keys ────────────────────────────────────────
        x_key = _LABEL_TO_KEY.get(self.x_combo.currentText())
        y_key = _LABEL_TO_KEY.get(self.y_combo.currentText())
        if x_key is None or y_key is None:
            return

        x_arr = data.get(x_key)
        y_arr = data.get(y_key)
        if x_arr is None or y_arr is None:
            self.info_label.setText(f"Missing data key: {x_key} or {y_key}")
            return

        x_obs = x_arr[mask]
        y_obs = y_arr[mask]
        valid = np.isfinite(x_obs) & np.isfinite(y_obs)
        x_obs, y_obs = x_obs[valid], y_obs[valid]

        # Axis labels
        x_label = self.x_combo.currentText()
        y_label = self.y_combo.currentText()
        self.pw.setLabel('bottom', x_label)
        self.pw.setLabel('left', y_label)

        # Observed scatter – color-coded by date (before/after 2025-10-20)
        point_size = self.size_slider.value()
        metadata = data.get('metadata_list', [])
        _DATE_CUTOFF = '20251020'  # YYYYMMDD format (matches match_date strings)
        _COLOR_BEFORE = (180, 180, 180, 100)  # grey
        _COLOR_AFTER  = ( 50, 150, 255, 100)  # blue

        # Build per-point boolean: True if date >= cutoff
        _is_after = np.zeros(n_total, dtype=bool)
        for i in range(n_total):
            if i < len(metadata):
                d = metadata[i].get('match_date', '')
                _is_after[i] = (d >= _DATE_CUTOFF) if d else False

        _is_after_filtered = _is_after[mask][valid]

        # Before cutoff
        _before = ~_is_after_filtered
        if np.any(_before):
            self.pw.addItem(pg.ScatterPlotItem(
                x_obs[_before], y_obs[_before], symbol='o', size=point_size,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(*_COLOR_BEFORE),
                name='Observed (< 2025-10-20)',
            ))
        # After cutoff
        _after = _is_after_filtered
        if np.any(_after):
            self.pw.addItem(pg.ScatterPlotItem(
                x_obs[_after], y_obs[_after], symbol='o', size=point_size,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(*_COLOR_AFTER),
                name='Observed (≥ 2025-10-20)',
            ))

        # ── Model overlays ───────────────────────────────────────────
        # Y is always a post-contact quantity → replace observed Y with
        # model-predicted Y, keeping X (pre) as observed.
        for label, (suffix, color) in _MODELS.items():
            cb = self._model_cbs.get(label)
            if cb is None or not cb.isChecked():
                continue

            model_y_key = y_key.replace('_post', f'_post_{suffix}')
            y_model_arr = data.get(model_y_key)
            if y_model_arr is None:
                continue

            y_m = y_model_arr[mask][valid]
            m_valid = np.isfinite(y_m)
            self.pw.addItem(pg.ScatterPlotItem(
                x_obs[m_valid], y_m[m_valid], symbol='o', size=point_size,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(*color),
                name=label,
            ))

        self.pw.addLegend(offset=(10, 10))
        n_before = int(np.sum(~_is_after_filtered))
        n_after = int(np.sum(_is_after_filtered))
        self.info_label.setText(f"{n} contacts (before: {n_before}, after: {n_after})")

    # ─── save ────────────────────────────────────────────────────────

    def _save_plot(self):
        if self.rcm_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            fname = f"rcm_restitution_{ts}.png"
            save_plot_as_image(self.pw, str(plots_dir / fname), publication_mode=False)
            QMessageBox.information(self, "Success", f"Plot saved to:\n{plots_dir / fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")
