# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Racket Contact Model (RCM) Box Plot Widget

Shows box-and-whisker plots of the absolute per-component prediction error
(``|post_data − pre_data|``) in both global and racket-local frames.

All selected models are drawn side-by-side in each subplot with distinct
colours.  Check-boxes allow toggling individual models on/off.

Layout
------
* **Global frame** – 2×4 grid: vx, vy, vz, v_mag (row 1) and wx, wy, wz,
  w_mag (row 2).
* **Local (racket) frame** – 2×3 grid: vrx, vry, vrz (row 1) and wrx, wry,
  wrz (row 2).

Reuses ``_draw_violin`` from ``fitness_boxplot.py``.
"""

from typing import Dict, Any, Optional, List
import csv
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QSlider, QGridLayout, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from .base_coefficient_plot import create_combo_selector, save_plot_as_image
from .fitness_boxplot import _draw_violin

# Model-prediction suffixes + colours
# label → (suffix, colour_rgb)
# "Observed" is omitted — it is the reference, not a model to evaluate.
_MODEL_DEFS = {
    'Nakashima':               ('default',        (  0, 114, 178)),
    'C++ residual':            ('cpp',            (200,  50, 200)),
    'Parametric 7p':           ('parametric_7p',  (128, 128, 128)),
    'RCM Tangential':          ('rcm_tangential', (  0, 200, 200)),
    'C++ tangential':          ('cpp_tangential', (  0, 128, 128)),
    'Nakashima (refined)':      ('nakashima_refined',     (230, 159,   0)),
    'C++ no-residual (refined)': ('cpp_refined',          (180,  80,  40)),
    'RCM Tangential (refined)': ('rcm_tangential_refined', (  0, 220, 180)),
    'RCM Tangential (polyfit)': ('rcm_tangential_polyfit', (100, 180, 255)),
    'ONNX alex':                  ('onnx_alex',             (180, 255,   0)),
    'ONNX alex (refined)':        ('onnx_alex_refined',     (  0, 158, 115)),
    'ONNX alex (polyfit)':        ('onnx_alex_polyfit',     (100, 190,  80)),
    'ONNX 0426':                   ('onnx_0426',             (255, 127,  14)),
}

# Keep the old dict for backward compat with anything that imports it
_MODEL_SUFFIXES = {k: v[0] for k, v in _MODEL_DEFS.items()}


class RcmBoxPlot(QWidget):
    """Box plots comparing per-component error across *all* models.

    Every checked model gets its own coloured box side-by-side in each
    subplot.  The user can toggle models via check-boxes.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Title
        title = QLabel("RCM – Per-Component Error Distribution")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ── Control bar ──────────────────────────────────────────────
        control_bar = QHBoxLayout()

        # Frame selector
        frame_layout, self.frame_combo = create_combo_selector(
            "Frame:", ["Global", "Local (Racket)"],
            self._on_frame_changed,
        )
        control_bar.addLayout(frame_layout)

        # Group-by selector
        group_layout, self.group_combo = create_combo_selector(
            "Group By:", ["Model", "Date", "Shot Type", "Contact Offset"],
            self._on_group_changed,
        )
        control_bar.addLayout(group_layout)

        # Model check-boxes (replace the old combo)
        control_bar.addWidget(QLabel("  Models:"))
        self._model_cbs: Dict[str, QCheckBox] = {}
        _DEFAULT_CHECKED = {'Nakashima', 'Nakashima (refined)', 'ONNX alex (refined)'}
        for label, (suffix, color) in _MODEL_DEFS.items():
            cb = QCheckBox(label)
            cb.setChecked(label in _DEFAULT_CHECKED)
            cb.stateChanged.connect(self._refresh)
            self._model_cbs[label] = cb
            control_bar.addWidget(cb)

        # Error type selector
        err_layout, self.error_combo = create_combo_selector(
            "Error Type:", ["Absolute |err|", "Signed (model−obs)"],
            self._refresh,
        )
        control_bar.addLayout(err_layout)

        # Info label
        self.info_label = QLabel("")
        control_bar.addWidget(self.info_label, stretch=1)

        # Legend toggle
        self._legend_cb = QCheckBox("Legend")
        self._legend_cb.setChecked(False)
        self._legend_cb.stateChanged.connect(self._refresh)
        control_bar.addWidget(self._legend_cb)

        # Publication mode
        self.publication_checkbox = QCheckBox("Publication")
        self.publication_checkbox.stateChanged.connect(self._refresh)
        control_bar.addWidget(self.publication_checkbox)

        # Magnitude-only export (v_mag + w_mag side-by-side)
        self._mag_only_cb = QCheckBox("Magnitude only")
        self._mag_only_cb.setToolTip(
            "When saving, export only v_mag and w_mag plots side-by-side"
        )
        control_bar.addWidget(self._mag_only_cb)

        # Save button
        save_btn = QPushButton("Save Plot")
        save_btn.setMaximumWidth(100)
        save_btn.clicked.connect(self._save_plot)
        control_bar.addWidget(save_btn)

        # CSV export button
        csv_btn = QPushButton("Export CSV")
        csv_btn.setMaximumWidth(100)
        csv_btn.clicked.connect(self._export_csv)
        control_bar.addWidget(csv_btn)

        layout.addLayout(control_bar)

        # ── Plot grid ────────────────────────────────────────────────
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        layout.addWidget(self.grid_container, stretch=1)

        # Stats label
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

        # Data
        self.rcm_data: Optional[Dict[str, np.ndarray]] = None
        self.plot_widgets: Dict = {}
        self._legends: list = []  # track (pw, legend) for cleanup

        # Debounce
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._do_update)

        # Build initial grid
        self._rebuild_grid()

    # ─── helpers ─────────────────────────────────────────────────────

    def _checked_models(self) -> List[tuple]:
        """Return list of (label, suffix, color) for checked models."""
        result = []
        for label, (suffix, color) in _MODEL_DEFS.items():
            cb = self._model_cbs.get(label)
            if cb is not None and cb.isChecked():
                result.append((label, suffix, color))
        return result

    # ─── grid management ─────────────────────────────────────────────

    def _is_global(self) -> bool:
        return self.frame_combo.currentText() == "Global"

    def _rebuild_grid(self):
        """Tear down and recreate the grid of PlotWidgets."""
        for pw in self.plot_widgets.values():
            self.grid_layout.removeWidget(pw)
            pw.deleteLater()
        self.plot_widgets.clear()

        if self._is_global():
            titles = [
                ("vx error (m/s)", 0, 0), ("vy error (m/s)", 0, 1),
                ("vz error (m/s)", 0, 2), ("v_mag error (m/s)", 0, 3),
                ("wx error (rad/s)", 1, 0), ("wy error (rad/s)", 1, 1),
                ("wz error (rad/s)", 1, 2), ("w_mag error (rad/s)", 1, 3),
            ]
        else:
            titles = [
                ("vrx post (m/s)", 0, 0), ("vry post (m/s)", 0, 1),
                ("vrz post (m/s)", 0, 2),
                ("wrx post (rad/s)", 1, 0), ("wry post (rad/s)", 1, 1),
                ("wrz post (rad/s)", 1, 2),
            ]

        for title_text, row, col in titles:
            pw = pg.PlotWidget(title=title_text)
            pw.showGrid(x=True, y=True)
            pw.getAxis('left').enableAutoSIPrefix(False)
            pw.getAxis('bottom').enableAutoSIPrefix(False)
            self.plot_widgets[(row, col)] = pw
            self.grid_layout.addWidget(pw, row, col)

    # ─── callbacks ───────────────────────────────────────────────────

    def _on_frame_changed(self, _text: str = ""):
        self._rebuild_grid()
        self._timer.start()

    def _on_group_changed(self, _text: str = ""):
        self._rebuild_grid()
        self._timer.start()

    def _refresh(self, *_args):
        self._timer.start()

    # ─── public API ──────────────────────────────────────────────────

    def plot_data(self, rcm_data: Dict[str, np.ndarray], base_folder: Optional[str] = None):
        self.rcm_data = rcm_data
        self.base_folder = base_folder
        self._rebuild_grid()
        self._do_update()

    # ─── update logic ────────────────────────────────────────────────

    def _build_mask(self):
        """Build the boolean mask from all active filters."""
        data = self.rcm_data
        n_total = len(data.get('vx_pre', []))
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

        return mask

    def _do_update(self):
        # Remove previous legends
        for _leg_pw, _leg in self._legends:
            try:
                _leg.clear()
                _leg_pw.removeItem(_leg)
                if hasattr(_leg_pw, 'plotItem') and getattr(_leg_pw.plotItem, 'legend', None) is _leg:
                    _leg_pw.plotItem.legend = None
            except Exception:
                pass
        self._legends.clear()

        for pw in self.plot_widgets.values():
            pw.clear()

        if self.rcm_data is None:
            self.info_label.setText("No data")
            return

        data = self.rcm_data
        n_total = len(data.get('vx_pre', []))
        if n_total == 0:
            self.info_label.setText("No RCM data")
            return

        mask = self._build_mask()
        if not np.any(mask):
            self.info_label.setText("No data in selected range")
            return

        n = int(np.sum(mask))
        checked = self._checked_models()
        n_models = len(checked)
        if n_models == 0:
            self.info_label.setText(f"{n} contacts — no models selected")
            return

        group_by = self.group_combo.currentText()
        if group_by == "Date":
            self._do_update_by_date(data, mask, n_total, n, checked)
        elif group_by == "Shot Type":
            self._do_update_by_shot_type(data, mask, n_total, n, checked)
        elif group_by == "Contact Offset":
            self._do_update_by_offset(data, mask, n_total, n, checked)
        else:
            self._do_update_by_model(data, mask, n_total, n, checked)

    # ─── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _whisker_extent(data):
        """Return (whisker_low, whisker_high) matching _draw_violin logic."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        within = data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]
        return (np.min(within) if len(within) > 0 else q1,
                np.max(within) if len(within) > 0 else q3)

    def _grid_map(self):
        if self._is_global():
            return [
                ('vx', 0, 0), ('vy', 0, 1), ('vz', 0, 2), ('vmag', 0, 3),
                ('wx', 1, 0), ('wy', 1, 1), ('wz', 1, 2), ('wmag', 1, 3),
            ]
        else:
            return [
                ('vrx', 0, 0), ('vry', 0, 1), ('vrz', 0, 2),
                ('wrx', 1, 0), ('wry', 1, 1), ('wrz', 1, 2),
            ]

    # ─── Model grouping (original) ───────────────────────────────────

    def _do_update_by_model(self, data, mask, n_total, n, checked):
        n_models = len(checked)
        bar_width = 0.6 / max(n_models, 1)
        stats_parts = []
        use_signed = self.error_combo.currentText().startswith("Signed")
        _pub = self.publication_checkbox.isChecked()

        grid_map = self._grid_map()

        for key, row, col in grid_map:
            pw = self.plot_widgets[(row, col)]
            is_velocity = (row == 0)

            tick_labels = []
            y_max_all = 0.01

            y_min_all = 0.0

            for m_idx, (label, suffix, color) in enumerate(checked):
                err_data = self._get_error_data(data, mask, key, suffix, n_total, use_signed)
                if err_data is None:
                    continue
                err_data = err_data[np.isfinite(err_data)]

                x_center = (m_idx - (n_models - 1) / 2.0) * bar_width * 1.3
                _draw_violin(pw, err_data, x_center, bar_width, color,
                             is_velocity=is_velocity, show_labels=not _pub)

                tick_labels.append((x_center, label.split('(')[0].strip()[:10]))

                if len(err_data) > 0:
                    wlo, whi = self._whisker_extent(err_data)
                    y_max_all = max(y_max_all, whi)
                    y_min_all = min(y_min_all, wlo)

                med = np.median(err_data) if len(err_data) > 0 else float('nan')
                if m_idx == 0:
                    stats_parts.append(f"{key}: {med:.3f}")

            pw.getPlotItem().getAxis('bottom').setTicks([tick_labels])
            half_span = n_models * bar_width * 0.8
            pw.setXRange(-half_span - 0.2, half_span + 0.2)
            margin = max(abs(y_max_all), abs(y_min_all)) * 0.1
            pw.setYRange(y_min_all - margin if use_signed else 0,
                         y_max_all + margin)

        # ── Optional legend ───────────────────────────────────────
        if self._legend_cb.isChecked() and n_models > 0:
            # Add legend to first plot widget
            pw0 = self.plot_widgets.get((0, 0))
            if pw0 is not None:
                legend = pw0.addLegend(offset=(5, 5), labelTextSize='8pt')
                self._legends.append((pw0, legend))
                for label, suffix, color in checked:
                    _dummy = pg.ScatterPlotItem(
                        x=[0], y=[0], size=10,
                        brush=pg.mkBrush(*color, 180),
                        pen=pg.mkPen(color=color, width=2),
                    )
                    legend.addItem(_dummy, label)
                    _dummy.clear()

        frame_str = "Global" if self._is_global() else "Local (Racket)"
        model_labels = ', '.join(l.split('(')[0].strip() for l, _, _ in checked)
        err_str = "signed" if use_signed else "|error|"
        self.info_label.setText(
            f"{n} contacts | {frame_str} | Models: {model_labels} | {err_str}"
        )
        self.stats_label.setText(f"Median {err_str} (first model) — " + "  |  ".join(stats_parts))

    # ─── Date grouping ───────────────────────────────────────────────

    def _do_update_by_date(self, data, mask, n_total, n, checked):
        use_signed = self.error_combo.currentText().startswith("Signed")
        _pub = self.publication_checkbox.isChecked()

        # Extract per-point date strings from metadata
        metadata = data.get('metadata_list', [])
        all_dates = [metadata[i].get('match_date', '') if i < len(metadata) else ''
                     for i in range(n_total)]
        filtered_indices = np.where(mask)[0]
        filtered_dates = [all_dates[i] for i in filtered_indices]
        unique_dates = sorted(set(filtered_dates))
        n_dates = len(unique_dates)
        if n_dates == 0:
            self.info_label.setText("No dates in selected range")
            return

        n_models = len(checked)
        # Each date position gets n_models sub-bars side-by-side
        bar_width = 0.6 / max(n_models, 1)
        date_spacing = max(n_models * bar_width * 1.3, 1.0) + 0.3

        grid_map = self._grid_map()

        for key, row, col in grid_map:
            pw = self.plot_widgets[(row, col)]
            is_velocity = (row == 0)

            tick_labels = []
            y_max_all = 0.01
            y_min_all = 0.0

            for d_idx, date_str in enumerate(unique_dates):
                date_x = d_idx * date_spacing
                # Build per-date mask (subset of the global mask)
                date_mask = np.zeros(n_total, dtype=bool)
                for fi in filtered_indices:
                    if all_dates[fi] == date_str:
                        date_mask[fi] = True

                tick_labels.append((date_x, date_str if date_str else '(unknown)'))

                for m_idx, (label, suffix, color) in enumerate(checked):
                    err_data = self._get_error_data(data, date_mask, key, suffix, n_total, use_signed)
                    if err_data is None:
                        continue
                    err_data = err_data[np.isfinite(err_data)]
                    if len(err_data) == 0:
                        continue

                    x_offset = (m_idx - (n_models - 1) / 2.0) * bar_width * 1.3
                    _draw_violin(pw, err_data, date_x + x_offset, bar_width,
                                 color, is_velocity=is_velocity,
                                 show_labels=not _pub)

                    if len(err_data) > 0:
                        wlo, whi = self._whisker_extent(err_data)
                        y_max_all = max(y_max_all, whi)
                        y_min_all = min(y_min_all, wlo)

            pw.getPlotItem().getAxis('bottom').setTicks([tick_labels])
            x_lo = -date_spacing * 0.5
            x_hi = (n_dates - 1) * date_spacing + date_spacing * 0.5
            pw.setXRange(x_lo, x_hi)
            margin = max(abs(y_max_all), abs(y_min_all)) * 0.1
            pw.setYRange(y_min_all - margin if use_signed else 0,
                         y_max_all + margin)

        # ── Optional legend ───────────────────────────────────────────
        if self._legend_cb.isChecked() and n_models > 0:
            pw0 = self.plot_widgets.get((0, 0))
            if pw0 is not None:
                legend = pw0.addLegend(offset=(5, 5), labelTextSize='8pt')
                self._legends.append((pw0, legend))
                for label, suffix, color in checked:
                    _dummy = pg.ScatterPlotItem(
                        x=[0], y=[0], size=10,
                        brush=pg.mkBrush(*color, 180),
                        pen=pg.mkPen(color=color, width=2),
                    )
                    legend.addItem(_dummy, label)
                    _dummy.clear()

        frame_str = "Global" if self._is_global() else "Local (Racket)"
        model_labels = ', '.join(l.split('(')[0].strip() for l, _, _ in checked)
        err_str = "signed" if use_signed else "|error|"
        self.info_label.setText(
            f"{n} contacts | {n_dates} dates | {frame_str} | "
            f"Models: {model_labels} | {err_str}"
        )
        self.stats_label.setText("")

    # ─── Shot Type grouping ──────────────────────────────────────────

    _SHOT_TYPE_LABELS = {0: 'Serve', 1: 'Rally', 2: 'Last Shot'}

    # Extended shot categories that split "Last Shot" by outcome
    _SHOT_CAT_LABELS = {
        0: 'Serve',
        1: 'Rally',
        3: 'Last (Won)',
        4: 'Last (Lost)',
    }

    @staticmethod
    def _build_shot_categories(shot_types, point_winner):
        """Build extended shot category array:
        0=serve, 1=rally, 3=last_won, 4=last_lost."""
        cats = shot_types.copy()
        last_mask = cats == 2
        cats[last_mask & (point_winner == 1)] = 3   # last + robot won
        cats[last_mask & (point_winner != 1)] = 4   # last + robot lost (or unknown)
        return cats

    def _do_update_by_shot_type(self, data, mask, n_total, n, checked):
        use_signed = self.error_combo.currentText().startswith("Signed")
        _pub = self.publication_checkbox.isChecked()

        shot_types = data.get('shot_type', np.full(n_total, -1, dtype=int))
        player = data.get('player', np.ones(n_total, dtype=int))
        point_winner = data.get('point_winner', np.zeros(n_total, dtype=int))
        shot_cats = self._build_shot_categories(shot_types, point_winner)

        filtered_indices = np.where(mask)[0]
        filtered_cats = shot_cats[filtered_indices]
        # Fixed display order
        cat_order = [0, 1, 3, 4]
        unique_cats = [c for c in cat_order if c in set(int(s) for s in filtered_cats)]
        n_groups = len(unique_cats)
        if n_groups == 0:
            self.info_label.setText("No shot types in selected range")
            return

        n_models = len(checked)
        bar_width = 0.6 / max(n_models, 1)
        group_spacing = max(n_models * bar_width * 1.3, 1.0) + 0.3

        grid_map = self._grid_map()

        for key, row, col in grid_map:
            pw = self.plot_widgets[(row, col)]
            is_velocity = (row == 0)

            tick_labels = []
            y_max_all = 0.01
            y_min_all = 0.0

            for g_idx, cat_val in enumerate(unique_cats):
                g_x = g_idx * group_spacing
                st_mask = mask & (shot_cats == cat_val)

                label_str = self._SHOT_CAT_LABELS.get(cat_val, f'Type {cat_val}')
                n_cat = int(np.sum(st_mask))
                tick_labels.append((g_x, f"{label_str}\nn={n_cat}"))

                for m_idx, (label, suffix, color) in enumerate(checked):
                    err_data = self._get_error_data(data, st_mask, key, suffix, n_total, use_signed)
                    if err_data is None:
                        continue
                    err_data = err_data[np.isfinite(err_data)]
                    if len(err_data) == 0:
                        continue

                    x_offset = (m_idx - (n_models - 1) / 2.0) * bar_width * 1.3
                    _draw_violin(pw, err_data, g_x + x_offset, bar_width,
                                 color, is_velocity=is_velocity,
                                 show_labels=not _pub)

                    if len(err_data) > 0:
                        wlo, whi = self._whisker_extent(err_data)
                        y_max_all = max(y_max_all, whi)
                        y_min_all = min(y_min_all, wlo)

            pw.getPlotItem().getAxis('bottom').setTicks([tick_labels])
            x_lo = -group_spacing * 0.5
            x_hi = (n_groups - 1) * group_spacing + group_spacing * 0.5
            pw.setXRange(x_lo, x_hi)
            margin = max(abs(y_max_all), abs(y_min_all)) * 0.1
            pw.setYRange(y_min_all - margin if use_signed else 0,
                         y_max_all + margin)

        # ── Optional legend ───────────────────────────────────────────
        if self._legend_cb.isChecked() and n_models > 0:
            pw0 = self.plot_widgets.get((0, 0))
            if pw0 is not None:
                legend = pw0.addLegend(offset=(5, 5), labelTextSize='8pt')
                self._legends.append((pw0, legend))
                for label, suffix, color in checked:
                    _dummy = pg.ScatterPlotItem(
                        x=[0], y=[0], size=10,
                        brush=pg.mkBrush(*color, 180),
                        pen=pg.mkPen(color=color, width=2),
                    )
                    legend.addItem(_dummy, label)
                    _dummy.clear()

        frame_str = "Global" if self._is_global() else "Local (Racket)"
        model_labels = ', '.join(l.split('(')[0].strip() for l, _, _ in checked)
        err_str = "signed" if use_signed else "|error|"
        self.info_label.setText(
            f"{n} contacts | {n_groups} shot types | {frame_str} | "
            f"Models: {model_labels} | {err_str}"
        )
        self.stats_label.setText("")

    # ─── Contact Offset grouping ──────────────────────────────────

    _OFFSET_BANDS_CM = [(0, 3), (3, 5), (5, 8), (8, 100)]
    _OFFSET_BAND_LABELS = ['0-3 cm', '3-5 cm', '5-8 cm', '8+ cm']

    def _do_update_by_offset(self, data, mask, n_total, n, checked):
        use_signed = self.error_combo.currentText().startswith("Signed")
        _pub = self.publication_checkbox.isChecked()

        # Compute contact offset in cm
        dy = data.get('dy_pre', np.full(n_total, np.nan))
        dz = data.get('dz_pre', np.full(n_total, np.nan))
        offset_cm = np.sqrt(dy**2 + dz**2) * 100.0  # m → cm

        n_models = len(checked)
        n_groups = len(self._OFFSET_BANDS_CM)
        bar_width = 0.6 / max(n_models, 1)
        group_spacing = max(n_models * bar_width * 1.3, 1.0) + 0.3

        grid_map = self._grid_map()

        for key, row, col in grid_map:
            pw = self.plot_widgets[(row, col)]
            is_velocity = (row == 0)

            tick_labels = []
            y_max_all = 0.01
            y_min_all = 0.0

            for g_idx, (lo, hi) in enumerate(self._OFFSET_BANDS_CM):
                g_x = g_idx * group_spacing
                band_mask = mask & (offset_cm >= lo) & (offset_cm < hi) & np.isfinite(offset_cm)
                n_band = int(np.sum(band_mask))
                label_str = self._OFFSET_BAND_LABELS[g_idx]
                tick_labels.append((g_x, f"{label_str}\nn={n_band}"))

                for m_idx, (label, suffix, color) in enumerate(checked):
                    err_data = self._get_error_data(data, band_mask, key, suffix, n_total, use_signed)
                    if err_data is None:
                        continue
                    err_data = err_data[np.isfinite(err_data)]
                    if len(err_data) == 0:
                        continue

                    x_offset = (m_idx - (n_models - 1) / 2.0) * bar_width * 1.3
                    _draw_violin(pw, err_data, g_x + x_offset, bar_width,
                                 color, is_velocity=is_velocity,
                                 show_labels=not _pub)

                    if len(err_data) > 0:
                        wlo, whi = self._whisker_extent(err_data)
                        y_max_all = max(y_max_all, whi)
                        y_min_all = min(y_min_all, wlo)

            pw.getPlotItem().getAxis('bottom').setTicks([tick_labels])
            x_lo = -group_spacing * 0.5
            x_hi = (n_groups - 1) * group_spacing + group_spacing * 0.5
            pw.setXRange(x_lo, x_hi)
            margin = max(abs(y_max_all), abs(y_min_all)) * 0.1
            pw.setYRange(y_min_all - margin if use_signed else 0,
                         y_max_all + margin)

        # ── Optional legend ───────────────────────────────────────────
        if self._legend_cb.isChecked() and n_models > 0:
            pw0 = self.plot_widgets.get((0, 0))
            if pw0 is not None:
                legend = pw0.addLegend(offset=(5, 5), labelTextSize='8pt')
                self._legends.append((pw0, legend))
                for label, suffix, color in checked:
                    _dummy = pg.ScatterPlotItem(
                        x=[0], y=[0], size=10,
                        brush=pg.mkBrush(*color, 180),
                        pen=pg.mkPen(color=color, width=2),
                    )
                    legend.addItem(_dummy, label)
                    _dummy.clear()

        frame_str = "Global" if self._is_global() else "Local (Racket)"
        model_labels = ', '.join(l.split('(')[0].strip() for l, _, _ in checked)
        err_str = "signed" if use_signed else "|error|"
        self.info_label.setText(
            f"{n} contacts | {n_groups} offset bands | {frame_str} | "
            f"Models: {model_labels} | {err_str}"
        )
        self.stats_label.setText("")

    def _get_error_data(self, data, mask, key, suffix, n_total, use_signed=False):
        """Get error array for a component key and model suffix.

        If *use_signed* is False (default), returns ``|model − observed|``.
        If *use_signed* is True, returns ``model − observed``.
        """
        _wrap = (lambda x: x) if use_signed else np.abs
        if self._is_global():
            comp_map = {'vx': 'vx_post', 'vy': 'vy_post', 'vz': 'vz_post',
                        'wx': 'wx_post', 'wy': 'wy_post', 'wz': 'wz_post'}
            if key in ('vmag', 'wmag'):
                cs = ['vx', 'vy', 'vz'] if key == 'vmag' else ['wx', 'wy', 'wz']
                obs_mag = np.sqrt(sum(data[f'{c}_post'][mask]**2 for c in cs))
                mod_keys = [f'{c}_post_{suffix}' for c in cs]
                if any(mk not in data for mk in mod_keys):
                    return None
                mod_mag = np.sqrt(sum(data[mk][mask]**2 for mk in mod_keys))
                return _wrap(mod_mag - obs_mag)
            dk = comp_map.get(key)
            mk = f'{key}_post_{suffix}'
            if dk and dk in data and mk in data:
                return _wrap(data[mk][mask] - data[dk][mask])
            return None
        else:
            # Local (racket) frame
            comp_map = {'vrx': 'vrx_post', 'vry': 'vry_post', 'vrz': 'vrz_post',
                        'wrx': 'wrx_post', 'wry': 'wry_post', 'wrz': 'wrz_post'}
            dk = comp_map.get(key)
            mk = f'{key}_post_{suffix}'
            if dk and dk in data and mk in data:
                return _wrap(data[mk][mask] - data.get(dk, np.zeros(n_total))[mask])
            return None

    # ─── save ────────────────────────────────────────────────────────

    def _save_plot(self):
        if self.rcm_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame = "global" if self._is_global() else "local"
        pub = self.publication_checkbox.isChecked()
        mag_only = self._mag_only_cb.isChecked() and self._is_global()
        ext = "pdf" if pub else "png"
        suffix = "_mag" if mag_only else ""
        fname = f"rcm_boxplot_{frame}{suffix}_{ts}.{ext}"
        output_path = plots_dir / fname

        try:
            from PySide6.QtGui import QImage, QPainter, QColor
            from PySide6.QtCore import QRect, QMarginsF
            from .fitness_boxplot import _export_publication_cell

            if not self.plot_widgets:
                return

            if mag_only:
                # Only v_mag (0,3) and w_mag (1,3) — side by side
                mag_keys = [(0, 3), (1, 3)]
                widgets = [(k, self.plot_widgets[k]) for k in mag_keys
                           if k in self.plot_widgets]
                if not widgets:
                    QMessageBox.warning(self, "No Data",
                                        "Magnitude plots not available in this frame.")
                    return
                n_rows, n_cols = 1, len(widgets)
            else:
                max_row = max(r for r, c in self.plot_widgets.keys()) + 1
                max_col = max(c for r, c in self.plot_widgets.keys()) + 1
                n_rows, n_cols = max_row, max_col
                widgets = list(self.plot_widgets.items())

            cell_w, cell_h = (800, 600) if pub else (640, 480)
            # Overlap cells to reduce inter-subplot spacing in publication mode
            gap_x = 70 if pub else 0   # absorb right+left margins
            gap_y = 60 if pub else 0   # absorb bottom+top margins
            total_w = cell_w * n_cols - gap_x * max(n_cols - 1, 0)
            total_h = cell_h * n_rows - gap_y * max(n_rows - 1, 0)

            def _cell_origin(row, col):
                return (col * (cell_w - gap_x), row * (cell_h - gap_y))

            if pub:
                # ── PDF output ──────────────────────────────────────
                from PySide6.QtGui import QPageLayout, QPageSize
                from PySide6.QtCore import QSizeF
                try:
                    from PySide6.QtGui import QPdfWriter
                except ImportError:
                    from PySide6.QtCore import QPdfWriter

                writer = QPdfWriter(str(output_path))
                page_size = QPageSize(QSizeF(total_w, total_h), QPageSize.Unit.Point)
                writer.setPageSize(page_size)
                writer.setPageMargins(QMarginsF(0, 0, 0, 0))
                writer.setResolution(150)

                painter = QPainter(writer)
                sx = writer.width() / total_w
                sy = writer.height() / total_h
                painter.scale(sx, sy)

                for (row, col), pw in widgets:
                    cell_img = _export_publication_cell(pw, cell_w, cell_h,
                                                       hide_x_labels=True)
                    if mag_only:
                        draw_col = [(0, 3), (1, 3)].index((row, col))
                        ox, oy = _cell_origin(0, draw_col)
                    else:
                        ox, oy = _cell_origin(row, col)
                    target = QRect(ox, oy, cell_w, cell_h)
                    painter.drawImage(target, cell_img)

                painter.end()
            else:
                # ── PNG output ──────────────────────────────────────
                bg_color = QColor(0, 0, 0, 255)
                combined = QImage(total_w, total_h, QImage.Format_ARGB32)
                combined.fill(bg_color)
                painter = QPainter(combined)

                for (row, col), pw in widgets:
                    exporter = ImageExporter(pw.plotItem)
                    exporter.parameters()['width'] = cell_w
                    cell_img = exporter.export(toBytes=True)
                    if mag_only:
                        draw_col = [(0, 3), (1, 3)].index((row, col))
                        ox, oy = _cell_origin(0, draw_col)
                    else:
                        ox, oy = _cell_origin(row, col)
                    target = QRect(ox, oy, cell_w, cell_h)
                    painter.drawImage(target, cell_img)

                painter.end()
                combined.save(str(output_path))

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")

    def _export_csv(self):
        """Export per-contact RCM data to CSV for comparison with publication_plots."""
        if self.rcm_data is None:
            QMessageBox.warning(self, "No Data", "No RCM data to export.")
            return

        data = self.rcm_data
        n_total = len(data.get('vx_pre', []))
        mask = self._build_mask()
        if not np.any(mask):
            QMessageBox.warning(self, "No Data", "No data in selected range.")
            return

        checked = self._checked_models()
        vel_comps = ['vx', 'vy', 'vz']
        spin_comps = ['wx', 'wy', 'wz']
        all_comps = vel_comps + spin_comps

        # Contact offset
        dy = data.get('dy_pre', np.full(n_total, np.nan))
        dz = data.get('dz_pre', np.full(n_total, np.nan))
        offset_cm = np.sqrt(dy**2 + dz**2) * 100.0

        indices = np.where(mask)[0]
        rows = []
        for idx in indices:
            row = {'index': int(idx), 'offset_cm': float(offset_cm[idx])}
            # Observed post values
            for comp in all_comps:
                key = f'{comp}_post'
                row[f'obs_{comp}'] = float(data[key][idx]) if key in data else float('nan')
            # Pre values
            for comp in all_comps:
                key = f'{comp}_pre'
                row[f'pre_{comp}'] = float(data[key][idx]) if key in data else float('nan')
            # Model post values
            for label, suffix, _color in checked:
                for comp in all_comps:
                    key = f'{comp}_post_{suffix}'
                    col_name = f'{label}_{comp}'
                    row[col_name] = float(data[key][idx]) if key in data else float('nan')
            # Confidence / RMSE / fitness
            row['confidence'] = float(data.get('confidence', np.full(n_total, np.nan))[idx])
            row['max_rmse_opt'] = float(data.get('max_rmse_opt', np.full(n_total, np.nan))[idx])
            row['fitness_post'] = float(data.get('fitness_post', np.full(n_total, np.nan))[idx])
            rows.append(row)

        if not rows:
            QMessageBox.warning(self, "No Data", "No rows to export.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = plots_dir / f'rcm_boxplot_data_{ts}.csv'

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        QMessageBox.information(
            self, "CSV Exported",
            f"Exported {len(rows)} contacts to:\n{output_path}",
        )
