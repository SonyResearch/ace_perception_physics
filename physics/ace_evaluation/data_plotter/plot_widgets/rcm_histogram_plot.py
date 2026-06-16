# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Racket Contact Model (RCM) Histogram Widget

Shows overlaid histograms of input and output quantity distributions,
split by shot type (Serve / Rally / Last Won / Last Lost).

Layout
------
Configurable grid of subplots – one per selected quantity.
Each subplot overlays one semi-transparent histogram per shot-type category.
"""

from typing import Dict, Any, Optional, List
import numpy as np

import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QGridLayout, QCheckBox, QPushButton, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from .base_coefficient_plot import create_combo_selector


# ── Quantity definitions ─────────────────────────────────────────────
# Layout: 3 columns × 4 rows for Global/Local (input vel, input spin,
#         output vel, output spin).  "Other" uses flexible layout.
# (display_label, data_key, unit)
_GLOBAL_QUANTITIES = [
    # row 0: input velocity
    ("vx pre",  "vx_pre",  "m/s"),
    ("vy pre",  "vy_pre",  "m/s"),
    ("vz pre",  "vz_pre",  "m/s"),
    # row 1: input spin
    ("ωx pre",  "wx_pre",  "rad/s"),
    ("ωy pre",  "wy_pre",  "rad/s"),
    ("ωz pre",  "wz_pre",  "rad/s"),
    # row 2: output velocity
    ("vx post", "vx_post", "m/s"),
    ("vy post", "vy_post", "m/s"),
    ("vz post", "vz_post", "m/s"),
    # row 3: output spin
    ("ωx post", "wx_post", "rad/s"),
    ("ωy post", "wy_post", "rad/s"),
    ("ωz post", "wz_post", "rad/s"),
]

_LOCAL_QUANTITIES = [
    # row 0: input velocity (2 components in racket frame)
    ("vrx pre",  "vrx_pre",  "m/s"),
    ("vry pre",  "vry_pre",  "m/s"),
    ("",         None,        ""),
    # row 1: input spin
    ("ωrx pre",  "wrx_pre",  "rad/s"),
    ("ωry pre",  "wry_pre",  "rad/s"),
    ("ωrz pre",  "wrz_pre",  "rad/s"),
    # row 2: output velocity
    ("vrx post", "vrx_post", "m/s"),
    ("vry post", "vry_post", "m/s"),
    ("vrz post", "vrz_post", "m/s"),
    # row 3: output spin
    ("ωrx post", "wrx_post", "rad/s"),
    ("ωry post", "wry_post", "rad/s"),
    ("ωrz post", "wrz_post", "rad/s"),
]

_OTHER_QUANTITIES = [
    ("θ angle",         "theta_angle",        "rad"),
    ("Racket open ∠",   "racket_open_angle",  "rad"),
    ("",                None,                  ""),
    ("Δx (ball disp.)", "dx_pre",             "m"),
    ("Δy (ball disp.)", "dy_pre",             "m"),
    ("Δz (ball disp.)", "dz_pre",             "m"),
    ("vrx racket",      "vrx_racket",         "m/s"),
    ("vry racket",      "vry_racket",         "m/s"),
    ("vrz racket",      "vrz_racket",         "m/s"),
    ("ωrx racket",      "wrx_racket",         "rad/s"),
    ("ωry racket",      "wry_racket",         "rad/s"),
    ("ωrz racket",      "wrz_racket",         "rad/s"),
]

_VIEW_QUANTITIES = {
    "Global Frame":   _GLOBAL_QUANTITIES,
    "Local Frame":    _LOCAL_QUANTITIES,
    "Other":          _OTHER_QUANTITIES,
}

# ── Shot-type categories ─────────────────────────────────────────────
_SHOT_CAT_LABELS = {
    0: "Serve",
    1: "Rally",
    3: "Last (Won)",
    4: "Last (Lost)",
}

_SHOT_CAT_COLORS = {
    0: (50, 180, 50, 100),     # serve – green
    1: (74, 144, 226, 100),    # rally – blue
    3: (220, 60, 60, 100),     # last won – red
    4: (255, 140, 0, 100),     # last lost – orange
}

_SHOT_CAT_COLORS_LINE = {
    0: (50, 180, 50, 220),
    1: (74, 144, 226, 220),
    3: (220, 60, 60, 220),
    4: (255, 140, 0, 220),
}

_CAT_ORDER = [0, 1, 3, 4]


class RcmHistogramPlot(QWidget):
    """Overlaid histograms of RCM input/output quantities split by shot type."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Debounce timer
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self._do_update)

        # ── Control bar ──────────────────────────────────────────────
        control_bar = QHBoxLayout()

        # View selector
        view_layout, self.view_combo = create_combo_selector(
            "View:",
            list(_VIEW_QUANTITIES.keys()),
            self._on_view_changed,
        )
        control_bar.addLayout(view_layout)

        # Bin count slider
        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Bins:"))
        self.bin_slider = QSlider(Qt.Horizontal)
        self.bin_slider.setMinimum(10)
        self.bin_slider.setMaximum(200)
        self.bin_slider.setValue(50)
        self.bin_slider.setFixedWidth(120)
        self.bin_slider.valueChanged.connect(self._schedule_update)
        self.bin_label = QLabel("50")
        bin_layout.addWidget(self.bin_slider)
        bin_layout.addWidget(self.bin_label)
        control_bar.addLayout(bin_layout)

        # Normalize toggle
        self.normalize_cb = QCheckBox("Normalize")
        self.normalize_cb.setChecked(True)
        self.normalize_cb.stateChanged.connect(self._schedule_update)
        control_bar.addWidget(self.normalize_cb)

        control_bar.addStretch()

        # Shot-type checkboxes
        control_bar.addWidget(QLabel("Shot Types:"))
        self._shot_cbs: Dict[int, QCheckBox] = {}
        for cat in _CAT_ORDER:
            cb = QCheckBox(_SHOT_CAT_LABELS[cat])
            cb.setChecked(True)
            cb.stateChanged.connect(self._schedule_update)
            control_bar.addWidget(cb)
            self._shot_cbs[cat] = cb

        # Save button
        save_btn = QPushButton("Save Plot")
        save_btn.setMaximumWidth(100)
        save_btn.clicked.connect(self._save_plot)
        control_bar.addWidget(save_btn)

        layout.addLayout(control_bar)

        # ── Info label ───────────────────────────────────────────────
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        # ── Plot grid ────────────────────────────────────────────────
        self.plot_grid = pg.GraphicsLayoutWidget()
        self.plot_grid.setBackground('w')
        layout.addWidget(self.plot_grid, stretch=1)

        self.setLayout(layout)

        # State
        self.rcm_data: Optional[Dict[str, Any]] = None
        self.plot_items: Dict[tuple, pg.PlotItem] = {}
        self._legends: list = []
        # Highlight state
        self._highlight_lines: list = []  # [(PlotItem, InfiniteLine)]
        self._cached_cell_keys: Dict[tuple, str] = {}  # (row,col) → data key

    # ── Public API ───────────────────────────────────────────────────

    def plot_data(self, rcm_data: Dict[str, Any]):
        self.rcm_data = rcm_data
        self._rebuild_grid()
        self._do_update()

    # ── Internal ─────────────────────────────────────────────────────

    def _schedule_update(self, *_args):
        self.bin_label.setText(str(self.bin_slider.value()))
        self._update_timer.start()

    def _on_view_changed(self, _text: str):
        self._rebuild_grid()
        self._do_update()

    def _current_quantities(self):
        return _VIEW_QUANTITIES.get(self.view_combo.currentText(), _GLOBAL_QUANTITIES)

    def _rebuild_grid(self):
        self.plot_grid.clear()
        self.plot_items.clear()
        self._legends = []

        quantities = self._current_quantities()
        n_cols = 3

        for i, (label, key, unit) in enumerate(quantities):
            if key is None:
                continue  # placeholder for empty cells
            row = i // n_cols
            col = i % n_cols
            pw = self.plot_grid.addPlot(row=row, col=col)
            pw.setTitle(f"{label} ({unit})", size="9pt")
            pw.setLabel("bottom", label)
            pw.setLabel("left", "Density" if self.normalize_cb.isChecked() else "Count")
            self.plot_items[(row, col)] = pw

    def _build_shot_categories(self, data):
        """Build extended shot category array: 0=serve, 1=rally, 3=last_won, 4=last_lost."""
        n = len(data.get('vx_pre', []))
        shot_types = data.get('shot_type', np.full(n, -1, dtype=int)).copy()
        pw_arr = data.get('point_winner', np.zeros(n, dtype=int))
        last_mask = shot_types == 2
        shot_types[last_mask & (pw_arr == 1)] = 3
        shot_types[last_mask & (pw_arr != 1)] = 4
        return shot_types

    def _do_update(self):
        if self.rcm_data is None:
            return

        data = self.rcm_data
        n_total = len(data.get('vx_pre', []))
        if n_total == 0:
            self.info_label.setText("No RCM data")
            return

        # Apply confidence / RMSE filters if sliders exist
        mask = np.ones(n_total, dtype=bool)
        confidence = data.get('confidence', np.ones(n_total))
        max_rmse = data.get('max_rmse_opt', np.full(n_total, np.nan))

        conf_min_slider = getattr(self, 'conf_min_slider', None)
        conf_max_slider = getattr(self, 'conf_max_slider', None)
        if conf_min_slider is not None and conf_min_slider.value() > 0:
            mask &= confidence >= (conf_min_slider.value() / 100.0)
        if conf_max_slider is not None and conf_max_slider.value() < 100:
            mask &= confidence <= (conf_max_slider.value() / 100.0)

        rmse_min_slider = getattr(self, 'rmse_min_slider', None)
        rmse_max_slider = getattr(self, 'rmse_slider', None)
        if rmse_min_slider is not None or rmse_max_slider is not None:
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

        shot_cats = self._build_shot_categories(data)
        n_bins = self.bin_slider.value()
        normalize = self.normalize_cb.isChecked()

        # Determine which categories are checked
        active_cats = [c for c in _CAT_ORDER if self._shot_cbs[c].isChecked()]
        if not active_cats:
            self.info_label.setText("No shot types selected")
            return

        quantities = self._current_quantities()
        n_cols = 3

        cat_counts = {}
        for cat in active_cats:
            cat_counts[cat] = int(np.sum(mask & (shot_cats == cat)))

        # Remove old data items from all plots (keep axes/labels)
        self._clear_highlights()
        self._cached_cell_keys.clear()
        for pw in self.plot_items.values():
            for item in list(pw.items):
                if isinstance(item, pg.PlotCurveItem):
                    pw.removeItem(item)
        # Remove old legends
        for pw, legend in self._legends:
            pw.scene().removeItem(legend)
        self._legends = []

        # Add legend to first plot only
        first_pw = None
        for i, (label, key, unit) in enumerate(quantities):
            if key is None:
                continue
            row = i // n_cols
            col = i % n_cols
            pw = self.plot_items.get((row, col))
            if pw is not None:
                first_pw = pw
                break

        legend = None
        if first_pw is not None:
            legend = first_pw.addLegend(offset=(5, 5), labelTextSize='7pt')
            self._legends.append((first_pw, legend))
            # Pre-populate legend with all active categories so it persists
            # even when some categories have no data on the first subplot
            for cat in active_cats:
                dummy = pg.PlotCurveItem(
                    pen=pg.mkPen(color=_SHOT_CAT_COLORS_LINE[cat], width=1.5),
                    brush=pg.mkBrush(*_SHOT_CAT_COLORS[cat]),
                )
                legend.addItem(dummy, f"{_SHOT_CAT_LABELS[cat]} (n={cat_counts[cat]})")

        for i, (label, key, unit) in enumerate(quantities):
            if key is None:
                continue
            row = i // n_cols
            col = i % n_cols
            pw = self.plot_items.get((row, col))
            if pw is None:
                continue

            # Cache data key for cross-highlight
            self._cached_cell_keys[(row, col)] = key

            arr = data.get(key)
            if arr is None:
                continue

            # Compute shared bin edges across all categories for this quantity
            all_valid = arr[mask & np.isfinite(arr)]
            if len(all_valid) == 0:
                continue
            bin_edges = np.linspace(np.min(all_valid), np.max(all_valid), n_bins + 1)

            for cat in active_cats:
                cat_mask = mask & (shot_cats == cat) & np.isfinite(arr)
                vals = arr[cat_mask]
                if len(vals) == 0:
                    continue

                counts, _ = np.histogram(vals, bins=bin_edges)
                if normalize and len(vals) > 0:
                    bin_width = bin_edges[1] - bin_edges[0]
                    if bin_width > 0:
                        y = counts.astype(float) / (len(vals) * bin_width)
                    else:
                        y = counts.astype(float)
                else:
                    y = counts.astype(float)

                fill_color = _SHOT_CAT_COLORS[cat]
                line_color = _SHOT_CAT_COLORS_LINE[cat]

                # stepMode='center': x must be bin_edges (len = n_bins+1), y = counts (len = n_bins)
                curve = pg.PlotCurveItem(
                    x=bin_edges, y=y,
                    stepMode='center',
                    pen=pg.mkPen(color=line_color, width=1.5),
                    fillLevel=0,
                    fillOutline=True,
                    brush=pg.mkBrush(*fill_color),
                )
                pw.addItem(curve)

            pw.setLabel("left", "Density" if normalize else "Count")

        parts = [f"{_SHOT_CAT_LABELS[c]}: {cat_counts[c]}" for c in active_cats]
        self.info_label.setText(
            f"Showing {int(np.sum(mask))}/{n_total} contacts | "
            + ", ".join(parts)
        )

    # ─── highlight helpers ───────────────────────────────────────────

    def _clear_highlights(self):
        """Remove all highlight overlays from histogram plots."""
        for pw, line in self._highlight_lines:
            try:
                pw.removeItem(line)
            except Exception:
                pass
        self._highlight_lines.clear()

    def highlight_point(self, global_idx: int):
        """Highlight the sample with the given global rcm_data index as vertical lines on each histogram."""
        self._clear_highlights()
        if self.rcm_data is None or len(self._cached_cell_keys) == 0:
            return
        data = self.rcm_data
        for (row, col), key in self._cached_cell_keys.items():
            pw = self.plot_items.get((row, col))
            if pw is None:
                continue
            arr = data.get(key)
            if arr is None or global_idx >= len(arr):
                continue
            val = arr[global_idx]
            if not np.isfinite(val):
                continue
            line = pg.InfiniteLine(
                pos=val, angle=90,
                pen=pg.mkPen(color=(255, 255, 0), width=2, style=Qt.DashLine),
            )
            pw.addItem(line)
            self._highlight_lines.append((pw, line))

    # ─── save ────────────────────────────────────────────────────────

    def _save_plot(self):
        if self.rcm_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        view = self.view_combo.currentText().lower().replace(" ", "_")
        fname = f"rcm_histogram_{view}_{ts}.png"
        output_path = plots_dir / fname

        try:
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtCore import QRect

            if not self.plot_items:
                return
            max_row = max(r for r, c in self.plot_items.keys()) + 1
            max_col = max(c for r, c in self.plot_items.keys()) + 1

            cell_w, cell_h = 640, 480
            combined = QImage(cell_w * max_col, cell_h * max_row, QImage.Format_ARGB32)
            combined.fill(0xFF000000)
            painter = QPainter(combined)

            for (row, col), pw in self.plot_items.items():
                exporter = ImageExporter(pw)
                exporter.parameters()['width'] = cell_w
                cell_img = exporter.export(toBytes=True)
                target = QRect(col * cell_w, row * cell_h, cell_w, cell_h)
                painter.drawImage(target, cell_img)

            painter.end()
            combined.save(str(output_path))
            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")
