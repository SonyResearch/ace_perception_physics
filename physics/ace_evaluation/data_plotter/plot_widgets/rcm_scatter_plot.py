# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Racket Contact Model (RCM) Scatter Plot Widget

Widget for visualising racket contact data: pre/post contact velocity and
spin in both global and racket-local frames.  Mirrors the structure of
``TableContactPlot`` (TCM) but uses physics-optimised data from
``extracted.csv`` loaded into :class:`RacketContactEvent` instances.

Supported view modes
--------------------
* **Global: Input vs Output** – 6×6 scatter matrix (vx/vy/vz/wx/wy/wz
  pre→post) in the global frame.
* **Global: Error** – same matrix but showing ``post − pre`` error.
* **Local: Input vs Output** – 6×5 scatter matrix in the racket frame
  (vrx/vry pre, wrx/wry/wrz pre → vrx/vry/vrz post, wrx/wry/wrz post).
* **Local: Error** – error version of the local scatter.

Colour-by options: *Racket Open Angle*, *Theta Angle*, *Fitness Pre*,
*Player*.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton,
    QMessageBox, QSlider, QGridLayout, QCheckBox, QDoubleSpinBox,
)
from PySide6.QtCore import Signal, Qt, QTimer

import pyqtgraph as pg
import matplotlib.pyplot as plt

from .base_coefficient_plot import create_combo_selector


# ═════════════════════════════════════════════════════════════════════════
# Grid definitions – mirroring plot_rcm.py
# ═════════════════════════════════════════════════════════════════════════

# Global frame: 6 rows × 6 cols
_GLOBAL_PRE_KEYS = ['vx_pre', 'vy_pre', 'vz_pre', 'wx_pre', 'wy_pre', 'wz_pre']
_GLOBAL_POST_KEYS = ['vx_post', 'vy_post', 'vz_post', 'wx_post', 'wy_post', 'wz_post']
_GLOBAL_PRE_DISPLAY = ['vx pre', 'vy pre', 'vz pre', 'wx pre', 'wy pre', 'wz pre']
_GLOBAL_POST_DISPLAY = ['vx post', 'vy post', 'vz post', 'wx post', 'wy post', 'wz post']
_GLOBAL_ERR_DISPLAY = ['Δvx', 'Δvy', 'Δvz', 'Δwx', 'Δwy', 'Δwz']

# Local (racket) frame: 6 rows × 5 cols
_LOCAL_PRE_KEYS = ['vrx_pre', 'vry_pre', 'wrx_pre', 'wry_pre', 'wrz_pre']
_LOCAL_POST_KEYS = ['vrx_post', 'vry_post', 'vrz_post', 'wrx_post', 'wry_post', 'wrz_post']
_LOCAL_PRE_DISPLAY = ['vᵣₓ pre', 'vᵣᵧ pre', 'wᵣₓ pre', 'wᵣᵧ pre', 'wᵣz pre']
_LOCAL_POST_DISPLAY = ['vᵣₓ post', 'vᵣᵧ post', 'vᵣz post', 'wᵣₓ post', 'wᵣᵧ post', 'wᵣz post']
_LOCAL_ERR_DISPLAY = ['Δvᵣₓ', 'Δvᵣᵧ', 'Δvᵣz', 'Δwᵣₓ', 'Δwᵣᵧ', 'Δwᵣz']

# Mapping from each post key to its matching pre key for delta computation.
# None when no corresponding pre key exists (e.g. vrz has no pre in the
# local frame because the normal approach speed is captured by theta_angle).
_GLOBAL_POST_TO_PRE = {
    'vx_post': 'vx_pre', 'vy_post': 'vy_pre', 'vz_post': 'vz_pre',
    'wx_post': 'wx_pre', 'wy_post': 'wy_pre', 'wz_post': 'wz_pre',
}
_LOCAL_POST_TO_PRE = {
    'vrx_post': 'vrx_pre', 'vry_post': 'vry_pre', 'vrz_post': None,
    'wrx_post': 'wrx_pre', 'wry_post': 'wry_pre', 'wrz_post': 'wrz_pre',
}

# Model-prediction suffixes → data key suffix
# 'Observed' maps to None (use the raw observed post keys)
_MODEL_SUFFIXES = {
    'Observed (pseudo-GT)': None,
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

# Global-frame model post keys (replace observed post with model post)
_GLOBAL_MODEL_POST_KEYS = {
    suffix: [f'vx_post_{suffix}', f'vy_post_{suffix}', f'vz_post_{suffix}',
             f'wx_post_{suffix}', f'wy_post_{suffix}', f'wz_post_{suffix}']
    for suffix in _MODEL_SUFFIXES.values() if suffix is not None
}

# Local (racket) frame model post keys
_LOCAL_MODEL_POST_KEYS = {
    suffix: [f'vrx_post_{suffix}', f'vry_post_{suffix}', f'vrz_post_{suffix}',
             f'wrx_post_{suffix}', f'wry_post_{suffix}', f'wrz_post_{suffix}']
    for suffix in _MODEL_SUFFIXES.values() if suffix is not None
}


class RcmScatterPlot(QWidget):
    """Scatter-matrix widget for racket contact model data (RCM).

    Provides a grid of :class:`pyqtgraph.ScatterPlotItem` cells arranged
    as *post* (rows) vs *pre* (columns), with selectable frame (global /
    local) and view mode (input-vs-output / error).
    """

    # ── Signals ──────────────────────────────────────────────────────
    contact_selected = Signal(object, object, object, object)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Debounce
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self._do_update)

        # ── Control bar ──────────────────────────────────────────────
        control_bar = QHBoxLayout()

        # View mode selector
        view_layout, self.view_combo = create_combo_selector(
            "View:",
            [
                "Global: Input vs Output",
                "Global: Error",
                "Local: Input vs Output",
                "Local: Error",
            ],
            self._on_view_changed,
        )
        control_bar.addLayout(view_layout)

        # Model selector (affects both Input vs Output and Error views)
        model_layout, self.model_combo = create_combo_selector(
            "Model:",
            list(_MODEL_SUFFIXES.keys()),
            self._schedule_update,
        )
        control_bar.addLayout(model_layout)

        # Color-by selector
        color_layout, self.color_combo = create_combo_selector(
            "Color By:",
            ["Racket Open Angle", "Theta Angle", "Player", "Shot Type", "Rally Winner", "Date",
             "Worst Spin Variance", "Worst Spin Density",
             "Contact X", "Contact Y", "Contact Z", "Contact Offset"],
            self._schedule_update,
        )
        control_bar.addLayout(color_layout)

        # Shot type filter
        shot_filter_layout, self.shot_filter_combo = create_combo_selector(
            "Shot Filter:",
            ["All", "Serve", "Rally", "Last (Won)", "Last (Lost)"],
            self._schedule_update,
        )
        control_bar.addLayout(shot_filter_layout)

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

        # Fitness max filter
        fitness_layout = QHBoxLayout()
        fitness_layout.addWidget(QLabel("Max Fitness:"))
        self.fitness_max_spin = QDoubleSpinBox()
        self.fitness_max_spin.setRange(0.0, 100.0)
        self.fitness_max_spin.setValue(0.1)
        self.fitness_max_spin.setSingleStep(0.01)
        self.fitness_max_spin.setDecimals(2)
        self.fitness_max_spin.setToolTip(
            "Exclude contacts with fitness_post > this value.\n"
            "fitness_post is the max position error of the post-contact\n"
            "trajectory fit. Set to 0 to disable."
        )
        self.fitness_max_spin.valueChanged.connect(self._schedule_update)
        fitness_layout.addWidget(self.fitness_max_spin)
        control_bar.addLayout(fitness_layout)

        # Relative ω checkbox
        self.relative_spin_cb = QCheckBox("Relative ω")
        self.relative_spin_cb.setChecked(False)
        self.relative_spin_cb.setToolTip("Subtract racket angular velocity from ball spin")
        self.relative_spin_cb.stateChanged.connect(self._on_relative_spin_changed)
        control_bar.addWidget(self.relative_spin_cb)

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self._save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Info label
        self.info_label = QLabel("Showing RCM scatter matrix")
        layout.addWidget(self.info_label)

        # Grid container
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.grid_container, stretch=1)

        self.setLayout(layout)

        # Data
        self.rcm_data: Optional[Dict[str, np.ndarray]] = None
        self.plot_items: Dict[Tuple[int, int], pg.PlotWidget] = {}
        self.scatter_items: Dict[Tuple[int, int], pg.ScatterPlotItem] = {}
        self.stat_lines: Dict[Tuple[int, int], list] = {}  # (row, col) -> list of InfiniteLine items (median, ±1σ)
        self.stat_text_items: Dict[Tuple[int, int], Any] = {}  # (row, col) -> pg.TextItem
        self._legends: list = []  # track legend items for cleanup
        self._header_labels: List[QLabel] = []  # keep ref so we can remove them
        self._current_n_rows = 0
        self._current_n_cols = 0
        self.current_filtered_indices: Optional[np.ndarray] = None
        # Highlight overlay state
        self._highlight_scatters: list = []  # [(PlotWidget, ScatterPlotItem)]
        self._cached_cell_xy: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}  # (row,col) → (x,y)

        # Build the initial matrix (will be rebuilt on view change)
        self._rebuild_grid()

    # ─── layout helpers ──────────────────────────────────────────────

    def _view_is_global(self) -> bool:
        return self.view_combo.currentText().startswith("Global")

    def _view_is_error(self) -> bool:
        return "Error" in self.view_combo.currentText()

    def _pre_keys(self) -> List[str]:
        return _GLOBAL_PRE_KEYS if self._view_is_global() else _LOCAL_PRE_KEYS

    def _post_keys(self) -> List[str]:
        return _GLOBAL_POST_KEYS if self._view_is_global() else _LOCAL_POST_KEYS

    def _pre_display(self) -> List[str]:
        labels = list(_GLOBAL_PRE_DISPLAY if self._view_is_global() else _LOCAL_PRE_DISPLAY)
        if self.relative_spin_cb.isChecked():
            labels = [f"{l} (rel)" if l.startswith(('w', 'ω')) else l for l in labels]
        return labels

    def _post_display(self) -> List[str]:
        if self._view_is_error():
            labels = list(_GLOBAL_ERR_DISPLAY if self._view_is_global() else _LOCAL_ERR_DISPLAY)
        else:
            labels = list(_GLOBAL_POST_DISPLAY if self._view_is_global() else _LOCAL_POST_DISPLAY)
        if self.relative_spin_cb.isChecked():
            labels = [f"{l} (rel)" if l.startswith(('w', 'ω', 'Δw', 'Δω')) else l for l in labels]
        return labels

    def _post_to_pre(self) -> Dict[str, Optional[str]]:
        return _GLOBAL_POST_TO_PRE if self._view_is_global() else _LOCAL_POST_TO_PRE

    # ─── grid management ─────────────────────────────────────────────

    def _rebuild_grid(self):
        """(Re)create the scatter-matrix grid to match current view dims."""
        pre_display = self._pre_display()
        post_display = self._post_display()
        n_rows = len(post_display)
        n_cols = len(pre_display)

        if n_rows == self._current_n_rows and n_cols == self._current_n_cols:
            return  # grid already correct size

        # Tear down old widgets
        for lbl in self._header_labels:
            self.grid_layout.removeWidget(lbl)
            lbl.deleteLater()
        self._header_labels.clear()

        for pw in self.plot_items.values():
            self.grid_layout.removeWidget(pw)
            pw.deleteLater()
        self.plot_items.clear()
        self.scatter_items.clear()

        # Column headers
        for col, label_text in enumerate(pre_display):
            lbl = QLabel(label_text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
            self.grid_layout.addWidget(lbl, 0, col + 1)
            self._header_labels.append(lbl)

        # Row headers + plots
        for row, row_label_text in enumerate(post_display):
            lbl = QLabel(row_label_text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
            self.grid_layout.addWidget(lbl, row + 1, 0)
            self._header_labels.append(lbl)

            for col in range(n_cols):
                pw = pg.PlotWidget()
                pw.setDefaultPadding(0.02)
                pw.showGrid(x=True, y=True, alpha=0.3)
                pw.getAxis('left').enableAutoSIPrefix(False)
                pw.getAxis('bottom').enableAutoSIPrefix(False)
                if row == n_rows - 1:
                    pw.setLabel('bottom', pre_display[col])
                if col == 0:
                    pw.setLabel('left', post_display[row])
                pw.enableAutoRange(axis='x', enable=True)
                pw.enableAutoRange(axis='y', enable=True)

                self.plot_items[(row, col)] = pw
                self.grid_layout.addWidget(pw, row + 1, col + 1)

                scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None))
                scatter.sigClicked.connect(self._on_point_clicked)
                pw.addItem(scatter)
                self.scatter_items[(row, col)] = scatter

        # Link axes – same column = same X, same row = same Y
        for col in range(n_cols):
            first = self.plot_items[(0, col)]
            for row in range(1, n_rows):
                self.plot_items[(row, col)].setXLink(first)
        for row in range(n_rows):
            first = self.plot_items[(row, 0)]
            for col in range(1, n_cols):
                self.plot_items[(row, col)].setYLink(first)

        self._current_n_rows = n_rows
        self._current_n_cols = n_cols

    # ─── callbacks ───────────────────────────────────────────────────

    def _on_view_changed(self, _text: str = ""):
        self._rebuild_grid()
        self._schedule_update()

    def _on_relative_spin_changed(self, _state: int = 0):
        """Force grid rebuild (labels change) and replot."""
        self._current_n_rows = 0  # force _rebuild_grid to actually rebuild
        self._current_n_cols = 0
        self._rebuild_grid()
        self._schedule_update()

    def _on_filter_changed(self, _=None):
        self._update_timer.start()

    def _schedule_update(self, _=None):
        self.size_label.setText(str(self.size_slider.value()))
        self._update_timer.start()

    # ─── public API ──────────────────────────────────────────────────

    def plot_data(self, rcm_data: Dict[str, np.ndarray]):
        """Load RCM arrays and render.

        ``rcm_data`` must contain at least the global-frame keys
        (``vx_pre … wz_post``) and, for local-frame views, the racket-frame
        keys (``vrx_pre … wrz_post``).  Optional keys: ``theta_angle``,
        ``racket_open_angle``, ``fitness_pre``, ``player``.
        """
        self.rcm_data = rcm_data
        self._rebuild_grid()
        self._do_update()

    # ─── update logic ────────────────────────────────────────────────

    def _do_update(self):
        """Recompute filtered data and refresh every scatter cell."""

        # Clear existing scatter data
        for sc in self.scatter_items.values():
            sc.clear()

        # Clear existing stat lines (median / ±1σ) and text labels
        for (r, c), lines in self.stat_lines.items():
            pw = self.plot_items.get((r, c))
            if pw:
                for line in lines:
                    pw.removeItem(line)
        self.stat_lines.clear()
        for (r, c), txts in self.stat_text_items.items():
            pw = self.plot_items.get((r, c))
            if pw:
                for txt in txts:
                    pw.removeItem(txt)
        self.stat_text_items.clear()

        # Remove any previous legends (shot-type / date)
        for _leg_pw, _leg in self._legends:
            try:
                _leg.clear()
                _leg_pw.removeItem(_leg)
                # Reset pyqtgraph's internal cached legend reference so
                # addLegend() will create a fresh one next time.
                if hasattr(_leg_pw, 'plotItem') and getattr(_leg_pw.plotItem, 'legend', None) is _leg:
                    _leg_pw.plotItem.legend = None
            except Exception:
                pass
        self._legends.clear()

        if self.rcm_data is None or len(self.rcm_data.get('vx_pre', [])) == 0:
            self.info_label.setText("No RCM data available")
            return

        data = self.rcm_data
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

        # RMSE filter (from global sliders, attached by app.py)
        rmse_min_slider = getattr(self, 'rmse_min_slider', None)
        rmse_max_slider = getattr(self, 'rmse_slider', None)
        if rmse_min_slider is not None or rmse_max_slider is not None:
            max_rmse = data.get('max_rmse_opt', np.full(n_total, np.nan))
            has_rmse = ~np.isnan(max_rmse)
            rmse_scale = 1000.0  # slider is 0–100 mapped to 0–0.1 m
            if rmse_min_slider is not None and rmse_min_slider.value() > 0:
                r_lo = rmse_min_slider.value() / rmse_scale
                mask &= (~has_rmse) | (max_rmse >= r_lo)
            if rmse_max_slider is not None and rmse_max_slider.value() < 100:
                r_hi = rmse_max_slider.value() / rmse_scale
                mask &= (~has_rmse) | (max_rmse <= r_hi)

        # Fitness post filter
        _fit_max = self.fitness_max_spin.value()
        if _fit_max > 0:
            _fp = data.get('fitness_post', np.full(n_total, np.nan))
            _has_fp = ~np.isnan(_fp)
            mask &= (~_has_fp) | (_fp <= _fit_max)

        # Shot type filter
        shot_filter = self.shot_filter_combo.currentText()
        if shot_filter != "All":
            shot_types_arr = data.get('shot_type', np.full(n_total, -1, dtype=int))
            pw_arr = data.get('point_winner', np.zeros(n_total, dtype=int))
            if shot_filter == "Serve":
                mask &= (shot_types_arr == 0)
            elif shot_filter == "Rally":
                mask &= (shot_types_arr == 1)
            elif shot_filter == "Last (Won)":
                mask &= (shot_types_arr == 2) & (pw_arr == 1)
            elif shot_filter == "Last (Lost)":
                mask &= (shot_types_arr == 2) & (pw_arr != 1)

        if not np.any(mask):
            self.info_label.setText("No data in selected filter range")
            return

        filtered_indices = np.where(mask)[0]
        n_points = len(filtered_indices)
        self.current_filtered_indices = filtered_indices

        # ── Gather axis data ─────────────────────────────────────────
        pre_keys = self._pre_keys()
        post_keys = self._post_keys()
        is_error = self._view_is_error()

        pre_arrays = {k: data[k][mask] for k in pre_keys}
        # Observed post values (always needed for error computation)
        observed_post = {k: data[k][mask] for k in post_keys}

        # ── Relative ω: subtract racket angular velocity from ball spin ──
        if self.relative_spin_cb.isChecked():
            if self._view_is_global():
                _spin_sub = {
                    'wx_pre': 'wgx_racket', 'wy_pre': 'wgy_racket', 'wz_pre': 'wgz_racket',
                    'wx_post': 'wgx_racket', 'wy_post': 'wgy_racket', 'wz_post': 'wgz_racket',
                }
            else:
                _spin_sub = {
                    'wrx_pre': 'wrx_racket', 'wry_pre': 'wry_racket', 'wrz_pre': 'wrz_racket',
                    'wrx_post': 'wrx_racket', 'wry_post': 'wry_racket', 'wrz_post': 'wrz_racket',
                }
            for bk, rk in _spin_sub.items():
                r_vals = data.get(rk, np.zeros(n_total))[mask]
                if bk in pre_arrays:
                    pre_arrays[bk] = pre_arrays[bk] - r_vals
                if bk in observed_post:
                    observed_post[bk] = observed_post[bk] - r_vals

        # Determine which model is selected
        model_label = self.model_combo.currentText()
        model_suffix = _MODEL_SUFFIXES.get(model_label)  # None = Observed

        # Select the right model-key mapping for the current frame
        model_key_map = _GLOBAL_MODEL_POST_KEYS if self._view_is_global() else _LOCAL_MODEL_POST_KEYS

        # Build the "output" arrays — either observed or model-predicted
        if model_suffix is None:
            # Observed (pseudo-GT)
            output_arrays = observed_post
            model_available = True
        else:
            model_keys = model_key_map[model_suffix]
            has_model = all(mk in data for mk in model_keys)
            if has_model:
                first_key = model_keys[0]
                model_available = not np.all(np.isnan(data[first_key]))
            else:
                model_available = False

            if model_available:
                output_arrays = {pk: data[mk][mask] for pk, mk in zip(post_keys, model_keys)}
                # Subtract racket ω from model-predicted spin too
                if self.relative_spin_cb.isChecked():
                    if self._view_is_global():
                        _model_spin_sub = {
                            'wx_post': 'wgx_racket', 'wy_post': 'wgy_racket', 'wz_post': 'wgz_racket',
                        }
                    else:
                        _model_spin_sub = {
                            'wrx_post': 'wrx_racket', 'wry_post': 'wry_racket', 'wrz_post': 'wrz_racket',
                        }
                    for bk, rk in _model_spin_sub.items():
                        if bk in output_arrays:
                            output_arrays[bk] = output_arrays[bk] - data.get(rk, np.zeros(n_total))[mask]
            else:
                # Fall back to observed, show warning
                output_arrays = observed_post

        # For error mode: delta = model_output − observed_post
        delta_arrays: Dict[str, np.ndarray] = {}
        if is_error:
            if model_suffix is None:
                # "Observed" selected in error mode → delta is always 0, not useful
                self.info_label.setText(
                    f"Select a model to see prediction error. "
                    f"'Observed' error is zero by definition."
                )
                return
            elif model_available:
                for pk in post_keys:
                    delta_arrays[pk] = output_arrays[pk] - observed_post[pk]
            else:
                self.info_label.setText(
                    f"No model data for '{model_label}'. "
                    f"Delete .data_plotter_cache.pkl and reload, or re-run extract_racket_contacts.py --no_cache."
                )
                return

        # Show info if model not available in Input vs Output mode
        if not is_error and not model_available and model_suffix is not None:
            self.info_label.setText(
                f"No model data for '{model_label}' — showing Observed (pseudo-GT). "
                f"Delete .data_plotter_cache.pkl and reload."
            )

        # ── Colour mapping ───────────────────────────────────────────
        color_by = self.color_combo.currentText()
        use_categorical = False

        if color_by == "Shot Type":
            # Categorical: split last shot by player and outcome
            _SHOT_CAT_COLORS = {
                0: (50, 180, 50),    # serve – green
                1: (74, 144, 226),   # rally – blue
                3: (220, 60, 60),    # last won – red
                4: (255, 140, 0),    # last lost – orange
            }
            _SHOT_CAT_LABELS = {0: "Serve", 1: "Rally", 3: "Last (Won)", 4: "Last (Lost)"}
            shot_types = data.get('shot_type', np.ones(n_total, dtype=int)).copy()
            pw_arr = data.get('point_winner', np.zeros(n_total, dtype=int))
            last_mask = shot_types == 2
            shot_types[last_mask & (pw_arr == 1)] = 3
            shot_types[last_mask & (pw_arr != 1)] = 4
            shot_cats_filtered = shot_types[mask]
            rgb = np.array([_SHOT_CAT_COLORS.get(int(sc), (128, 128, 128)) for sc in shot_cats_filtered], dtype=np.int32)
            use_categorical = True
        elif color_by == "Rally Winner":
            _WINNER_COLORS = {
                1: (50, 180, 50),    # robot won – green
                2: (220, 60, 60),    # player won – red
                0: (128, 128, 128),  # unknown – grey
            }
            _WINNER_LABELS = {1: "Robot Won", 2: "Player Won", 0: "Unknown"}
            pw_arr = data.get('point_winner', np.zeros(n_total, dtype=int))[mask]
            rgb = np.array([_WINNER_COLORS.get(int(w), (128, 128, 128)) for w in pw_arr], dtype=np.int32)
            use_categorical = True
        elif color_by == "Date":
            # Categorical: one colour per unique date string
            metadata = data.get('metadata_list', [])
            all_dates = [metadata[i].get('match_date', '') if i < len(metadata) else ''
                         for i in range(n_total)]
            filtered_dates = [all_dates[i] for i in filtered_indices]
            unique_dates = sorted(set(filtered_dates))
            _DATE_CMAP = plt.get_cmap('tab20' if len(unique_dates) <= 20 else 'turbo')
            _date_color_map = {}
            for di, d in enumerate(unique_dates):
                rgba = _DATE_CMAP(di / max(len(unique_dates) - 1, 1))
                _date_color_map[d] = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            rgb = np.array([_date_color_map.get(d, (128, 128, 128)) for d in filtered_dates], dtype=np.int32)
            _DATE_LABELS = {d: d if d else "(unknown)" for d in unique_dates}
            use_categorical = True
        else:
            if color_by == "Racket Open Angle":
                color_vals = data.get('racket_open_angle', np.full(n_total, np.nan))[mask]
                c_min, c_max = -60.0, 60.0
            elif color_by == "Theta Angle":
                color_vals = data.get('theta_angle', np.full(n_total, np.nan))[mask]
                c_min, c_max = 0.0, 90.0
            elif color_by == "Worst Spin Variance":
                color_vals = data.get('worst_spin_variance', np.full(n_total, np.nan))[mask]
                _valid = color_vals[np.isfinite(color_vals)]
                c_min = float(np.min(_valid)) if len(_valid) > 0 else 0.0
                c_max = float(np.percentile(_valid, 95)) if len(_valid) > 0 else 1.0
            elif color_by == "Worst Spin Density":
                color_vals = data.get('worst_spin_density', np.full(n_total, np.nan))[mask]
                c_min, c_max = 0.0, 100.0
            elif color_by == "Contact X":
                color_vals = data.get('rx_global', np.full(n_total, np.nan))[mask]
                _valid = color_vals[np.isfinite(color_vals)]
                c_min = float(np.min(_valid)) if len(_valid) > 0 else 0.0
                c_max = float(np.max(_valid)) if len(_valid) > 0 else 1.0
            elif color_by == "Contact Y":
                color_vals = data.get('ry_global', np.full(n_total, np.nan))[mask]
                _valid = color_vals[np.isfinite(color_vals)]
                c_min = float(np.min(_valid)) if len(_valid) > 0 else 0.0
                c_max = float(np.max(_valid)) if len(_valid) > 0 else 1.0
            elif color_by == "Contact Z":
                color_vals = data.get('rz_global', np.full(n_total, np.nan))[mask]
                _valid = color_vals[np.isfinite(color_vals)]
                c_min = float(np.min(_valid)) if len(_valid) > 0 else 0.0
                c_max = float(np.max(_valid)) if len(_valid) > 0 else 1.0
            elif color_by == "Contact Offset":
                _dy = data.get('dy_pre', np.full(n_total, np.nan))[mask]
                _dz = data.get('dz_pre', np.full(n_total, np.nan))[mask]
                color_vals = np.sqrt(_dy**2 + _dz**2)
                _valid = color_vals[np.isfinite(color_vals)]
                c_min = 0.0
                c_max = float(np.percentile(_valid, 95)) if len(_valid) > 0 else 0.1
            else:  # Player
                color_vals = data.get('player', np.ones(n_total))[mask]
                c_min, c_max = 1.0, 2.0

            cmap = plt.get_cmap('turbo')
            clipped = np.clip(np.nan_to_num(color_vals, nan=c_min), c_min, c_max)
            denom = (c_max - c_min) if c_max > c_min else 1.0
            norm = (clipped - c_min) / denom
            rgba = cmap(norm)
            rgb = (rgba[:, :3] * 255).astype(np.int32)

        point_size = self.size_slider.value()

        # ── Clear previous highlight overlays ────────────────────────
        self._clear_highlights()
        self._cached_cell_xy.clear()

        # ── Fill scatter cells ───────────────────────────────────────
        for row, post_key in enumerate(post_keys):
            for col, pre_key in enumerate(pre_keys):
                scatter = self.scatter_items[(row, col)]
                x = pre_arrays[pre_key]

                if is_error:
                    y = delta_arrays[post_key]
                else:
                    y = output_arrays[post_key]

                # Cache x/y for cross-highlight
                self._cached_cell_xy[(row, col)] = (x, y)

                scatter.setData(
                    x=x, y=y,
                    size=point_size,
                    pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                    brush=[
                        pg.mkBrush(int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), 180)
                        for i in range(n_points)
                    ],
                    data=filtered_indices,
                )

                # ── Add median and ±1σ horizontal lines (error views only) ──
                if is_error and len(y) > 0:
                    _med = float(np.nanmedian(y))
                    _std = float(np.nanstd(y))
                    pw = self.plot_items[(row, col)]
                    _lines = []
                    for _val, _style, _w in [
                        (_med, Qt.SolidLine, 2),
                        (_med + _std, Qt.DashLine, 1),
                        (_med - _std, Qt.DashLine, 1),
                    ]:
                        _line = pg.InfiniteLine(
                            pos=_val, angle=0,
                            pen=pg.mkPen(color=(255, 255, 0, 180), width=_w, style=_style),
                        )
                        pw.addItem(_line)
                        _lines.append(_line)
                    self.stat_lines[(row, col)] = _lines

                    # Show numeric labels next to each line (first column only)
                    if col == 0:
                        vb = pw.getViewBox()
                        vr = vb.viewRange()
                        _x_right = vr[0][1]  # right edge of view
                        _texts = []
                        for _label, _ypos in [
                            (f'med={_med:.4g}', _med),
                            (f'+σ={_med + _std:.4g}', _med + _std),
                            (f'−σ={_med - _std:.4g}', _med - _std),
                        ]:
                            _txt = pg.TextItem(
                                html=f'<span style="color:#ffff00;font-size:8pt">{_label}</span>',
                                anchor=(1, 1),
                            )
                            _txt.setZValue(100)
                            pw.addItem(_txt, ignoreBounds=True)
                            _txt.setPos(_x_right, _ypos)
                            _texts.append(_txt)
                        self.stat_text_items[(row, col)] = _texts

        view = self.view_combo.currentText()
        info_text = (
            f"Showing {n_points}/{n_total} contacts | View: {view} | "
            f"Model: {model_label} | Color: {color_by}"
        )
        if use_categorical:
            if color_by == "Shot Type":
                counts = {c: int(np.sum(shot_cats_filtered == c)) for c in _SHOT_CAT_LABELS}
                parts = [f"{_SHOT_CAT_LABELS[c]}: {counts[c]}" for c in _SHOT_CAT_LABELS if counts[c] > 0]
                info_text += "  [" + ", ".join(parts) + "]"
                _legend_items = [(_SHOT_CAT_COLORS[sc], lbl) for sc, lbl in _SHOT_CAT_LABELS.items() if counts.get(sc, 0) > 0]
            elif color_by == "Rally Winner":
                counts = {w: int(np.sum(pw_arr == w)) for w in _WINNER_LABELS}
                parts = [f"{_WINNER_LABELS[w]}: {counts[w]}" for w in _WINNER_LABELS if counts[w] > 0]
                info_text += "  [" + ", ".join(parts) + "]"
                _legend_items = [(_WINNER_COLORS[w], lbl) for w, lbl in _WINNER_LABELS.items() if counts.get(w, 0) > 0]
            elif color_by == "Date":
                counts = {}
                for d in filtered_dates:
                    counts[d] = counts.get(d, 0) + 1
                parts = [f"{_DATE_LABELS[d]}: {counts.get(d, 0)}" for d in unique_dates]
                info_text += "  [" + ", ".join(parts) + "]"
                _legend_items = [(_date_color_map[d], _DATE_LABELS[d]) for d in unique_dates]
            else:
                _legend_items = []

            # Add legend items in top-right plot cell
            pw0 = self.plot_items.get((0, self._current_n_cols - 1))
            if pw0 is not None:
                legend = pw0.addLegend(offset=(5, 5), labelTextSize='8pt')
                self._legends.append((pw0, legend))
                for _c, _label in _legend_items:
                    _dummy = pg.ScatterPlotItem(
                        x=[0], y=[0], size=8,
                        brush=pg.mkBrush(*_c, 180),
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                    )
                    legend.addItem(_dummy, _label)
                    _dummy.clear()
        self.info_label.setText(info_text)

    # ─── highlight helpers ───────────────────────────────────────────

    def _clear_highlights(self):
        """Remove all highlight overlays from scatter cells."""
        for pw, item in self._highlight_scatters:
            try:
                pw.removeItem(item)
            except Exception:
                pass
        self._highlight_scatters.clear()

    def highlight_point(self, global_idx: int):
        """Highlight the sample with the given global rcm_data index across all scatter cells."""
        self._clear_highlights()
        if self.current_filtered_indices is None or len(self._cached_cell_xy) == 0:
            return
        positions = np.where(self.current_filtered_indices == global_idx)[0]
        if len(positions) == 0:
            return
        pos = positions[0]

        highlight_size = self.size_slider.value() * 3
        highlight_pen = pg.mkPen(color=(255, 255, 255), width=2)
        highlight_brush = pg.mkBrush(255, 255, 0, 220)

        for (row, col), (x_arr, y_arr) in self._cached_cell_xy.items():
            pw = self.plot_items.get((row, col))
            if pw is None:
                continue
            x_val = x_arr[pos]
            y_val = y_arr[pos]
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue
            sc = pg.ScatterPlotItem(
                x=[x_val], y=[y_val],
                size=highlight_size, pen=highlight_pen,
                brush=highlight_brush, symbol='o',
            )
            pw.addItem(sc)
            self._highlight_scatters.append((pw, sc))

    # ─── click handler ───────────────────────────────────────────────

    def _on_point_clicked(self, _plot_item, points):
        if not points:
            return
        pt = points[0]
        idx = pt.data()
        if idx is None or self.rcm_data is None:
            return
        idx = int(idx)
        self.highlight_point(idx)
        metadata = self.rcm_data.get('metadata_list', [])
        shot_list = self.rcm_data.get('shot_list', [])
        rally_list = self.rcm_data.get('rally_list', [])
        fs_pre_list = self.rcm_data.get('fs_pre_list', [])
        fs_post_list = self.rcm_data.get('fs_post_list', [])
        md = metadata[idx] if idx < len(metadata) else None
        shot = shot_list[idx] if idx < len(shot_list) else None
        rally = rally_list[idx] if idx < len(rally_list) else None

        contact_point_data = {'_rcm_global_idx': idx}
        for k in ['vx_pre', 'vy_pre', 'vz_pre', 'wx_pre', 'wy_pre', 'wz_pre',
                   'vx_post', 'vy_post', 'vz_post', 'wx_post', 'wy_post', 'wz_post']:
            contact_point_data[k] = float(self.rcm_data[k][idx])
        # Attach flight segments for trajectory display
        if idx < len(fs_pre_list):
            contact_point_data['fs_pre'] = fs_pre_list[idx]
        if idx < len(fs_post_list):
            contact_point_data['fs_post'] = fs_post_list[idx]
        self.contact_selected.emit(contact_point_data, md, shot, rally)

    # ─── save ────────────────────────────────────────────────────────

    def _save_plot(self):
        if self.rcm_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save.")
            return
        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"rcm_scatter_matrix_{ts}.png"
        out = plots_dir / fname
        try:
            px = self.grid_container.grab()
            px.save(str(out), "PNG")
            QMessageBox.information(self, "Success", f"Plot saved to:\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")
