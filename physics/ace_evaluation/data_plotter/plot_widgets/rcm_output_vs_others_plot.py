# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
RCM Output vs Others Plot Widget

Scatter-matrix widget for viewing post-contact quantities (local frame)
vs selectable x-axis groups:
  * **Contact Location** – drx_pre, dry_pre, drz_pre
  * **Racket Velocity (Local Frame)** – vrx_racket, vry_racket, vrz_racket
  * **Racket Angular Velocity (Local Frame)** – wrx_racket, wry_racket, wrz_racket

Mirrors the ``RCM_racket_frame_output_error_vs_others.html`` plot from
``plot_rcm.py`` but as an interactive pyqtgraph widget inside the
data_plotter application.  Supports both *output* and *error* views, a
model selector, colour-by options, fitness and size sliders.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton,
    QMessageBox, QSlider, QGridLayout,
)
from PySide6.QtCore import Signal, Qt, QTimer

import pyqtgraph as pg
import matplotlib.pyplot as plt

from .base_coefficient_plot import create_combo_selector


# ═════════════════════════════════════════════════════════════════════════
# X-axis group definitions
# ═════════════════════════════════════════════════════════════════════════

_X_GROUPS = {
    "Contact Location": {
        "keys":    ["drx_pre", "dry_pre", "drz_pre"],
        "display": ["dᵣₓ pre", "dᵣᵧ pre", "dᵣz pre"],
    },
    "Racket Velocity (Local Frame)": {
        "keys":    ["vrx_racket", "vry_racket", "vrz_racket"],
        "display": ["vᵣₓ racket", "vᵣᵧ racket", "vᵣz racket"],
    },
    "Racket Angular Velocity (Local Frame)": {
        "keys":    ["wrx_racket", "wry_racket", "wrz_racket", "w_mag_racket"],
        "display": ["wᵣₓ racket", "wᵣᵧ racket", "wᵣz racket", "|ω| racket"],
    },
    "Racket Angular Velocity (Global Frame)": {
        "keys":    ["wgx_racket", "wgy_racket", "wgz_racket"],
        "display": ["ωₓ global", "ωᵧ global", "ωz global"],
    },
    "Racket Angular Velocity (Body Frame)": {
        "keys":    ["wbx_racket", "wby_racket", "wbz_racket"],
        "display": ["ωᵦₓ body", "ωᵦᵧ body", "ωᵦz body"],
    },
    "ω × r (Contact Tangential Vel)": {
        "keys":    ["wxr_x", "wxr_y", "wxr_z", "wxr_mag"],
        "display": ["(ω×r)ₓ", "(ω×r)ᵧ", "(ω×r)z", "|ω×r|"],
    },
}

# Y-axis: always the local-frame post-contact quantities
_POST_KEYS = ["vrx_post", "vry_post", "vrz_post", "wrx_post", "wry_post", "wrz_post"]
_POST_DISPLAY = ["vᵣₓ post", "vᵣᵧ post", "vᵣz post", "wᵣₓ post", "wᵣᵧ post", "wᵣz post"]
_ERR_DISPLAY  = ["Δvᵣₓ", "Δvᵣᵧ", "Δvᵣz", "Δwᵣₓ", "Δwᵣᵧ", "Δwᵣz"]

# Model-prediction suffixes → data key suffix
_MODEL_SUFFIXES = {
    "Observed (pseudo-GT)": None,
    "Default (constant COR)": "default",
    "RCM Tangential": "rcm_tangential",
    "C++ tangential": "cpp_tangential",
    "Nakashima (refined)": "nakashima_refined",
    "C++ no-residual (refined)": "cpp_refined",
    "RCM Tangential (refined)": "rcm_tangential_refined",
    "RCM Tangential (polyfit)": "rcm_tangential_polyfit",
    "ONNX alex": "onnx_alex",
    "ONNX alex (refined)": "onnx_alex_refined",
    "ONNX alex (polyfit)": "onnx_alex_polyfit",
}

# Local (racket) frame model post keys
_LOCAL_MODEL_POST_KEYS = {
    suffix: [f"vrx_post_{suffix}", f"vry_post_{suffix}", f"vrz_post_{suffix}",
             f"wrx_post_{suffix}", f"wry_post_{suffix}", f"wrz_post_{suffix}"]
    for suffix in _MODEL_SUFFIXES.values() if suffix is not None
}


class RcmOutputVsOthersPlot(QWidget):
    """Scatter-matrix widget: post-contact output (local frame) vs selectable
    x-axis groups (contact location, racket velocity, racket angular velocity).
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

        # X-axis group selector
        xgroup_layout, self.xgroup_combo = create_combo_selector(
            "X Axis:",
            list(_X_GROUPS.keys()),
            self._on_xgroup_changed,
        )
        control_bar.addLayout(xgroup_layout)

        # View mode: Output vs Error
        view_layout, self.view_combo = create_combo_selector(
            "View:",
            ["Output", "Error"],
            self._schedule_update,
        )
        control_bar.addLayout(view_layout)

        # Model selector
        model_layout, self.model_combo = create_combo_selector(
            "Model:",
            list(_MODEL_SUFFIXES.keys()),
            self._schedule_update,
        )
        control_bar.addLayout(model_layout)

        # Color-by selector
        color_layout, self.color_combo = create_combo_selector(
            "Color By:",
            ["Racket Open Angle", "Theta Angle", "Player", "Shot Type", "Rally Winner", "Date"],
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

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self._save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Info label
        self.info_label = QLabel("Showing RCM output vs others")
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
        self.stat_lines: Dict[Tuple[int, int], list] = {}
        self.stat_text_items: Dict[Tuple[int, int], Any] = {}
        self._legends: list = []
        self._header_labels: List[QLabel] = []
        self._current_n_rows = 0
        self._current_n_cols = 0
        self.current_filtered_indices: Optional[np.ndarray] = None

        # Build the initial matrix
        self._rebuild_grid()

    # ─── helpers ─────────────────────────────────────────────────────

    def _current_x_group(self) -> Dict:
        return _X_GROUPS[self.xgroup_combo.currentText()]

    def _is_error(self) -> bool:
        return self.view_combo.currentText() == "Error"

    def _x_keys(self) -> List[str]:
        return self._current_x_group()["keys"]

    def _x_display(self) -> List[str]:
        return self._current_x_group()["display"]

    def _y_display(self) -> List[str]:
        return _ERR_DISPLAY if self._is_error() else _POST_DISPLAY

    # ─── grid management ─────────────────────────────────────────────

    def _rebuild_grid(self):
        x_display = self._x_display()
        y_display = self._y_display()
        n_rows = len(y_display)
        n_cols = len(x_display)

        if n_rows == self._current_n_rows and n_cols == self._current_n_cols:
            # Just update labels (they may change between Output/Error)
            self._update_labels()
            return

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
        for col, label_text in enumerate(x_display):
            lbl = QLabel(label_text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
            self.grid_layout.addWidget(lbl, 0, col + 1)
            self._header_labels.append(lbl)

        # Row headers + plots
        for row, row_label_text in enumerate(y_display):
            lbl = QLabel(row_label_text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
            self.grid_layout.addWidget(lbl, row + 1, 0)
            self._header_labels.append(lbl)

            for col in range(n_cols):
                pw = pg.PlotWidget()
                pw.setDefaultPadding(0.02)
                pw.showGrid(x=True, y=True, alpha=0.3)
                pw.getAxis("left").enableAutoSIPrefix(False)
                pw.getAxis("bottom").enableAutoSIPrefix(False)
                if row == n_rows - 1:
                    pw.setLabel("bottom", x_display[col])
                if col == 0:
                    pw.setLabel("left", y_display[row])
                pw.enableAutoRange(axis="x", enable=True)
                pw.enableAutoRange(axis="y", enable=True)

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

    def _update_labels(self):
        """Update header labels without rebuilding the grid."""
        x_display = self._x_display()
        y_display = self._y_display()
        n_cols = len(x_display)
        idx = 0
        for col in range(n_cols):
            if idx < len(self._header_labels):
                self._header_labels[idx].setText(x_display[col])
            idx += 1
        for row in range(len(y_display)):
            if idx < len(self._header_labels):
                self._header_labels[idx].setText(y_display[row])
            idx += 1

    # ─── callbacks ───────────────────────────────────────────────────

    def _on_xgroup_changed(self, _text: str = ""):
        # Force a full grid rebuild since column count may change
        self._current_n_rows = 0
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
        """Load RCM arrays and render."""
        self.rcm_data = rcm_data
        self._rebuild_grid()
        self._do_update()

    # ─── update logic ────────────────────────────────────────────────

    def _do_update(self):
        """Recompute filtered data and refresh every scatter cell."""

        # Clear existing scatter data
        for sc in self.scatter_items.values():
            sc.clear()

        # Clear stat lines / text
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

        # Remove legends
        for _leg_pw, _leg in self._legends:
            try:
                _leg.clear()
                _leg_pw.removeItem(_leg)
                if hasattr(_leg_pw, "plotItem") and getattr(_leg_pw.plotItem, "legend", None) is _leg:
                    _leg_pw.plotItem.legend = None
            except Exception:
                pass
        self._legends.clear()

        if self.rcm_data is None or len(self.rcm_data.get("vx_pre", [])) == 0:
            self.info_label.setText("No RCM data available")
            return

        data = self.rcm_data
        n_total = len(data["vx_pre"])

        # ── Build mask ───────────────────────────────────────────────
        mask = np.ones(n_total, dtype=bool)

        # Confidence filter (from global sliders, attached by app.py)
        conf_min_slider = getattr(self, "conf_min_slider", None)
        conf_max_slider = getattr(self, "conf_max_slider", None)
        if conf_min_slider is not None or conf_max_slider is not None:
            confidence = data.get("confidence", np.full(n_total, np.nan))
            has_conf = ~np.isnan(confidence)
            if conf_min_slider is not None and conf_min_slider.value() > 0:
                c_lo = conf_min_slider.value() / 100.0
                mask &= (~has_conf) | (confidence >= c_lo)
            if conf_max_slider is not None and conf_max_slider.value() < 100:
                c_hi = conf_max_slider.value() / 100.0
                mask &= (~has_conf) | (confidence <= c_hi)

        # RMSE filter
        rmse_min_slider = getattr(self, "rmse_min_slider", None)
        rmse_max_slider = getattr(self, "rmse_slider", None)
        if rmse_min_slider is not None or rmse_max_slider is not None:
            max_rmse = data.get("max_rmse_opt", np.full(n_total, np.nan))
            has_rmse = ~np.isnan(max_rmse)
            rmse_scale = 1000.0
            if rmse_min_slider is not None and rmse_min_slider.value() > 0:
                r_lo = rmse_min_slider.value() / rmse_scale
                mask &= (~has_rmse) | (max_rmse >= r_lo)
            if rmse_max_slider is not None and rmse_max_slider.value() < 100:
                r_hi = rmse_max_slider.value() / rmse_scale
                mask &= (~has_rmse) | (max_rmse <= r_hi)

        # Fitness post filter (shared via app.py from scatter plot)
        _fit_max_spin = getattr(self, "fitness_max_spin", None)
        if _fit_max_spin is not None:
            _fit_max = _fit_max_spin.value()
            if _fit_max > 0:
                _fp = data.get("fitness_post", np.full(n_total, np.nan))
                _has_fp = ~np.isnan(_fp)
                mask &= (~_has_fp) | (_fp <= _fit_max)

        # Shot type filter
        shot_filter = self.shot_filter_combo.currentText()
        if shot_filter != "All":
            shot_types_arr = data.get("shot_type", np.full(n_total, -1, dtype=int))
            pw_arr = data.get("point_winner", np.zeros(n_total, dtype=int))
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
        x_keys = self._x_keys()
        is_error = self._is_error()

        # Check x-axis keys exist
        _COMPUTED_KEYS = {"w_mag_racket", "wxr_x", "wxr_y", "wxr_z", "wxr_mag"}
        for xk in x_keys:
            if xk not in data and xk not in _COMPUTED_KEYS:
                self.info_label.setText(
                    f"Missing data key '{xk}'. "
                    f"Delete .data_plotter_cache.pkl and reload to regenerate."
                )
                return

        x_arrays = {k: data[k][mask] for k in x_keys if k in data}
        # Computed keys
        if "w_mag_racket" in x_keys and "w_mag_racket" not in x_arrays:
            _wx = data.get("wrx_racket", np.zeros(n_total))[mask]
            _wy = data.get("wry_racket", np.zeros(n_total))[mask]
            _wz = data.get("wrz_racket", np.zeros(n_total))[mask]
            x_arrays["w_mag_racket"] = np.sqrt(_wx**2 + _wy**2 + _wz**2)
        # ω × r: tangential velocity at contact point from angular velocity
        _wxr_keys = {"wxr_x", "wxr_y", "wxr_z", "wxr_mag"}
        if _wxr_keys & set(x_keys):
            # Use body-frame ω and body-frame contact location (both pure R^T)
            _wx = data.get("wbx_racket", np.zeros(n_total))[mask]
            _wy = data.get("wby_racket", np.zeros(n_total))[mask]
            _wz = data.get("wbz_racket", np.zeros(n_total))[mask]
            # dx/dy/dz_pre: contact location in pure body frame (no v_ref alignment)
            # x=0 (on surface), y/z = position on racket face
            _rx = data.get("dx_pre", np.zeros(n_total))[mask]
            _ry = data.get("dy_pre", np.zeros(n_total))[mask]
            _rz = data.get("dz_pre", np.zeros(n_total))[mask]
            # cross product ω × r
            x_arrays["wxr_x"] = _wy * _rz - _wz * _ry
            x_arrays["wxr_y"] = _wz * _rx - _wx * _rz
            x_arrays["wxr_z"] = _wx * _ry - _wy * _rx
            x_arrays["wxr_mag"] = np.sqrt(
                x_arrays["wxr_x"]**2 + x_arrays["wxr_y"]**2 + x_arrays["wxr_z"]**2
            )
        observed_post = {k: data[k][mask] for k in _POST_KEYS}

        # Determine which model is selected
        model_label = self.model_combo.currentText()
        model_suffix = _MODEL_SUFFIXES.get(model_label)

        # Build output arrays (observed or model-predicted)
        if model_suffix is None:
            output_arrays = observed_post
            model_available = True
        else:
            model_keys = _LOCAL_MODEL_POST_KEYS.get(model_suffix, [])
            has_model = all(mk in data for mk in model_keys)
            if has_model:
                model_available = not np.all(np.isnan(data[model_keys[0]]))
            else:
                model_available = False

            if model_available:
                output_arrays = {pk: data[mk][mask] for pk, mk in zip(_POST_KEYS, model_keys)}
            else:
                output_arrays = observed_post

        # For error mode: delta = model_output − observed_post
        delta_arrays: Dict[str, np.ndarray] = {}
        if is_error:
            if model_suffix is None:
                self.info_label.setText(
                    "Select a model to see prediction error. "
                    "'Observed' error is zero by definition."
                )
                return
            elif model_available:
                for pk in _POST_KEYS:
                    delta_arrays[pk] = output_arrays[pk] - observed_post[pk]
            else:
                self.info_label.setText(
                    f"No model data for '{model_label}'. "
                    f"Delete .data_plotter_cache.pkl and reload."
                )
                return

        if not is_error and not model_available and model_suffix is not None:
            self.info_label.setText(
                f"No model data for '{model_label}' — showing Observed (pseudo-GT). "
                f"Delete .data_plotter_cache.pkl and reload."
            )

        # ── Colour mapping ───────────────────────────────────────────
        color_by = self.color_combo.currentText()
        use_categorical = False

        if color_by == "Shot Type":
            _SHOT_CAT_COLORS = {
                0: (50, 180, 50),    # serve – green
                1: (74, 144, 226),   # rally – blue
                3: (220, 60, 60),    # last won – red
                4: (255, 140, 0),    # last lost – orange
            }
            _SHOT_CAT_LABELS = {0: "Serve", 1: "Rally", 3: "Last (Won)", 4: "Last (Lost)"}
            shot_types = data.get("shot_type", np.ones(n_total, dtype=int)).copy()
            pw_arr = data.get("point_winner", np.zeros(n_total, dtype=int))
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
            pw_arr = data.get("point_winner", np.zeros(n_total, dtype=int))[mask]
            rgb = np.array([_WINNER_COLORS.get(int(w), (128, 128, 128)) for w in pw_arr], dtype=np.int32)
            use_categorical = True
        elif color_by == "Date":
            metadata = data.get("metadata_list", [])
            all_dates = [metadata[i].get("match_date", "") if i < len(metadata) else "" for i in range(n_total)]
            filtered_dates = [all_dates[i] for i in filtered_indices]
            unique_dates = sorted(set(filtered_dates))
            _DATE_CMAP = plt.get_cmap("tab20" if len(unique_dates) <= 20 else "turbo")
            _date_color_map = {}
            for di, d in enumerate(unique_dates):
                rgba = _DATE_CMAP(di / max(len(unique_dates) - 1, 1))
                _date_color_map[d] = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            rgb = np.array([_date_color_map.get(d, (128, 128, 128)) for d in filtered_dates], dtype=np.int32)
            _DATE_LABELS = {d: d if d else "(unknown)" for d in unique_dates}
            use_categorical = True
        else:
            if color_by == "Racket Open Angle":
                color_vals = data.get("racket_open_angle", np.full(n_total, np.nan))[mask]
                c_min, c_max = -60.0, 60.0
            elif color_by == "Theta Angle":
                color_vals = data.get("theta_angle", np.full(n_total, np.nan))[mask]
                c_min, c_max = 0.0, 90.0
            else:  # Player
                color_vals = data.get("player", np.ones(n_total))[mask]
                c_min, c_max = 1.0, 2.0

            cmap = plt.get_cmap("turbo")
            clipped = np.clip(np.nan_to_num(color_vals, nan=c_min), c_min, c_max)
            denom = (c_max - c_min) if c_max > c_min else 1.0
            norm = (clipped - c_min) / denom
            rgba = cmap(norm)
            rgb = (rgba[:, :3] * 255).astype(np.int32)

        point_size = self.size_slider.value()

        # ── Fill scatter cells ───────────────────────────────────────
        for row, post_key in enumerate(_POST_KEYS):
            for col, x_key in enumerate(x_keys):
                scatter = self.scatter_items[(row, col)]
                x = x_arrays[x_key]
                y = delta_arrays[post_key] if is_error else output_arrays[post_key]

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

                # Median and ±1σ lines (error view only)
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

                    if col == 0:
                        vb = pw.getViewBox()
                        vr = vb.viewRange()
                        _x_right = vr[0][1]
                        _texts = []
                        for _label, _ypos in [
                            (f"med={_med:.4g}", _med),
                            (f"+σ={_med + _std:.4g}", _med + _std),
                            (f"−σ={_med - _std:.4g}", _med - _std),
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

        xgroup_name = self.xgroup_combo.currentText()
        view_mode = self.view_combo.currentText()
        info_text = (
            f"Showing {n_points}/{n_total} contacts | X: {xgroup_name} | "
            f"View: {view_mode} | Model: {model_label} | Color: {color_by}"
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

            pw0 = self.plot_items.get((0, self._current_n_cols - 1))
            if pw0 is not None:
                legend = pw0.addLegend(offset=(5, 5), labelTextSize="8pt")
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

    # ─── click handler ───────────────────────────────────────────────

    def _on_point_clicked(self, _plot_item, points):
        if not points:
            return
        pt = points[0]
        idx = pt.data()
        if idx is None or self.rcm_data is None:
            return
        idx = int(idx)
        metadata = self.rcm_data.get("metadata_list", [])
        shot_list = self.rcm_data.get("shot_list", [])
        rally_list = self.rcm_data.get("rally_list", [])
        fs_pre_list = self.rcm_data.get("fs_pre_list", [])
        fs_post_list = self.rcm_data.get("fs_post_list", [])
        md = metadata[idx] if idx < len(metadata) else None
        shot = shot_list[idx] if idx < len(shot_list) else None
        rally = rally_list[idx] if idx < len(rally_list) else None

        contact_point_data = {}
        for k in ["vx_pre", "vy_pre", "vz_pre", "wx_pre", "wy_pre", "wz_pre",
                   "vx_post", "vy_post", "vz_post", "wx_post", "wy_post", "wz_post"]:
            contact_point_data[k] = float(self.rcm_data[k][idx])
        if idx < len(fs_pre_list):
            contact_point_data["fs_pre"] = fs_pre_list[idx]
        if idx < len(fs_post_list):
            contact_point_data["fs_post"] = fs_post_list[idx]
        self.contact_selected.emit(contact_point_data, md, shot, rally)

    # ─── save ────────────────────────────────────────────────────────

    def _save_plot(self):
        if self.rcm_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save.")
            return
        plots_dir = pathlib.Path(getattr(self, "plots_folder", None) or ".")
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"rcm_output_vs_others_{ts}.png"
        out = plots_dir / fname
        try:
            px = self.grid_container.grab()
            px.save(str(out), "PNG")
            QMessageBox.information(self, "Success", f"Plot saved to:\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{e}")
