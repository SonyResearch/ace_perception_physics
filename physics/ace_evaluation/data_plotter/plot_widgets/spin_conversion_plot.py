# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Spin Conversion Plot Widget

Three side-by-side scatter plots showing pre-contact quantities (vx_pre,
vz_pre, wy_pre) vs post-contact wy_post for racket contacts.
Points are coloured by the player who hit the ball (robot / human) and
can be filtered by confidence.

Inspired by spin_conversion_focus.html from compare_racket_spin_player_vs_robot.py.
"""

from typing import Dict, Any, Optional
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSlider, QPushButton, QMessageBox,
)
from PySide6.QtCore import Signal, Qt, QTimer

import pathlib
from datetime import datetime

import pyqtgraph as pg
import matplotlib.pyplot as plt

from .base_coefficient_plot import create_range_slider, create_combo_selector


class SpinConversionPlot(QWidget):
    """Three-panel scatter plot: pre-contact quantity vs post-contact wy.

    Columns:
        1. vx_pre  vs  wy_post
        2. vz_pre  vs  wy_post
        3. wy_pre  vs  wy_post

    Points coloured by player (robot P1 = blue, human P2 = orange).
    """

    # Signal when user clicks a point – emits (FlightSegment, metadata, shot_data, rally_data, shot, rally)
    flight_segment_selected = Signal(object, object, object, object, object, object)

    # Columns shown per subplot (x-axis label, key into data dict)
    _PANELS = [
        ("vx_pre (m/s)", "vx_pre"),
        ("vz_pre (m/s)", "vz_pre"),
        ("wy_pre (rad/s)", "wy_pre"),
    ]

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Debounce timer for filter/slider updates (prevents lag with large datasets)
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(120)  # 120ms debounce
        self._update_timer.timeout.connect(self._apply_filters_now)

        # --- control bar ---
        control_bar = QHBoxLayout()

        # Y-axis selector
        yaxis_layout, self.yaxis_combo = create_combo_selector(
            "Y Axis:",
            ["wy_post", "wx_post", "wz_post", "vx_post", "vy_post", "vz_post",
             "spin_mag_post"],
            self.update_plot,
        )
        control_bar.addLayout(yaxis_layout)

        # Color-by selector
        color_layout, self.color_combo = create_combo_selector(
            "Color By:", ["Player", "Point Winner", "Confidence", "wy_pre", "Date", "Policy"], self.update_plot,
        )
        control_bar.addLayout(color_layout)

        # Confidence range slider (default 50-100 %)
        self._conf_layout, self.conf_min_slider, self.conf_max_slider, self.conf_min_label, self.conf_max_label = \
            create_range_slider("Confidence (%):", 0, 100, 50, 100, 10, self._on_filter_changed, "{}", 1.0, 25)
        control_bar.addLayout(self._conf_layout)

        # OPT RMSE filter (mm) – min/max range
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
        self.rmse_min_slider.valueChanged.connect(self._on_filter_changed)
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
        self.rmse_slider.valueChanged.connect(self._on_filter_changed)
        rmse_max_row.addWidget(self.rmse_slider)
        self.rmse_label = QLabel("100")
        self.rmse_label.setMinimumWidth(25)
        rmse_max_row.addWidget(self.rmse_label)
        self._rmse_layout.addLayout(rmse_max_row)
        control_bar.addLayout(self._rmse_layout)

        # Point size slider
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(2)
        self.size_slider.setMaximum(15)
        self.size_slider.setValue(6)
        self.size_slider.valueChanged.connect(self.update_plot)
        size_layout.addWidget(self.size_slider)
        self.size_label = QLabel("6")
        self.size_label.setMinimumWidth(20)
        size_layout.addWidget(self.size_label)
        control_bar.addLayout(size_layout)

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.setMaximumWidth(80)
        self.save_button.clicked.connect(self.save_plot)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # --- 3 plot panels side-by-side ---
        plot_row = QHBoxLayout()
        self.plot_widgets = []
        self._scatters = []

        for label, _ in self._PANELS:
            pw = pg.PlotWidget()
            pw.setBackground('#1e1e1e')
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setLabel('bottom', label)
            pw.setLabel('left', 'wy_post (rad/s)')
            plot_row.addWidget(pw)
            self.plot_widgets.append(pw)
            self._scatters.append(None)

        layout.addLayout(plot_row)

        # Info label
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        self.setLayout(layout)

        # Data (set via plot_data)
        self._data: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_data(self, data: Dict[str, Any]):
        """Store racket contact data and refresh the plot."""
        self._data = data
        self.update_plot()

    def save_plot(self):
        """Save the three panels as a single wide PNG image."""
        if self._data is None or len(self._data.get('vx_pre', [])) == 0:
            QMessageBox.warning(self, "No Data", "No plot data to save.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spin_conversion_{timestamp}.png"
        output_path = plots_dir / filename

        try:
            # Grab each panel and stitch horizontally
            from PySide6.QtGui import QPixmap, QPainter
            pixmaps = [pw.grab() for pw in self.plot_widgets]
            total_w = sum(p.width() for p in pixmaps)
            max_h = max(p.height() for p in pixmaps)
            combined = QPixmap(total_w, max_h)
            combined.fill(Qt.black)
            painter = QPainter(combined)
            x_offset = 0
            for p in pixmaps:
                painter.drawPixmap(x_offset, 0, p)
                x_offset += p.width()
            painter.end()
            combined.save(str(output_path), "PNG")
            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_filter_changed(self, _=None):
        """Debounced handler for slider/filter changes."""
        self.conf_min_label.setText(str(self.conf_min_slider.value()))
        self.conf_max_label.setText(str(self.conf_max_slider.value()))
        self.rmse_min_label.setText(str(self.rmse_min_slider.value()))
        self.rmse_label.setText(str(self.rmse_slider.value()))
        self._update_timer.start()  # debounce – actual update happens after 120ms

    def _apply_filters_now(self):
        """Called by debounce timer – performs the actual plot update."""
        self.update_plot()

    @staticmethod
    def _velocity_aligned_transform(vx, vy, vz, wx, wy, wz):
        """Transform spin into a velocity-aligned local coordinate frame.

        Local frame (right-handed):
            x' = unit velocity direction (horizontal plane, from vx/vy)
            z' = gravity direction (world z-up, unchanged)
            y' = z' × x'  (perpendicular to velocity in horizontal plane)

        Returns (wx', wy', wz') where:
            wx' = gyroscopic / rifle-spin component (spin about velocity axis)
            wy' = topspin component (spin about horizontal axis ⊥ velocity)
            wz' = sidespin component (spin about vertical axis)
        """
        # Horizontal velocity direction
        v_horiz = np.sqrt(vx**2 + vy**2)
        # Avoid division by zero – fall back to world frame
        safe = v_horiz > 1e-6
        cos_a = np.where(safe, vx / np.maximum(v_horiz, 1e-9), 1.0)
        sin_a = np.where(safe, vy / np.maximum(v_horiz, 1e-9), 0.0)

        # Rotation about z-axis by -alpha  (world→local)
        #   x'_local =  cos(a)*x + sin(a)*y
        #   y'_local = -sin(a)*x + cos(a)*y
        #   z'_local =  z
        wx_local =  cos_a * wx + sin_a * wy
        wy_local = -sin_a * wx + cos_a * wy
        wz_local =  wz  # unchanged
        return wx_local, wy_local, wz_local

    def update_plot(self, _=None):
        for pw in self.plot_widgets:
            pw.clear()
            # Remove any existing legend
            if hasattr(pw.plotItem, 'legend') and pw.plotItem.legend is not None:
                pw.plotItem.legend.scene().removeItem(pw.plotItem.legend)
                pw.plotItem.legend = None
        self._scatters = [None] * len(self._PANELS)
        self.size_label.setText(str(self.size_slider.value()))

        if self._data is None or len(self._data.get('vx_pre', [])) == 0:
            self.info_label.setText("No racket contact data available")
            return

        # Read controls
        y_key = self.yaxis_combo.currentText()
        color_by = self.color_combo.currentText()
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0
        rmse_max = self.rmse_slider.value() / 1000.0  # mm -> meters
        point_size = self.size_slider.value()

        # Filter mask
        confidence = self._data['confidence']
        mask = (
            ~np.isnan(confidence) &
            (confidence >= conf_min) &
            (confidence <= conf_max)
        )

        # OPT RMSE filter
        rmse_opt = self._data.get('max_rmse_opt')
        if rmse_opt is not None:
            rmse_min = self.rmse_min_slider.value() / 1000.0  # mm -> meters
            rmse_valid = ~np.isnan(rmse_opt)
            if self.rmse_slider.value() < self.rmse_slider.maximum():  # apply max if not at slider max
                mask = mask & rmse_valid & (rmse_opt <= rmse_max)
            if self.rmse_min_slider.value() > 0:  # apply min if above 0
                mask = mask & rmse_valid & (rmse_opt >= rmse_min)
        if not np.any(mask):
            self.info_label.setText("No data matching current filters")
            return

        player = self._data['player'][mask]
        conf_valid = confidence[mask]
        indices_valid = np.where(mask)[0]
        n_points = int(np.sum(mask))

        # ---- Build working arrays in velocity-aligned local frame ----
        # Always mirror P2 so robot and player share the same coordinate frame
        is_p2 = (player == 2)

        # Pre-contact velocity (mirrored for P2)
        vx_pre = self._data['vx_pre'][mask].copy()
        vy_pre = self._data['vy_pre'][mask].copy()
        vz_pre = self._data['vz_pre'][mask].copy()
        if np.any(is_p2):
            vx_pre[is_p2] *= -1
            vy_pre[is_p2] *= -1

        # Pre-contact spin (mirrored for P2)
        wx_pre_raw = self._data['wx_pre'][mask].copy()
        wy_pre_raw = self._data['wy_pre'][mask].copy()
        wz_pre_raw = self._data['wz_pre'][mask].copy()
        if np.any(is_p2):
            wx_pre_raw[is_p2] *= -1
            wy_pre_raw[is_p2] *= -1

        wx_pre_loc, wy_pre_loc, wz_pre_loc = self._velocity_aligned_transform(
            vx_pre, vy_pre, vz_pre, wx_pre_raw, wy_pre_raw, wz_pre_raw
        )

        # Post-contact velocity (mirrored for P2) – used as frame direction for post data
        vx_post_raw = self._data['vx_post'][mask].copy()
        vy_post_raw = self._data['vy_post'][mask].copy()
        vz_post_raw = self._data['vz_post'][mask].copy()
        if np.any(is_p2):
            vx_post_raw[is_p2] *= -1
            vy_post_raw[is_p2] *= -1

        # Post-contact spin (mirrored for P2)
        wx_post_raw = self._data['wx_post'][mask].copy()
        wy_post_raw = self._data['wy_post'][mask].copy()
        wz_post_raw = self._data['wz_post'][mask].copy()
        if np.any(is_p2):
            wx_post_raw[is_p2] *= -1
            wy_post_raw[is_p2] *= -1

        wx_post_loc, wy_post_loc, wz_post_loc = self._velocity_aligned_transform(
            vx_post_raw, vy_post_raw, vz_post_raw, wx_post_raw, wy_post_raw, wz_post_raw
        )

        # Transform velocities into their respective local frames
        vx_pre_loc = np.sqrt(vx_pre**2 + vy_pre**2)   # ||v_horiz|| (always positive)
        vy_pre_loc = np.zeros_like(vx_pre_loc)          # by definition
        vz_pre_loc = vz_pre.copy()

        vx_post_loc = np.sqrt(vx_post_raw**2 + vy_post_raw**2)  # ||v_horiz||
        vy_post_loc = np.zeros_like(vx_post_loc)                  # by definition
        vz_post_loc = vz_post_raw.copy()

        # Spin magnitude (frame-invariant)
        spin_mag_post = np.sqrt(wx_post_loc**2 + wy_post_loc**2 + wz_post_loc**2)

        # Local-frame data dict for easy key-based access
        local_data = {
            'vx_pre': vx_pre_loc, 'vy_pre': vy_pre_loc, 'vz_pre': vz_pre_loc,
            'wx_pre': wx_pre_loc, 'wy_pre': wy_pre_loc, 'wz_pre': wz_pre_loc,
            'vx_post': vx_post_loc, 'vy_post': vy_post_loc, 'vz_post': vz_post_loc,
            'wx_post': wx_post_loc, 'wy_post': wy_post_loc, 'wz_post': wz_post_loc,
            'spin_mag_post': spin_mag_post,
        }
        y_data = local_data[y_key]

        # ---- Determine brush colors (vectorized) ----
        use_categorical = False
        color_label = None

        if color_by == "Player":
            use_categorical = True
            # robot=blue, human=orange  – build uint8 RGBA array
            brush_colors = np.where(
                (player == 1)[:, None],
                np.array([80, 150, 255, 200], dtype=np.uint8),
                np.array([255, 140, 60, 200], dtype=np.uint8),
            )

        elif color_by == "Point Winner":
            use_categorical = True
            rally_list = self._data.get('rally_list', [])
            # Determine point winner per sample
            pw_colors = np.empty((n_points, 4), dtype=np.uint8)
            _PW_ROBOT  = np.array([80, 200, 80, 200], dtype=np.uint8)   # green = robot won
            _PW_PLAYER = np.array([220, 80, 80, 200], dtype=np.uint8)   # red   = player won
            _PW_UNKNOWN = np.array([160, 160, 160, 200], dtype=np.uint8) # gray  = unknown
            for i, idx in enumerate(indices_valid):
                pw_val = None
                if idx < len(rally_list) and rally_list[idx] is not None:
                    pw_val = getattr(rally_list[idx], 'point_winner', None)
                if pw_val == 'player1':
                    pw_colors[i] = _PW_ROBOT
                elif pw_val == 'player2':
                    pw_colors[i] = _PW_PLAYER
                else:
                    pw_colors[i] = _PW_UNKNOWN
            brush_colors = pw_colors
            self._pw_legend_info = [
                ("Robot Won", (80, 200, 80)),
                ("Player Won", (220, 80, 80)),
                ("Unknown", (160, 160, 160)),
            ]

        elif color_by == "Date":
            metadata_list = self._data.get('metadata_list', [])
            # Collect dates and assign numeric indices
            date_strs = []
            for idx in indices_valid:
                md = metadata_list[idx] if idx < len(metadata_list) else {}
                date_strs.append(md.get('match_date', 'unknown'))
            unique_dates = sorted(set(date_strs))
            n_dates = max(len(unique_dates), 1)
            date_to_idx = {d: i for i, d in enumerate(unique_dates)}
            date_indices = np.array([date_to_idx[d] for d in date_strs])
            # Vectorized colormap
            cmap = plt.get_cmap('turbo')
            norm_vals = date_indices / max(n_dates - 1, 1)
            rgba_array = cmap(norm_vals)
            brush_colors = (rgba_array[:, :4] * 255).astype(np.uint8)
            brush_colors[:, 3] = 200
            color_label = "Date"
            self._date_legend_info = [(d, (rgba_array[date_indices == date_to_idx[d]][0][:3] * 255).astype(int)) for d in unique_dates]

        elif color_by == "Confidence":
            cmap = plt.get_cmap('turbo')
            c_min, c_max = conf_min, conf_max
            norm_vals = np.clip(conf_valid, c_min, c_max)
            if c_max > c_min:
                norm_vals = (norm_vals - c_min) / (c_max - c_min)
            else:
                norm_vals = np.full(n_points, 0.5)
            rgba_array = cmap(norm_vals)
            brush_colors = (rgba_array[:, :4] * 255).astype(np.uint8)
            brush_colors[:, 3] = 200

        elif color_by == "Policy":
            use_categorical = True
            policy_all = self._data.get('policy', [])
            policy_filtered = [policy_all[i] for i in indices_valid] if len(policy_all) > 0 else ['unknown'] * n_points
            unique_policies = sorted(set(policy_filtered))
            _POLICY_COLORS = [
                (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
                (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
                (188, 189, 34), (23, 190, 207),
            ]
            policy_color_map = {p: _POLICY_COLORS[i % len(_POLICY_COLORS)] for i, p in enumerate(unique_policies)}
            brush_colors = np.array([(*policy_color_map[p], 200) for p in policy_filtered], dtype=np.uint8)
            self._policy_legend_info = [(p, policy_color_map[p]) for p in unique_policies]

        else:
            # wy_pre
            vals = local_data['wy_pre'].copy()
            cmap = plt.get_cmap('turbo')
            c_min, c_max = float(np.nanmin(vals)), float(np.nanmax(vals))
            norm_vals = np.clip(vals, c_min, c_max)
            if c_max > c_min:
                norm_vals = (norm_vals - c_min) / (c_max - c_min)
            else:
                norm_vals = np.full(n_points, 0.5)
            rgba_array = cmap(norm_vals)
            brush_colors = (rgba_array[:, :4] * 255).astype(np.uint8)
            brush_colors[:, 3] = 200

        # Build brush list from (N,4) uint8 array
        brushes = [pg.mkBrush(int(c[0]), int(c[1]), int(c[2]), int(c[3])) for c in brush_colors]

        # Symbols: circle=robot, triangle=player (vectorized)
        symbols = np.where(player == 1, 'o', 't')

        # Update Y-axis label
        if y_key.startswith('spin_mag'):
            y_label = f"{y_key} (rad/s)"
        elif y_key.startswith('w'):
            y_label = f"{y_key} (local, rad/s)"
        else:
            y_label = f"{y_key} (local, m/s)"

        # Shared pen for all points
        shared_pen = pg.mkPen(color=(50, 50, 50), width=1)

        # Plot each panel
        for panel_idx, (x_label_orig, x_key) in enumerate(self._PANELS):
            pw = self.plot_widgets[panel_idx]
            pw.setLabel('left', y_label)

            x_data = local_data[x_key]

            # Update x-axis label for local frame
            x_label_display = x_label_orig.replace("(m/s)", "(local, m/s)").replace("(rad/s)", "(local, rad/s)")
            pw.setLabel('bottom', x_label_display)

            # Vectorized ScatterPlotItem construction
            scatter = pg.ScatterPlotItem(
                x=x_data.astype(float), y=y_data.astype(float),
                size=point_size,
                pen=shared_pen,
                brush=brushes,
                symbol=list(symbols),
                data=indices_valid.astype(int),
                hoverable=True, tip=None,
            )
            scatter.sigClicked.connect(lambda plot, pts, pi=panel_idx: self._on_point_clicked(plot, pts))
            pw.addItem(scatter)
            self._scatters[panel_idx] = scatter

            pw.enableAutoRange()

            # Add legend only on first panel
            if panel_idx == 0:
                p2_label = "Player (P2 mirrored)"
                if color_by == "Policy" and hasattr(self, '_policy_legend_info'):
                    legend = pw.addLegend(offset=(10, 10))
                    for policy_name, color in self._policy_legend_info:
                        legend.addItem(
                            pg.ScatterPlotItem([0],[0], pen=pg.mkPen(50,50,50,1),
                                               brush=pg.mkBrush(*color, 200), size=8, symbol='o'),
                            policy_name
                        )
                elif color_by == "Point Winner" and hasattr(self, '_pw_legend_info'):
                    legend = pw.addLegend(offset=(10, 10))
                    for pw_name, pw_color in self._pw_legend_info:
                        legend.addItem(
                            pg.ScatterPlotItem([0],[0], pen=pg.mkPen(50,50,50,1),
                                               brush=pg.mkBrush(*pw_color, 200), size=8, symbol='o'),
                            pw_name
                        )
                elif color_by == "Date" and hasattr(self, '_date_legend_info'):
                    legend = pw.addLegend(offset=(10, 10))
                    for date_name, date_color in self._date_legend_info:
                        legend.addItem(
                            pg.ScatterPlotItem([0],[0], pen=pg.mkPen(50,50,50,1),
                                               brush=pg.mkBrush(int(date_color[0]), int(date_color[1]), int(date_color[2]), 200), size=8, symbol='o'),
                            date_name
                        )
                elif use_categorical:
                    legend = pw.addLegend(offset=(10, 10))
                    legend.addItem(
                        pg.ScatterPlotItem([0],[0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(80,150,255,200), size=8, symbol='o'),
                        "Robot (P1)"
                    )
                    legend.addItem(
                        pg.ScatterPlotItem([0],[0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(255,140,60,200), size=8, symbol='t'),
                        p2_label
                    )
                else:
                    legend = pw.addLegend(offset=(10, 10))
                    legend.addItem(
                        pg.ScatterPlotItem([0],[0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(180,180,180,200), size=8, symbol='o'),
                        "Robot (P1)"
                    )
                    legend.addItem(
                        pg.ScatterPlotItem([0],[0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(180,180,180,200), size=8, symbol='t'),
                        p2_label
                    )

        n_robot = int(np.sum(player == 1))
        n_player = int(np.sum(player == 2))
        rmse_active = self.rmse_min_slider.value() > 0 or self.rmse_slider.value() < 200
        rmse_tag = f" | OPT RMSE: [{self.rmse_min_slider.value()}, {self.rmse_slider.value()}]mm" if rmse_active else ""
        self.info_label.setText(
            f"Showing {n_points} racket contacts (Robot: {n_robot}, Player: {n_player}) | "
            f"Conf: [{int(conf_min*100)}, {int(conf_max*100)}]%{rmse_tag} | Velocity-aligned, P2 mirrored"
        )

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _on_point_clicked(self, plot, points):
        if len(points) == 0:
            return
        pt = points[0]
        idx = pt.data()
        if idx is None or self._data is None:
            return

        fs_post_list = self._data.get('fs_post_list', [])
        if idx < 0 or idx >= len(fs_post_list):
            return

        fs = fs_post_list[idx]
        metadata = self._data['metadata_list'][idx] if idx < len(self._data['metadata_list']) else None
        shot = self._data['shot_list'][idx] if idx < len(self._data['shot_list']) else None
        rally = self._data['rally_list'][idx] if idx < len(self._data['rally_list']) else None

        # Build shot data
        shot_data = None
        if shot is not None:
            import pandas as pd
            try:
                shot_data = pd.concat([seg.data for seg in shot.flight_segments], ignore_index=False)
            except Exception:
                pass

        # Build rally data
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

        self.flight_segment_selected.emit(fs, metadata, shot_data, rally_data, shot, rally)
