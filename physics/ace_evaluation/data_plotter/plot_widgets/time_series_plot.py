#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Stacked Multi-Metric Time Series Plot Widget  (aligned overlay)

Overlays rallies on three vertically stacked subplots (|vx|, wy, z).

Filtering & alignment
  • The anchor mode determines which shot is placed at x = 0:
    - "Last Player Shot": last human shot matching vx/wy filters
    - "Last Robot Shot": last robot shot matching vx/wy filters
    - "Player Serve": human serve (rally.shots[1])
    - "Robot Serve": robot serve (rally.shots[1])
  • For shot-based modes, an optional shot −2 vx/wy filter is also applied.
  • The alignment shot, up to 2 shots before and 2 shots after are
    displayed (x ∈ {-2, -1, 0, 1, 2}).
  • Tick labels indicate Robot (R) or Player (P) for each shot.

Colouring
  Lines are coloured by the human **player name** (from match.player),
  so different opponents get distinct colours.
"""

import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox,
    QSlider, QGridLayout, QSizePolicy, QGroupBox, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor


# Alternating column colours (RGBA)
_COL_EVEN = QColor(255, 255, 255, 12)
_COL_ODD  = QColor(255, 255, 255, 25)

# Line alpha (0-255)
_LINE_ALPHA = 80
_LINE_ALPHA_HIGHLIGHT = 255
_LINE_WIDTH = 1.5
_LINE_WIDTH_HIGHLIGHT = 4.0
_LINE_ALPHA_DIMMED = 30

# Extended colour palette for player names  (r, g, b)
_PLAYER_PALETTE = [
    (31, 119, 180),   # blue
    (255, 127, 14),   # orange
    (44, 160, 44),    # green
    (214, 39, 40),    # red
    (148, 103, 189),  # purple
    (140, 86, 75),    # brown
    (227, 119, 194),  # pink
    (188, 189, 34),   # olive
    (23, 190, 207),   # cyan
]
_COLOR_UNKNOWN = (127, 127, 127)

# Point-winner colours
_COLOR_ROBOT_WON   = (44, 160, 44)     # green – robot (player1) won
_COLOR_HUMAN_WON   = (214, 39, 40)     # red   – human (player2) won
_COLOR_WINNER_UNKNOWN = (127, 127, 127)  # grey  – unknown winner

# Default filter ranges  (anchor = shot 0)
_VX_MIN_DEFAULT, _VX_MAX_DEFAULT = 15.0, 35.0   # m/s  (|vx|)
_WY_MIN_DEFAULT, _WY_MAX_DEFAULT = -1000.0, -500.0   # rad/s (signed wy)

# Default filter ranges  (shot -2)
_VX2_MIN_DEFAULT, _VX2_MAX_DEFAULT = 0.0, 7.0   # m/s  (|vx|)
_WY2_MIN_DEFAULT, _WY2_MAX_DEFAULT = -1000.0, 1000.0   # rad/s – full range

# Default filter ranges  (shot -1)
_VX1_MIN_DEFAULT, _VX1_MAX_DEFAULT = 0.0, 10.0   # m/s  (|vx|)
_WY1_MIN_DEFAULT, _WY1_MAX_DEFAULT = -1000.0, 1000.0   # rad/s – full range

# Slider granularity
_VX_SLIDER_SCALE = 1      # 1 slider tick = 1 m/s
_WY_SLIDER_SCALE = 0.02   # 1 slider tick = 50 rad/s  (value / 0.02 = value * 50)

# Absolute slider bounds
_VX_ABS_MIN, _VX_ABS_MAX = 0.0, 40.0
_WY_ABS_MIN, _WY_ABS_MAX = -1000.0, 1000.0

# Extra quantity filter configuration per component.
# (abs_min, abs_max, slider_scale, use_abs_for_filter, fmt)
_EXTRA_COMPONENTS = ['wx', 'wz', 'x', 'y', 'vy', 'vz']
_EXTRA_FILTER_CFG = {
    'wx':  (_WY_ABS_MIN, _WY_ABS_MAX, _WY_SLIDER_SCALE, False, '.0f'),
    'wz':  (_WY_ABS_MIN, _WY_ABS_MAX, _WY_SLIDER_SCALE, False, '.0f'),
    'x':   (-3.0,  3.0,  10.0, False, '.1f'),
    'y':   (-1.5,  1.5,  10.0, False, '.1f'),
    'vy':  (_VX_ABS_MIN, _VX_ABS_MAX, _VX_SLIDER_SCALE, True,  '.0f'),
    'vz':  (_VX_ABS_MIN, _VX_ABS_MAX, _VX_SLIDER_SCALE, True,  '.0f'),
}

# How many shots before the anchor to show
_MAX_LOOKBACK = 2
# How many shots after the anchor to show
_MAX_LOOKAHEAD = 2


# Anchor mode options
_ANCHOR_MODES = [
    "Last Player Shot",
    "Last Robot Shot",
    "Player Serve",
    "Robot Serve",
]

# Selectable extra quantities for the 4th time-series row.
# Each entry: (label, component, y-axis label, units, use_abs)
#   component is passed to _col(df, component, pri, fb)
_EXTRA_QUANTITIES = [
    ("None",              None,  None,  None,    None),   # hidden
    ("wx  (Spin X)",      "wx",  "wx",  "rad/s", False),
    ("wz  (Spin Z)",      "wz",  "wz",  "rad/s", False),
    ("x   (Position X)",  "x",   "x",   "m",     False),
    ("y   (Position Y)",  "y",   "y",   "m",     False),
    ("vy  (Velocity Y)",  "vy",  "vy",  "m/s",   False),
    ("vz  (Velocity Z)",  "vz",  "vz",  "m/s",   False),
]


class TimeSeriesPlot(QWidget):
    """Stacked multi-metric time series aligned to a configurable anchor shot.

    Three vertically stacked subplots (|vx|, wy, z) sharing the x-axis.
    X = 0 is the anchor shot, chosen by the selected anchor mode.
    Up to 2 preceding and 2 following shots shown (x = -2, -1, 0, 1, 2).
    Tick labels indicate Robot (R) or Player (P).
    Lines are coloured by the human player name or point winner.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── Control bar ──────────────────────────────────────────────
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Anchor:"))
        self.anchor_combo = QComboBox()
        self.anchor_combo.addItems(_ANCHOR_MODES)
        self.anchor_combo.currentIndexChanged.connect(self._on_anchor_mode_changed)
        ctrl.addWidget(self.anchor_combo)

        ctrl.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["OPT → GT200", "GT200 only"])
        self.source_combo.currentIndexChanged.connect(self._apply_filters)
        ctrl.addWidget(self.source_combo)

        ctrl.addWidget(QLabel("Color by:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["Player name", "Point winner", "Date"])
        self.color_by_combo.currentIndexChanged.connect(self._on_color_by_changed)
        ctrl.addWidget(self.color_by_combo)

        ctrl.addWidget(QLabel("Extra:"))
        self.extra_combo = QComboBox()
        self.extra_combo.addItems([lbl for lbl, *_ in _EXTRA_QUANTITIES])
        self.extra_combo.currentIndexChanged.connect(self._on_extra_changed)
        ctrl.addWidget(self.extra_combo)

        self.save_button = QPushButton("💾 Save")
        self.save_button.setToolTip("Save current plot as PNG")
        self.save_button.clicked.connect(self._save_plot)
        ctrl.addWidget(self.save_button)

        self._skip_precontact_cb = QCheckBox("Skip pre-contact rows")
        self._skip_precontact_cb.setToolTip(
            "When checked, filtering uses the first post-contact value\n"
            "(detected via velocity discontinuity) instead of row[0].\n"
            "Enable this to work around duplicate boundary rows in\n"
            "existing HDF5 data that hasn't been reprocessed yet."
        )
        self._skip_precontact_cb.setChecked(False)
        self._skip_precontact_cb.stateChanged.connect(self._on_skip_precontact_changed)
        ctrl.addWidget(self._skip_precontact_cb)

        self._reset_filters_btn = QPushButton("\u21ba Reset Filters")
        self._reset_filters_btn.setToolTip(
            "Set all vx/wy filter sliders to their full range.\n"
            "When a filter is at full range it passes every value\n"
            "(including NaN and out-of-range data)."
        )
        self._reset_filters_btn.clicked.connect(self._reset_filters)
        ctrl.addWidget(self._reset_filters_btn)

        ctrl.addStretch()

        # Dynamic legend (populated in _replot)
        self.legend_label = QLabel("")
        self.legend_label.setStyleSheet("font-size: 10pt;")
        ctrl.addWidget(self.legend_label)

        layout.addLayout(ctrl)

        # ── Player filter row ────────────────────────────────────────
        player_row = QHBoxLayout()
        player_row.addWidget(QLabel("Players:"))
        self._player_cb_container = QHBoxLayout()
        player_row.addLayout(self._player_cb_container)
        btn_all = QPushButton("All")
        btn_all.setFixedWidth(40)
        btn_all.clicked.connect(lambda: self._set_all_players(True))
        player_row.addWidget(btn_all)
        btn_none = QPushButton("None")
        btn_none.setFixedWidth(46)
        btn_none.clicked.connect(lambda: self._set_all_players(False))
        player_row.addWidget(btn_none)
        player_row.addStretch()
        layout.addLayout(player_row)
        self._player_checkboxes: Dict[str, QCheckBox] = {}

        # ── Filter panels (above their respective plot columns) ─────
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(60, 0, 0, 0)  # approx y-axis margin

        # --- Shot −2 filter group ---
        grp_m2 = QGroupBox("Shot −2")
        g2 = QGridLayout()
        g2.setContentsMargins(4, 2, 4, 2)
        g2.setSpacing(2)

        g2.addWidget(QLabel("|vx|"), 0, 0)
        self.vx2_min_slider = self._make_slider(
            _VX_ABS_MIN, _VX_ABS_MAX, _VX2_MIN_DEFAULT, _VX_SLIDER_SCALE)
        g2.addWidget(self.vx2_min_slider, 0, 1)
        self.vx2_min_label = QLabel(f"{_VX2_MIN_DEFAULT:.0f}")
        self.vx2_min_label.setFixedWidth(36)
        g2.addWidget(self.vx2_min_label, 0, 2)
        g2.addWidget(QLabel("–"), 0, 3)
        self.vx2_max_slider = self._make_slider(
            _VX_ABS_MIN, _VX_ABS_MAX, _VX2_MAX_DEFAULT, _VX_SLIDER_SCALE)
        g2.addWidget(self.vx2_max_slider, 0, 4)
        self.vx2_max_label = QLabel(f"{_VX2_MAX_DEFAULT:.0f}")
        self.vx2_max_label.setFixedWidth(36)
        g2.addWidget(self.vx2_max_label, 0, 5)

        g2.addWidget(QLabel("wy"), 1, 0)
        self.wy2_min_slider = self._make_slider(
            _WY_ABS_MIN, _WY_ABS_MAX, _WY2_MIN_DEFAULT, _WY_SLIDER_SCALE)
        g2.addWidget(self.wy2_min_slider, 1, 1)
        self.wy2_min_label = QLabel(f"{_WY2_MIN_DEFAULT:.0f}")
        self.wy2_min_label.setFixedWidth(44)
        g2.addWidget(self.wy2_min_label, 1, 2)
        g2.addWidget(QLabel("–"), 1, 3)
        self.wy2_max_slider = self._make_slider(
            _WY_ABS_MIN, _WY_ABS_MAX, _WY2_MAX_DEFAULT, _WY_SLIDER_SCALE)
        g2.addWidget(self.wy2_max_slider, 1, 4)
        self.wy2_max_label = QLabel(f"{_WY2_MAX_DEFAULT:.0f}")
        self.wy2_max_label.setFixedWidth(44)
        g2.addWidget(self.wy2_max_label, 1, 5)

        grp_m2.setLayout(g2)
        self._filter_grp_m2 = grp_m2
        filter_row.addWidget(grp_m2, stretch=1)

        # --- Shot −1 filter group ---
        grp_m1 = QGroupBox("Shot −1")
        g1 = QGridLayout()
        g1.setContentsMargins(4, 2, 4, 2)
        g1.setSpacing(2)

        g1.addWidget(QLabel("|vx|"), 0, 0)
        self.vx1_min_slider = self._make_slider(
            _VX_ABS_MIN, _VX_ABS_MAX, _VX1_MIN_DEFAULT, _VX_SLIDER_SCALE)
        g1.addWidget(self.vx1_min_slider, 0, 1)
        self.vx1_min_label = QLabel(f"{_VX1_MIN_DEFAULT:.0f}")
        self.vx1_min_label.setFixedWidth(36)
        g1.addWidget(self.vx1_min_label, 0, 2)
        g1.addWidget(QLabel("–"), 0, 3)
        self.vx1_max_slider = self._make_slider(
            _VX_ABS_MIN, _VX_ABS_MAX, _VX1_MAX_DEFAULT, _VX_SLIDER_SCALE)
        g1.addWidget(self.vx1_max_slider, 0, 4)
        self.vx1_max_label = QLabel(f"{_VX1_MAX_DEFAULT:.0f}")
        self.vx1_max_label.setFixedWidth(36)
        g1.addWidget(self.vx1_max_label, 0, 5)

        g1.addWidget(QLabel("wy"), 1, 0)
        self.wy1_min_slider = self._make_slider(
            _WY_ABS_MIN, _WY_ABS_MAX, _WY1_MIN_DEFAULT, _WY_SLIDER_SCALE)
        g1.addWidget(self.wy1_min_slider, 1, 1)
        self.wy1_min_label = QLabel(f"{_WY1_MIN_DEFAULT:.0f}")
        self.wy1_min_label.setFixedWidth(44)
        g1.addWidget(self.wy1_min_label, 1, 2)
        g1.addWidget(QLabel("–"), 1, 3)
        self.wy1_max_slider = self._make_slider(
            _WY_ABS_MIN, _WY_ABS_MAX, _WY1_MAX_DEFAULT, _WY_SLIDER_SCALE)
        g1.addWidget(self.wy1_max_slider, 1, 4)
        self.wy1_max_label = QLabel(f"{_WY1_MAX_DEFAULT:.0f}")
        self.wy1_max_label.setFixedWidth(44)
        g1.addWidget(self.wy1_max_label, 1, 5)

        grp_m1.setLayout(g1)
        self._filter_grp_m1 = grp_m1
        filter_row.addWidget(grp_m1, stretch=1)

        # --- Shot 0 (anchor) filter group ---
        grp_0 = QGroupBox("Shot 0 (anchor)")
        g0 = QGridLayout()
        g0.setContentsMargins(4, 2, 4, 2)
        g0.setSpacing(2)

        g0.addWidget(QLabel("|vx|"), 0, 0)
        self.vx_min_slider = self._make_slider(
            _VX_ABS_MIN, _VX_ABS_MAX, _VX_MIN_DEFAULT, _VX_SLIDER_SCALE)
        g0.addWidget(self.vx_min_slider, 0, 1)
        self.vx_min_label = QLabel(f"{_VX_MIN_DEFAULT:.0f}")
        self.vx_min_label.setFixedWidth(36)
        g0.addWidget(self.vx_min_label, 0, 2)
        g0.addWidget(QLabel("–"), 0, 3)
        self.vx_max_slider = self._make_slider(
            _VX_ABS_MIN, _VX_ABS_MAX, _VX_MAX_DEFAULT, _VX_SLIDER_SCALE)
        g0.addWidget(self.vx_max_slider, 0, 4)
        self.vx_max_label = QLabel(f"{_VX_MAX_DEFAULT:.0f}")
        self.vx_max_label.setFixedWidth(36)
        g0.addWidget(self.vx_max_label, 0, 5)

        g0.addWidget(QLabel("wy"), 1, 0)
        self.wy_min_slider = self._make_slider(
            _WY_ABS_MIN, _WY_ABS_MAX, _WY_MIN_DEFAULT, _WY_SLIDER_SCALE)
        g0.addWidget(self.wy_min_slider, 1, 1)
        self.wy_min_label = QLabel(f"{_WY_MIN_DEFAULT:.0f}")
        self.wy_min_label.setFixedWidth(44)
        g0.addWidget(self.wy_min_label, 1, 2)
        g0.addWidget(QLabel("–"), 1, 3)
        self.wy_max_slider = self._make_slider(
            _WY_ABS_MIN, _WY_ABS_MAX, _WY_MAX_DEFAULT, _WY_SLIDER_SCALE)
        g0.addWidget(self.wy_max_slider, 1, 4)
        self.wy_max_label = QLabel(f"{_WY_MAX_DEFAULT:.0f}")
        self.wy_max_label.setFixedWidth(44)
        g0.addWidget(self.wy_max_label, 1, 5)

        # Extra quantity filter row (hidden by default)
        self.eq_filter_label = QLabel("—")
        g0.addWidget(self.eq_filter_label, 2, 0)
        self.eq_min_slider = QSlider(Qt.Horizontal)
        self.eq_min_slider.setMinimum(0)
        self.eq_min_slider.setMaximum(1)
        g0.addWidget(self.eq_min_slider, 2, 1)
        self.eq_min_label = QLabel("")
        self.eq_min_label.setFixedWidth(44)
        g0.addWidget(self.eq_min_label, 2, 2)
        self._eq_dash = QLabel("–")
        g0.addWidget(self._eq_dash, 2, 3)
        self.eq_max_slider = QSlider(Qt.Horizontal)
        self.eq_max_slider.setMinimum(0)
        self.eq_max_slider.setMaximum(1)
        g0.addWidget(self.eq_max_slider, 2, 4)
        self.eq_max_label = QLabel("")
        self.eq_max_label.setFixedWidth(44)
        g0.addWidget(self.eq_max_label, 2, 5)
        self._eq_filter_widgets = [
            self.eq_filter_label, self.eq_min_slider, self.eq_min_label,
            self._eq_dash, self.eq_max_slider, self.eq_max_label,
        ]
        for w in self._eq_filter_widgets:
            w.setVisible(False)
        self._eq_filter_cfg = None  # current (abs_min, abs_max, scale, use_abs, fmt)

        grp_0.setLayout(g0)
        self._filter_grp_0 = grp_0
        filter_row.addWidget(grp_0, stretch=1)

        # spacer for shot +1
        filter_row.addStretch(1)

        # spacer for shot +2
        filter_row.addStretch(1)

        # Connect all sliders → live update
        for s in (self.vx_min_slider, self.vx_max_slider,
                  self.wy_min_slider, self.wy_max_slider,
                  self.vx1_min_slider, self.vx1_max_slider,
                  self.wy1_min_slider, self.wy1_max_slider,
                  self.vx2_min_slider, self.vx2_max_slider,
                  self.wy2_min_slider, self.wy2_max_slider,
                  self.eq_min_slider, self.eq_max_slider):
            s.valueChanged.connect(self._on_slider_changed)

        # Debounce timer – full refilter is expensive with many rallies
        self._refilter_timer = QTimer()
        self._refilter_timer.setSingleShot(True)
        self._refilter_timer.setInterval(150)  # 150ms debounce
        self._refilter_timer.timeout.connect(self._apply_filters_now)

        layout.addLayout(filter_row)

        # ── Info label ───────────────────────────────────────────────
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #aaa; font-size: 10pt;")
        layout.addWidget(self.info_label)

        # ── Detail label (shown when a trajectory is selected) ───────
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet(
            "color: #ddd; font-size: 10pt; background: rgba(40,40,80,180); "
            "border-radius: 4px; padding: 2px 6px;"
        )
        self.detail_label.setVisible(False)
        layout.addWidget(self.detail_label)

        # ── Three stacked pyqtgraph plots ────────────────────────────
        self.plot_vx = pg.PlotWidget()
        self.plot_wy = pg.PlotWidget()
        self.plot_z  = pg.PlotWidget()

        self.plot_vx.setTitle("|Velocity X|  (|vx|)")
        self.plot_wy.setTitle("Spin Y  (wy – topspin / backspin)")
        self.plot_z.setTitle("Height  (z)")

        self.plot_vx.setLabel('left', '|vx|', units='m/s')
        self.plot_wy.setLabel('left', 'wy', units='rad/s')
        self.plot_z.setLabel('left',  'z',  units='m')

        # Extra selectable quantity (4th time-series row)
        self.plot_extra = pg.PlotWidget()
        self._update_extra_labels()  # set title / axis labels from combo
        self.plot_extra.setLabel('bottom', 'Shot (relative to anchor)')

        # Sankey flow plot (5th row)
        self.plot_sankey = pg.PlotWidget()
        self.plot_sankey.setTitle("Filter flow  (inside / outside)")
        self.plot_sankey.setLabel('bottom', 'Shot (relative to anchor)')
        self.plot_sankey.getAxis('left').setStyle(showValues=False)
        self.plot_sankey.getAxis('left').setTicks([])
        self.plot_sankey.setMouseEnabled(x=False, y=False)

        # Extra row hidden by default ("None" is the first combo entry)
        self.plot_extra.setVisible(False)

        # Hide x-axis tick labels on upper plots; show on last visible row.
        # When extra is hidden, plot_z is the bottom time-series plot.
        self.plot_vx.getAxis('bottom').setStyle(showValues=False)
        self.plot_wy.getAxis('bottom').setStyle(showValues=False)
        self.plot_z.getAxis('bottom').setStyle(showValues=True)
        self.plot_z.setLabel('bottom', 'Shot (relative to anchor)')

        # Link x-axes (time-series plots only; Sankey has its own range)
        self.plot_wy.setXLink(self.plot_vx)
        self.plot_z.setXLink(self.plot_vx)
        self.plot_extra.setXLink(self.plot_vx)

        self._ts_plots = (self.plot_vx, self.plot_wy, self.plot_z, self.plot_extra)
        for pw in self._ts_plots:
            pw.showGrid(x=False, y=True, alpha=0.25)
            pw.setMouseEnabled(x=False, y=True)

        layout.addWidget(self.plot_vx, stretch=2)
        layout.addWidget(self.plot_wy, stretch=2)
        layout.addWidget(self.plot_z,  stretch=2)
        layout.addWidget(self.plot_extra, stretch=2)
        layout.addWidget(self.plot_sankey, stretch=1)

        # ── Internal state ───────────────────────────────────────────
        self._match_collection = None
        self._enabled_indices = None
        # Pre-computed per-rally anchor data (populated by _precompute_rally_data)
        self._precomputed_rallies: List[dict] = []
        # List of (rally, anchor_shot_idx, player_name)
        self._filtered_rallies: List[Tuple[object, int, str]] = []
        # Parallel metadata: one dict per filtered rally
        self._rally_metadata: List[Dict] = []
        # Colour map: player_name → (r, g, b)
        self._player_colors: Dict[str, Tuple[int, int, int]] = {}
        self._date_colors: Dict[str, Tuple[int, int, int]] = {}
        self.plots_folder: Optional[str] = None

        # Curve-selection state
        self._rally_curves: Dict[int, List[pg.PlotDataItem]] = {}   # rally_idx → curves
        self._curve_to_rally: Dict[int, int] = {}                   # id(curve) → rally_idx
        self._selected_rally_idx: Optional[int] = None

        # Sankey data – set per _apply_filters, consumed by _replot
        self._sankey_data: Optional[Dict] = None

        # Click on empty area to deselect
        for pw in self._ts_plots:
            pw.scene().sigMouseClicked.connect(self._on_scene_clicked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_match_collection(self, mc, enabled_indices=None):
        """Discover players, assign colours, rebuild checkboxes, then filter."""
        need_precompute = (mc is not self._match_collection)
        self._match_collection = mc
        self._enabled_indices = enabled_indices
        self._filtered_rallies = []
        self._rally_metadata = []

        if mc is None:
            self._precomputed_rallies = []
            self._player_colors = {}
            self._date_colors = {}
            self._rebuild_player_checkboxes([])
            self._clear_plots()
            self.info_label.setText("No data loaded.")
            return

        # Pre-compute per-shot anchor values once when the data changes.
        # This avoids repeated DataFrame access in _apply_filters.
        if need_precompute:
            self._precompute_rally_data()

        # Discover all player names from precomputed data (fast scan)
        enabled_set = set(enabled_indices) if enabled_indices is not None else None
        seen_matches: set = set()
        all_player_names: List[str] = []
        for rinfo in self._precomputed_rallies:
            m_idx = rinfo['match_idx']
            if m_idx in seen_matches:
                continue
            if enabled_set is not None and m_idx not in enabled_set:
                continue
            seen_matches.add(m_idx)
            name = rinfo['player_name']
            if name not in all_player_names:
                all_player_names.append(name)

        # Assign stable colours to every player
        self._player_colors = {}
        for i, name in enumerate(sorted(all_player_names)):
            self._player_colors[name] = _PLAYER_PALETTE[i % len(_PLAYER_PALETTE)]

        # Discover all dates and assign colours
        all_dates: List[str] = []
        for rinfo in self._precomputed_rallies:
            if enabled_set is not None and rinfo['match_idx'] not in enabled_set:
                continue
            d = rinfo.get('match_date', 'unknown')
            if d not in all_dates:
                all_dates.append(d)
        self._date_colors = {}
        for i, d in enumerate(sorted(all_dates)):
            self._date_colors[d] = _PLAYER_PALETTE[i % len(_PLAYER_PALETTE)]

        # Rebuild checkboxes (preserves checked state for known players)
        self._rebuild_player_checkboxes(all_player_names)

        # Apply vx/wy + player filters and replot
        self._apply_filters()

    def _apply_filters(self, _=None):
        """Recompute *_filtered_rallies* from sliders + player checkboxes, then replot.

        Uses pre-computed per-shot anchor values (populated by
        ``_precompute_rally_data``) so that this hot path—invoked on
        every slider change—does **no** DataFrame access at all.

        Also computes per-filter statistics and Sankey flow data so
        the UI can show how many rallies/segments satisfy each filter
        independently and combined.
        """
        mc = self._match_collection
        if mc is None:
            return

        self._filtered_rallies = []
        self._rally_metadata = []
        checked = self._get_checked_players()
        enabled_set = set(self._enabled_indices) if self._enabled_indices is not None else None

        anchor_mode = self.anchor_combo.currentText()
        vx_range = self._get_vx_range()
        wy_range = self._get_wy_range()
        vx1_range = self._get_vx1_range()
        wy1_range = self._get_wy1_range()
        vx2_range = self._get_vx2_range()
        wy2_range = self._get_wy2_range()
        extra_comp, _ = self._get_extra_config()
        extra_range = self._get_extra_filter_range()
        extra_use_abs = False
        if extra_comp and extra_comp in _EXTRA_FILTER_CFG:
            extra_use_abs = _EXTRA_FILTER_CFG[extra_comp][3]
        src = self.source_combo.currentText()
        pri = "opt" if src.startswith("OPT") else "gt200"
        fb = "gt200" if src.startswith("OPT") else None
        filters_visible = self._filter_grp_m2.isVisible()  # all 3 share visibility

        # ── Per-filter counters for causal analysis instrumentation ──
        n_eligible = 0          # rallies passing player/enabled checks
        n_valid_anchor = 0      # eligible rallies with a valid anchor shot (non-NaN data)
        n_pass_anchor = 0       # rallies with a valid anchor (pass shot 0 filter)
        n_pass_m1 = 0           # anchored rallies whose shot −1 passes its filter
        n_pass_m2 = 0           # anchored rallies whose shot −2 passes its filter

        # ── Sankey flow accumulator ──────────────────────────────────
        # Records a 4-element tuple
        #   (in_m2, in_m1, in_0, winner)
        # for EVERY rally that has a valid unfiltered anchor.
        # *winner* is 'robot', 'human', or 'unknown'.
        sankey_flows: Dict[Tuple[bool, bool, bool, str], int] = {}

        skip_precontact = self._skip_precontact_cb.isChecked()

        for rinfo_raw in self._precomputed_rallies:
            if enabled_set is not None and rinfo_raw['match_idx'] not in enabled_set:
                continue
            if rinfo_raw['player_name'] not in checked:
                continue

            # If the checkbox is active, swap in the corrected arrays
            # so every _fast helper reads post-contact values.
            if skip_precontact:
                rinfo = {**rinfo_raw}
                for _tag in ('opt', 'gt200'):
                    for _comp in ('vx', 'wy'):
                        _k = f'shot_{_comp}_{_tag}'
                        _ck = f'{_k}_corr'
                        if _ck in rinfo:
                            rinfo[_k] = rinfo[_ck]
            else:
                rinfo = rinfo_raw

            n_eligible += 1

            # ── Sankey: use an unfiltered anchor so we can evaluate
            #    all three columns independently ──────────────────────
            uf_anchor = self._find_anchor_unfiltered_fast(
                rinfo, pri, fb, anchor_mode)
            if uf_anchor is not None:
                n_valid_anchor += 1
            if uf_anchor is not None and filters_visible:
                # Evaluate each filter independently at the unfiltered anchor
                s0_in = self._check_shot_filter_fast(
                    rinfo, uf_anchor, vx_range, wy_range, pri, fb)
                if s0_in and extra_comp and extra_range:
                    s0_in = self._check_extra_filter_fast(
                        rinfo, uf_anchor, extra_comp, extra_range, extra_use_abs)

                seg_m2_idx = uf_anchor - _MAX_LOOKBACK
                m2_in_sk = True
                if seg_m2_idx >= 0:
                    m2_in_sk = self._check_shot_filter_fast(
                        rinfo, seg_m2_idx, vx2_range, wy2_range, pri, fb)

                seg_m1_idx = uf_anchor - 1
                m1_in_sk = True
                if seg_m1_idx >= 0:
                    m1_in_sk = self._check_shot_filter_fast(
                        rinfo, seg_m1_idx, vx1_range, wy1_range, pri, fb)

                pw_raw = rinfo.get('point_winner', None)
                if pw_raw == 'player1':
                    winner = 'robot'
                elif pw_raw == 'player2':
                    winner = 'human'
                else:
                    winner = 'unknown'
                key = (m2_in_sk, m1_in_sk, s0_in, winner)
                sankey_flows[key] = sankey_flows.get(key, 0) + 1

            # ── Filtered anchor for time-series display ──────────────
            anchor = self._find_anchor_fast(
                rinfo, vx_range, wy_range, pri, fb, anchor_mode,
                extra_comp=extra_comp, extra_range=extra_range,
                extra_use_abs=extra_use_abs)
            if anchor is None:
                continue

            n_pass_anchor += 1

            # ── Evaluate shot filters for the filtered rallies ───────
            seg_m2_idx = anchor - _MAX_LOOKBACK
            seg_m1_idx = anchor - 1

            m2_in = True
            if seg_m2_idx >= 0 and filters_visible:
                m2_in = self._check_shot_filter_fast(
                    rinfo, seg_m2_idx, vx2_range, wy2_range, pri, fb,
                    nan_lenient=False)
            if m2_in:
                n_pass_m2 += 1

            m1_in = True
            if seg_m1_idx >= 0 and filters_visible:
                m1_in = self._check_shot_filter_fast(
                    rinfo, seg_m1_idx, vx1_range, wy1_range, pri, fb,
                    nan_lenient=False)
            if m1_in:
                n_pass_m1 += 1

            # Only include rally in the output if ALL filters pass
            if not m2_in or not m1_in:
                continue

            self._filtered_rallies.append((rinfo['rally'], anchor, rinfo['player_name']))
            self._rally_metadata.append({
                'date': rinfo['match_date'],
                'player': rinfo['player_name'],
                'policy': rinfo['match_policy'],
                'game_name': rinfo['game_name'],
                'game_id': rinfo['game_id'],
                'rally_id': rinfo['rally_id'],
                'point_winner': rinfo['point_winner'],
                'anchor_shot': anchor,
                'num_shots': rinfo['num_shots'],
            })

        # ── Store Sankey data for _replot ─────────────────────────────
        n_sankey_total = sum(sankey_flows.values())
        self._sankey_data = {
            'flows': sankey_flows,
            'n_total': n_sankey_total,
        }

        # ── Update filter group box titles with per-filter stats ─────
        self._update_filter_stats(
            n_eligible, n_valid_anchor, n_pass_anchor,
            n_pass_m1, n_pass_m2, filters_visible)

        if self._filtered_rallies:
            self._replot()
        else:
            self._clear_plots()
            # Still draw the Sankey even when no rallies pass all filters
            if n_sankey_total > 0 and filters_visible:
                self._draw_sankey()
            anchor_mode = self.anchor_combo.currentText()
            if anchor_mode in ("Last Player Shot", "Last Robot Shot"):
                vx_lo, vx_hi = vx_range
                wy_lo, wy_hi = wy_range
                self.info_label.setText(
                    f"No rallies found with |vx| \u2208 [{vx_lo:.0f}, {vx_hi:.0f}] m/s  "
                    f"and wy \u2208 [{wy_lo:.0f}, {wy_hi:.0f}] rad/s  "
                    f"(anchor: {anchor_mode})."
                )
            else:
                self.info_label.setText(
                    f"No rallies found for anchor mode: {anchor_mode}."
                )

    # ── Filter statistics display ────────────────────────────────
    def _update_filter_stats(self, n_eligible: int, n_valid_anchor: int,
                             n_pass_anchor: int,
                             n_pass_m1: int, n_pass_m2: int,
                             filters_visible: bool):
        """Update filter group box titles and info with per-filter counts."""
        # Store for use by _replot info label
        self._filter_n_eligible = n_eligible
        self._filter_n_valid_anchor = n_valid_anchor
        self._filter_n_pass_anchor = n_pass_anchor
        self._filter_n_pass_m1 = n_pass_m1
        self._filter_n_pass_m2 = n_pass_m2
        self._filter_filters_visible = filters_visible

        if n_valid_anchor > 0:
            pct_anchor = 100.0 * n_pass_anchor / n_valid_anchor
            self._filter_grp_0.setTitle(
                f"Shot 0 (anchor) \u2014 {n_pass_anchor}/{n_valid_anchor} valid rallies "
                f"({pct_anchor:.1f}%)"
            )
            if filters_visible and n_pass_anchor > 0:
                pct_m1 = 100.0 * n_pass_m1 / n_pass_anchor
                self._filter_grp_m1.setTitle(
                    f"Shot −1 — {n_pass_m1}/{n_pass_anchor} rallies "
                    f"({pct_m1:.1f}% of anchor-passing)"
                )
                pct_m2 = 100.0 * n_pass_m2 / n_pass_anchor
                self._filter_grp_m2.setTitle(
                    f"Shot −2 — {n_pass_m2}/{n_pass_anchor} rallies "
                    f"({pct_m2:.1f}% of anchor-passing)"
                )
            else:
                self._filter_grp_m1.setTitle("Shot −1")
                self._filter_grp_m2.setTitle("Shot −2")
        else:
            self._filter_grp_0.setTitle("Shot 0 (anchor)")
            self._filter_grp_m1.setTitle("Shot −1")
            self._filter_grp_m2.setTitle("Shot −2")

    # ── Player filter helpers ──────────────────────────────────
    def _rebuild_player_checkboxes(self, player_names: List[str]):
        """(Re)create one checkbox per player, preserving checked state."""
        prev_state = {n: cb.isChecked() for n, cb in self._player_checkboxes.items()}
        # Clear old checkboxes
        while self._player_cb_container.count():
            item = self._player_cb_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._player_checkboxes.clear()
        # Create new ones (signal connected *after* setChecked to avoid spurious updates)
        for name in sorted(player_names):
            cb = QCheckBox(name)
            cb.setChecked(prev_state.get(name, True))
            r, g, b = self._player_colors.get(name, _COLOR_UNKNOWN)
            cb.setStyleSheet(f"QCheckBox {{ color: rgb({r},{g},{b}); }}")
            cb.toggled.connect(self._on_player_filter_changed)
            self._player_cb_container.addWidget(cb)
            self._player_checkboxes[name] = cb

    def _get_checked_players(self) -> set:
        """Return the set of player names whose checkboxes are checked."""
        if not self._player_checkboxes:
            return set(self._player_colors.keys())
        return {n for n, cb in self._player_checkboxes.items() if cb.isChecked()}

    def _set_all_players(self, checked: bool):
        """Check or uncheck all player checkboxes, then re-filter once."""
        for cb in self._player_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        if self._match_collection is not None:
            self._apply_filters()

    def _on_player_filter_changed(self, _=None):
        """Re-filter when a player checkbox is toggled."""
        if self._match_collection is not None:
            self._apply_filters()

    # ------------------------------------------------------------------
    # Pre-computed anchor data  (avoids DataFrame access in hot path)
    # ------------------------------------------------------------------

    def _precompute_rally_data(self):
        """Pre-extract per-shot vx/wy anchor values for all rallies.

        Called once when the MatchCollection *object* changes.  Stores a
        lightweight dict per rally with numpy arrays of per-shot starting
        vx/wy for both ``opt`` and ``gt200`` sources so that
        ``_apply_filters`` (the hot path on every slider change) does **no**
        DataFrame / ``.iloc`` access at all.
        """
        mc = self._match_collection
        self._precomputed_rallies = []
        if mc is None:
            return

        for m_idx, match in enumerate(mc.matches):
            player_name = getattr(match, 'player', None) or 'unknown'
            match_date = getattr(match, 'date', None)
            match_policy = getattr(match, 'policy', '') or ''

            for game in match.games:
                game_name = getattr(game, 'game_name', '') or ''
                game_id = getattr(game, 'game_id', '?')

                for rally in game.rallies:
                    n_shots = len(rally.shots)
                    shot_players = np.zeros(n_shots, dtype=np.int8)
                    shot_vx_opt = np.full(n_shots, np.nan)
                    shot_wy_opt = np.full(n_shots, np.nan)
                    shot_vx_gt200 = np.full(n_shots, np.nan)
                    shot_wy_gt200 = np.full(n_shots, np.nan)

                    # Corrected arrays (discontinuity-detected post-contact)
                    shot_vx_opt_corr = np.full(n_shots, np.nan)
                    shot_wy_opt_corr = np.full(n_shots, np.nan)
                    shot_vx_gt200_corr = np.full(n_shots, np.nan)
                    shot_wy_gt200_corr = np.full(n_shots, np.nan)

                    # Extra quantity arrays (bare columns)
                    shot_extra = {c: np.full(n_shots, np.nan)
                                  for c in _EXTRA_COMPONENTS}

                    for s_idx, shot in enumerate(rally.shots):
                        shot_players[s_idx] = getattr(shot, 'player', 0) or 0
                        for fs in shot.flight_segments:
                            te = getattr(fs, 'trigger_event', None)
                            if getattr(te, 'type', None) == 'START':
                                continue
                            df = fs.data
                            if df is None or len(df) == 0:
                                continue
                            cols = df.columns
                            vals = df.values
                            n_rows = len(vals)
                            for tag, vx_arr, wy_arr, vx_arr_c, wy_arr_c in [
                                ('opt', shot_vx_opt, shot_wy_opt,
                                 shot_vx_opt_corr, shot_wy_opt_corr),
                                ('gt200', shot_vx_gt200, shot_wy_gt200,
                                 shot_vx_gt200_corr, shot_wy_gt200_corr),
                            ]:
                                vx_col = f'vx_{tag}'
                                wy_col = f'wy_{tag}'
                                # Detect pre-contact boundary row via
                                # velocity discontinuity (|Δvx| > 1 m/s).
                                pc_idx = 0  # post-contact index
                                if vx_col in cols and n_rows > 1:
                                    vx_ci = cols.get_loc(vx_col)
                                    for ri in range(1, min(n_rows, 5)):
                                        prev_v = vals[ri - 1, vx_ci]
                                        curr_v = vals[ri, vx_ci]
                                        if (not np.isnan(prev_v)
                                                and not np.isnan(curr_v)
                                                and abs(curr_v - prev_v) > 1.0):
                                            pc_idx = ri
                                            print(f"\033[93m[time_series_plot] WARNING: "
                                                  f"game {game_id} shot {s_idx} "
                                                  f"{vx_col}: row[0] is pre-contact "
                                                  f"(vx={prev_v:.2f}→{curr_v:.2f} "
                                                  f"at row {ri})\033[0m")
                                            break
                                # Raw: always row[0]
                                row0 = vals[0]
                                if vx_col in cols:
                                    v = row0[cols.get_loc(vx_col)]
                                    if not np.isnan(v):
                                        vx_arr[s_idx] = float(v)
                                if wy_col in cols:
                                    v = row0[cols.get_loc(wy_col)]
                                    if not np.isnan(v):
                                        wy_arr[s_idx] = float(v)
                                # Corrected: first post-contact row
                                row_c = vals[pc_idx]
                                if vx_col in cols:
                                    v = row_c[cols.get_loc(vx_col)]
                                    if not np.isnan(v):
                                        vx_arr_c[s_idx] = float(v)
                                if wy_col in cols:
                                    v = row_c[cols.get_loc(wy_col)]
                                    if not np.isnan(v):
                                        wy_arr_c[s_idx] = float(v)
                            # Extra quantities (wx, wz, x, y, vy, vz)
                            # Resolve with source suffix: opt → gt200 → bare,
                            # same as _col() does for plotting.
                            # 'x' uses the last row (final position); others
                            # use row[0] (start of segment).
                            row_last = vals[-1]
                            for _ecomp, _earr in shot_extra.items():
                                _row = row_last if _ecomp == 'x' else row0
                                # Try primary source, then fallback, then bare
                                _resolved = None
                                for _src_tag in ('opt', 'gt200'):
                                    _cname = f'{_ecomp}_{_src_tag}'
                                    if _cname in cols:
                                        _ev = _row[cols.get_loc(_cname)]
                                        if not np.isnan(_ev):
                                            _resolved = float(_ev)
                                            break
                                if _resolved is None and _ecomp in cols:
                                    _ev = _row[cols.get_loc(_ecomp)]
                                    if not np.isnan(_ev):
                                        _resolved = float(_ev)
                                if _resolved is not None:
                                    _earr[s_idx] = _resolved
                            break  # only first valid segment for vx/wy

                    self._precomputed_rallies.append({
                        'rally': rally,
                        'match_idx': m_idx,
                        'player_name': player_name,
                        'match_date': str(match_date) if match_date else 'unknown',
                        'match_policy': match_policy,
                        'game_name': game_name,
                        'game_id': game_id,
                        'rally_id': getattr(rally, 'rally_id', '?'),
                        'point_winner': getattr(rally, 'point_winner', None),
                        'num_shots': n_shots,
                        'shot_players': shot_players,
                        'shot_vx_opt': shot_vx_opt,
                        'shot_wy_opt': shot_wy_opt,
                        'shot_vx_gt200': shot_vx_gt200,
                        'shot_wy_gt200': shot_wy_gt200,
                        'shot_vx_opt_corr': shot_vx_opt_corr,
                        'shot_wy_opt_corr': shot_wy_opt_corr,
                        'shot_vx_gt200_corr': shot_vx_gt200_corr,
                        'shot_wy_gt200_corr': shot_wy_gt200_corr,
                        'shot_extra': shot_extra,
                    })

    @staticmethod
    def _find_anchor_fast(
        rinfo: dict,
        vx_range: Tuple[float, float],
        wy_range: Tuple[float, float],
        pri: str,
        fb: Optional[str],
        mode: str,
        extra_comp: Optional[str] = None,
        extra_range: Optional[Tuple[float, float]] = None,
        extra_use_abs: bool = False,
    ) -> Optional[int]:
        """Find the anchor shot index using pre-computed arrays (no DataFrame access)."""
        n = rinfo['num_shots']

        if mode == "Player Serve":
            if n > 1 and rinfo['shot_players'][1] == 2:
                return 1
            return None

        if mode == "Robot Serve":
            if n > 1 and rinfo['shot_players'][1] == 1:
                return 1
            return None

        # Filtered modes: apply vx/wy range
        target_player = 2 if mode == "Last Player Shot" else 1
        players = rinfo['shot_players']

        # Resolve vx/wy with primary → fallback source
        vx_pri = rinfo[f'shot_vx_{pri}']
        wy_pri = rinfo[f'shot_wy_{pri}']
        if fb:
            vx_fb = rinfo[f'shot_vx_{fb}']
            wy_fb = rinfo[f'shot_wy_{fb}']
            vx = np.where(~np.isnan(vx_pri), vx_pri, vx_fb)
            wy = np.where(~np.isnan(wy_pri), wy_pri, wy_fb)
        else:
            vx = vx_pri
            wy = wy_pri

        vx_lo, vx_hi = vx_range
        wy_lo, wy_hi = wy_range

        # When a range is (-inf, +inf) the filter is fully open:
        # skip both the NaN check and the range check for that component
        # so rallies with missing data are still eligible.
        vx_full = (vx_lo == -float('inf') and vx_hi == float('inf'))
        wy_full = (wy_lo == -float('inf') and wy_hi == float('inf'))

        valid = (players == target_player)
        if not vx_full:
            valid = valid & ~np.isnan(vx) & (np.abs(vx) >= vx_lo) & (np.abs(vx) <= vx_hi)
        if not wy_full:
            valid = valid & ~np.isnan(wy) & (wy >= wy_lo) & (wy <= wy_hi)

        # Extra quantity filter (shot 0 only)
        if extra_comp and extra_range:
            eq_lo, eq_hi = extra_range
            eq_full = (eq_lo == -float('inf') and eq_hi == float('inf'))
            if not eq_full:
                eq_vals = rinfo['shot_extra'][extra_comp]
                if extra_use_abs:
                    valid = valid & ~np.isnan(eq_vals) & (np.abs(eq_vals) >= eq_lo) & (np.abs(eq_vals) <= eq_hi)
                else:
                    valid = valid & ~np.isnan(eq_vals) & (eq_vals >= eq_lo) & (eq_vals <= eq_hi)

        matching = np.where(valid)[0]
        if len(matching) == 0:
            return None
        return int(matching[-1])

    @staticmethod
    def _find_anchor_unfiltered_fast(
        rinfo: dict,
        pri: str,
        fb: Optional[str],
        mode: str,
    ) -> Optional[int]:
        """Find the anchor shot index *without* vx/wy filtering.

        Used by the Sankey to evaluate all eligible rallies regardless
        of whether they pass the shot-0 filter.
        """
        n = rinfo['num_shots']

        if mode == "Player Serve":
            if n > 1 and rinfo['shot_players'][1] == 2:
                return 1
            return None

        if mode == "Robot Serve":
            if n > 1 and rinfo['shot_players'][1] == 1:
                return 1
            return None

        # Non-serve modes: find the last shot by the target player
        # that has valid vx/wy data (but ignore the range filter).
        target_player = 2 if mode == "Last Player Shot" else 1
        players = rinfo['shot_players']

        vx_pri = rinfo[f'shot_vx_{pri}']
        wy_pri = rinfo[f'shot_wy_{pri}']
        if fb:
            vx_fb = rinfo[f'shot_vx_{fb}']
            wy_fb = rinfo[f'shot_wy_{fb}']
            vx = np.where(~np.isnan(vx_pri), vx_pri, vx_fb)
            wy = np.where(~np.isnan(wy_pri), wy_pri, wy_fb)
        else:
            vx = vx_pri
            wy = wy_pri

        valid = (
            (players == target_player)
            & ~np.isnan(vx) & ~np.isnan(wy)
        )
        matching = np.where(valid)[0]
        if len(matching) == 0:
            return None
        return int(matching[-1])

    @staticmethod
    def _check_shot_filter_fast(
        rinfo: dict,
        shot_idx: int,
        vx_range: Tuple[float, float],
        wy_range: Tuple[float, float],
        pri: str,
        fb: Optional[str],
        nan_lenient: bool = True,
    ) -> bool:
        """Check a single shot against vx/wy filters using pre-computed arrays.

        *nan_lenient* controls how missing data (NaN in both sources) is
        handled:
          True  – NaN passes (used by Sankey so max-limits → all “in”).
          False – NaN fails  (original display behaviour: unknown → exclude).
        """
        # When a range is (-inf, +inf) the filter is fully open: skip check
        # entirely so that NaN values pass regardless of nan_lenient.
        vx_full = (vx_range[0] == -float('inf') and vx_range[1] == float('inf'))
        wy_full = (wy_range[0] == -float('inf') and wy_range[1] == float('inf'))

        if not vx_full:
            vx = rinfo[f'shot_vx_{pri}'][shot_idx]
            if np.isnan(vx) and fb:
                vx = rinfo[f'shot_vx_{fb}'][shot_idx]
            if nan_lenient:
                if not np.isnan(vx) and not (vx_range[0] <= abs(vx) <= vx_range[1]):
                    return False
            else:
                if np.isnan(vx) or not (vx_range[0] <= abs(vx) <= vx_range[1]):
                    return False

        if not wy_full:
            wy = rinfo[f'shot_wy_{pri}'][shot_idx]
            if np.isnan(wy) and fb:
                wy = rinfo[f'shot_wy_{fb}'][shot_idx]
            if nan_lenient:
                if not np.isnan(wy) and not (wy_range[0] <= wy <= wy_range[1]):
                    return False
            else:
                if np.isnan(wy) or not (wy_range[0] <= wy <= wy_range[1]):
                    return False

        return True

    @staticmethod
    def _check_extra_filter_fast(
        rinfo: dict,
        shot_idx: int,
        extra_comp: str,
        extra_range: Tuple[float, float],
        extra_use_abs: bool,
    ) -> bool:
        """Check a single shot against the extra quantity filter.

        Returns True (pass) when *extra_range* is ``(-inf, +inf)``.
        """
        eq_lo, eq_hi = extra_range
        if eq_lo == -float('inf') and eq_hi == float('inf'):
            return True
        val = rinfo['shot_extra'][extra_comp][shot_idx]
        if np.isnan(val):
            return False
        if extra_use_abs:
            val = abs(val)
        return eq_lo <= val <= eq_hi

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _on_color_by_changed(self, _=None):
        """Re-draw when the user switches colour mode."""
        if self._match_collection is not None and self._filtered_rallies:
            self._replot()

    def _on_skip_precontact_changed(self, _=None):
        """Re-filter when the 'Skip pre-contact rows' checkbox changes."""
        if self._match_collection is not None:
            self._apply_filters()

    def _on_extra_changed(self, _=None):
        """Show/hide the extra row, reconfigure filter sliders, and re-filter."""
        show = self.extra_combo.currentIndex() > 0
        self.plot_extra.setVisible(show)
        # Swap x-axis labels to the last visible time-series row
        self.plot_z.getAxis('bottom').setStyle(showValues=not show)
        if show:
            self.plot_z.setLabel('bottom', '')
        else:
            self.plot_z.setLabel('bottom', 'Shot (relative to anchor)')
        self._update_extra_labels()

        # Reconfigure extra filter sliders
        comp, _ = self._get_extra_config()
        if comp and comp in _EXTRA_FILTER_CFG:
            cfg = _EXTRA_FILTER_CFG[comp]
            abs_min, abs_max, scale, use_abs, fmt = cfg
            self._eq_filter_cfg = cfg
            # Block signals while reconfiguring
            for s in (self.eq_min_slider, self.eq_max_slider):
                s.blockSignals(True)
            self.eq_min_slider.setMinimum(int(abs_min * scale))
            self.eq_min_slider.setMaximum(int(abs_max * scale))
            self.eq_min_slider.setValue(int(abs_min * scale))
            self.eq_max_slider.setMinimum(int(abs_min * scale))
            self.eq_max_slider.setMaximum(int(abs_max * scale))
            self.eq_max_slider.setValue(int(abs_max * scale))
            for s in (self.eq_min_slider, self.eq_max_slider):
                s.blockSignals(False)
            # Update filter label
            _, _, y_label, _, _ = _EXTRA_QUANTITIES[self.extra_combo.currentIndex()]
            self.eq_filter_label.setText(f"|{y_label}|" if use_abs else y_label)
            self.eq_min_label.setText(f"{abs_min:{fmt}}")
            self.eq_max_label.setText(f"{abs_max:{fmt}}")
            for w in self._eq_filter_widgets:
                w.setVisible(True)
        else:
            self._eq_filter_cfg = None
            for w in self._eq_filter_widgets:
                w.setVisible(False)

        # Re-filter (extra filter may have changed)
        if self._match_collection is not None:
            self._apply_filters()

    def _update_extra_labels(self):
        """Set the title / y-axis label on plot_extra from the combo."""
        idx = self.extra_combo.currentIndex()
        if idx == 0:  # "None" – extra row hidden
            return
        label, component, y_label, units, _ = _EXTRA_QUANTITIES[idx]
        self.plot_extra.setTitle(label)
        self.plot_extra.setLabel('left', y_label, units=units)

    def _get_extra_config(self):
        """Return (component, use_abs) for the currently selected extra quantity.

        Returns ``(None, False)`` when the extra row is disabled ("None").
        """
        idx = self.extra_combo.currentIndex()
        if idx == 0:  # "None" – extra row hidden
            return None, False
        _, component, _, _, use_abs = _EXTRA_QUANTITIES[idx]
        return component, use_abs

    def _on_anchor_mode_changed(self, _=None):
        """Show/hide filter groups depending on anchor mode, then re-filter."""
        mode = self.anchor_combo.currentText()
        uses_filter = mode in ("Last Player Shot", "Last Robot Shot")
        self._filter_grp_0.setVisible(uses_filter)
        self._filter_grp_m1.setVisible(uses_filter)
        self._filter_grp_m2.setVisible(uses_filter)
        if self._match_collection is not None:
            self._apply_filters()

    def _rally_color(self, rally, player_name,
                     rally_idx: Optional[int] = None) -> Tuple[int, int, int]:
        """Return (r, g, b) for a rally based on the current 'Color by' mode."""
        mode = self.color_by_combo.currentText()
        if mode == "Point winner":
            pw = getattr(rally, 'point_winner', None)
            if pw == "player1":
                return _COLOR_ROBOT_WON
            elif pw == "player2":
                return _COLOR_HUMAN_WON
            else:
                return _COLOR_WINNER_UNKNOWN
        if mode == "Date" and rally_idx is not None:
            if rally_idx < len(self._rally_metadata):
                d = self._rally_metadata[rally_idx].get('date', 'unknown')
                return self._date_colors.get(d, _COLOR_UNKNOWN)
            return _COLOR_UNKNOWN
        return self._player_colors.get(player_name, _COLOR_UNKNOWN)

    # ── Slider helpers ────────────────────────────────────────────
    @staticmethod
    def _make_slider(lo: float, hi: float, default: float, scale: float) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setMinimum(int(lo * scale))
        s.setMaximum(int(hi * scale))
        s.setValue(int(default * scale))
        s.setSingleStep(1)
        s.setPageStep(max(1, int((hi - lo) * scale / 10)))
        return s

    def _slider_val(self, slider: QSlider, scale: float) -> float:
        return slider.value() / scale

    def _get_vx_range(self) -> Tuple[float, float]:
        lo = self._slider_val(self.vx_min_slider, _VX_SLIDER_SCALE)
        hi = self._slider_val(self.vx_max_slider, _VX_SLIDER_SCALE)
        lo, hi = min(lo, hi), max(lo, hi)
        if self._is_full_range_vx(lo, hi):
            return (-float('inf'), float('inf'))
        return (lo, hi)

    def _get_wy_range(self) -> Tuple[float, float]:
        lo = self._slider_val(self.wy_min_slider, _WY_SLIDER_SCALE)
        hi = self._slider_val(self.wy_max_slider, _WY_SLIDER_SCALE)
        lo, hi = min(lo, hi), max(lo, hi)
        if self._is_full_range_wy(lo, hi):
            return (-float('inf'), float('inf'))
        return (lo, hi)

    def _get_vx1_range(self) -> Tuple[float, float]:
        lo = self._slider_val(self.vx1_min_slider, _VX_SLIDER_SCALE)
        hi = self._slider_val(self.vx1_max_slider, _VX_SLIDER_SCALE)
        lo, hi = min(lo, hi), max(lo, hi)
        if self._is_full_range_vx(lo, hi):
            return (-float('inf'), float('inf'))
        return (lo, hi)

    def _get_wy1_range(self) -> Tuple[float, float]:
        lo = self._slider_val(self.wy1_min_slider, _WY_SLIDER_SCALE)
        hi = self._slider_val(self.wy1_max_slider, _WY_SLIDER_SCALE)
        lo, hi = min(lo, hi), max(lo, hi)
        if self._is_full_range_wy(lo, hi):
            return (-float('inf'), float('inf'))
        return (lo, hi)

    def _get_vx2_range(self) -> Tuple[float, float]:
        lo = self._slider_val(self.vx2_min_slider, _VX_SLIDER_SCALE)
        hi = self._slider_val(self.vx2_max_slider, _VX_SLIDER_SCALE)
        lo, hi = min(lo, hi), max(lo, hi)
        if self._is_full_range_vx(lo, hi):
            return (-float('inf'), float('inf'))
        return (lo, hi)

    def _get_wy2_range(self) -> Tuple[float, float]:
        lo = self._slider_val(self.wy2_min_slider, _WY_SLIDER_SCALE)
        hi = self._slider_val(self.wy2_max_slider, _WY_SLIDER_SCALE)
        lo, hi = min(lo, hi), max(lo, hi)
        if self._is_full_range_wy(lo, hi):
            return (-float('inf'), float('inf'))
        return (lo, hi)

    def _get_extra_filter_range(self) -> Optional[Tuple[float, float]]:
        """Return (lo, hi) for the active extra filter, or None if disabled."""
        cfg = self._eq_filter_cfg
        if cfg is None:
            return None
        abs_min, abs_max, scale, use_abs, _fmt = cfg
        lo = self._slider_val(self.eq_min_slider, scale)
        hi = self._slider_val(self.eq_max_slider, scale)
        lo, hi = min(lo, hi), max(lo, hi)
        if lo <= abs_min and hi >= abs_max:
            return (-float('inf'), float('inf'))
        return (lo, hi)

    def _reset_filters(self):
        """Set every vx/wy slider to its absolute min/max (full range)."""
        # Block signals while bulk-updating to avoid per-slider refilters
        sliders = [
            (self.vx_min_slider,  _VX_ABS_MIN,  _VX_SLIDER_SCALE),
            (self.vx_max_slider,  _VX_ABS_MAX,  _VX_SLIDER_SCALE),
            (self.wy_min_slider,  _WY_ABS_MIN,  _WY_SLIDER_SCALE),
            (self.wy_max_slider,  _WY_ABS_MAX,  _WY_SLIDER_SCALE),
            (self.vx1_min_slider, _VX_ABS_MIN,  _VX_SLIDER_SCALE),
            (self.vx1_max_slider, _VX_ABS_MAX,  _VX_SLIDER_SCALE),
            (self.wy1_min_slider, _WY_ABS_MIN,  _WY_SLIDER_SCALE),
            (self.wy1_max_slider, _WY_ABS_MAX,  _WY_SLIDER_SCALE),
            (self.vx2_min_slider, _VX_ABS_MIN,  _VX_SLIDER_SCALE),
            (self.vx2_max_slider, _VX_ABS_MAX,  _VX_SLIDER_SCALE),
            (self.wy2_min_slider, _WY_ABS_MIN,  _WY_SLIDER_SCALE),
            (self.wy2_max_slider, _WY_ABS_MAX,  _WY_SLIDER_SCALE),
        ]
        # Also reset the extra quantity filter if active
        if self._eq_filter_cfg:
            abs_min, abs_max, scale, _, _ = self._eq_filter_cfg
            sliders.append((self.eq_min_slider, abs_min, scale))
            sliders.append((self.eq_max_slider, abs_max, scale))
        for s, val, scale in sliders:
            s.blockSignals(True)
            s.setValue(int(val * scale))
            s.blockSignals(False)
        # Update labels + trigger a single refilter
        self._on_slider_changed()

    def _is_full_range_vx(self, lo: float, hi: float) -> bool:
        """True if vx range covers the full slider extent."""
        return lo <= _VX_ABS_MIN and hi >= _VX_ABS_MAX

    def _is_full_range_wy(self, lo: float, hi: float) -> bool:
        """True if wy range covers the full slider extent."""
        return lo <= _WY_ABS_MIN and hi >= _WY_ABS_MAX

    def _on_slider_changed(self, _=None):
        """Update labels and schedule a debounced re-filter."""
        vx_lo, vx_hi = self._get_vx_range()
        wy_lo, wy_hi = self._get_wy_range()
        self.vx_min_label.setText(f"{vx_lo:.0f}")
        self.vx_max_label.setText(f"{vx_hi:.0f}")
        self.wy_min_label.setText(f"{wy_lo:.0f}")
        self.wy_max_label.setText(f"{wy_hi:.0f}")
        vx1_lo, vx1_hi = self._get_vx1_range()
        wy1_lo, wy1_hi = self._get_wy1_range()
        self.vx1_min_label.setText(f"{vx1_lo:.0f}")
        self.vx1_max_label.setText(f"{vx1_hi:.0f}")
        self.wy1_min_label.setText(f"{wy1_lo:.0f}")
        self.wy1_max_label.setText(f"{wy1_hi:.0f}")
        vx2_lo, vx2_hi = self._get_vx2_range()
        wy2_lo, wy2_hi = self._get_wy2_range()
        self.vx2_min_label.setText(f"{vx2_lo:.0f}")
        self.vx2_max_label.setText(f"{vx2_hi:.0f}")
        self.wy2_min_label.setText(f"{wy2_lo:.0f}")
        self.wy2_max_label.setText(f"{wy2_hi:.0f}")
        # Extra filter labels
        if self._eq_filter_cfg:
            _, _, scale, _, fmt = self._eq_filter_cfg
            eq_lo = self._slider_val(self.eq_min_slider, scale)
            eq_hi = self._slider_val(self.eq_max_slider, scale)
            self.eq_min_label.setText(f"{eq_lo:{fmt}}")
            self.eq_max_label.setText(f"{eq_hi:{fmt}}")
        if self._match_collection is not None:
            self._refilter_timer.start()  # debounced

    def _apply_filters_now(self):
        """Execute filter + replot (called by debounce timer)."""
        if self._match_collection is not None:
            self._apply_filters()

    def _clear_plots(self):
        for pw in (*self._ts_plots, self.plot_sankey):
            pw.clear()
        self.detail_label.setVisible(False)

    @staticmethod
    def _col(df, component: str, primary: str, fallback: str):
        """Return column values with primary→fallback source resolution."""
        c1 = f"{component}_{primary}"
        if c1 in df.columns:
            vals = df[c1].values
            if not np.all(np.isnan(vals)):
                return vals
        if fallback:
            c2 = f"{component}_{fallback}"
            if c2 in df.columns:
                return df[c2].values
        # Bare column name (e.g. 'wx', 'wz' from GCS have no source suffix)
        if component in df.columns:
            vals = df[component].values
            if not np.all(np.isnan(vals)):
                return vals
        return None

    @staticmethod
    def _shot_start_vx(shot, pri: str, fb: str,
                       skip_precontact: bool = False) -> Optional[float]:
        """Return the starting vx of the first valid segment.

        When *skip_precontact* is True, skips past pre-contact boundary
        rows (detected via |Δvx| > 1 m/s discontinuity).  Otherwise
        uses row[0].
        """
        for fs in shot.flight_segments:
            if getattr(getattr(fs, 'trigger_event', None), 'type', None) == 'START':
                continue
            df = fs.data
            if df is None or len(df) == 0:
                continue
            for colname in (f"vx_{pri}", f"vx_{fb}" if fb else None):
                if colname is None or colname not in df.columns:
                    continue
                vals = df[colname].values
                idx = 0
                if skip_precontact:
                    for ri in range(1, min(len(vals), 5)):
                        if (not np.isnan(vals[ri - 1])
                                and not np.isnan(vals[ri])
                                and abs(vals[ri] - vals[ri - 1]) > 1.0):
                            idx = ri
                            break
                v = vals[idx]
                if not np.isnan(v):
                    return float(v)
        return None

    @staticmethod
    def _shot_start_wy(shot, pri: str, fb: str,
                       skip_precontact: bool = False) -> Optional[float]:
        """Return the starting wy of the first valid segment.

        When *skip_precontact* is True, uses the same post-contact row
        index as vx (detected via |Δvx| > 1 m/s discontinuity).
        Otherwise uses row[0].
        """
        for fs in shot.flight_segments:
            if getattr(getattr(fs, 'trigger_event', None), 'type', None) == 'START':
                continue
            df = fs.data
            if df is None or len(df) == 0:
                continue
            idx = 0
            if skip_precontact:
                for vx_col in (f"vx_{pri}", f"vx_{fb}" if fb else None):
                    if vx_col is None or vx_col not in df.columns:
                        continue
                    vxv = df[vx_col].values
                    for ri in range(1, min(len(vxv), 5)):
                        if (not np.isnan(vxv[ri - 1])
                                and not np.isnan(vxv[ri])
                                and abs(vxv[ri] - vxv[ri - 1]) > 1.0):
                            idx = ri
                            break
                    break
            for colname in (f"wy_{pri}", f"wy_{fb}" if fb else None):
                if colname is None or colname not in df.columns:
                    continue
                v = df[colname].iloc[idx]
                if not np.isnan(v):
                    return float(v)
        return None

    @classmethod
    def _find_anchor(cls, rally, vx_range: Tuple[float, float],
                     wy_range: Tuple[float, float], pri: str, fb: str,
                     mode: str = "Last Player Shot (filtered)") -> Optional[int]:
        """Return the anchor shot index in *rally* based on the selected mode."""
        if mode == "Player Serve":
            if len(rally.shots) > 1 and getattr(rally.shots[1], 'player', None) == 2:
                return 1
            return None

        if mode == "Robot Serve":
            if len(rally.shots) > 1 and getattr(rally.shots[1], 'player', None) == 1:
                return 1
            return None

        # Filtered modes: apply vx/wy range
        target_player = 2 if mode == "Last Player Shot" else 1
        vx_lo, vx_hi = vx_range
        wy_lo, wy_hi = wy_range
        anchor = None
        for shot_idx, shot in enumerate(rally.shots):
            player = getattr(shot, 'player', None)
            if player != target_player:
                continue
            vx = cls._shot_start_vx(shot, pri, fb)
            if vx is None or not (vx_lo <= abs(vx) <= vx_hi):
                continue
            wy = cls._shot_start_wy(shot, pri, fb)
            if wy is None or not (wy_lo <= wy <= wy_hi):
                continue
            anchor = shot_idx
        return anchor

    @classmethod
    def _check_shot_filter(cls, shot, vx_range: Tuple[float, float],
                           wy_range: Tuple[float, float],
                           pri: str, fb: str) -> bool:
        """Return True if *shot*'s starting |vx| and wy fall in the given ranges."""
        vx = cls._shot_start_vx(shot, pri, fb)
        if vx is None or not (vx_range[0] <= abs(vx) <= vx_range[1]):
            return False
        wy = cls._shot_start_wy(shot, pri, fb)
        if wy is None or not (wy_range[0] <= wy <= wy_range[1]):
            return False
        return True

    # ------------------------------------------------------------------
    def _replot(self, _=None):
        """Redraw filtered rallies, aligned at x = 0 = anchor shot."""
        self._clear_plots()

        if not self._filtered_rallies:
            return

        src = self.source_combo.currentText()
        pri, fb = ("opt", "gt200") if src.startswith("OPT") else ("gt200", None)

        # x-axis spans [-_MAX_LOOKBACK, _MAX_LOOKAHEAD]
        x_min = -_MAX_LOOKBACK                # -2
        x_max = _MAX_LOOKAHEAD + 1            # right edge of last column (+3)

        extra_comp, extra_abs = self._get_extra_config()

        # Only operate on visible time-series plots (excludes hidden extra)
        active_plots = tuple(pw for pw in self._ts_plots if pw.isVisible())

        # ── Alternating column shading ───────────────────────────────
        for i in range(x_min, x_max):
            abs_i = i - x_min
            col = _COL_ODD if (abs_i % 2) else _COL_EVEN
            for pw in active_plots:
                r = pg.LinearRegionItem(
                    values=[i, i + 1], orientation='vertical',
                    movable=False, brush=QBrush(col),
                )
                r.setZValue(-100)
                r.lines[0].setPen(pg.mkPen(None))
                r.lines[1].setPen(pg.mkPen(None))
                pw.addItem(r)

        # ── Shot boundary lines ──────────────────────────────────────
        for i in range(x_min, x_max + 1):
            for pw in active_plots:
                pw.addItem(pg.InfiniteLine(
                    pos=i, angle=90,
                    pen=pg.mkPen((255, 255, 255, 160), width=2, style=Qt.SolidLine),
                ))

        # ── Highlight the anchor column (x=0) with a slightly brighter fill ─
        for pw in active_plots:
            anchor_rect = pg.LinearRegionItem(
                values=[0, 1], orientation='vertical',
                movable=False, brush=QBrush(QColor(255, 200, 50, 20)),
            )
            anchor_rect.setZValue(-90)
            anchor_rect.lines[0].setPen(pg.mkPen(None))
            anchor_rect.lines[1].setPen(pg.mkPen(None))
            pw.addItem(anchor_rect)

        # ── Draw each rally ──────────────────────────────────────────
        total_rallies = 0
        total_pts = 0
        shot_player_map: Dict[int, set] = {}  # rel_idx → {player_values}
        self._rally_curves = {}
        self._curve_to_rally = {}
        self._selected_rally_idx = None

        for rally, anchor_idx, player_name in self._filtered_rallies:
            color_rgb = self._rally_color(rally, player_name,
                                          rally_idx=total_rallies)

            # Determine which shots to draw: up to 2 before + 2 after
            first_shot = max(0, anchor_idx - _MAX_LOOKBACK)
            last_shot = min(len(rally.shots) - 1, anchor_idx + _MAX_LOOKAHEAD)

            for abs_shot_idx in range(first_shot, last_shot + 1):
                shot = rally.shots[abs_shot_idx]
                # Relative position: anchor → 0, anchor-1 → -1, etc.
                rel = abs_shot_idx - anchor_idx

                # Track which player performed this shot at each position
                _player = getattr(shot, 'player', None)
                if _player is not None:
                    shot_player_map.setdefault(rel, set()).add(_player)

                segs = [
                    fs for fs in shot.flight_segments
                    if getattr(getattr(fs, 'trigger_event', None), 'type', None) != 'START'
                ]
                if not segs:
                    continue

                t_parts: List[np.ndarray] = []
                vx_parts: List[np.ndarray] = []
                wy_parts: List[np.ndarray] = []
                z_parts: List[np.ndarray] = []
                extra_parts: List[np.ndarray] = []

                for fs in segs:
                    df = fs.data
                    if df is None or len(df) == 0:
                        continue
                    t  = df['time'].values.copy()
                    vx = self._col(df, 'vx', pri, fb)
                    wy = self._col(df, 'wy', pri, fb)
                    z  = self._col(df, 'z',  pri, fb)
                    eq = self._col(df, extra_comp, pri, fb) if extra_comp else None
                    if vx is None and wy is None and z is None and eq is None:
                        continue
                    n = len(t)
                    t_parts.append(t)
                    vx_parts.append(vx if vx is not None else np.full(n, np.nan))
                    wy_parts.append(wy if wy is not None else np.full(n, np.nan))
                    z_parts.append(z  if z  is not None else np.full(n, np.nan))
                    if extra_comp:
                        extra_parts.append(eq if eq is not None else np.full(n, np.nan))

                if not t_parts:
                    continue

                all_t  = np.concatenate(t_parts)
                all_vx = np.concatenate(vx_parts)
                all_wy = np.concatenate(wy_parts)
                all_z  = np.concatenate(z_parts)

                # Normalise time → [0, 1] within the shot
                t_min, t_max = all_t.min(), all_t.max()
                norm = (all_t - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(all_t)
                x = norm + rel  # rel is ≤ 0, so x ∈ [rel, rel+1)
                total_pts += len(x)

                pen = pg.mkPen(color=(*color_rgb, _LINE_ALPHA), width=_LINE_WIDTH)
                rally_idx = total_rallies
                pairs = [
                    (self.plot_vx, np.abs(all_vx)),
                    (self.plot_wy, all_wy),
                    (self.plot_z,  all_z),
                ]
                if extra_comp and extra_parts:
                    all_eq = np.concatenate(extra_parts)
                    eq_y = np.abs(all_eq) if extra_abs else all_eq
                    pairs.append((self.plot_extra, eq_y))
                for pw, y_data in pairs:
                    curve = pw.plot(x, y_data, pen=pen)
                    curve.setCurveClickable(True, width=8)
                    curve.sigClicked.connect(self._on_curve_clicked)
                    self._rally_curves.setdefault(rally_idx, []).append(curve)
                    self._curve_to_rally[id(curve)] = rally_idx
                    # pyqtgraph's signal-to-signal forwarding passes
                    # the inner PlotCurveItem, so register that too.
                    if hasattr(curve, 'curve'):
                        self._curve_to_rally[id(curve.curve)] = rally_idx

            total_rallies += 1

        # ── X-axis tick labels with player info ──────────────────────
        def _player_tag(pset):
            if pset == {1}: return "R"
            if pset == {2}: return "P"
            if pset == {1, 2}: return "R/P"
            return "?"

        ticks = [[(i + 0.5, f"{i}\n({_player_tag(shot_player_map.get(i, set()))})")
                  for i in range(x_min, x_max)]]
        for pw in active_plots:
            pw.getAxis('bottom').setTicks(ticks)

        # ── Set x-range ──────────────────────────────────────────────
        for pw in active_plots:
            pw.setXRange(x_min - 0.05, x_max + 0.05, padding=0)

        # ── Set y-ranges to match the plotted data ─────────────────
        for pw in active_plots:
            pw.enableAutoRange(axis='y')
        color_mode = self.color_by_combo.currentText()
        if color_mode == "Point winner":
            legend_items = [
                (_COLOR_ROBOT_WON,      "Robot won"),
                (_COLOR_HUMAN_WON,      "Human won"),
                (_COLOR_WINNER_UNKNOWN, "Unknown"),
            ]
            parts = [
                f'<span style="color: rgb({r},{g},{b});">■</span> {label}'
                for (r, g, b), label in legend_items
            ]
        elif color_mode == "Date":
            parts = []
            for d in sorted(self._date_colors):
                r, g, b = self._date_colors[d]
                parts.append(f'<span style="color: rgb({r},{g},{b});">■</span> {d}')
        else:
            checked = self._get_checked_players()
            parts = []
            for name in sorted(self._player_colors):
                if name not in checked:
                    continue
                r, g, b = self._player_colors[name]
                parts.append(f'<span style="color: rgb({r},{g},{b});">■</span> {name}')
        self.legend_label.setText("  ".join(parts))

        vx_lo, vx_hi = self._get_vx_range()
        wy_lo, wy_hi = self._get_wy_range()
        anchor_mode = self.anchor_combo.currentText()
        if anchor_mode in ("Last Player Shot", "Last Robot Shot"):
            filter_info = (
                f"|vx| ∈ [{vx_lo:.0f}, {vx_hi:.0f}] m/s  •  "
                f"wy ∈ [{wy_lo:.0f}, {wy_hi:.0f}] rad/s  •  "
            )
        else:
            filter_info = ""
        # ── Per-filter causal analysis stats ─────────────────────
        n_elig = getattr(self, '_filter_n_eligible', 0)
        n_all  = total_rallies  # rallies that passed all filters

        if n_elig > 0 and anchor_mode in ("Last Player Shot", "Last Robot Shot"):
            pct_all = 100.0 * n_all / n_elig
            causal_info = f"All filters: {n_all}/{n_elig} rallies ({pct_all:.1f}%)  •  "
        else:
            causal_info = ""

        self.info_label.setText(
            f"{total_rallies} rallies  •  "
            f"anchor: {anchor_mode}  •  "
            f"{filter_info}"
            f"{causal_info}"
            f"aligned at shot 0  •  {total_pts:,} pts"
        )

        # ── Draw Sankey flow diagram ─────────────────────────────────
        if anchor_mode in ("Last Player Shot", "Last Robot Shot"):
            self._draw_sankey()

    # ------------------------------------------------------------------
    # Sankey flow diagram
    # ------------------------------------------------------------------
    def _draw_sankey(self):
        """Draw a Sankey-style flow diagram on ``self.plot_sankey``.

        Four columns: t=−2, t=−1, t=0 (filter in/out) and t=+1
        (rally winner: robot / human / unknown).  All eligible rallies
        are represented, not just those that pass the anchor filter.
        """
        pw = self.plot_sankey
        pw.clear()

        sd = self._sankey_data
        if sd is None:
            return
        flows = sd['flows']          # {(m2_in, m1_in, s0_in, winner): count}
        n_total = sd['n_total']
        if n_total == 0:
            return

        # ── Colours ──────────────────────────────────────────────────
        c_in  = (44, 160, 44)       # green
        c_out = (214, 39, 40)       # red
        c_robot   = (214, 39, 40)   # red    – robot won
        c_human   = (44, 160, 44)   # green  – human (player) won
        c_unknown = (127, 127, 127) # grey   – unknown

        gap    = 0.06   # gap between nodes
        node_w = 0.25   # half-width of node rectangles

        # ── Per-column totals ────────────────────────────────────────
        m2_in_n   = sum(v for (m2, m1, s0, w), v in flows.items() if m2)
        m2_out_n  = sum(v for (m2, m1, s0, w), v in flows.items() if not m2)
        m1_in_n   = sum(v for (m2, m1, s0, w), v in flows.items() if m1)
        m1_out_n  = sum(v for (m2, m1, s0, w), v in flows.items() if not m1)
        s0_in_n   = sum(v for (m2, m1, s0, w), v in flows.items() if s0)
        s0_out_n  = sum(v for (m2, m1, s0, w), v in flows.items() if not s0)
        w_robot_n   = sum(v for (m2, m1, s0, w), v in flows.items() if w == 'robot')
        w_human_n   = sum(v for (m2, m1, s0, w), v in flows.items() if w == 'human')
        w_unknown_n = sum(v for (m2, m1, s0, w), v in flows.items() if w == 'unknown')

        # ── Build binary (in/out) columns ────────────────────────────
        bin_cols = [
            (-1.5, m2_in_n, m2_out_n, "t=−2"),
            (-0.5, m1_in_n, m1_out_n, "t=−1"),
            ( 0.5, s0_in_n, s0_out_n, "t=0"),
        ]

        node_rects = {}  # (col_idx, node_key) → (x_centre, y_bot, y_top)

        for ci, (cx, n_in, n_out, label) in enumerate(bin_cols):
            total = n_in + n_out
            f_in  = n_in / total if total else 0.5
            f_out = n_out / total if total else 0.5
            y_out_top = f_out * (1 - gap)
            y_in_bot  = y_out_top + gap
            y_in_top  = y_in_bot + f_in * (1 - gap)
            node_rects[(ci, 'in')]  = (cx, y_in_bot, y_in_top)
            node_rects[(ci, 'out')] = (cx, 0.0, y_out_top)

            for key, colour in [('in', c_in), ('out', c_out)]:
                _, yb, yt = node_rects[(ci, key)]
                if yt - yb < 0.001:
                    continue
                r = pg.QtWidgets.QGraphicsRectItem(
                    cx - node_w, yb, node_w * 2, yt - yb)
                r.setBrush(QBrush(QColor(*colour, 200)))
                r.setPen(pg.mkPen(QColor(255, 255, 255, 120), width=1))
                r.setZValue(10)
                pw.addItem(r)

            for key, colour, count in [('in', c_in, n_in), ('out', c_out, n_out)]:
                _, yb, yt = node_rects[(ci, key)]
                if count == 0:
                    continue
                pct = 100.0 * count / n_total
                txt = pg.TextItem(
                    f"{count} ({pct:.0f}%)",
                    color=(255, 255, 255), anchor=(0.5, 0.5),
                )
                txt.setPos(cx, (yb + yt) / 2)
                txt.setZValue(20)
                pw.addItem(txt)

            hdr = pg.TextItem(label, color=(200, 200, 200), anchor=(0.5, 0))
            hdr.setPos(cx, 1.05)
            hdr.setZValue(20)
            pw.addItem(hdr)

        # ── Build winner column (col index 3) ────────────────────────
        ci_w = 3
        cx_w = 1.5
        w_counts = [('robot', w_robot_n, c_robot),
                    ('unknown', w_unknown_n, c_unknown),
                    ('human', w_human_n, c_human)]
        # Layout: stack bottom→top: human, unknown, robot (robot on top)
        w_total = w_robot_n + w_human_n + w_unknown_n
        usable = 1.0 - gap * (len([c for _, c, _ in w_counts if c]) - 1)
        if usable < 0.3:
            usable = 0.3
        # order bottom→top: human, unknown, robot
        w_order = [('human', w_human_n, c_human),
                   ('unknown', w_unknown_n, c_unknown),
                   ('robot', w_robot_n, c_robot)]
        y_cur = 0.0
        n_drawn = 0
        for wk, wn, wc in w_order:
            if wn == 0:
                node_rects[(ci_w, wk)] = (cx_w, y_cur, y_cur)
                continue
            if n_drawn > 0:
                y_cur += gap
            frac = wn / w_total if w_total else 0.0
            h = frac * usable
            node_rects[(ci_w, wk)] = (cx_w, y_cur, y_cur + h)
            r = pg.QtWidgets.QGraphicsRectItem(
                cx_w - node_w, y_cur, node_w * 2, h)
            r.setBrush(QBrush(QColor(*wc, 200)))
            r.setPen(pg.mkPen(QColor(255, 255, 255, 120), width=1))
            r.setZValue(10)
            pw.addItem(r)
            pct = 100.0 * wn / n_total
            lbl = {'robot': 'Robot', 'human': 'Human'}.get(wk, '')
            txt = pg.TextItem(
                f"{lbl}\n{wn} ({pct:.0f}%)" if lbl else f"{wn} ({pct:.0f}%)",
                color=(255, 255, 255), anchor=(0.5, 0.5),
            )
            txt.setPos(cx_w, y_cur + h / 2)
            txt.setZValue(20)
            pw.addItem(txt)
            y_cur += h
            n_drawn += 1

        hdr = pg.TextItem("Winner", color=(200, 200, 200), anchor=(0.5, 0))
        hdr.setPos(cx_w, 1.05)
        hdr.setZValue(20)
        pw.addItem(hdr)

        # ── Generic node-key resolver for the first 3 columns ────────
        def _node_key_for_col(flow_key, col_idx):
            return 'in' if flow_key[col_idx] else 'out'

        def _node_key_for_winner(flow_key):
            return flow_key[3]  # 'robot'/'human'/'unknown'

        # ── Draw flow bands ──────────────────────────────────────────
        def _draw_band(ci_from, key_from, ci_to, key_to, count, colour_rgba,
                       key_fn_from=None, key_fn_to=None):
            if count == 0:
                return
            _, yb_f, yt_f = node_rects[(ci_from, key_from)]
            _, yb_t, yt_t = node_rects[(ci_to, key_to)]
            h_f = yt_f - yb_f
            h_t = yt_t - yb_t
            kf_from = key_fn_from or (lambda k: _node_key_for_col(k, ci_from))
            kf_to   = key_fn_to   or (lambda k: _node_key_for_col(k, ci_to))
            total_from = sum(v for k, v in flows.items() if kf_from(k) == key_from)
            total_to   = sum(v for k, v in flows.items() if kf_to(k) == key_to)
            if total_from == 0 or total_to == 0:
                return
            band_h_f = h_f * count / total_from
            band_h_t = h_t * count / total_to

            y_off_f = _offsets.get((ci_from, key_from), 0.0)
            y_off_t = _offsets.get((ci_to, key_to), 0.0)

            x_from = node_rects[(ci_from, key_from)][0] + node_w
            x_to   = node_rects[(ci_to, key_to)][0] - node_w

            y_f_bot = yt_f - y_off_f - band_h_f
            y_f_top = yt_f - y_off_f
            y_t_bot = yt_t - y_off_t - band_h_t
            y_t_top = yt_t - y_off_t

            fill = pg.FillBetweenItem(
                pg.PlotDataItem([x_from, x_to], [y_f_top, y_t_top]),
                pg.PlotDataItem([x_from, x_to], [y_f_bot, y_t_bot]),
                brush=QBrush(QColor(*colour_rgba)),
            )
            fill.setZValue(5)
            pw.addItem(fill)

            _offsets[(ci_from, key_from)] = y_off_f + band_h_f
            _offsets[(ci_to, key_to)]     = y_off_t + band_h_t

        def _band_colour(from_in: bool, to_in: bool):
            if from_in and to_in:
                return (*c_in, 80)
            elif not from_in and not to_in:
                return (*c_out, 80)
            return (255, 165, 0, 80)  # orange for mixed

        _offsets: Dict = {}

        # ── Bands: column 0 (t=−2) → column 1 (t=−1) ────────────────
        m2_m1_agg: Dict[Tuple[bool, bool], int] = {}
        for (m2, m1, s0, w), cnt in flows.items():
            m2_m1_agg[(m2, m1)] = m2_m1_agg.get((m2, m1), 0) + cnt

        for (m2, m1), cnt in sorted(m2_m1_agg.items(),
                                     key=lambda kv: (not kv[0][0], not kv[0][1])):
            _draw_band(0, 'in' if m2 else 'out',
                       1, 'in' if m1 else 'out',
                       cnt, _band_colour(m2, m1))

        # ── Bands: column 1 (t=−1) → column 2 (t=0) ─────────────────
        _offsets[(1, 'in')]  = 0.0
        _offsets[(1, 'out')] = 0.0
        _offsets[(2, 'in')]  = 0.0
        _offsets[(2, 'out')] = 0.0

        m1_s0_agg: Dict[Tuple[bool, bool], int] = {}
        for (m2, m1, s0, w), cnt in flows.items():
            m1_s0_agg[(m1, s0)] = m1_s0_agg.get((m1, s0), 0) + cnt

        for (m1, s0), cnt in sorted(m1_s0_agg.items(),
                                     key=lambda kv: (not kv[0][0], not kv[0][1])):
            _draw_band(1, 'in' if m1 else 'out',
                       2, 'in' if s0 else 'out',
                       cnt, _band_colour(m1, s0))

        # ── Bands: column 2 (t=0) → column 3 (winner) ────────────────
        _offsets[(2, 'in')]  = 0.0
        _offsets[(2, 'out')] = 0.0
        for wk in ('human', 'unknown', 'robot'):
            _offsets[(ci_w, wk)] = 0.0

        def _winner_band_colour(s0_in: bool, wk: str):
            wc = {'robot': c_robot, 'human': c_human, 'unknown': c_unknown}[wk]
            return (*wc, 80)

        s0_w_agg: Dict[Tuple[bool, str], int] = {}
        for (m2, m1, s0, w), cnt in flows.items():
            s0_w_agg[(s0, w)] = s0_w_agg.get((s0, w), 0) + cnt

        for (s0, wk), cnt in sorted(
                s0_w_agg.items(),
                key=lambda kv: (not kv[0][0],
                                {'human': 0, 'unknown': 1, 'robot': 2}.get(kv[0][1], 3))):
            _draw_band(2, 'in' if s0 else 'out',
                       ci_w, wk,
                       cnt, _winner_band_colour(s0, wk),
                       key_fn_from=lambda k: _node_key_for_col(k, 2),
                       key_fn_to=lambda k: _node_key_for_winner(k))

        # ── Axes setup ───────────────────────────────────────────────
        x_min = -_MAX_LOOKBACK
        x_max = _MAX_LOOKAHEAD + 1
        pw.setXRange(x_min - 0.05, x_max + 0.05, padding=0)
        pw.setYRange(-0.15, 1.15, padding=0)
        pw.getAxis('bottom').setTicks([[]])

        # ── Win-rate annotation for "in" at t=0 ─────────────────────
        # Count wins among rallies whose anchor shot passes the filter.
        s0_in_robot   = sum(v for (m2, m1, s0, w), v in flows.items()
                            if s0 and w == 'robot')
        s0_in_human   = sum(v for (m2, m1, s0, w), v in flows.items()
                            if s0 and w == 'human')
        s0_in_unknown = sum(v for (m2, m1, s0, w), v in flows.items()
                            if s0 and w == 'unknown')
        s0_in_total = s0_in_robot + s0_in_human + s0_in_unknown
        if s0_in_total > 0:
            parts = []
            if s0_in_robot:
                parts.append(f"Robot {100*s0_in_robot/s0_in_total:.0f}%")
            if s0_in_human:
                parts.append(f"Human {100*s0_in_human/s0_in_total:.0f}%")
            if s0_in_unknown:
                parts.append(f"Unk {100*s0_in_unknown/s0_in_total:.0f}%")
            win_txt = pg.TextItem(
                f"Anchor-in win rate ({s0_in_total}):  "
                + "  /  ".join(parts),
                color=(220, 220, 180), anchor=(0.5, 0.5),
            )
            win_txt.setPos(0.0, -0.08)
            win_txt.setZValue(20)
            pw.addItem(win_txt)

    # ------------------------------------------------------------------
    # Curve selection
    # ------------------------------------------------------------------
    def _on_curve_clicked(self, curve, ev=None):
        """Toggle highlight for the rally that owns *curve*."""
        rally_idx = self._curve_to_rally.get(id(curve))
        if rally_idx is None:
            return
        if self._selected_rally_idx == rally_idx:
            self._selected_rally_idx = None          # deselect
        else:
            self._selected_rally_idx = rally_idx     # select
        self._update_highlight()

    def _on_scene_clicked(self, ev):
        """Deselect when the user clicks on empty space."""
        if ev.button() != Qt.LeftButton:
            return
        # If a curve already accepted this event, do nothing
        if ev.isAccepted():
            return
        if self._selected_rally_idx is not None:
            self._selected_rally_idx = None
            self._update_highlight()

    def _update_highlight(self):
        """Adjust pen of every curve based on the current selection."""
        sel = self._selected_rally_idx
        for r_idx, curves in self._rally_curves.items():
            # Determine the colour for this rally
            rally, _, player_name = self._filtered_rallies[r_idx]
            color_rgb = self._rally_color(rally, player_name, rally_idx=r_idx)

            if sel is None:
                # No selection → default appearance
                alpha = _LINE_ALPHA
                width = _LINE_WIDTH
            elif r_idx == sel:
                alpha = _LINE_ALPHA_HIGHLIGHT
                width = _LINE_WIDTH_HIGHLIGHT
            else:
                alpha = _LINE_ALPHA_DIMMED
                width = _LINE_WIDTH

            pen = pg.mkPen(color=(*color_rgb, alpha), width=width)
            for c in curves:
                c.setPen(pen)

        # Show / hide detail label
        if sel is not None and sel < len(self._rally_metadata):
            m = self._rally_metadata[sel]
            pw_str = {
                'player1': 'Robot', 'player2': 'Human',
            }.get(m['point_winner'] or '', 'unknown')
            self.detail_label.setText(
                f"📅 {m['date']}  •  "
                f"👤 {m['player']}  •  "
                f"🎮 Game {m['game_id']} ({m['game_name']})  •  "
                f"🏓 Rally {m['rally_id']}  •  "
                f"{m['num_shots']} shots  •  "
                f"anchor shot {m['anchor_shot']}  •  "
                f"point winner: {pw_str}"
            )
            self.detail_label.setVisible(True)
        else:
            self.detail_label.setVisible(False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _save_plot(self):
        """Export the three sub-plots as a single PNG via matplotlib."""
        if not self._filtered_rallies:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.gridspec import GridSpec
            from matplotlib.patches import Rectangle as MplRect

            src = self.source_combo.currentText()
            pri, fb = ("opt", "gt200") if src.startswith("OPT") else ("gt200", None)

            x_min = -_MAX_LOOKBACK
            x_max = _MAX_LOOKAHEAD + 1
            extra_comp, extra_abs = self._get_extra_config()
            show_extra = extra_comp is not None
            if show_extra:
                extra_label = self.extra_combo.currentText()
                _, _, extra_ylabel, extra_units, _ = _EXTRA_QUANTITIES[
                    self.extra_combo.currentIndex()]

            n_data_rows = 4 if show_extra else 3
            gs = GridSpec(n_data_rows + 1, 1,
                         height_ratios=[2] * n_data_rows + [1.2], hspace=0.35)
            fig = plt.figure(figsize=(10, 13 if show_extra else 11))
            axes = [fig.add_subplot(gs[0])]
            for _gi in range(1, n_data_rows):
                axes.append(fig.add_subplot(gs[_gi], sharex=axes[0]))
            ax_sankey = fig.add_subplot(gs[n_data_rows])

            # Alternating column shading
            for i in range(x_min, x_max):
                abs_i = i - x_min
                c = '#222240' if (abs_i % 2) else '#1a1a30'
                for ax in axes:
                    ax.axvspan(i, i + 1, color=c, zorder=0)
            # Highlight anchor
            for ax in axes:
                ax.axvspan(0, 1, color='#2a2a20', zorder=0)

            # Shot boundary lines
            for i in range(x_min, x_max + 1):
                for ax in axes:
                    ax.axvline(i, color='white', lw=1.0, alpha=0.6)

            # Player colour map (normalised) – used in "Player name" mode
            mpl_colors_player = {
                name: tuple(c / 255.0 for c in rgb)
                for name, rgb in self._player_colors.items()
            }

            def _mpl_rally_color(rally, player_name, rally_idx=None):
                rgb = self._rally_color(rally, player_name, rally_idx=rally_idx)
                return tuple(c / 255.0 for c in rgb)

            # Draw
            save_player_map: dict = {}
            for _fi, (rally, anchor_idx, player_name) in enumerate(self._filtered_rallies):
                c_rgb = _mpl_rally_color(rally, player_name, rally_idx=_fi)
                first_shot = max(0, anchor_idx - _MAX_LOOKBACK)
                last_shot = min(len(rally.shots) - 1, anchor_idx + _MAX_LOOKAHEAD)
                for abs_shot_idx in range(first_shot, last_shot + 1):
                    shot = rally.shots[abs_shot_idx]
                    rel = abs_shot_idx - anchor_idx
                    _p = getattr(shot, 'player', None)
                    if _p is not None:
                        save_player_map.setdefault(rel, set()).add(_p)
                    segs = [
                        fs for fs in shot.flight_segments
                        if getattr(getattr(fs, 'trigger_event', None), 'type', None) != 'START'
                    ]
                    if not segs:
                        continue
                    t_all, vx_all, wy_all, z_all, eq_all = [], [], [], [], []
                    for fs in segs:
                        df = fs.data
                        if df is None or len(df) == 0:
                            continue
                        t = df['time'].values
                        vx = self._col(df, 'vx', pri, fb)
                        wy = self._col(df, 'wy', pri, fb)
                        z  = self._col(df, 'z',  pri, fb)
                        eq = self._col(df, extra_comp, pri, fb) if extra_comp else None
                        if vx is None and wy is None and z is None and eq is None:
                            continue
                        n = len(t)
                        t_all.append(t)
                        vx_all.append(vx if vx is not None else np.full(n, np.nan))
                        wy_all.append(wy if wy is not None else np.full(n, np.nan))
                        z_all.append(z  if z  is not None else np.full(n, np.nan))
                        if extra_comp:
                            eq_all.append(eq if eq is not None else np.full(n, np.nan))
                    if not t_all:
                        continue
                    at = np.concatenate(t_all)
                    t0, t1 = at.min(), at.max()
                    nt = (at - t0) / (t1 - t0) if t1 > t0 else np.zeros_like(at)
                    x = nt + rel
                    axes[0].plot(x, np.abs(np.concatenate(vx_all)), color=(*c_rgb, 0.35), lw=0.8)
                    axes[1].plot(x, np.concatenate(wy_all), color=(*c_rgb, 0.35), lw=0.8)
                    axes[2].plot(x, np.concatenate(z_all),  color=(*c_rgb, 0.35), lw=0.8)
                    if show_extra and eq_all:
                        eq_concat = np.concatenate(eq_all)
                        axes[3].plot(x, np.abs(eq_concat) if extra_abs else eq_concat,
                                    color=(*c_rgb, 0.35), lw=0.8)

            # ── Sankey subplot ────────────────────────────────────
            sd = self._sankey_data
            if sd is not None and sd['n_total'] > 0:
                flows_sk = sd['flows']
                n_total_sk = sd['n_total']
                c_in_m  = (44 / 255, 160 / 255, 44 / 255)
                c_out_m = (214 / 255, 39 / 255, 40 / 255)
                c_robot_m   = (214 / 255, 39 / 255, 40 / 255)
                c_human_m   = (44 / 255, 160 / 255, 44 / 255)
                c_unknown_m = (127 / 255, 127 / 255, 127 / 255)
                sk_gap = 0.06
                sk_node_w = 0.25

                m2_in_n  = sum(v for (m2, m1, s0, w), v in flows_sk.items() if m2)
                m2_out_n = sum(v for (m2, m1, s0, w), v in flows_sk.items() if not m2)
                m1_in_n  = sum(v for (m2, m1, s0, w), v in flows_sk.items() if m1)
                m1_out_n = sum(v for (m2, m1, s0, w), v in flows_sk.items() if not m1)
                s0_in_n  = sum(v for (m2, m1, s0, w), v in flows_sk.items() if s0)
                s0_out_n = sum(v for (m2, m1, s0, w), v in flows_sk.items() if not s0)
                w_robot_n   = sum(v for (m2, m1, s0, w), v in flows_sk.items() if w == 'robot')
                w_human_n   = sum(v for (m2, m1, s0, w), v in flows_sk.items() if w == 'human')
                w_unknown_n = sum(v for (m2, m1, s0, w), v in flows_sk.items() if w == 'unknown')

                bin_cols_sk = [
                    (-1.5, m2_in_n, m2_out_n, "t=\u22122"),
                    (-0.5, m1_in_n, m1_out_n, "t=\u22121"),
                    (0.5, s0_in_n, s0_out_n, "t=0"),
                ]

                node_rects_sk: dict = {}
                for ci, (cx, n_in, n_out, col_label) in enumerate(bin_cols_sk):
                    col_total = n_in + n_out
                    f_in  = n_in / col_total if col_total else 0.5
                    f_out = n_out / col_total if col_total else 0.5
                    y_out_top = f_out * (1 - sk_gap)
                    y_in_bot = y_out_top + sk_gap
                    y_in_top = y_in_bot + f_in * (1 - sk_gap)
                    node_rects_sk[(ci, 'in')] = (cx, y_in_bot, y_in_top)
                    node_rects_sk[(ci, 'out')] = (cx, 0.0, y_out_top)

                    for key, colour, count in [('in', c_in_m, n_in), ('out', c_out_m, n_out)]:
                        _, yb, yt = node_rects_sk[(ci, key)]
                        if yt - yb < 0.001:
                            continue
                        rect = MplRect(
                            (cx - sk_node_w, yb), sk_node_w * 2, yt - yb,
                            facecolor=(*colour, 0.78), edgecolor='white',
                            linewidth=0.8, zorder=10)
                        ax_sankey.add_patch(rect)
                        if count > 0:
                            pct = 100.0 * count / n_total_sk
                            ax_sankey.text(
                                cx, (yb + yt) / 2,
                                f"{count} ({pct:.0f}%)",
                                ha='center', va='center',
                                color='white', fontsize=8, zorder=20)
                    ax_sankey.text(
                        cx, 1.07, col_label, ha='center', va='bottom',
                        color='#c8c8c8', fontsize=9, zorder=20)

                # ── Winner column (index 3) ───────────────────────────
                ci_w_sk = 3
                cx_w_sk = 1.5
                w_total = w_robot_n + w_human_n + w_unknown_n
                usable_sk = 1.0 - sk_gap * 2  # max 3 nodes → 2 gaps
                if usable_sk < 0.3:
                    usable_sk = 0.3
                w_order_sk = [('robot', w_robot_n, c_robot_m),
                              ('unknown', w_unknown_n, c_unknown_m),
                              ('human', w_human_n, c_human_m)]
                y_cur_sk = 0.0
                n_drawn_sk = 0
                for wk, wn, wc in w_order_sk:
                    if wn == 0:
                        node_rects_sk[(ci_w_sk, wk)] = (cx_w_sk, y_cur_sk, y_cur_sk)
                        continue
                    if n_drawn_sk > 0:
                        y_cur_sk += sk_gap
                    frac = wn / w_total if w_total else 0.0
                    h = frac * usable_sk
                    node_rects_sk[(ci_w_sk, wk)] = (cx_w_sk, y_cur_sk, y_cur_sk + h)
                    rect = MplRect(
                        (cx_w_sk - sk_node_w, y_cur_sk), sk_node_w * 2, h,
                        facecolor=(*wc, 0.78), edgecolor='white',
                        linewidth=0.8, zorder=10)
                    ax_sankey.add_patch(rect)
                    pct = 100.0 * wn / n_total_sk
                    lbl = {'robot': 'Robot', 'human': 'Human'}.get(wk, '')
                    ax_sankey.text(
                        cx_w_sk, y_cur_sk + h / 2,
                        f"{lbl}\n{wn} ({pct:.0f}%)" if lbl else f"{wn} ({pct:.0f}%)",
                        ha='center', va='center',
                        color='white', fontsize=8, zorder=20)
                    y_cur_sk += h
                    n_drawn_sk += 1
                ax_sankey.text(
                    cx_w_sk, 1.07, 'Winner', ha='center', va='bottom',
                    color='#c8c8c8', fontsize=9, zorder=20)

                # Flow band helpers
                def _nk_sk(flow_key, col_idx):
                    return 'in' if flow_key[col_idx] else 'out'

                def _nk_w_sk(flow_key):
                    return flow_key[3]

                def _bc_sk(from_in, to_in):
                    if from_in and to_in:
                        return (*c_in_m, 0.30)
                    if not from_in and not to_in:
                        return (*c_out_m, 0.30)
                    return (1.0, 165 / 255, 0.0, 0.30)

                _off_sk: dict = {}

                def _draw_band_mpl(ci_f, kf, ci_t, kt, cnt, clr,
                                   kfn_from=None, kfn_to=None):
                    if cnt == 0:
                        return
                    _, yb_f, yt_f = node_rects_sk[(ci_f, kf)]
                    _, yb_t, yt_t = node_rects_sk[(ci_t, kt)]
                    h_f = yt_f - yb_f
                    h_t = yt_t - yb_t
                    _kf = kfn_from or (lambda k: _nk_sk(k, ci_f))
                    _kt = kfn_to   or (lambda k: _nk_sk(k, ci_t))
                    tot_f = sum(v for k, v in flows_sk.items() if _kf(k) == kf)
                    tot_t = sum(v for k, v in flows_sk.items() if _kt(k) == kt)
                    if tot_f == 0 or tot_t == 0:
                        return
                    bh_f = h_f * cnt / tot_f
                    bh_t = h_t * cnt / tot_t
                    yo_f = _off_sk.get((ci_f, kf), 0.0)
                    yo_t = _off_sk.get((ci_t, kt), 0.0)
                    xf = node_rects_sk[(ci_f, kf)][0] + sk_node_w
                    xt = node_rects_sk[(ci_t, kt)][0] - sk_node_w
                    yf_bot = yt_f - yo_f - bh_f
                    yf_top = yt_f - yo_f
                    yt_bot = yt_t - yo_t - bh_t
                    yt_top = yt_t - yo_t
                    ax_sankey.fill_between(
                        [xf, xt], [yf_bot, yt_bot], [yf_top, yt_top],
                        color=clr, zorder=5)
                    _off_sk[(ci_f, kf)] = yo_f + bh_f
                    _off_sk[(ci_t, kt)] = yo_t + bh_t

                # Bands: col 0 (t=-2) → col 1 (t=-1)
                m2_m1 = {}
                for (m2, m1, s0, w), cnt in flows_sk.items():
                    m2_m1[(m2, m1)] = m2_m1.get((m2, m1), 0) + cnt
                for (m2, m1), cnt in sorted(
                        m2_m1.items(), key=lambda kv: (not kv[0][0], not kv[0][1])):
                    _draw_band_mpl(0, 'in' if m2 else 'out',
                                   1, 'in' if m1 else 'out',
                                   cnt, _bc_sk(m2, m1))

                # Bands: col 1 (t=-1) → col 2 (t=0)
                _off_sk[(1, 'in')] = 0.0
                _off_sk[(1, 'out')] = 0.0
                _off_sk[(2, 'in')] = 0.0
                _off_sk[(2, 'out')] = 0.0
                m1_s0 = {}
                for (m2, m1, s0, w), cnt in flows_sk.items():
                    m1_s0[(m1, s0)] = m1_s0.get((m1, s0), 0) + cnt
                for (m1, s0), cnt in sorted(
                        m1_s0.items(), key=lambda kv: (not kv[0][0], not kv[0][1])):
                    _draw_band_mpl(1, 'in' if m1 else 'out',
                                   2, 'in' if s0 else 'out',
                                   cnt, _bc_sk(m1, s0))

                # Bands: col 2 (t=0) → col 3 (winner)
                _off_sk[(2, 'in')] = 0.0
                _off_sk[(2, 'out')] = 0.0
                for wk_ in ('human', 'unknown', 'robot'):
                    _off_sk[(ci_w_sk, wk_)] = 0.0
                s0_w = {}
                for (m2, m1, s0, w), cnt in flows_sk.items():
                    s0_w[(s0, w)] = s0_w.get((s0, w), 0) + cnt
                for (s0, wk_), cnt in sorted(
                        s0_w.items(),
                        key=lambda kv: (not kv[0][0],
                                        {'human': 0, 'unknown': 1, 'robot': 2}.get(kv[0][1], 3))):
                    wc_ = {'robot': c_robot_m, 'human': c_human_m,
                           'unknown': c_unknown_m}[wk_]
                    _draw_band_mpl(
                        2, 'in' if s0 else 'out',
                        ci_w_sk, wk_, cnt, (*wc_, 0.30),
                        kfn_from=lambda k: _nk_sk(k, 2),
                        kfn_to=lambda k: _nk_w_sk(k))

                ax_sankey.set_xlim(x_min - 0.05, x_max + 0.05)
                ax_sankey.set_ylim(-0.15, 1.2)
                ax_sankey.set_title('Filter Flow (Sankey)', color='white', fontsize=10)

                # Win-rate annotation for "in" at t=0
                _s0r = sum(v for (m2, m1, s0, w), v in flows_sk.items()
                           if s0 and w == 'robot')
                _s0h = sum(v for (m2, m1, s0, w), v in flows_sk.items()
                           if s0 and w == 'human')
                _s0u = sum(v for (m2, m1, s0, w), v in flows_sk.items()
                           if s0 and w == 'unknown')
                _s0t = _s0r + _s0h + _s0u
                if _s0t > 0:
                    _wp = []
                    if _s0r:
                        _wp.append(f"Robot {100*_s0r/_s0t:.0f}%")
                    if _s0h:
                        _wp.append(f"Human {100*_s0h/_s0t:.0f}%")
                    if _s0u:
                        _wp.append(f"Unk {100*_s0u/_s0t:.0f}%")
                    ax_sankey.text(
                        0.0, -0.08,
                        f"Anchor-in win rate ({_s0t}):  "
                        + "  /  ".join(_wp),
                        ha='center', va='center',
                        color='#dddcb4', fontsize=8)
            else:
                ax_sankey.text(
                    0.5, 0.5, 'No Sankey data', ha='center', va='center',
                    color='gray', fontsize=10, transform=ax_sankey.transAxes)

            ax_sankey.set_facecolor('#16162a')
            ax_sankey.set_xticks([])
            ax_sankey.set_yticks([])
            for _sp in ax_sankey.spines.values():
                _sp.set_visible(False)

            # Style
            for ax in axes:
                ax.set_facecolor('#16162a')
                ax.tick_params(colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('white')
            fig.patch.set_facecolor('#16162a')

            axes[0].set_ylabel('|vx| (m/s)')
            axes[1].set_ylabel('wy (rad/s)')
            axes[2].set_ylabel('z (m)')
            if show_extra:
                axes[3].set_ylabel(f'{extra_ylabel} ({extra_units})')
            bottom_ax = axes[-1]  # last data axis gets x-tick labels
            bottom_ax.set_xlabel('Shot (relative to anchor)')
            def _ptag(pset):
                if pset == {1}: return "R"
                if pset == {2}: return "P"
                if pset == {1, 2}: return "R/P"
                return "?"
            ticks_x = list(range(x_min, x_max))
            bottom_ax.set_xticks([i + 0.5 for i in ticks_x])
            bottom_ax.set_xticklabels(
                [f"{i}\n({_ptag(save_player_map.get(i, set()))})" for i in ticks_x],
                color='white',
            )

            # Legend
            if self.color_by_combo.currentText() == "Point winner":
                legend_els = [
                    Line2D([0], [0], color=tuple(c / 255.0 for c in _COLOR_ROBOT_WON),
                           lw=2, label="Robot won"),
                    Line2D([0], [0], color=tuple(c / 255.0 for c in _COLOR_HUMAN_WON),
                           lw=2, label="Human won"),
                    Line2D([0], [0], color=tuple(c / 255.0 for c in _COLOR_WINNER_UNKNOWN),
                           lw=2, label="Unknown"),
                ]
            elif self.color_by_combo.currentText() == "Date":
                legend_els = [
                    Line2D([0], [0],
                           color=tuple(c / 255.0 for c in self._date_colors[d]),
                           lw=2, label=d)
                    for d in sorted(self._date_colors)
                ]
            else:
                legend_els = [
                    Line2D([0], [0], color=mpl_colors_player[n], lw=2, label=n)
                    for n in sorted(mpl_colors_player)
                ]
            axes[0].legend(handles=legend_els, loc='upper right', fontsize=9,
                           facecolor='#2a2a4a', edgecolor='white', labelcolor='white')

            vx_lo, vx_hi = self._get_vx_range()
            wy_lo, wy_hi = self._get_wy_range()
            anchor_mode = self.anchor_combo.currentText()
            if anchor_mode in ("Last Player Shot", "Last Robot Shot"):
                filter_str = f"|vx| ∈ [{vx_lo:.0f}, {vx_hi:.0f}],  wy ∈ [{wy_lo:.0f}, {wy_hi:.0f}],  "
            else:
                filter_str = ""
            fig.suptitle(
                f"Time Series — {len(self._filtered_rallies)} rallies  "
                f"({filter_str}anchor: {anchor_mode})",
                fontsize=11, color='white',
            )
            fig.tight_layout()

            save_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = save_dir / f"time_series_aligned_{timestamp}.png"
            fig.savefig(str(fname), dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            QMessageBox.information(self, "Saved", f"Plot saved to:\n{fname}")
        except Exception as e:
            import traceback; tb = traceback.format_exc()
            QMessageBox.warning(self, "Save Error", f"Failed to save plot:\n{e}\n\n{tb}")
