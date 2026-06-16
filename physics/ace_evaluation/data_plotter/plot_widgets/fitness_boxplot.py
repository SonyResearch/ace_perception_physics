# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Fitness Box Plot Widget

Shows 2x3 box plots comparing NakashimaITTF, NakashimaPaper and Residual model
fitness errors for all 6 post-contact quantities (vx, vy, vz, wx, wy, wz).

Fitness is defined as the per-component error magnitude:
  fitness_v*_eps = |v*_post_model - v*_post_data|
  fitness_w*_eps = |w*_post_model - w*_post_data|
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QCheckBox, QGridLayout, QSlider
)
from PySide6.QtCore import Qt, QTimer

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from .base_coefficient_plot import save_plot_as_image
from .table_contact_plot import (
    contact_model_residual_vectorized,
    contact_model_residual0805_vectorized,
    contact_model_pysr_vectorized,
    contact_model_0426_vectorized,
)


def _draw_box(pw, data: np.ndarray, x_center: float, bar_width: float,
              color, is_velocity: bool = True, show_labels: bool = True):
    """
    Draw a single box-and-whisker on a PlotWidget, matching the style of
    AeroSummaryErrorPlot (FillBetweenItem, mean diamond, jittered outliers,
    value labels).
    """
    if len(data) == 0:
        return

    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    # Whiskers at the most extreme data points within fences (standard convention)
    within = data[(data >= fence_low) & (data <= fence_high)]
    whisker_low = np.min(within) if len(within) > 0 else q1
    whisker_high = np.max(within) if len(within) > 0 else q3
    mean = np.mean(data)

    hw = bar_width / 2.0

    # Box outline
    box_x = [x_center - hw, x_center + hw, x_center + hw, x_center - hw, x_center - hw]
    box_y = [q1, q1, q3, q3, q1]
    pw.plot(box_x, box_y, pen=pg.mkPen(color=color, width=2), fillLevel=None)

    # Filled box (semi-transparent)
    brush_color = (*color, 100)
    fill_item = pg.FillBetweenItem(
        pg.PlotCurveItem([x_center - hw, x_center + hw], [q1, q1]),
        pg.PlotCurveItem([x_center - hw, x_center + hw], [q3, q3]),
        brush=pg.mkBrush(*brush_color)
    )
    pw.addItem(fill_item)

    # Median line
    pw.plot([x_center - hw, x_center + hw], [median, median],
            pen=pg.mkPen(color=color, width=3))

    # Whiskers
    pw.plot([x_center, x_center], [q1, whisker_low],
            pen=pg.mkPen(color=color, width=1))
    pw.plot([x_center, x_center], [q3, whisker_high],
            pen=pg.mkPen(color=color, width=1))
    pw.plot([x_center - hw / 2, x_center + hw / 2], [whisker_low, whisker_low],
            pen=pg.mkPen(color=color, width=2))
    pw.plot([x_center - hw / 2, x_center + hw / 2], [whisker_high, whisker_high],
            pen=pg.mkPen(color=color, width=2))

    # Mean diamond marker
    scatter = pg.ScatterPlotItem([x_center], [mean], symbol='d', size=10,
                                  pen=pg.mkPen(color='white', width=1),
                                  brush=pg.mkBrush(*color))
    pw.addItem(scatter)

    # Outliers (jittered, capped at 100)
    outliers = data[(data < whisker_low) | (data > whisker_high)]
    if len(outliers) > 0:
        if len(outliers) > 100:
            outliers = np.random.choice(outliers, 100, replace=False)
        x_jitter = np.random.uniform(x_center - hw / 2, x_center + hw / 2, len(outliers))
        pw.addItem(pg.ScatterPlotItem(
            x_jitter, outliers, symbol='o', size=3,
            pen=pg.mkPen(color=(*color, 150), width=1),
            brush=pg.mkBrush(*color, 80)
        ))

    # Quartile value labels to the right of the box
    if show_labels:
        text_x = x_center + hw + 0.05
        fmt = ".3f" if is_velocity else ".1f"
        for val in [q3, median, q1]:
            txt = pg.TextItem(f'{val:{fmt}}', color=color, anchor=(0, 0.5))
            txt.setPos(text_x, val)
            pw.addItem(txt)


def _draw_violin(pw, data: np.ndarray, x_center: float, bar_width: float,
                 color, is_velocity: bool = True, show_labels: bool = True):
    """
    Draw a single violin on a PlotWidget with inner quartile lines
    and a mean diamond.
    """
    if len(data) == 0:
        return

    from scipy.stats import gaussian_kde

    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    mean = np.mean(data)
    lo, hi = np.percentile(data, [2, 98])
    clipped = data[(data >= lo) & (data <= hi)]
    if len(clipped) < 2:
        clipped = data  # not enough inliers, use all

    hw = bar_width / 2.0

    # KDE on inlier data only
    try:
        kde = gaussian_kde(clipped, bw_method='scott')
    except (np.linalg.LinAlgError, ValueError):
        # Fall back to box if KDE fails (e.g. constant data)
        _draw_box(pw, data, x_center, bar_width, color, is_velocity, show_labels)
        return

    y_min, y_max = np.min(clipped), np.max(clipped)
    y_pad = (y_max - y_min) * 0.05 or 0.1
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 200)
    density = kde(y_grid)
    # Normalise so max width = bar_width
    max_d = np.max(density) or 1.0
    density_scaled = density / max_d * hw

    # Draw filled violin (mirrored)
    x_left = x_center - density_scaled
    x_right = x_center + density_scaled

    brush_color = (*color, 80)
    fill_item = pg.FillBetweenItem(
        pg.PlotCurveItem(x_left, y_grid, pen=pg.mkPen(color=color, width=1.5)),
        pg.PlotCurveItem(x_right, y_grid, pen=pg.mkPen(color=color, width=1.5)),
        brush=pg.mkBrush(*brush_color),
    )
    pw.addItem(fill_item)

    # Inner quartile lines (Q1, median, Q3)
    for val, width in [(q1, 1), (median, 2.5), (q3, 1)]:
        d_at_val = float(kde(val)[0])
        d_scaled = d_at_val / max_d * hw
        pw.plot([x_center - d_scaled, x_center + d_scaled], [val, val],
                pen=pg.mkPen(color=color, width=width))

    # Mean diamond marker
    scatter = pg.ScatterPlotItem([x_center], [mean], symbol='d', size=10,
                                  pen=pg.mkPen(color='white', width=1),
                                  brush=pg.mkBrush(*color))
    pw.addItem(scatter)

    # Quartile value labels
    if show_labels:
        text_x = x_center + hw + 0.05
        fmt = ".3f" if is_velocity else ".1f"
        for val in [q3, median, q1]:
            txt = pg.TextItem(f'{val:{fmt}}', color=color, anchor=(0, 0.5))
            txt.setPos(text_x, val)
            pw.addItem(txt)


def _export_publication_cell(pw, cell_w: int, cell_h: int,
                             hide_x_labels: bool = False):
    """Render a single PlotWidget as a publication-quality QImage.

    White background, larger tick/axis fonts, black axes, no grid.
    Font sizes are tuned for single-column publication readability.

    Parameters
    ----------
    hide_x_labels : bool
        If True, suppress x-axis tick labels in the exported image.
    """
    from PySide6.QtGui import QImage, QPainter, QColor, QFont, QFontMetrics, QPen
    from PySide6.QtCore import QRectF

    plot_item = pw.plotItem
    left_axis = plot_item.getAxis('left')
    bottom_axis = plot_item.getAxis('bottom')
    view_range = plot_item.vb.viewRange()

    # Save original state
    orig_bg = pw.backgroundBrush().color()
    orig_vb_bg = plot_item.vb.background.brush() if plot_item.vb.background else None
    orig_grid_x = plot_item.ctrl.xGridCheck.isChecked()
    orig_grid_y = plot_item.ctrl.yGridCheck.isChecked()
    orig_title_text = plot_item.titleLabel.text
    orig_left_vis = left_axis.isVisible()
    orig_bottom_vis = bottom_axis.isVisible()

    try:
        pw.setBackground('w')
        plot_item.vb.setBackgroundColor('w')
        plot_item.showGrid(x=False, y=False)
        plot_item.setTitle('')          # hide title so it doesn't bleed through
        left_axis.hide()
        bottom_axis.hide()

        exporter = ImageExporter(plot_item)
        exporter.parameters()['width'] = cell_w
        base_img = exporter.export(toBytes=True)
        source_rect = exporter.getSourceRect()
    finally:
        pw.setBackground(orig_bg)
        if orig_vb_bg is not None:
            plot_item.vb.background.setBrush(orig_vb_bg)
        else:
            plot_item.vb.setBackgroundColor(None)
        plot_item.showGrid(x=orig_grid_x, y=orig_grid_y)
        plot_item.setTitle(orig_title_text)   # restore title
        if orig_left_vis:
            left_axis.show()
        if orig_bottom_vis:
            bottom_axis.show()
        pw.update()

    scale = cell_w / source_rect.width()
    base_h = int(source_rect.height() * scale)

    margin_left = 80
    margin_bottom = 60
    margin_top = 35
    margin_right = 10
    img = QImage(cell_w, cell_h, QImage.Format.Format_ARGB32)
    img.fill(QColor('white'))

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Compute available area and offsets to center the base image
    avail_w = cell_w - margin_left - margin_right
    avail_h = cell_h - margin_top - margin_bottom
    draw_scale = min(avail_w / base_img.width(), avail_h / base_img.height())
    draw_w = int(base_img.width() * draw_scale)
    draw_h = int(base_img.height() * draw_scale)
    draw_x = margin_left
    draw_y = margin_top

    from PySide6.QtCore import QRect
    painter.drawImage(QRect(draw_x, draw_y, draw_w, draw_h), base_img)

    # Plot data area in composited image
    vb_scene_rect = plot_item.vb.mapRectToScene(plot_item.vb.rect())
    plot_x = (vb_scene_rect.x() - source_rect.x()) / source_rect.width() * draw_w + draw_x
    plot_y_top = (vb_scene_rect.y() - source_rect.y()) / source_rect.height() * draw_h + draw_y
    plot_w = vb_scene_rect.width() / source_rect.width() * draw_w
    plot_h = vb_scene_rect.height() / source_rect.height() * draw_h

    # y = 0 reference line (gray) if zero is within view range
    y_range = view_range[1]
    if y_range[0] < 0 < y_range[1]:
        frac_zero = (0 - y_range[0]) / (y_range[1] - y_range[0])
        py_zero = plot_y_top + plot_h - frac_zero * plot_h
        painter.setPen(QPen(QColor(160, 160, 160), 1))
        painter.drawLine(int(plot_x), int(py_zero), int(plot_x + plot_w), int(py_zero))

    # Axis lines
    painter.setPen(QPen(QColor('black'), 2))
    painter.drawLine(int(plot_x), int(plot_y_top), int(plot_x), int(plot_y_top + plot_h))
    painter.drawLine(int(plot_x), int(plot_y_top + plot_h), int(plot_x + plot_w), int(plot_y_top + plot_h))

    # Publication-sized fonts for single-column readability
    tick_font = QFont('Arial', 18)
    label_font = QFont('Arial', 20)
    title_font = QFont('Arial', 20, QFont.Bold)
    painter.setFont(tick_font)
    painter.setPen(QColor('black'))
    fm = QFontMetrics(tick_font)

    # Y-axis ticks
    y_ticks = left_axis.tickValues(y_range[0], y_range[1], plot_h)
    if y_ticks:
        _, major_ticks = y_ticks[0]
        for val in major_ticks:
            if val < y_range[0] - 1e-9 or val > y_range[1] + 1e-9:
                continue
            frac = (val - y_range[0]) / (y_range[1] - y_range[0]) if y_range[1] != y_range[0] else 0
            py = plot_y_top + plot_h - frac * plot_h
            painter.drawLine(int(plot_x - 5), int(py), int(plot_x), int(py))
            text = f"{val:g}"
            tw = fm.horizontalAdvance(text)
            painter.drawText(int(plot_x - 8 - tw), int(py + fm.height() / 3), text)

    # X-axis ticks — skip entirely if hide_x_labels is set
    if not hide_x_labels:
        x_range = view_range[0]
        # Retrieve custom tick label mapping: {position: label_text}
        _custom_tick_map = {}
        _raw_ticks = getattr(bottom_axis, '_tickLevels', None)
        if _raw_ticks is not None:
            for level in _raw_ticks:
                for pos, label in level:
                    _custom_tick_map[pos] = label

        if _custom_tick_map:
            # Draw custom text labels at their positions
            for pos, label_text in _custom_tick_map.items():
                if pos < x_range[0] - 1e-9 or pos > x_range[1] + 1e-9:
                    continue
                frac = (pos - x_range[0]) / (x_range[1] - x_range[0]) if x_range[1] != x_range[0] else 0
                px = plot_x + frac * plot_w
                painter.drawLine(int(px), int(plot_y_top + plot_h), int(px), int(plot_y_top + plot_h + 5))
                # Handle multi-line labels (e.g. "Serve\nn=42")
                lines = label_text.split('\n')
                for li, line in enumerate(lines):
                    tw = fm.horizontalAdvance(line)
                    painter.drawText(
                        int(px - tw / 2),
                        int(plot_y_top + plot_h + 8 + fm.height() * (li + 1)),
                        line,
                    )
        else:
            # Fallback: numeric x-axis ticks
            x_tick_data = bottom_axis.tickValues(x_range[0], x_range[1], plot_w)
            if x_tick_data:
                _, major_ticks = x_tick_data[0]
                for val in major_ticks:
                    if val < x_range[0] - 1e-9 or val > x_range[1] + 1e-9:
                        continue
                    frac = (val - x_range[0]) / (x_range[1] - x_range[0]) if x_range[1] != x_range[0] else 0
                    px = plot_x + frac * plot_w
                    painter.drawLine(int(px), int(plot_y_top + plot_h), int(px), int(plot_y_top + plot_h + 5))
                    text = f"{val:g}"
                    tw = fm.horizontalAdvance(text)
                    painter.drawText(int(px - tw / 2), int(plot_y_top + plot_h + 8 + fm.height()), text)

    # Title
    title_text = orig_title_text
    if title_text:
        painter.setFont(title_font)
        fm_title = QFontMetrics(title_font)
        tw = fm_title.horizontalAdvance(title_text)
        painter.drawText(int(draw_x + draw_w / 2 - tw / 2), int(margin_top - 8), title_text)

    painter.end()
    return img


# Model definitions: label → (key_suffix, colour_rgb)
_TCM_MODEL_DEFS = {
    'NakashimaITTF':  ('ittf',          ( 52, 168,  83)),
    'NakashimaPaper': ('paper',         (255, 165,   0)),
    'Residual':       ('res',           ( 66, 133, 244)),
    'Residual0805':   ('res0805',       (200,  50, 200)),
    'PySR':           ('pysr',          (  0, 200, 200)),
    '0426':            ('0426',          (255, 127,  14)),
}


class FitnessBoxPlot(QWidget):
    """Widget displaying 2x3 box plots comparing table contact model
    fitness errors with toggleable model checkboxes.

    Row 1: vx, vy, vz fitness errors (velocity, m/s range)
    Row 2: wx, wy, wz fitness errors (spin, rad/s range)
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Fitness Error: Table Contact Models")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Control bar
        control_bar = QHBoxLayout()

        # Confidence filter
        self._conf_layout = QHBoxLayout()
        self._conf_layout.addWidget(QLabel("Min Confidence (%):"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self._refresh)
        self._conf_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("50")
        self.conf_label.setMinimumWidth(25)
        self._conf_layout.addWidget(self.conf_label)
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
        self.rmse_min_slider.valueChanged.connect(self._refresh)
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
        self.rmse_slider.valueChanged.connect(self._refresh)
        rmse_max_row.addWidget(self.rmse_slider)
        self.rmse_label = QLabel("100")
        self.rmse_label.setMinimumWidth(25)
        rmse_max_row.addWidget(self.rmse_label)
        self._rmse_layout.addLayout(rmse_max_row)
        control_bar.addLayout(self._rmse_layout)

        # Model check-boxes
        control_bar.addWidget(QLabel("  Models:"))
        self._model_cbs: Dict[str, QCheckBox] = {}
        _DEFAULT_CHECKED = {'Residual', 'Residual0805', 'PySR'}
        for label in _TCM_MODEL_DEFS:
            cb = QCheckBox(label)
            cb.setChecked(label in _DEFAULT_CHECKED)
            cb.stateChanged.connect(self._refresh)
            self._model_cbs[label] = cb
            control_bar.addWidget(cb)

        # Error type selector
        self._signed_cb = QCheckBox("Signed error")
        self._signed_cb.setChecked(False)
        self._signed_cb.stateChanged.connect(self._refresh)
        control_bar.addWidget(self._signed_cb)

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

        # Save button
        save_btn = QPushButton("Save Plot")
        save_btn.setMaximumWidth(100)
        save_btn.clicked.connect(self._save_plot)
        control_bar.addWidget(save_btn)

        layout.addLayout(control_bar)

        # 2x3 grid of plot widgets
        grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        grid_widget.setLayout(self.grid_layout)

        self.plot_widgets = {}
        titles = [
            ("vx error (m/s)", 0, 0), ("vy error (m/s)", 0, 1), ("vz error (m/s)", 0, 2),
            ("wx error (rad/s)", 1, 0), ("wy error (rad/s)", 1, 1), ("wz error (rad/s)", 1, 2),
        ]
        for title_text, row, col in titles:
            pw = pg.PlotWidget(title=title_text)
            pw.showGrid(x=True, y=True)
            pw.getAxis('left').enableAutoSIPrefix(False)
            pw.getAxis('bottom').enableAutoSIPrefix(False)
            self.plot_widgets[(row, col)] = pw
            self.grid_layout.addWidget(pw, row, col)

        layout.addWidget(grid_widget, stretch=1)

        # Stats label
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

        # Data storage
        self.contact_data = None
        self.base_folder = None
        self._legends: list = []

        # Debounce timer
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._do_update)

    def _checked_models(self):
        """Return list of (label, suffix, color) for checked models."""
        result = []
        for label, (suffix, color) in _TCM_MODEL_DEFS.items():
            cb = self._model_cbs.get(label)
            if cb is not None and cb.isChecked():
                result.append((label, suffix, color))
        return result

    def _refresh(self):
        self.conf_label.setText(str(self.conf_slider.value()))
        self.rmse_min_label.setText(str(self.rmse_min_slider.value()))
        self.rmse_label.setText(str(self.rmse_slider.value()))
        self._timer.start()

    def plot_data(self, contact_data: Dict[str, np.ndarray], base_folder: Optional[str] = None):
        """Load contact data and render box plots.

        Pre-computes both Nakashima and Residual model predictions and all 12
        fitness error arrays once, so that _do_update() only needs to apply
        mask filtering.
        """
        self.contact_data = contact_data
        self.base_folder = base_folder

        # ── Pre-compute model predictions and fitness errors once ──
        self._precomputed = None
        if contact_data is not None and len(contact_data.get('vx_pre', [])) > 0:
            vx_pre = contact_data['vx_pre']
            vy_pre = contact_data['vy_pre']
            vz_pre = contact_data['vz_pre']
            wx_pre = contact_data['wx_pre']
            wy_pre = contact_data['wy_pre']
            wz_pre = contact_data['wz_pre']
            vx_post = contact_data['vx_post']
            vy_post = contact_data['vy_post']
            vz_post = contact_data['vz_post']
            wx_post = contact_data['wx_post']
            wy_post = contact_data['wy_post']
            wz_post = contact_data['wz_post']

            epsilon = vz_pre * 0.02 + 0.98

            # Compute all analytical models once
            post_gt = np.array([vx_post, vy_post, vz_post, wx_post, wy_post, wz_post])

            def _fitness(model_post):
                return {c: model_post[i] - post_gt[i]
                        for i, c in enumerate(['vx', 'vy', 'vz', 'wx', 'wy', 'wz'])}

            vx_res, vy_res, vz_res, wx_res, wy_res, wz_res = contact_model_residual_vectorized(
                vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre, epsilon)
            vx_r05, vy_r05, vz_r05, wx_r05, wy_r05, wz_r05 = contact_model_residual0805_vectorized(
                vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre, epsilon)
            vx_psr, vy_psr, vz_psr, wx_psr, wy_psr, wz_psr = contact_model_pysr_vectorized(
                vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre, epsilon)
            vx_0426, vy_0426, vz_0426, wx_0426, wy_0426, wz_0426 = contact_model_0426_vectorized(
                vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre)

            self._precomputed = {
                'fitness_res':    _fitness(np.array([vx_res, vy_res, vz_res, wx_res, wy_res, wz_res])),
                'fitness_res0805': _fitness(np.array([vx_r05, vy_r05, vz_r05, wx_r05, wy_r05, wz_r05])),
                'fitness_pysr':   _fitness(np.array([vx_psr, vy_psr, vz_psr, wx_psr, wy_psr, wz_psr])),
                'fitness_0426':   _fitness(np.array([vx_0426, vy_0426, vz_0426, wx_0426, wy_0426, wz_0426])),
            }

            # HDF5-based model predictions (NakashimaITTF, NakashimaPaper)
            for model_suffix, model_key in [('ittf', 'fitness_ittf'), ('paper', 'fitness_paper')]:
                vx_m = contact_data.get(f'vx_post_{model_suffix}')
                if vx_m is not None and len(vx_m) == len(vx_post):
                    self._precomputed[model_key] = _fitness(np.array([
                        vx_m,
                        contact_data[f'vy_post_{model_suffix}'],
                        contact_data[f'vz_post_{model_suffix}'],
                        contact_data[f'wx_post_{model_suffix}'],
                        contact_data[f'wy_post_{model_suffix}'],
                        contact_data[f'wz_post_{model_suffix}'],
                    ]))

        self._do_update()

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

        # Clear all plots
        for pw in self.plot_widgets.values():
            pw.clear()

        if self.contact_data is None:
            self.info_label.setText("No data")
            return

        data = self.contact_data
        n_total = len(data['vx_pre'])
        if n_total == 0:
            self.info_label.setText("No contact data")
            return

        # Apply confidence filter
        conf_min = self.conf_slider.value() / 100.0
        confidence = np.array(data.get('confidence', np.ones(n_total)))
        mask = confidence >= conf_min
        if hasattr(self, 'conf_max_slider'):
            conf_max = self.conf_max_slider.value() / 100.0
            mask = mask & (confidence <= conf_max)

        # Apply OPT RMSE filter (slider in mm, data in meters)
        rmse_max = self.rmse_slider.value() / 1000.0
        rmse_min = self.rmse_min_slider.value() / 1000.0
        max_rmse_opt = np.array(data.get('max_rmse_opt', np.full(n_total, np.nan)))
        rmse_valid = ~np.isnan(max_rmse_opt)
        if self.rmse_slider.value() < self.rmse_slider.maximum():
            mask = mask & (rmse_valid & (max_rmse_opt <= rmse_max))
        if self.rmse_min_slider.value() > 0:
            mask = mask & (rmse_valid & (max_rmse_opt >= rmse_min))

        if not np.any(mask):
            self.info_label.setText("No data in selected confidence range")
            return

        n = int(np.sum(mask))
        use_signed = self._signed_cb.isChecked()
        _wrap = (lambda x: x) if use_signed else np.abs
        _pub = self.publication_checkbox.isChecked()

        # Determine which models are checked
        checked = self._checked_models()
        n_models = len(checked)
        if n_models == 0:
            self.info_label.setText(f"{n} contacts — no models selected")
            return

        # Build per-model fitness dicts from precomputed signed errors
        fitness_dicts = {}
        if self._precomputed is not None:
            suffix_to_key = {
                'ittf': 'fitness_ittf', 'paper': 'fitness_paper',
                'res': 'fitness_res', 'res0805': 'fitness_res0805',
                'pysr': 'fitness_pysr', '0426': 'fitness_0426',
            }
            for label, suffix, color in checked:
                pk = suffix_to_key.get(suffix)
                if pk and pk in self._precomputed:
                    raw = {k: v[mask] for k, v in self._precomputed[pk].items()}
                    wrapped = {k: _wrap(v) for k, v in raw.items()}
                    fitness_dicts[suffix] = wrapped

        bar_width = 0.6 / max(n_models, 1)

        grid_map = [
            ('vx', 0, 0), ('vy', 0, 1), ('vz', 0, 2),
            ('wx', 1, 0), ('wy', 1, 1), ('wz', 1, 2),
        ]

        def _whisker_ext(d):
            q1 = np.percentile(d, 25)
            q3 = np.percentile(d, 75)
            iqr = q3 - q1
            w = d[(d >= q1 - 1.5 * iqr) & (d <= q3 + 1.5 * iqr)]
            return (np.min(w) if len(w) > 0 else q1,
                    np.max(w) if len(w) > 0 else q3)

        stats_parts = []

        for key, row, col in grid_map:
            pw = self.plot_widgets[(row, col)]
            is_velocity = (row == 0)
            tick_labels = []
            y_max_all = 0.01
            y_min_all = 0.0

            for m_idx, (label, suffix, color) in enumerate(checked):
                fd = fitness_dicts.get(suffix)
                if fd is None:
                    continue
                model_data = fd.get(key)
                if model_data is None:
                    continue
                model_data = model_data[np.isfinite(model_data)]
                if len(model_data) == 0:
                    continue

                x_center = (m_idx - (n_models - 1) / 2.0) * bar_width * 1.3
                _draw_violin(pw, model_data, x_center, bar_width, color,
                             is_velocity=is_velocity, show_labels=not _pub)
                tick_labels.append((x_center, label[:12]))

                wlo, whi = _whisker_ext(model_data)
                y_max_all = max(y_max_all, whi)
                y_min_all = min(y_min_all, wlo)

                if m_idx == 0:
                    stats_parts.append(f"{key}={np.median(model_data):.3f}")

            pw.getPlotItem().getAxis('bottom').setTicks([tick_labels])
            half_span = n_models * bar_width * 0.8
            pw.setXRange(-half_span - 0.3, half_span + 0.3)
            margin = max(abs(y_max_all), abs(y_min_all)) * 0.1
            pw.setYRange(y_min_all - margin if use_signed else 0,
                         y_max_all + margin)

        # Optional legend
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

        err_str = "signed" if use_signed else "|error|"
        model_labels = ', '.join(l for l, _, _ in checked)
        self.info_label.setText(f"{n} contacts | Conf≥{int(conf_min*100)}% | OPT RMSE: [{self.rmse_min_slider.value()}, {self.rmse_slider.value()}]mm | {err_str} | Models: {model_labels}")
        self.stats_label.setText("Median (first model) — " + "  |  ".join(stats_parts))

    def _save_plot(self):
        """Save all plots as a single combined image (PDF in publication mode)."""
        if self.contact_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pub = self.publication_checkbox.isChecked()
        ext = "pdf" if pub else "png"
        filename = f"fitness_boxplot_{timestamp}.{ext}"
        output_path = plots_dir / filename

        try:
            from PySide6.QtGui import QImage, QPainter, QColor, QFont, QFontMetrics
            from PySide6.QtCore import QRect, QMarginsF

            # Grid dimensions from actual plot widgets
            max_row = max(r for r, c in self.plot_widgets.keys()) + 1
            max_col = max(c for r, c in self.plot_widgets.keys()) + 1
            n_rows, n_cols = max_row, max_col
            cell_w, cell_h = (800, 600) if pub else (640, 480)
            # Overlap cells to reduce inter-subplot spacing in publication mode
            gap_x = 70 if pub else 0
            gap_y = 60 if pub else 0
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

                for (row, col), pw in self.plot_widgets.items():
                    cell_img = _export_publication_cell(pw, cell_w, cell_h,
                                                       hide_x_labels=True)
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

                for (row, col), pw in self.plot_widgets.items():
                    exporter = ImageExporter(pw.plotItem)
                    exporter.parameters()['width'] = cell_w
                    cell_img = exporter.export(toBytes=True)
                    ox, oy = _cell_origin(row, col)
                    target = QRect(ox, oy, cell_w, cell_h)
                    painter.drawImage(target, cell_img)

                painter.end()
                combined.save(str(output_path))

            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
