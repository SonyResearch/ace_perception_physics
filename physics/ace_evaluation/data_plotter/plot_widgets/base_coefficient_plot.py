# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Base Coefficient Plot Widget

Abstract base class containing shared functionality for coefficient plots.
DragCoefficientPlot and MagnusCoefficientPlot inherit from this class.
"""

from typing import List, Tuple
import numpy as np
import pathlib
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QSlider
from PySide6.QtCore import Qt

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter


# =============================================================================
# Helper functions for creating UI controls
# =============================================================================

def create_range_slider(label_text: str, min_val: int, max_val: int,
                        default_min: int, default_max: int,
                        tick_interval: int, on_changed_callback,
                        label_format: str = "{}",
                        scale_factor: float = 1.0,
                        min_label_width: int = 35) -> Tuple[QVBoxLayout, QSlider, QSlider, QLabel, QLabel]:
    """
    Create a min/max range slider pair with labels.

    Args:
        label_text: Header label for the slider group
        min_val: Minimum slider value
        max_val: Maximum slider value
        default_min: Default value for min slider
        default_max: Default value for max slider
        tick_interval: Tick interval for sliders
        on_changed_callback: Callback function for value changes
        label_format: Format string for labels (e.g., "{:.2f}")
        scale_factor: Factor to divide slider value by for display
        min_label_width: Minimum width for value labels

    Returns:
        Tuple of (layout, min_slider, max_slider, min_label, max_label)
    """
    range_layout = QVBoxLayout()
    range_label = QLabel(label_text)
    range_layout.addWidget(range_label)

    # Min slider
    min_layout = QHBoxLayout()
    min_layout.addWidget(QLabel("Min:"))
    min_slider = QSlider(Qt.Horizontal)
    min_slider.setMinimum(min_val)
    min_slider.setMaximum(max_val)
    min_slider.setValue(default_min)
    min_slider.setTickPosition(QSlider.TicksBelow)
    min_slider.setTickInterval(tick_interval)
    min_slider.valueChanged.connect(on_changed_callback)
    min_layout.addWidget(min_slider)
    min_label = QLabel(label_format.format(default_min / scale_factor if scale_factor != 1.0 else default_min))
    min_label.setMinimumWidth(min_label_width)
    min_layout.addWidget(min_label)
    range_layout.addLayout(min_layout)

    # Max slider
    max_layout = QHBoxLayout()
    max_layout.addWidget(QLabel("Max:"))
    max_slider = QSlider(Qt.Horizontal)
    max_slider.setMinimum(min_val)
    max_slider.setMaximum(max_val)
    max_slider.setValue(default_max)
    max_slider.setTickPosition(QSlider.TicksBelow)
    max_slider.setTickInterval(tick_interval)
    max_slider.valueChanged.connect(on_changed_callback)
    max_layout.addWidget(max_slider)
    max_label = QLabel(label_format.format(default_max / scale_factor if scale_factor != 1.0 else default_max))
    max_label.setMinimumWidth(min_label_width)
    max_layout.addWidget(max_label)
    range_layout.addLayout(max_layout)

    return range_layout, min_slider, max_slider, min_label, max_label


def create_combo_selector(label_text: str, items: List[str],
                          on_changed_callback, default_index: int = 0) -> Tuple[QVBoxLayout, QComboBox]:
    """
    Create a labeled combo box selector.

    Args:
        label_text: Label for the combo box
        items: List of items to add
        on_changed_callback: Callback for selection changes
        default_index: Default selected index

    Returns:
        Tuple of (layout, combo_box)
    """
    layout = QVBoxLayout()
    label = QLabel(label_text)
    combo = QComboBox()
    combo.addItems(items)
    combo.setCurrentIndex(default_index)
    combo.currentTextChanged.connect(on_changed_callback)
    layout.addWidget(label)
    layout.addWidget(combo)
    return layout, combo


def save_plot_as_image(plot_widget, output_path: str, publication_mode: bool = False,
                       width: int = 1920, hide_x_labels: bool = False):
    """
    Save a pyqtgraph PlotWidget as an image.

    Args:
        plot_widget: A pyqtgraph PlotWidget instance.
        output_path: Path to save the image.
        publication_mode: If True, render a publication-friendly copy
            (white background, larger axis labels/ticks, no title, no grid)
            and save as PDF instead of PNG.
        width: Image width in pixels.
        hide_x_labels: If True, suppress x-axis tick labels in the export.
    """
    from PySide6.QtGui import QPen, QImage, QPainter, QFont, QColor, QFontMetrics
    from PySide6.QtCore import QRectF, QRect

    plot_item = plot_widget.plotItem

    if publication_mode:
        # Switch extension to .pdf for publication output
        output_path = str(pathlib.Path(output_path).with_suffix('.pdf'))

    if not publication_mode:
        exporter = ImageExporter(plot_item)
        exporter.parameters()['width'] = width
        exporter.export(str(output_path))
        return

    # --- Publication mode: render to image without modifying the live widget ---
    # Save widget state we'll temporarily change, then restore after export
    left_axis = plot_item.getAxis('left')
    bottom_axis = plot_item.getAxis('bottom')
    view_range = plot_item.vb.viewRange()

    # Save original state
    orig_bg = plot_widget.backgroundBrush().color()
    orig_vb_bg_brush = plot_item.vb.background.brush() if plot_item.vb.background else None
    orig_grid_x = plot_item.ctrl.xGridCheck.isChecked()
    orig_grid_y = plot_item.ctrl.yGridCheck.isChecked()
    orig_title = plot_item.titleLabel.text
    orig_left_visible = left_axis.isVisible()
    orig_bottom_visible = bottom_axis.isVisible()

    try:
        # Temporarily: white bg, no grid, no title, hide axes (we draw our own)
        plot_widget.setBackground('w')
        plot_item.vb.setBackgroundColor('w')
        plot_item.showGrid(x=False, y=False)
        plot_item.setTitle('')
        left_axis.hide()
        bottom_axis.hide()

        # Export to QImage — NO processEvents!
        exporter = ImageExporter(plot_item)
        exporter.parameters()['width'] = width
        base_img = exporter.export(toBytes=True)  # returns QImage

        source_rect = exporter.getSourceRect()
    finally:
        # Restore everything immediately
        plot_widget.setBackground(orig_bg)
        if orig_vb_bg_brush is not None:
            plot_item.vb.background.setBrush(orig_vb_bg_brush)
        else:
            plot_item.vb.setBackgroundColor(None)
        plot_item.showGrid(x=orig_grid_x, y=orig_grid_y)
        plot_item.setTitle(orig_title)
        if orig_left_visible:
            left_axis.show()
        if orig_bottom_visible:
            bottom_axis.show()
        plot_widget.update()

    # 2) Compute scale and dimensions
    scale = width / source_rect.width()
    height = int(source_rect.height() * scale)

    # 3) Create a new white-background image with extra margins for large labels
    margin_left = 160   # space for y-axis label + tick labels
    margin_bottom = 100  # space for x-axis label + tick labels
    margin_top = 20
    margin_right = 30
    img_w = width + margin_left + margin_right
    img_h = height + margin_top + margin_bottom

    img = QImage(img_w, img_h, QImage.Format.Format_ARGB32)
    img.fill(QColor('white'))

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # 4) Draw the clean plot image (white bg, no grid, no axes) offset by margins
    painter.drawImage(margin_left, margin_top, base_img)

    # 5) Compute the plot data area bounds in the composited image
    vb_scene_rect = plot_item.vb.mapRectToScene(plot_item.vb.rect())
    plot_x = (vb_scene_rect.x() - source_rect.x()) * scale + margin_left
    plot_y = (vb_scene_rect.y() - source_rect.y()) * scale + margin_top
    plot_w = vb_scene_rect.width() * scale
    plot_h = vb_scene_rect.height() * scale

    # 6) Draw axis lines
    painter.setPen(pg.mkPen(color='k', width=2))
    # Left axis line
    painter.drawLine(int(plot_x), int(plot_y), int(plot_x), int(plot_y + plot_h))
    # Bottom axis line
    painter.drawLine(int(plot_x), int(plot_y + plot_h), int(plot_x + plot_w), int(plot_y + plot_h))

    # 7) Draw tick marks and labels — ONLY major ticks (first/largest spacing)
    tick_font = QFont('Arial', 22)
    label_font = QFont('Arial', 28)
    painter.setPen(pg.mkPen(color='k', width=1))
    painter.setFont(tick_font)
    fm = QFontMetrics(tick_font)

    # Y-axis ticks — only the major level
    y_range = view_range[1]
    y_ticks = left_axis.tickValues(y_range[0], y_range[1], plot_h)
    if y_ticks:
        # First entry has the largest spacing = major ticks
        major_spacing, major_tick_list = y_ticks[0]
        for val in major_tick_list:
            if val < y_range[0] - 1e-9 or val > y_range[1] + 1e-9:
                continue
            frac = (val - y_range[0]) / (y_range[1] - y_range[0])
            py = plot_y + plot_h - frac * plot_h
            # Tick mark
            painter.drawLine(int(plot_x - 6), int(py), int(plot_x), int(py))
            # Label
            text = f"{val:g}"
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            painter.drawText(int(plot_x - 12 - tw), int(py + th / 3), text)

    # X-axis ticks — only the major level (skip if hide_x_labels)
    if not hide_x_labels:
        x_range = view_range[0]
        x_ticks = bottom_axis.tickValues(x_range[0], x_range[1], plot_w)
        if x_ticks:
            major_spacing, major_tick_list = x_ticks[0]
            for val in major_tick_list:
                if val < x_range[0] - 1e-9 or val > x_range[1] + 1e-9:
                    continue
                frac = (val - x_range[0]) / (x_range[1] - x_range[0])
                px = plot_x + frac * plot_w
                # Tick mark
                painter.drawLine(int(px), int(plot_y + plot_h), int(px), int(plot_y + plot_h + 6))
                # Label
                text = f"{val:g}"
                tw = fm.horizontalAdvance(text)
                painter.drawText(int(px - tw / 2), int(plot_y + plot_h + 10 + fm.height()), text)

    # 8) Draw axis labels
    painter.setFont(label_font)
    painter.setPen(pg.mkPen(color='k', width=1))
    fm_label = QFontMetrics(label_font)

    # Y-axis label (rotated)
    y_label = left_axis.labelText or ""
    if left_axis.labelUnits:
        y_label += f" ({left_axis.labelUnits})"
    painter.save()
    painter.translate(fm_label.height() + 5, plot_y + plot_h / 2 + fm_label.horizontalAdvance(y_label) / 2)
    painter.rotate(-90)
    painter.drawText(0, 0, y_label)
    painter.restore()

    # X-axis label
    if not hide_x_labels:
        x_label = bottom_axis.labelText or ""
        if bottom_axis.labelUnits:
            x_label += f" ({bottom_axis.labelUnits})"
        tw = fm_label.horizontalAdvance(x_label)
        painter.drawText(int(plot_x + plot_w / 2 - tw / 2), int(img_h - 10), x_label)

    painter.end()

    # 9) Save — PDF for publication, PNG otherwise
    output_p = pathlib.Path(output_path)
    if output_p.suffix.lower() in ('.pdf',):
        # Already PDF path — write via QPdfWriter
        from PySide6.QtCore import QMarginsF, QSizeF
        from PySide6.QtGui import QPageLayout, QPageSize
        try:
            from PySide6.QtGui import QPdfWriter
        except ImportError:
            from PySide6.QtCore import QPdfWriter

        writer = QPdfWriter(str(output_p))
        page_size = QPageSize(QSizeF(img_w, img_h), QPageSize.Unit.Point)
        writer.setPageSize(page_size)
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        writer.setResolution(150)
        pdf_painter = QPainter(writer)
        sx = writer.width() / img_w
        sy = writer.height() / img_h
        pdf_painter.scale(sx, sy)
        pdf_painter.drawImage(0, 0, img)
        pdf_painter.end()
    else:
        img.save(str(output_path))
