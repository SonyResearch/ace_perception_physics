# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Table Tennis Evaluation App Package

Interactive visualization tool for table tennis match data.
"""

try:
    from .app import MainWindow, main
    from .data_processor import DataProcessor
    from .plot_widgets import FlightSegmentPlot, SpinSpeedScatterPlot
except ImportError:  # PySide6 / pyqtgraph not installed
    MainWindow = None  # type: ignore[assignment,misc]
    main = None  # type: ignore[assignment]
    DataProcessor = None  # type: ignore[assignment,misc]
    FlightSegmentPlot = None  # type: ignore[assignment,misc]
    SpinSpeedScatterPlot = None  # type: ignore[assignment,misc]

__all__ = ["MainWindow", "main", "DataProcessor", "FlightSegmentPlot", "SpinSpeedScatterPlot"]
