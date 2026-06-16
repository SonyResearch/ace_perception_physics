# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Coefficient Plot Widgets - Backward Compatibility

This module re-exports DragCoefficientPlot and MagnusCoefficientPlot
for backward compatibility. The actual implementations are now in:
  - drag_coefficient_plot.py
  - magnus_coefficient_plot.py
"""

from .drag_coefficient_plot import DragCoefficientPlot
from .magnus_coefficient_plot import MagnusCoefficientPlot

__all__ = ['DragCoefficientPlot', 'MagnusCoefficientPlot']
