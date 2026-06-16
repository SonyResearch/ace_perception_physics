# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Plot Widgets Module

Contains plot widgets for visualizing table tennis data.
Split into multiple files for better maintainability.
"""

try:
    from .flight_segment_plot import FlightSegmentPlot
    from .scatter_plots import SpinSpeedScatterPlot
    from .histogram_plots import HistogramPlot, ConfidenceDistributionPlot, GCSSpinConfidenceHistogramPlot
    from .drag_coefficient_plot import DragCoefficientPlot
    from .magnus_coefficient_plot import MagnusCoefficientPlot
    from .rally_analysis_plot import RallyAnalysisPlot
    from .aero_summary_error_plot import AeroSummaryErrorPlot
    from .table_contact_plot import TableContactPlot
    from .confidence_vs_error_plot import ConfidenceVsErrorPlot
    from .epsilon_vs_vz_plot import EpsilonVsVzPlot
    from .fitness_boxplot import FitnessBoxPlot
    from .spin_observation_plot import SpinObservationPlot
    from .spin_conversion_plot import SpinConversionPlot
    from .time_series_plot import TimeSeriesPlot
    from .rcm_scatter_plot import RcmScatterPlot
    from .rcm_boxplot import RcmBoxPlot
    from .rcm_restitution_plot import RcmRestitutionPlot
    from .net_contact_plot import NetContactPlot
    from .rcm_contact_location_plot import RcmContactLocationPlot
    from .rcm_output_vs_others_plot import RcmOutputVsOthersPlot
    from .rcm_histogram_plot import RcmHistogramPlot
except ImportError:  # PySide6 / pyqtgraph not installed
    pass

__all__ = [
    'FlightSegmentPlot',
    'SpinSpeedScatterPlot',
    'HistogramPlot',
    'ConfidenceDistributionPlot',
    'GCSSpinConfidenceHistogramPlot',
    'DragCoefficientPlot',
    'MagnusCoefficientPlot',
    'RallyAnalysisPlot',
    'AeroSummaryErrorPlot',
    'TableContactPlot',
    'ConfidenceVsErrorPlot',
    'EpsilonVsVzPlot',
    'FitnessBoxPlot',
    'SpinObservationPlot',
    'SpinConversionPlot',
    'TimeSeriesPlot',
    'RcmScatterPlot',
    'RcmBoxPlot',
    'RcmRestitutionPlot',
    'NetContactPlot',
    'RcmContactLocationPlot',
    'RcmOutputVsOthersPlot',
    'RcmHistogramPlot',
]
