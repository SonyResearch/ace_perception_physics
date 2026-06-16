#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Table Tennis Evaluation App

A PySide6 application with PyQtGraph for interactive visualization of table tennis match data.
Features:
- Load HDF5 files containing match data
- Interactive scatter plot of spin magnitude vs speed magnitude
- Click on points to view the corresponding flight segment trajectory

HOW TO ADD NEW PLOTS TO THE LEFT PANEL:
==========================================

1. Create a new plot widget class in plot_widgets.py:
   - Inherit from QWidget
   - Implement __init__ to set up the UI (controls, plot widgets, etc.)
   - Add a plot_data() or update() method to handle data updates
   - Optionally emit flight_segment_selected signal if clicking should show details

2. In app.py __init__ method:
   a. Import your new plot widget class at the top
   b. Create an instance: self.my_new_plot = MyNewPlotWidget()
   c. Add it to the splitter: self.splitter.addWidget(self.my_new_plot)
   d. Hide it initially: self.my_new_plot.hide()
   e. Add the plot name to left_view_combo: self.left_view_combo.addItems(["Scatter Plot", "Histograms", "My New Plot"])

3. Update on_left_view_changed() method:
   - Add an elif branch for your new plot
   - Show your plot widget, hide others
   - Call your data update method if data is loaded

4. Create a data update method (e.g., update_my_new_plot):
   - Extract needed data from self.match_collection using self.data_processor
   - Call your plot widget's plot_data() method with the extracted data

5. Call your update method in:
   - auto_load_default_data() - to populate on startup
   - load_hdf5_files() - to populate after manual load
   - on_left_view_changed() - when switching to your view

Example skeleton:
-----------------
# In plot_widgets.py:
class MyNewPlotWidget(QWidget):
    flight_segment_selected = Signal(FlightSegment, dict, object, object, object, object)

    def __init__(self):
        super().__init__()
        # Set up UI, controls, plots

    def plot_data(self, data1, data2):
        # Update plots with data

# In app.py __init__:
self.my_new_plot = MyNewPlotWidget()
self.my_new_plot.hide()
self.splitter.addWidget(self.my_new_plot)
self.left_view_combo.addItems(["Scatter Plot", "Histograms", "My New Plot"])

# In on_left_view_changed():
elif view_type == "My New Plot":
    self.scatter_plot.hide()
    self.histogram_plot.hide()
    self.my_new_plot.show()
    if self.match_collection is not None:
        self.update_my_new_plot()

# Add update method:
def update_my_new_plot(self):
    data = self.data_processor.extract_my_data(self.match_collection)
    self.my_new_plot.plot_data(data)
"""

# =============================================================================
# PERFORMANCE TODOs
# =============================================================================
# NOTE [PERF-RESOLVED] extract_drag/magnus_coefficient_data — cache paths
#   already added.  Both methods use FlightSegmentCache when available.
#
# NOTE [PERF-RESOLVED] time_series_plot._apply_filters — per-shot anchor
#   values (vx, wy) are now pre-computed once in _precompute_rally_data()
#   when the MatchCollection changes.  _apply_filters only scans lightweight
#   dicts + numpy arrays (no DataFrame access on the hot path).
#
# TODO [PERF-MED] drag_coefficient_plot / magnus_coefficient_plot build spots
#   and brushes in Python for-loops (same pattern already fixed in
#   spin_conversion_plot).  Vectorize with numpy RGBA arrays and pass arrays
#   directly to ScatterPlotItem.
#   (Files: drag_coefficient_plot.py, magnus_coefficient_plot.py,
#    base_coefficient_plot.py)
#
# TODO [PERF-MED] time_series_plot._replot creates ~12 PlotDataItem objects
#   per rally (3 panels × ~4 shots).  With 100+ filtered rallies that's 1200+
#   individual curve objects.  Consider downsampling, batch-plotting with
#   connect='finite', or limiting the maximum number of displayed rallies.
#   (Files: time_series_plot.py)
#
# TODO [PERF-LOW] Duration and x-travel are recomputed from raw DataFrames on
#   every slider change in the coefficient plots.  Pre-compute these once in
#   plot_data() and store alongside the other cached arrays.
#   (Files: drag_coefficient_plot.py, magnus_coefficient_plot.py,
#    base_coefficient_plot.py)
#
# NOTE [PERF-RESOLVED] "Parallel plot widget updates" (#3) — Investigation
#   showed that _refresh_current_view() only updates the ONE currently visible
#   widget, so QThreadPool parallelism across widgets has no benefit.  Tab
#   switches (on_left_view_changed) also update a single widget.  The real
#   bottleneck was rapid slider-drag signals, addressed by debounce timers
#   in _on_global_filter_changed, _on_date_filter_changed, and
#   on_match_selection_changed.
# =============================================================================

import os
import sys

# ============================================================================
# WSL2 OpenGL Fix - MUST be set before importing Qt/pyqtgraph
# ============================================================================
# On WSL2, the D3D12-based OpenGL translation can cause segfaults when
# initializing GLViewWidget. Force software rendering on WSL2 to fix this.
def _is_wsl_environment() -> bool:
    """Check if running in WSL environment."""
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            return 'microsoft' in version_info or 'wsl' in version_info
    except Exception:
        return False

if _is_wsl_environment():
    # Force software OpenGL rendering to avoid D3D12 segfaults
    # Both NVIDIA and Intel D3D12 paths can crash on WSL2
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    print("WSL2 detected - using software OpenGL rendering for stability")
# ============================================================================

import contextlib
import pathlib
import numpy as np
from typing import List, Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QSplitter,
    QMessageBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QSlider,
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal as QSignal, QObject

# Configure PyQtGraph for performance AFTER importing PySide6
import pyqtgraph as pg
pg.setConfigOptions(
    antialias=False,          # Disable antialiasing for speed
)

# Add the parent directory to the path to import ace_evaluation
eval_path = Path(__file__).parent.parent
if str(eval_path) not in sys.path:
    sys.path.insert(0, str(eval_path))

from ace_evaluation.utilities.data_classes import FlightSegment, MatchCollection

# Import local modules
from data_processor import DataProcessor
from plot_widgets import FlightSegmentPlot, SpinSpeedScatterPlot, HistogramPlot, RallyAnalysisPlot, DragCoefficientPlot, MagnusCoefficientPlot, ConfidenceDistributionPlot, AeroSummaryErrorPlot, TableContactPlot, GCSSpinConfidenceHistogramPlot, ConfidenceVsErrorPlot, EpsilonVsVzPlot, FitnessBoxPlot, SpinObservationPlot, SpinConversionPlot, TimeSeriesPlot, RcmScatterPlot, RcmBoxPlot, RcmRestitutionPlot, NetContactPlot, RcmContactLocationPlot, RcmOutputVsOthersPlot, RcmHistogramPlot


# ---------------------------------------------------------------------------
# Background data-loading worker
# ---------------------------------------------------------------------------
class _DataLoadWorker(QObject):
    """Loads HDF5 data in a background thread so the UI stays responsive."""
    finished = QSignal(object)   # emits MatchCollection on success
    error = QSignal(str)         # emits error message on failure
    progress = QSignal(str)      # emits status text updates

    def __init__(self, data_processor: DataProcessor, data_folder: pathlib.Path, gcs_fallback: bool = False, clear_cache: bool = False):
        super().__init__()
        self._dp = data_processor
        self._folder = data_folder
        self._gcs_fallback = gcs_fallback
        self._clear_cache = clear_cache

    def run(self):
        try:
            if self._clear_cache:
                cache_file = self._folder / ".data_plotter_cache.pkl"
                if cache_file.exists():
                    size_mb = cache_file.stat().st_size / 1e6
                    cache_file.unlink()
                    print(f"Cleared cache: {cache_file} ({size_mb:.1f} MB)")
                    self.progress.emit("Cache cleared — reloading from HDF5...")
            self.progress.emit(f"Loading files from {self._folder}...")
            mc = self._dp.load_data(self._folder, gcs_fallback=self._gcs_fallback)
            self.finished.emit(mc)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Move back to the application thread while we are still on
            # the worker thread (moveToThread requires the calling thread
            # to be the object's current thread).
            app = QApplication.instance()
            if app is not None:
                self.moveToThread(app.thread())


class _JsonLoadWorker(QObject):
    """Loads specific rallies from a JSON manifest in a background thread."""
    finished = QSignal(object)   # emits MatchCollection on success
    error = QSignal(str)         # emits error message on failure
    progress = QSignal(str)      # emits status text updates

    def __init__(self, data_processor: DataProcessor, json_path: str, gcs_fallback: bool = False):
        super().__init__()
        self._dp = data_processor
        self._json_path = json_path
        self._gcs_fallback = gcs_fallback

    def run(self):
        try:
            self.progress.emit(f"Loading rallies from {self._json_path}...")
            mc = self._dp.load_data_from_json(self._json_path, gcs_fallback=self._gcs_fallback)
            self.finished.emit(mc)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            app = QApplication.instance()
            if app is not None:
                self.moveToThread(app.thread())


class MainWindow(QMainWindow):
    """Main application window"""

    # ── Lazy widget instantiation configuration ──────────────────────────
    # Only scatter_plot and flight_plot are created eagerly.  All other
    # plot widgets are created on first use to speed up startup.
    _LAZY_WIDGET_CONFIG = {
        "histogram_plot":           (HistogramPlot,                  1),
        "confidence_dist_plot":     (ConfidenceDistributionPlot,     2),
        "gcs_spin_conf_plot":       (GCSSpinConfidenceHistogramPlot, 3),
        "rally_analysis_plot":      (RallyAnalysisPlot,              4),
        "drag_coeff_plot":          (DragCoefficientPlot,            5),
        "magnus_coeff_plot":        (MagnusCoefficientPlot,          6),
        "aero_error_plot":          (AeroSummaryErrorPlot,           7),
        "conf_vs_error_plot":       (ConfidenceVsErrorPlot,          8),
        "table_contact_plot":       (TableContactPlot,               9),
        "epsilon_vs_vz_plot":       (EpsilonVsVzPlot,                10),
        "fitness_box_plot":         (FitnessBoxPlot,                 11),
        "spin_observation_plot":    (SpinObservationPlot,            12),
        "spin_conversion_plot":     (SpinConversionPlot,             13),
        "time_series_plot":         (TimeSeriesPlot,                 14),
        "rcm_scatter_plot":         (RcmScatterPlot,                 15),
        "rcm_boxplot":              (RcmBoxPlot,                     16),
        "rcm_restitution_plot":     (RcmRestitutionPlot,             17),
        "net_contact_plot":         (NetContactPlot,                 18),
        "rcm_contact_location_plot": (RcmContactLocationPlot,         19),
        "rcm_output_vs_others_plot": (RcmOutputVsOthersPlot,          20),
        "rcm_histogram_plot":        (RcmHistogramPlot,               21),
    }
    _VIEW_TO_ATTR = {
        "Aero - Spin vs Speed":           "scatter_plot",
        "Histograms":                      "histogram_plot",
        "Confidence Distribution":          "confidence_dist_plot",
        "GCS Spin Confidence":              "gcs_spin_conf_plot",
        "Rally Analysis":                   "rally_analysis_plot",
        "Aero - Drag Coefficient":          "drag_coeff_plot",
        "Aero - Magnus Coefficient":        "magnus_coeff_plot",
        "Aero - Error Box Plots":           "aero_error_plot",
        "Aero - Error Analysis":            "conf_vs_error_plot",
        "TCM - Input vs Output":            "table_contact_plot",
        "TCM - Epsilon vs Vz":              "epsilon_vs_vz_plot",
        "TCM - Fitness Box Plot":           "fitness_box_plot",
        "Game Analysis - Spin Observation": "spin_observation_plot",
        "RCM - Spin Conversion":            "spin_conversion_plot",
        "Game Analysis - Time Series":      "time_series_plot",
        "RCM - Input vs Output":            "rcm_scatter_plot",
        "RCM - Box Plots":                  "rcm_boxplot",
        "RCM - Restitution":                "rcm_restitution_plot",
        "Game Analysis - Net Contacts":     "net_contact_plot",
        "RCM - Contact Location":               "rcm_contact_location_plot",
        "RCM - Output vs Others":                "rcm_output_vs_others_plot",
        "RCM - Distributions":                   "rcm_histogram_plot",
    }
    _ALL_LEFT_PLOT_ATTRS = [
        "scatter_plot", "histogram_plot", "confidence_dist_plot",
        "gcs_spin_conf_plot", "rally_analysis_plot", "drag_coeff_plot",
        "magnus_coeff_plot", "aero_error_plot", "conf_vs_error_plot",
        "table_contact_plot", "epsilon_vs_vz_plot", "fitness_box_plot",
        "spin_observation_plot",
        "spin_conversion_plot", "time_series_plot",
        "rcm_scatter_plot", "rcm_boxplot", "rcm_restitution_plot",
        "net_contact_plot",
        "rcm_contact_location_plot",
        "rcm_output_vs_others_plot",
        "rcm_histogram_plot",
    ]
    _CONF_FILTER_ATTRS = [
        'scatter_plot', 'aero_error_plot', 'conf_vs_error_plot',
        'table_contact_plot', 'epsilon_vs_vz_plot', 'drag_coeff_plot',
        'magnus_coeff_plot', 'spin_observation_plot', 'spin_conversion_plot',
        'fitness_box_plot',
        'rcm_scatter_plot', 'rcm_boxplot', 'rcm_restitution_plot',
        'rcm_contact_location_plot',
        'rcm_output_vs_others_plot',
        'rcm_histogram_plot',
    ]
    _RMSE_FILTER_ATTRS = [
        'spin_observation_plot', 'spin_conversion_plot', 'fitness_box_plot',
        'conf_vs_error_plot',
        'drag_coeff_plot', 'magnus_coeff_plot', 'table_contact_plot',
        'epsilon_vs_vz_plot',
        'scatter_plot', 'aero_error_plot',
        'rcm_scatter_plot', 'rcm_boxplot', 'rcm_restitution_plot',
        'rcm_contact_location_plot',
        'rcm_output_vs_others_plot',
        'rcm_histogram_plot',
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Table Tennis Evaluation Viewer")
        self.showMaximized()  # Start in full screen (maximized window)

        # Create data processor
        self.data_processor = DataProcessor()
        self.match_collection: Optional[MatchCollection] = None
        self.rally_analysis_loaded = False  # Track if rally analysis has been computed

        # Hidden global fitness-max spinbox for RCM widgets (default 0.1)
        from PySide6.QtWidgets import QDoubleSpinBox
        self._rcm_fitness_max_spin = QDoubleSpinBox()
        self._rcm_fitness_max_spin.setRange(0.0, 100.0)
        self._rcm_fitness_max_spin.setValue(0.1)
        self._rcm_fitness_max_spin.setDecimals(2)
        self._rcm_fitness_max_spin.hide()

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Left column - Control panel
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(250)

        # Load button
        self.load_button = QPushButton("Load HDF5 File(s)")
        self.load_button.clicked.connect(self.load_hdf5_files)
        left_layout.addWidget(self.load_button)

        # Load from JSON button
        self.load_json_button = QPushButton("Load from JSON")
        self.load_json_button.setToolTip(
            "Load specific rallies from a JSON manifest file.\n"
            "The JSON should be a list of objects with keys:\n"
            "  hdf5_filepath, game_id, rally_id, sequence_number (optional)"
        )
        self.load_json_button.clicked.connect(self.load_from_json)
        left_layout.addWidget(self.load_json_button)

        # Save as JSON button
        self.save_json_button = QPushButton("Save as JSON")
        self.save_json_button.setToolTip(
            "Export the currently loaded rallies as a JSON manifest\n"
            "compatible with 'Load from JSON'."
        )
        self.save_json_button.setEnabled(False)
        self.save_json_button.clicked.connect(self.save_as_json)
        left_layout.addWidget(self.save_json_button)

        # Clear cache checkbox
        self.clear_cache_checkbox = QCheckBox("Clear cache on load")
        self.clear_cache_checkbox.setChecked(False)
        self.clear_cache_checkbox.setToolTip(
            "Delete the pickle cache before loading so data is re-read from HDF5.\n"
            "Use this after updating HDF5 files or when the cache may be stale."
        )
        left_layout.addWidget(self.clear_cache_checkbox)

        # Status label
        self.status_label = QLabel("No file loaded")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        # Separator
        separator = QLabel("─" * 30)
        left_layout.addWidget(separator)

        # Matches list
        matches_label = QLabel("Loaded Matches:")
        matches_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(matches_label)

        self.matches_list = QListWidget()
        self.matches_list.setMaximumHeight(350)
        self.matches_list.setStyleSheet("QListWidget { font-size: 10pt; }")
        self.matches_list.itemChanged.connect(self.on_match_selection_changed)
        left_layout.addWidget(self.matches_list)

        # Separator
        separator2 = QLabel("─" * 30)
        left_layout.addWidget(separator2)

        # Left panel view selector
        view_label = QLabel("Left Panel View:")
        left_layout.addWidget(view_label)

        self.left_view_combo = QComboBox()
        self.left_view_combo.addItems([
            "Aero - Spin vs Speed", "Aero - Drag Coefficient", "Aero - Magnus Coefficient", "Aero - Error Box Plots", "Aero - Error Analysis",
            "TCM - Input vs Output", "TCM - Epsilon vs Vz", "TCM - Fitness Box Plot",
            "RCM - Spin Conversion", "RCM - Input vs Output", "RCM - Box Plots", "RCM - Restitution", "RCM - Contact Location", "RCM - Output vs Others", "RCM - Distributions",
            "Game Analysis - Spin Observation", "Game Analysis - Time Series", "Game Analysis - Net Contacts",
            "Histograms", "Confidence Distribution", "GCS Spin Confidence", "Rally Analysis",
        ])
        self.left_view_combo.currentTextChanged.connect(self.on_left_view_changed)
        left_layout.addWidget(self.left_view_combo)

        # Checkbox to show/hide left analysis panel (scatter, drag, magnus, etc.)
        self.show_left_panel_checkbox = QCheckBox("Show Analysis Panel")
        self.show_left_panel_checkbox.setChecked(True)
        self.show_left_panel_checkbox.stateChanged.connect(self.on_left_panel_toggle)
        left_layout.addWidget(self.show_left_panel_checkbox)

        # Checkbox to show/hide right panel (flight segment plot)
        self.show_right_panel_checkbox = QCheckBox("Show Flight Segment Panel")
        self.show_right_panel_checkbox.setChecked(True)
        self.show_right_panel_checkbox.stateChanged.connect(self.on_right_panel_toggle)
        left_layout.addWidget(self.show_right_panel_checkbox)

        # ── Global Filters ──────────────────────────────────
        separator_filter = QLabel("─" * 30)
        left_layout.addWidget(separator_filter)

        filter_header = QLabel("Global Filters:")
        filter_header.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(filter_header)

        # Confidence range – min slider
        left_layout.addWidget(QLabel("Confidence (%):"))
        conf_min_row = QHBoxLayout()
        conf_min_row.addWidget(QLabel("Min:"))
        self.global_conf_min_slider = QSlider(Qt.Horizontal)
        self.global_conf_min_slider.setMinimum(0)
        self.global_conf_min_slider.setMaximum(100)
        self.global_conf_min_slider.setValue(50)
        self.global_conf_min_slider.setTickPosition(QSlider.TicksBelow)
        self.global_conf_min_slider.setTickInterval(10)
        self.global_conf_min_slider.valueChanged.connect(self._on_global_filter_changed)
        conf_min_row.addWidget(self.global_conf_min_slider)
        self.global_conf_min_label = QLabel("50")
        self.global_conf_min_label.setMinimumWidth(25)
        conf_min_row.addWidget(self.global_conf_min_label)
        left_layout.addLayout(conf_min_row)

        # Confidence range – max slider
        conf_max_row = QHBoxLayout()
        conf_max_row.addWidget(QLabel("Max:"))
        self.global_conf_max_slider = QSlider(Qt.Horizontal)
        self.global_conf_max_slider.setMinimum(0)
        self.global_conf_max_slider.setMaximum(100)
        self.global_conf_max_slider.setValue(100)
        self.global_conf_max_slider.setTickPosition(QSlider.TicksBelow)
        self.global_conf_max_slider.setTickInterval(10)
        self.global_conf_max_slider.valueChanged.connect(self._on_global_filter_changed)
        conf_max_row.addWidget(self.global_conf_max_slider)
        self.global_conf_max_label = QLabel("100")
        self.global_conf_max_label.setMinimumWidth(25)
        conf_max_row.addWidget(self.global_conf_max_label)
        left_layout.addLayout(conf_max_row)

        # OPT RMSE range
        left_layout.addWidget(QLabel("OPT RMSE (mm):"))
        rmse_min_row = QHBoxLayout()
        rmse_min_row.addWidget(QLabel("Min:"))
        self.global_rmse_min_slider = QSlider(Qt.Horizontal)
        self.global_rmse_min_slider.setMinimum(0)
        self.global_rmse_min_slider.setMaximum(100)
        self.global_rmse_min_slider.setValue(0)
        self.global_rmse_min_slider.setTickPosition(QSlider.TicksBelow)
        self.global_rmse_min_slider.setTickInterval(10)
        self.global_rmse_min_slider.valueChanged.connect(self._on_global_filter_changed)
        rmse_min_row.addWidget(self.global_rmse_min_slider)
        self.global_rmse_min_label = QLabel("0")
        self.global_rmse_min_label.setMinimumWidth(25)
        rmse_min_row.addWidget(self.global_rmse_min_label)
        left_layout.addLayout(rmse_min_row)

        rmse_max_row = QHBoxLayout()
        rmse_max_row.addWidget(QLabel("Max:"))
        self.global_rmse_max_slider = QSlider(Qt.Horizontal)
        self.global_rmse_max_slider.setMinimum(1)
        self.global_rmse_max_slider.setMaximum(100)
        self.global_rmse_max_slider.setValue(20)
        self.global_rmse_max_slider.setTickPosition(QSlider.TicksBelow)
        self.global_rmse_max_slider.setTickInterval(10)
        self.global_rmse_max_slider.valueChanged.connect(self._on_global_filter_changed)
        rmse_max_row.addWidget(self.global_rmse_max_slider)
        self.global_rmse_max_label = QLabel("20")
        self.global_rmse_max_label.setMinimumWidth(25)
        rmse_max_row.addWidget(self.global_rmse_max_label)
        left_layout.addLayout(rmse_max_row)

        # Date range filter
        left_layout.addWidget(QLabel("Date range:"))
        date_from_row = QHBoxLayout()
        date_from_row.addWidget(QLabel("From:"))
        self.global_date_from_combo = QComboBox()
        self.global_date_from_combo.addItem("(all)")
        self.global_date_from_combo.currentIndexChanged.connect(self._on_date_filter_changed)
        date_from_row.addWidget(self.global_date_from_combo)
        left_layout.addLayout(date_from_row)

        date_to_row = QHBoxLayout()
        date_to_row.addWidget(QLabel("To:"))
        self.global_date_to_combo = QComboBox()
        self.global_date_to_combo.addItem("(all)")
        self.global_date_to_combo.currentIndexChanged.connect(self._on_date_filter_changed)
        date_to_row.addWidget(self.global_date_to_combo)
        left_layout.addLayout(date_to_row)

        # H5 version filter
        left_layout.addWidget(QLabel("H5 Version:"))
        self.global_version_combo = QComboBox()
        self.global_version_combo.addItem("(all)")
        self.global_version_combo.setToolTip(
            "Filter matches by file/version_data_processing.\n"
            "Only matches processed with the selected version are shown."
        )
        self.global_version_combo.currentIndexChanged.connect(self._on_version_filter_changed)
        left_layout.addWidget(self.global_version_combo)

        # Add stretch to push controls to top
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # Right side - Plots area with splitter
        self.splitter = QSplitter(Qt.Horizontal)

        # Left plot: Only scatter plot is created eagerly (default view).
        # All other plot widgets are created lazily on first use via _ensure_widget().
        self.scatter_plot = SpinSpeedScatterPlot()
        self.splitter.addWidget(self.scatter_plot)

        # Lazy widget slots — initialised to None; lightweight placeholders
        # hold their positions in the splitter so indices remain stable.
        for _attr_name in self._LAZY_WIDGET_CONFIG:
            setattr(self, _attr_name, None)
        for _attr, (_cls, _idx) in self._LAZY_WIDGET_CONFIG.items():
            _ph = QWidget()
            _ph.hide()
            self.splitter.addWidget(_ph)

        # Right plot: Flight segment plot
        self.flight_plot = FlightSegmentPlot()
        self.flight_plot.hide()  # Hide initially until a point is clicked
        self.splitter.addWidget(self.flight_plot)

        # Set stretch factors - all left plots have same stretch, flight plot has same stretch
        # Index: 0=scatter, 1=histogram, 2=confidence, 3=gcs_spin_conf, 4=rally, 5=drag, 6=magnus, 7=aero_error, 8=conf_vs_error, 9=table_contact, 10=epsilon_vs_vz, 11=fitness_box, 12=spin_obs, 13=spin_conv, 14=time_series, 15=rcm_scatter, 16=rcm_box, 17=rcm_restitution, 18=net_contact, 19=rcm_contact_loc, 20=rcm_output_vs_others, 21=rcm_histogram, 22=flight
        for i in range(23):
            self.splitter.setStretchFactor(i, 1)
            # Allow collapsing to 0 for hidden widgets
            self.splitter.setCollapsible(i, True)

        # Set initial sizes for all 23 widgets (only scatter visible initially)
        self.splitter.setSizes([500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        main_layout.addWidget(self.splitter, stretch=1)

        # Connect signals for eagerly-created widgets only.
        # Lazy widgets get their signals connected in _setup_lazy_widget().
        self.scatter_plot.flight_segment_selected.connect(self.on_flight_segment_selected)

        # ── Per-widget slider aliases and filter hiding are set up lazily
        #    in _setup_lazy_widget() when each widget is first created. ──

        # Filter widget lists — only eagerly-created widgets initially;
        # rebuilt by _rebuild_filter_widget_lists() when lazy widgets are created.
        self._conf_filter_widgets = [self.scatter_plot]
        self._rmse_filter_widgets = [self.scatter_plot]
        self._hide_per_widget_filters()

        # Store current view
        self.current_left_view = "Aero - Spin vs Speed"

        # ------------------------------------------------------------------
        # Extract-data cache: avoids re-extracting from HDF5 on view switches
        # when the selection (enabled_indices) hasn't changed.
        # Keys: view name str  →  (enabled_indices_key, extracted_data)
        # Invalidated on data reload, date-filter change, or match-checkbox change.
        # ------------------------------------------------------------------
        self._extract_cache: dict = {}
        self._cache_indices_key = None  # hashable snapshot of current enabled_indices

        # ------------------------------------------------------------------
        # Debounce timers – coalesce rapid slider / filter changes so that
        # expensive plot refreshes only happen once the user pauses briefly.
        # ------------------------------------------------------------------
        self._filter_debounce = QTimer(self)
        self._filter_debounce.setSingleShot(True)
        self._filter_debounce.setInterval(120)  # ms
        self._filter_debounce.timeout.connect(self._refresh_current_view)

        self._date_debounce = QTimer(self)
        self._date_debounce.setSingleShot(True)
        self._date_debounce.setInterval(200)  # ms
        self._date_debounce.timeout.connect(self._apply_date_filter)

        self._selection_debounce = QTimer(self)
        self._selection_debounce.setSingleShot(True)
        self._selection_debounce.setInterval(200)  # ms
        self._selection_debounce.timeout.connect(self._apply_selection_change)

        # Auto-load default data path on startup
        self.auto_load_default_data()

    def auto_load_default_data(self):
        """Automatically load data from default path on startup (background thread)."""
        #default_path = pathlib.Path("/home/cconti/SonyAI/Data/Nature_data/tyo01/")
        default_path = pathlib.Path("/home/christian/SonyAI/Data/hdf5_122025/tyo01/20251201/ace_vs_yoshimura_match_0")

        if default_path.exists() and default_path.is_dir():
            self.status_label.setText(f"Loading files from {default_path}...")
            self.load_button.setEnabled(False)

            # Create worker + thread
            self._load_thread = QThread()
            self._load_worker = _DataLoadWorker(
                self.data_processor, default_path,
                gcs_fallback=getattr(self, '_gcs_fallback', False),
                clear_cache=self.clear_cache_checkbox.isChecked(),
            )
            self._load_worker.moveToThread(self._load_thread)

            # Wire signals
            self._load_thread.started.connect(self._load_worker.run)
            self._load_worker.progress.connect(self.status_label.setText)
            self._load_worker.finished.connect(self._on_data_loaded)
            self._load_worker.error.connect(self._on_data_load_error)
            # Clean up thread when done — quit first, then move worker back
            self._load_worker.finished.connect(self._load_thread.quit)
            self._load_worker.error.connect(self._load_thread.quit)
            self._load_thread.finished.connect(self._cleanup_load_thread)

            self._load_thread.start()
        else:
            self.status_label.setText(f"Default path not found: {default_path}")

    # ------------------------------------------------------------------
    # Progress / busy-cursor helper
    # ------------------------------------------------------------------

    def _cleanup_load_thread(self):
        """Move the worker back to the main thread and schedule cleanup.

        Without this, Python's GC may destroy the worker while it still
        belongs to the (now-stopped) QThread, which triggers
        ``QObject::setParent: Cannot set parent, new parent is in a
        different thread`` followed by a segfault.
        """
        if hasattr(self, '_load_worker') and self._load_worker is not None:
            self._load_worker.deleteLater()
            self._load_worker = None
        if hasattr(self, '_load_thread') and self._load_thread is not None:
            self._load_thread.deleteLater()
            self._load_thread = None

    @contextlib.contextmanager
    def _busy_status(self, message: str):
        """Context manager: show *message* in the status bar with a wait cursor.

        The previous status text is restored when the block exits (even on
        exception).  A single ``processEvents`` call is made on entry so that
        the label update is flushed to the screen before the heavy work starts.
        """
        prev_text = self.status_label.text()
        self.status_label.setText(message)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()          # flush label + cursor change
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()
            self.status_label.setText(prev_text)

    # ------------------------------------------------------------------
    # Global filter helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hide_layout_contents(layout):
        """Recursively hide all widgets inside a QLayout."""
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().hide()
            if item.layout():
                MainWindow._hide_layout_contents(item.layout())

    def _hide_per_widget_filters(self):
        """Hide the confidence / RMSE filter controls that live inside each plot widget."""
        for w in self._conf_filter_widgets:
            # Widgets that stored their filter layout as self._conf_layout
            layout = getattr(w, '_conf_layout', None)
            if layout is not None:
                self._hide_layout_contents(layout)
            # scatter_plots uses a QGroupBox
            group = getattr(w, '_confidence_group', None)
            if group is not None:
                group.hide()

        for w in self._rmse_filter_widgets:
            layout = getattr(w, '_rmse_layout', None)
            if layout is not None:
                self._hide_layout_contents(layout)
            # Also hide _error_layout used by drag/magnus/table_contact/epsilon_vs_vz
            error_layout = getattr(w, '_error_layout', None)
            if error_layout is not None:
                self._hide_layout_contents(error_layout)

    # ------------------------------------------------------------------
    # Lazy widget instantiation helpers
    # ------------------------------------------------------------------

    def _ensure_widget(self, attr_name: str):
        """Lazily create a plot widget on first access.

        If the widget has already been created this is a no-op.  Otherwise the
        widget is instantiated, swapped into the splitter (replacing its
        placeholder), and wired up with signals / slider aliases / global
        filter values.
        """
        if getattr(self, attr_name, None) is not None:
            return  # Already created

        config = self._LAZY_WIDGET_CONFIG.get(attr_name)
        if config is None:
            return  # Not a lazy widget (e.g. scatter_plot, flight_plot)

        cls, splitter_idx = config
        widget = cls()
        widget.hide()

        # Swap the placeholder in the splitter
        old = self.splitter.replaceWidget(splitter_idx, widget)
        if old is not None:
            old.deleteLater()
        self.splitter.setStretchFactor(splitter_idx, 1)
        self.splitter.setCollapsible(splitter_idx, True)

        setattr(self, attr_name, widget)
        self._setup_lazy_widget(attr_name, widget)

    def _setup_lazy_widget(self, attr_name: str, widget):
        """Wire up a lazily-created widget: signals, slider aliases, filters, plots_folder."""
        # ── Signal connections ──
        if hasattr(widget, 'flight_segment_selected'):
            widget.flight_segment_selected.connect(self.on_flight_segment_selected)
        if hasattr(widget, 'contact_selected'):
            widget.contact_selected.connect(self.on_contact_selected)

        # ── Widget-specific slider aliases ──
        if attr_name == "fitness_box_plot":
            widget.conf_min_slider = widget.conf_slider
            _fb_max = QSlider()
            _fb_max.setMinimum(0)
            _fb_max.setMaximum(100)
            _fb_max.setValue(100)
            widget.conf_max_slider = _fb_max
        elif attr_name == "drag_coeff_plot":
            widget.rmse_min_slider = widget.error_min_slider
            widget.rmse_slider = widget.error_max_slider
        elif attr_name == "magnus_coeff_plot":
            widget.rmse_min_slider = widget.error_min_slider
            widget.rmse_slider = widget.error_max_slider
        elif attr_name == "table_contact_plot":
            widget.rmse_min_slider = widget.error_min_slider
            widget.rmse_slider = widget.error_max_slider
            # Re-extract data when the user toggles "Use extracted TCM" checkbox
            widget.data_source_changed.connect(self._on_tcm_data_source_changed)
        elif attr_name == "epsilon_vs_vz_plot":
            widget.rmse_min_slider = widget.opt_err_min_slider
            widget.rmse_slider = widget.opt_err_max_slider
        elif attr_name in ("rcm_scatter_plot", "rcm_boxplot", "rcm_restitution_plot", "rcm_output_vs_others_plot", "rcm_histogram_plot", "rcm_contact_location_plot"):
            # RCM widgets have no native confidence/RMSE sliders;
            # create hidden proxy sliders so the global filter can push
            # values that _do_update reads.
            for _slider_attr in ('conf_min_slider', 'conf_max_slider',
                                 'rmse_min_slider', 'rmse_slider'):
                _s = QSlider()
                _s.setMinimum(0)
                _s.setMaximum(100)
                _s.setValue(100 if 'max' in _slider_attr else 0)
                setattr(widget, _slider_attr, _s)

            # Share the global fitness_max_spin to all RCM widgets.
            if attr_name == "rcm_scatter_plot":
                # The scatter plot owns a visible spinbox; sync it
                # bidirectionally with the hidden global spinbox.
                widget.fitness_max_spin.setValue(self._rcm_fitness_max_spin.value())
                widget.fitness_max_spin.valueChanged.connect(
                    lambda v: self._rcm_fitness_max_spin.setValue(v))
            else:
                widget.fitness_max_spin = self._rcm_fitness_max_spin

        # ── Hide per-widget filter controls ──
        self._hide_widget_filters(widget)

        # ── Sync current global filter values into the new widget's sliders ──
        self._sync_global_filters_to_widget(widget)

        # ── Set plots_folder if data has already been loaded ──
        if self.match_collection is not None:
            plots_folder = str(self.match_collection.base_folder / "plots")
            widget.plots_folder = plots_folder

        # ── Rebuild filter widget lists so _on_global_filter_changed sees this widget ──
        self._rebuild_filter_widget_lists()

    def _hide_widget_filters(self, widget):
        """Hide per-widget filter controls for a single widget."""
        for layout_attr in ('_conf_layout', '_rmse_layout', '_error_layout'):
            layout = getattr(widget, layout_attr, None)
            if layout is not None:
                self._hide_layout_contents(layout)
        group = getattr(widget, '_confidence_group', None)
        if group is not None:
            group.hide()

    def _sync_global_filters_to_widget(self, widget):
        """Push current global slider values into a widget's own (hidden) sliders."""
        conf_min = self.global_conf_min_slider.value()
        conf_max = self.global_conf_max_slider.value()
        rmse_min = self.global_rmse_min_slider.value()
        rmse_max = self.global_rmse_max_slider.value()
        for attr, val in [('conf_min_slider', conf_min), ('conf_max_slider', conf_max)]:
            slider = getattr(widget, attr, None)
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(val)
                slider.blockSignals(False)
        for attr, val in [('rmse_min_slider', rmse_min), ('rmse_slider', rmse_max)]:
            slider = getattr(widget, attr, None)
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(val)
                slider.blockSignals(False)

    def _rebuild_filter_widget_lists(self):
        """Rebuild _conf_filter_widgets / _rmse_filter_widgets from created widgets."""
        self._conf_filter_widgets = [
            getattr(self, a) for a in self._CONF_FILTER_ATTRS
            if getattr(self, a, None) is not None
        ]
        self._rmse_filter_widgets = [
            getattr(self, a) for a in self._RMSE_FILTER_ATTRS
            if getattr(self, a, None) is not None
        ]

    def _sync_global_filters_to_all_widgets(self):
        """Push current global filter slider values into every already-created widget.

        This ensures eagerly-created widgets (e.g. scatter_plot) receive the
        correct filter range at data-load time, not just lazily-created ones.
        """
        for attr in self._ALL_LEFT_PLOT_ATTRS:
            widget = getattr(self, attr, None)
            if widget is not None:
                self._sync_global_filters_to_widget(widget)

    def _on_global_filter_changed(self, _=None):
        """Sync global filter sliders to all per-widget sliders, then refresh the active view."""
        conf_min = self.global_conf_min_slider.value()
        conf_max = self.global_conf_max_slider.value()
        rmse_min_val = self.global_rmse_min_slider.value()
        rmse_max_val = self.global_rmse_max_slider.value()

        # Ensure min ≤ max for confidence
        if conf_min > conf_max:
            self.global_conf_min_slider.blockSignals(True)
            self.global_conf_min_slider.setValue(conf_max)
            self.global_conf_min_slider.blockSignals(False)
            conf_min = conf_max

        # Ensure min ≤ max for RMSE
        if rmse_min_val > rmse_max_val:
            self.global_rmse_min_slider.blockSignals(True)
            self.global_rmse_min_slider.setValue(rmse_max_val)
            self.global_rmse_min_slider.blockSignals(False)
            rmse_min_val = rmse_max_val

        # Update global labels
        self.global_conf_min_label.setText(str(conf_min))
        self.global_conf_max_label.setText(str(conf_max))
        self.global_rmse_min_label.setText(str(rmse_min_val))
        self.global_rmse_max_label.setText(str(rmse_max_val))

        # Push values into every widget's own (hidden) sliders
        for w in self._conf_filter_widgets:
            for attr, val in [('conf_min_slider', conf_min), ('conf_max_slider', conf_max)]:
                slider = getattr(w, attr, None)
                if slider is not None:
                    slider.blockSignals(True)
                    slider.setValue(val)
                    slider.blockSignals(False)

        for w in self._rmse_filter_widgets:
            for attr, val in [('rmse_min_slider', rmse_min_val), ('rmse_slider', rmse_max_val)]:
                slider = getattr(w, attr, None)
                if slider is not None:
                    slider.blockSignals(True)
                    slider.setValue(val)
                    slider.blockSignals(False)

        # Debounce the expensive refresh — restart the timer so only the
        # last change in a rapid sequence triggers an actual replot.
        self._filter_debounce.start()

    # ------------------------------------------------------------------
    # Extract-data cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _indices_key(enabled_indices):
        """Return a hashable key for the given enabled_indices list."""
        if enabled_indices is None:
            return None
        return tuple(enabled_indices)

    def _invalidate_extract_cache(self):
        """Clear the extract cache (call when data selection changes)."""
        self._extract_cache.clear()

    def _get_cached(self, view_name: str):
        """Return cached extracted data for *view_name*, or None if stale/missing."""
        entry = self._extract_cache.get(view_name)
        if entry is None:
            return None
        cached_key, cached_data = entry
        if cached_key != self._cache_indices_key:
            return None  # stale
        return cached_data

    def _set_cached(self, view_name: str, data):
        """Store extracted data in the cache for *view_name*."""
        self._extract_cache[view_name] = (self._cache_indices_key, data)

    def _on_date_filter_changed(self, _=None):
        """Debounce date filter changes — the actual work is in _apply_date_filter."""
        if self.match_collection is None:
            return
        self._date_debounce.start()

    def _apply_date_filter(self):
        """Re-extract data and refresh the active view after debounce settles."""
        if self.match_collection is None:
            return
        # Date change → invalidate extract cache (selection changed)
        self._invalidate_extract_cache()
        view = self.current_left_view
        enabled_indices = self._get_enabled_indices()
        # Date filtering changes which matches are included → always re-extract
        reload_map = {
            "Aero - Spin vs Speed":              lambda: self.update_scatter_plot(enabled_indices),
            "Histograms":                        lambda: self.update_histograms(enabled_indices),
            "Confidence Distribution":            lambda: self.update_confidence_distribution(enabled_indices),
            "GCS Spin Confidence":                lambda: self.update_gcs_spin_confidence(enabled_indices),
            "Rally Analysis":                     lambda: self.update_rally_analysis(enabled_indices),
            "Game Analysis - Time Series":        lambda: self.update_time_series_plot(enabled_indices),
            "Aero - Drag Coefficient":            lambda: self.update_drag_coefficient_plot(enabled_indices),
            "Aero - Magnus Coefficient":          lambda: self.update_magnus_coefficient_plot(enabled_indices),
            "Aero - Error Box Plots":             lambda: self.update_aero_error_plot(enabled_indices),
            "Aero - Error Analysis":              lambda: self.update_conf_vs_error_plot(enabled_indices),
            "TCM - Input vs Output":              lambda: self.update_table_contact_plot(enabled_indices),
            "TCM - Epsilon vs Vz":                lambda: self.update_epsilon_vs_vz_plot(enabled_indices),
            "TCM - Fitness Box Plot":             lambda: self.update_fitness_box_plot(enabled_indices),
            "Game Analysis - Spin Observation":   lambda: self.update_spin_observation_plot(enabled_indices),
            "RCM - Spin Conversion":              lambda: self.update_spin_conversion_plot(enabled_indices),
            "RCM - Input vs Output":               lambda: self.update_rcm_scatter_plot(enabled_indices),
            "RCM - Box Plots":                     lambda: self.update_rcm_boxplot(enabled_indices),
            "RCM - Restitution":                   lambda: self.update_rcm_restitution_plot(enabled_indices),
            "RCM - Contact Location":              lambda: self.update_rcm_contact_location_plot(enabled_indices),
            "RCM - Output vs Others":              lambda: self.update_rcm_output_vs_others_plot(enabled_indices),
            "RCM - Distributions":                   lambda: self.update_rcm_histogram_plot(enabled_indices),
            "Game Analysis - Net Contacts":       lambda: self.update_net_contact_plot(enabled_indices),
            "Game Analysis - Spin Observation":   lambda: self.update_spin_observation_plot(enabled_indices),
        }
        fn = reload_map.get(view)
        if fn is not None:
            fn()

    def _get_enabled_indices(self) -> Optional[List[int]]:
        """Return match indices enabled by checkboxes, date range, and version filter.

        Returns None when all matches are included (for efficiency).
        """
        if self.match_collection is None:
            return None

        # 1. Checkbox filter
        checked: List[int] = []
        for i in range(self.matches_list.count()):
            item = self.matches_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(i)

        # 2. Date range filter
        date_from = self.global_date_from_combo.currentText()
        date_to = self.global_date_to_combo.currentText()
        apply_date = date_from != "(all)" or date_to != "(all)"

        if apply_date:
            date_filtered: List[int] = []
            for i, match in enumerate(self.match_collection.matches):
                d = str(getattr(match, 'date', ''))
                if date_from != "(all)" and d < date_from:
                    continue
                if date_to != "(all)" and d > date_to:
                    continue
                date_filtered.append(i)
            # Intersect with checkboxes
            combined = sorted(set(checked) & set(date_filtered))
        else:
            combined = checked

        # 3. H5 version filter
        version_sel = self.global_version_combo.currentText()
        if version_sel != "(all)":
            version_filtered: List[int] = []
            for i, match in enumerate(self.match_collection.matches):
                v = getattr(match, 'version_data_processing', None) or '(unknown)'
                if v == version_sel:
                    version_filtered.append(i)
            combined = sorted(set(combined) & set(version_filtered))

        total = len(self.match_collection.matches)
        if len(combined) == 0 or len(combined) == total:
            result = None if len(combined) == total else combined if combined else []
        else:
            result = combined
        # Update the cache key so _get_cached / _set_cached know the current selection
        self._cache_indices_key = self._indices_key(result)
        return result

    def _populate_date_combos(self):
        """Populate the From / To date combo boxes from the loaded matches."""
        if self.match_collection is None:
            return
        dates = sorted({
            str(getattr(m, 'date', ''))
            for m in self.match_collection.matches
            if getattr(m, 'date', None) is not None
        })
        for combo in (self.global_date_from_combo, self.global_date_to_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(all)")
            combo.addItems(dates)
            combo.blockSignals(False)
        # Default: From = earliest (all), To = latest (all)
        self.global_date_from_combo.setCurrentIndex(0)
        self.global_date_to_combo.setCurrentIndex(0)

    @staticmethod
    def _backfill_version_data_processing(match_collection):
        """Backfill version_data_processing on matches loaded from a stale pickle cache.

        When the cache was written before the field existed, Match objects
        will have version_data_processing=None.  This reads the attribute
        from each HDF5 file and sets it on the Match in-place.
        """
        import h5py as _h5py

        needs_backfill = [
            m for m in match_collection.matches
            if getattr(m, 'version_data_processing', None) is None
        ]
        if not needs_backfill:
            return

        filled = 0
        for match in needs_backfill:
            fpath = match.file
            if fpath is None or not pathlib.Path(fpath).exists():
                continue
            try:
                with _h5py.File(str(fpath), 'r') as f:
                    raw = f.attrs.get('file/version_data_processing', None)
                    if isinstance(raw, bytes):
                        raw = raw.decode('utf-8', errors='replace')
                    match.version_data_processing = str(raw) if raw is not None else None
                    filled += 1
            except Exception:
                pass
        if filled:
            print(f"Backfilled version_data_processing for {filled}/{len(needs_backfill)} matches from HDF5")

    def _populate_version_combo(self):
        """Populate the H5 version combo box from the loaded matches."""
        if self.match_collection is None:
            return
        versions = sorted({
            getattr(m, 'version_data_processing', None) or '(unknown)'
            for m in self.match_collection.matches
        })
        self.global_version_combo.blockSignals(True)
        self.global_version_combo.clear()
        self.global_version_combo.addItem("(all)")
        self.global_version_combo.addItems(versions)
        self.global_version_combo.blockSignals(False)
        self.global_version_combo.setCurrentIndex(0)

    def _on_version_filter_changed(self, _=None):
        """Handle version filter changes — same flow as date filter."""
        if self.match_collection is None:
            return
        self._date_debounce.start()  # reuse date debounce → _apply_date_filter

    def _refresh_current_view(self):
        """Trigger a replot on the currently visible analysis widget."""
        if self.match_collection is None:
            return
        view = self.current_left_view
        # Only views that have confidence / RMSE filtering need refreshing
        refresh_map = {
            "Aero - Spin vs Speed":              lambda: self.scatter_plot.update_plot(),
            "Aero - Drag Coefficient":            lambda: self.drag_coeff_plot.update_plot(),
            "Aero - Magnus Coefficient":          lambda: self.magnus_coeff_plot.update_plot(),
            "Aero - Error Box Plots":             lambda: self.aero_error_plot.update_plot(),
            "Aero - Error Analysis":              lambda: self.conf_vs_error_plot.update_plot(),
            "TCM - Input vs Output":              lambda: self.table_contact_plot.update_plot(),
            "TCM - Epsilon vs Vz":                lambda: self.epsilon_vs_vz_plot.update_plot(),
            "TCM - Fitness Box Plot":             lambda: self.fitness_box_plot._do_update(),
            "Game Analysis - Spin Observation":   lambda: self.spin_observation_plot.update_plot(),
            "RCM - Spin Conversion":              lambda: self.spin_conversion_plot.update_plot(),
            "RCM - Input vs Output":               lambda: self.rcm_scatter_plot._do_update(),
            "RCM - Box Plots":                     lambda: self.rcm_boxplot._do_update(),
            "RCM - Restitution":                   lambda: self.rcm_restitution_plot._do_update(),
            "RCM - Output vs Others":              lambda: self.rcm_output_vs_others_plot._do_update(),
            "RCM - Distributions":                   lambda: self.rcm_histogram_plot._do_update(),
        }
        fn = refresh_map.get(view)
        if fn is not None:
            fn()

    # ------------------------------------------------------------------
    # Callbacks for background loading
    # ------------------------------------------------------------------
    def _on_data_loaded(self, match_collection):
        """Called on the main thread when background loading finishes."""
        self.match_collection = match_collection
        self.load_button.setEnabled(True)
        self.load_json_button.setEnabled(True)
        self.save_json_button.setEnabled(True)
        self._invalidate_extract_cache()  # new data → stale cache

        # Backfill version_data_processing for matches loaded from stale cache
        self._backfill_version_data_processing(match_collection)

        # Set the central plots folder on every *created* widget
        plots_folder = str(match_collection.base_folder / "plots")
        for widget in [
            self.scatter_plot, self.histogram_plot,
            self.confidence_dist_plot, self.gcs_spin_conf_plot,
            self.drag_coeff_plot, self.magnus_coeff_plot,
            self.aero_error_plot, self.conf_vs_error_plot,
            self.table_contact_plot, self.epsilon_vs_vz_plot,
            self.fitness_box_plot,
            self.spin_observation_plot, self.spin_conversion_plot,
            self.time_series_plot, self.rcm_scatter_plot,
            self.rcm_boxplot, self.net_contact_plot,
            self.rcm_contact_location_plot, self.flight_plot,
        ]:
            if widget is not None:
                widget.plots_folder = plots_folder

        # Warn about rallies not using gcs_offline
        self._warn_non_gcs_offline(match_collection)

        # Get statistics
        stats = self.data_processor.get_statistics(self.match_collection)
        self.status_label.setText(self._build_status_text(stats))

        # Update matches list and date combos first (they affect enabled_indices)
        self.update_matches_list()
        self._populate_date_combos()
        self._populate_version_combo()

        # Sync global filter values into all already-created widget sliders
        # so that the initial replot uses the correct filter range.
        self._sync_global_filters_to_all_widgets()

        # Trigger a full update of the currently visible view.
        # on_left_view_changed ensures the widget is created, shown, data
        # is extracted (respecting enabled_indices), and filters are applied
        # in a single pass.  Lazy widgets for other views will be populated
        # when the user switches to them.
        self.on_left_view_changed(self.current_left_view)

    def _on_data_load_error(self, error_msg: str):
        """Called on the main thread when background loading fails."""
        self.load_button.setEnabled(True)
        self.load_json_button.setEnabled(True)
        self.status_label.setText(f"Failed to auto-load default data: {error_msg}")

    @staticmethod
    def _warn_non_gcs_offline(match_collection):
        """Print warnings for rallies whose gcs_source is not gcs_offline.

        Groups issues by category so the user can see at a glance which
        files are missing ground_truth_200, which lack gcs_offline, etc.
        """
        from collections import defaultdict

        # Collect (file, game_id, rally_id, gcs_source) for every non-gcs_offline rally
        issues = []  # list of (file, game_id, rally_id, gcs_source)
        for match in match_collection.matches:
            for game in match.games:
                for rally in game.rallies:
                    src = getattr(rally, 'gcs_source', None)
                    if src is None:
                        # Stale pickle cache — derive reason from rally data
                        if isinstance(rally.rally, list) and len(rally.rally) == 0:
                            src = 'empty_rally'
                        elif hasattr(rally.rally, 'attrs'):
                            src = rally.rally.attrs.get('gcs_source', 'unknown')
                        else:
                            src = 'unknown'
                    if src != 'gcs_offline':
                        issues.append((match.file, game.game_id, rally.rally_id, src))

        if not issues:
            return

        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"

        # Bucket by reason category
        no_gt200 = [(f, g, r, s) for f, g, r, s in issues if 'no_ground_truth_200' in s]
        no_racket = [(f, g, r, s) for f, g, r, s in issues if 'no_racket1' in s]
        excluded_gcs = [(f, g, r, s) for f, g, r, s in issues if 'excluded' in s]
        no_gcs_sensor = [(f, g, r, s) for f, g, r, s in issues if s == 'none']
        empty_rally = [(f, g, r, s) for f, g, r, s in issues if s == 'empty_rally']
        other = [(f, g, r, s) for f, g, r, s in issues
                 if 'no_ground_truth_200' not in s and 'no_racket1' not in s
                 and 'excluded' not in s and s not in ('empty_rally', 'none')]

        def _print_grouped(label, color, entries):
            if not entries:
                return
            by_file = defaultdict(list)
            for fpath, gid, rid, src in entries:
                by_file[fpath].append((gid, rid, src))
            print(f"\n{color}⚠  {label} ({len(entries)} rally(ies)):")
            for fpath, file_entries in by_file.items():
                print(f"  {fpath}")
                for gid, rid, src in file_entries:
                    print(f"    game {gid}, rally {rid}: {src}")
            print(RESET)

        _print_grouped("Missing ground_truth_200 group", RED, no_gt200)
        _print_grouped("Missing racket1 in ground_truth_200 (missing musashi ace logs)", RED, no_racket)
        _print_grouped(
            "No gcs sensor found — spin columns will be NaN",
            YELLOW, no_gcs_sensor,
        )
        _print_grouped(
            "gcs_offline not available — excluded (use --gcs-fallback to include)",
            YELLOW, excluded_gcs,
        )
        _print_grouped(
            "Empty rally (missing ground_truth_200 or excluded — delete cache to see details)",
            YELLOW, empty_rally,
        )
        _print_grouped("Other gcs_source issues", YELLOW, other)

    def on_left_panel_toggle(self, state: int):
        """Handle left panel (analysis plots) visibility toggle"""
        view_to_index = {
            "Scatter Plot": 0,
            "Histograms": 1,
            "Confidence Distribution": 2,
            "GCS Spin Confidence": 3,
            "Rally Analysis": 4,
            "Aero - Drag Coefficient": 5,
            "Aero - Magnus Coefficient": 6,
            "Aero - Error Box Plots": 7,
            "Aero - Error Analysis": 8,
            "TCM - Input vs Output": 9,
            "TCM - Epsilon vs Vz": 10,
            "TCM - Fitness Box Plot": 11,
            "Game Analysis - Spin Observation": 12,
            "RCM - Spin Conversion": 13,
            "Game Analysis - Time Series": 14,
            "RCM - Input vs Output": 15,
            "RCM - Box Plots": 16,
            "RCM - Restitution": 17,
            "Game Analysis - Net Contacts": 18,
            "RCM - Contact Location": 19,
            "RCM - Output vs Others": 20,
            "RCM - Distributions": 21,
        }
        left_index = view_to_index.get(self.current_left_view, 0)

        if state == Qt.Checked.value:
            # Show left panel - restore the current view
            self._show_current_left_view()
            # Update splitter
            include_flight = self.show_right_panel_checkbox.isChecked() and self.flight_plot.isVisible()
            self._set_splitter_for_view(left_index, include_flight=include_flight, include_left=True)
        else:
            # Hide all left panel plots (skip lazy widgets not yet created)
            for _attr in self._ALL_LEFT_PLOT_ATTRS:
                _w = getattr(self, _attr, None)
                if _w is not None:
                    _w.hide()
            # Update splitter to give full width to flight plot
            self._set_splitter_for_view(left_index, include_flight=True, include_left=False)

    def _show_current_left_view(self):
        """Show the currently selected left view widget (creating it lazily if needed)."""
        attr = self._VIEW_TO_ATTR.get(self.current_left_view)
        if attr:
            self._ensure_widget(attr)
            widget = getattr(self, attr)
            if widget is not None:
                widget.show()

    def on_right_panel_toggle(self, state: int):
        """Handle right panel (flight segment plot) visibility toggle"""
        if state == Qt.Checked.value:
            # Right panel can be shown - no immediate action needed
            # It will appear when a point is clicked
            pass
        else:
            # Hide right panel immediately
            self.flight_plot.hide()
            # Update splitter to give full width to left panel
            view_to_index = {
                "Scatter Plot": 0,
                "Histograms": 1,
                "Confidence Distribution": 2,
                "GCS Spin Confidence": 3,
                "Rally Analysis": 4,
                "Aero - Drag Coefficient": 5,
                "Aero - Magnus Coefficient": 6,
                "Aero - Error Box Plots": 7,
                "Aero - Error Analysis": 8,
                "TCM - Input vs Output": 9,
                "TCM - Epsilon vs Vz": 10,
                "TCM - Fitness Box Plot": 11,
                "Game Analysis - Spin Observation": 12,
                "RCM - Spin Conversion": 13,
                "Game Analysis - Time Series": 14,
                "RCM - Input vs Output": 15,
                "RCM - Box Plots": 16,
                "RCM - Restitution": 17,
                "Game Analysis - Net Contacts": 18,
                "RCM - Contact Location": 19,
                "RCM - Output vs Others": 20,
                "RCM - Distributions": 21,
            }
            left_index = view_to_index.get(self.current_left_view, 0)
            self._set_splitter_for_view(left_index, include_flight=False)

    def on_left_view_changed(self, view_type: str):
        """Handle left panel view selection change"""
        self.current_left_view = view_type

        # Check if left panel should be visible
        left_panel_visible = self.show_left_panel_checkbox.isChecked()

        # Get enabled match indices (respects checkboxes + date filter)
        enabled_indices = self._get_enabled_indices()

        # Hide all left plots first (skip lazy widgets not yet created)
        for _attr in self._ALL_LEFT_PLOT_ATTRS:
            _w = getattr(self, _attr, None)
            if _w is not None:
                _w.hide()

        # Only show left panel widget if checkbox is checked
        if not left_panel_visible:
            return

        # Ensure the widget for the selected view is created (lazy instantiation)
        _view_attr = self._VIEW_TO_ATTR.get(view_type)
        if _view_attr:
            self._ensure_widget(_view_attr)

        if view_type == "Aero - Spin vs Speed":
            self.scatter_plot.show()
            self._set_splitter_for_view(0, include_flight=self.flight_plot.isVisible())
            if self.match_collection is not None:
                with self._busy_status("Updating Spin vs Speed…"):
                    self.update_scatter_plot(enabled_indices)
        elif view_type == "Histograms":
            self.histogram_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(1)
            if self.match_collection is not None:
                with self._busy_status("Updating Histograms…"):
                    self.update_histograms(enabled_indices)
        elif view_type == "Confidence Distribution":
            self.confidence_dist_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(2)
            if self.match_collection is not None:
                with self._busy_status("Updating Confidence Distribution…"):
                    self.update_confidence_distribution(enabled_indices)
        elif view_type == "GCS Spin Confidence":
            self.gcs_spin_conf_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(3)
            if self.match_collection is not None:
                with self._busy_status("Updating GCS Spin Confidence…"):
                    self.update_gcs_spin_confidence(enabled_indices)
        elif view_type == "Rally Analysis":
            self.rally_analysis_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(4)
            if self.match_collection is not None:
                msg = "Computing rally analysis… This may take a moment." if not self.rally_analysis_loaded else "Updating Rally Analysis…"
                with self._busy_status(msg):
                    self.update_rally_analysis(enabled_indices)
                    self.rally_analysis_loaded = True
        elif view_type == "Aero - Drag Coefficient":
            self.drag_coeff_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(5)
            if self.match_collection is not None:
                with self._busy_status("Updating Drag Coefficient…"):
                    self.update_drag_coefficient_plot(enabled_indices)
        elif view_type == "Aero - Magnus Coefficient":
            self.magnus_coeff_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(6)
            if self.match_collection is not None:
                with self._busy_status("Updating Magnus Coefficient…"):
                    self.update_magnus_coefficient_plot(enabled_indices)
        elif view_type == "Aero - Error Box Plots":
            self.aero_error_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(7)
            if self.match_collection is not None:
                with self._busy_status("Updating Error Box Plots…"):
                    self.update_aero_error_plot(enabled_indices)
        elif view_type == "Aero - Error Analysis":
            self.conf_vs_error_plot.show()
            self._set_splitter_for_view(8, include_flight=self.flight_plot.isVisible())
            if self.match_collection is not None:
                with self._busy_status("Updating Error Analysis…"):
                    self.update_conf_vs_error_plot(enabled_indices)
        elif view_type == "TCM - Input vs Output":
            self.table_contact_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(9)
            if self.match_collection is not None:
                with self._busy_status("Updating Table Contact…"):
                    self.update_table_contact_plot(enabled_indices)
        elif view_type == "TCM - Epsilon vs Vz":
            self.epsilon_vs_vz_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(10)
            if self.match_collection is not None:
                with self._busy_status("Updating Epsilon vs Vz…"):
                    self.update_epsilon_vs_vz_plot(enabled_indices)
        elif view_type == "TCM - Fitness Box Plot":
            self.fitness_box_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(11)
            if self.match_collection is not None:
                with self._busy_status("Updating Fitness Box Plot…"):
                    self.update_fitness_box_plot(enabled_indices)
        elif view_type == "Game Analysis - Spin Observation":
            self.spin_observation_plot.show()
            self._set_splitter_for_view(12, include_flight=self.flight_plot.isVisible())
            if self.match_collection is not None:
                with self._busy_status("Updating Spin Observation…"):
                    self.update_spin_observation_plot(enabled_indices)
        elif view_type == "RCM - Spin Conversion":
            self.spin_conversion_plot.show()
            self._set_splitter_for_view(13, include_flight=self.flight_plot.isVisible())
            if self.match_collection is not None:
                with self._busy_status("Updating Spin Conversion…"):
                    self.update_spin_conversion_plot(enabled_indices)
        elif view_type == "Game Analysis - Time Series":
            self.time_series_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(14)
            if self.match_collection is not None:
                with self._busy_status("Updating Time Series…"):
                    self.update_time_series_plot(enabled_indices)
        elif view_type == "RCM - Input vs Output":
            self.rcm_scatter_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(15)
            if self.match_collection is not None:
                with self._busy_status("Updating RCM Scatter…"):
                    self.update_rcm_scatter_plot(enabled_indices)
        elif view_type == "RCM - Box Plots":
            self.rcm_boxplot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(16)
            if self.match_collection is not None:
                with self._busy_status("Updating RCM Box Plots…"):
                    self.update_rcm_boxplot(enabled_indices)
        elif view_type == "RCM - Restitution":
            self.rcm_restitution_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(17)
            if self.match_collection is not None:
                with self._busy_status("Updating RCM Restitution…"):
                    self.update_rcm_restitution_plot(enabled_indices)
        elif view_type == "Game Analysis - Net Contacts":
            self.net_contact_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(18)
            if self.match_collection is not None:
                with self._busy_status("Updating Net Contacts…"):
                    self.update_net_contact_plot(enabled_indices)
        elif view_type == "RCM - Contact Location":
            self.rcm_contact_location_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(19)
            if self.match_collection is not None:
                with self._busy_status("Updating RCM Contact Location…"):
                    self.update_rcm_contact_location_plot(enabled_indices)
        elif view_type == "RCM - Output vs Others":
            self.rcm_output_vs_others_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(20)
            if self.match_collection is not None:
                with self._busy_status("Updating RCM Output vs Others…"):
                    self.update_rcm_output_vs_others_plot(enabled_indices)
        elif view_type == "RCM - Distributions":
            self.rcm_histogram_plot.show()
            self.flight_plot.hide()
            self._set_splitter_for_view(21)
            if self.match_collection is not None:
                with self._busy_status("Updating RCM Distributions…"):
                    self.update_rcm_histogram_plot(enabled_indices)

    def _set_splitter_for_view(self, view_index: int, include_flight: bool = False, include_left: bool = True):
        """Set splitter sizes for the given view index.

        Args:
            view_index: Index of the left panel widget (0-11)
            include_flight: If True, include flight plot in the splitter
            include_left: If True, include left analysis panel in the splitter
        """
        total = self.splitter.width()
        if total <= 0:
            total = 1000

        sizes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 23 elements (22 plots + flight)

        if include_left and include_flight:
            # Both panels visible - split 50/50
            sizes[view_index] = total // 2
            sizes[22] = total // 2  # Flight plot is at index 22
        elif include_left:
            # Only left panel visible
            sizes[view_index] = total
        elif include_flight:
            # Only flight plot visible
            sizes[22] = total  # Flight plot is at index 22
        # else: both hidden (shouldn't happen normally)

        self.splitter.setSizes(sizes)

    def update_histograms(self, enabled_indices=None):
        """Extract and plot histogram data"""
        speeds_robot, spins_robot = self.data_processor.extract_histogram_data(self.match_collection, player_type="robot", enabled_indices=enabled_indices)
        speeds_player, spins_player = self.data_processor.extract_histogram_data(self.match_collection, player_type="player", enabled_indices=enabled_indices)
        self.histogram_plot.plot_data(speeds_robot, spins_robot, speeds_player, spins_player)

    def update_net_contact_plot(self, enabled_indices=None):
        """Extract and plot net-contact histogram data"""
        if self.match_collection is not None:
            df = self.data_processor.extract_net_contact_data(
                self.match_collection, enabled_indices=enabled_indices
            )
            self.net_contact_plot.plot_data(df)

    def update_rcm_contact_location_plot(self, enabled_indices=None):
        """Update the RCM contact-location scatter matrix plot."""
        if self.match_collection is not None:
            try:
                data = self._get_rcm_data(enabled_indices)
                if len(data.get('vx_pre', [])) == 0:
                    print("Warning: No RCM data found")
                    return
                self.rcm_contact_location_plot.plot_data(data)
            except Exception as e:
                print(f"Error updating RCM contact location plot: {e}")
                import traceback
                traceback.print_exc()

    def update_confidence_distribution(self, enabled_indices=None):
        """Extract and plot confidence distribution data"""
        if self.match_collection is not None:
            vel_high, spin_high, vel_low, spin_low, duration_high, duration_low, dates_high, dates_low = self.data_processor.extract_confidence_distribution_data(
                self.match_collection, enabled_indices
            )
            match_file = self.match_collection.matches[0].file if self.match_collection and len(self.match_collection.matches) > 0 else None
            self.confidence_dist_plot.plot_data(vel_high, spin_high, vel_low, spin_low, duration_high, duration_low, dates_high, dates_low, match_file)

    def update_gcs_spin_confidence(self, enabled_indices=None):
        """Extract and plot GCS spin confidence histogram data"""
        if self.match_collection is not None:
            median_w_confidences, median_wx_confidences = self.data_processor.extract_gcs_spin_confidence_data(
                self.match_collection, enabled_indices
            )
            match_file = self.match_collection.matches[0].file if self.match_collection and len(self.match_collection.matches) > 0 else None
            self.gcs_spin_conf_plot.plot_data(median_w_confidences, match_file, median_wx_confidences)

    def update_rally_analysis(self, enabled_indices=None):
        """Extract and plot rally analysis data"""
        data_rally = self.data_processor.extract_rally_analysis_data(self.match_collection, enabled_indices)
        self.rally_analysis_plot.plot_data(data_rally)

    def update_scatter_plot(self, enabled_indices=None):
        """Update scatter plot with both robot and player data"""
        if self.match_collection is not None:
            cached = self._get_cached('scatter')
            if cached is not None:
                robot_data, player_data = cached
            else:
                robot_data = self.data_processor.extract_spin_speed_data(self.match_collection, player_type="robot", enabled_indices=enabled_indices)
                player_data = self.data_processor.extract_spin_speed_data(self.match_collection, player_type="player", enabled_indices=enabled_indices)
                self._set_cached('scatter', (robot_data, player_data))
            self.scatter_plot.plot_data(robot_data, player_data)

    def update_drag_coefficient_plot(self, enabled_indices=None):
        """Update drag coefficient plot with unified robot and player data"""
        if self.match_collection is not None:
            try:
                cached = self._get_cached('drag_coeff')
                if cached is not None:
                    unified_data = cached
                else:
                    unified_data = self.data_processor.extract_drag_coefficient_data(self.match_collection, player_type=None, enabled_indices=enabled_indices)
                    self._set_cached('drag_coeff', unified_data)

                # Check if we have any data
                if len(unified_data[0]) == 0:
                    print("Warning: No aerodynamics data found for drag coefficient plot")
                    # Clear stale data from previous dataset
                    self.drag_coeff_plot.robot_data = None
                    self.drag_coeff_plot.player_data = None
                    self.drag_coeff_plot.plot_widget_2d.clear()
                    self.drag_coeff_plot.info_label.setText("No aerodynamics data available")
                    return

                # Plot the unified data (pass same data for both arguments)
                self.drag_coeff_plot.plot_data(unified_data, None)
            except Exception as e:
                print(f"Error updating drag coefficient plot: {e}")

    def update_magnus_coefficient_plot(self, enabled_indices=None):
        """Update magnus coefficient plot with unified robot and player data"""
        if self.match_collection is not None:
            try:
                cached = self._get_cached('magnus_coeff')
                if cached is not None:
                    unified_data = cached
                else:
                    unified_data = self.data_processor.extract_magnus_coefficient_data(self.match_collection, player_type=None, enabled_indices=enabled_indices)
                    self._set_cached('magnus_coeff', unified_data)

                # Check if we have any data
                if len(unified_data[0]) == 0:
                    print("Warning: No aerodynamics data found for magnus coefficient plot")
                    # Clear stale data from previous dataset
                    self.magnus_coeff_plot.robot_data = None
                    self.magnus_coeff_plot.player_data = None
                    self.magnus_coeff_plot.plot_widget_2d.clear()
                    self.magnus_coeff_plot.info_label.setText("No aerodynamics data available")
                    return

                # Plot the unified data (pass same data for both arguments)
                self.magnus_coeff_plot.plot_data(unified_data, None)
            except Exception as e:
                print(f"Error updating magnus coefficient plot: {e}")

    def update_aero_error_plot(self, enabled_indices=None):
        """Update aero error summary plot"""
        if self.match_collection is not None:
            try:
                # Extract error data
                error_dict, confidences, durations, base_folder, match_dates, shot_types = self.data_processor.extract_aero_summary_error_data(
                    self.match_collection, enabled_indices=enabled_indices
                )

                # Check if we have any data
                n_gt200 = np.sum(~np.isnan(error_dict.get('gt200', np.array([]))))
                n_opt = np.sum(~np.isnan(error_dict.get('opt', np.array([]))))

                if n_gt200 == 0 and n_opt == 0:
                    print("Warning: No trajectory comparison data found for aero error summary plot")
                    return

                # Plot the error data (also clears any previous data)
                self.aero_error_plot.plot_data(error_dict, confidences, durations, base_folder,
                                                match_dates=match_dates, shot_types=shot_types)
            except Exception as e:
                print(f"Error updating aero error summary plot: {e}")

    def update_conf_vs_error_plot(self, enabled_indices=None):
        """Update Aerodynamics Error Analysis scatter plot"""
        if self.match_collection is not None:
            try:
                # Extract data for Aerodynamics Error Analysis (with flight segment references for click handling)
                data = self.data_processor.extract_confidence_vs_error_data(
                    self.match_collection, enabled_indices=enabled_indices
                )

                # Check if we have any data
                n_points = len(data.get('confidence', []))
                if n_points == 0:
                    print("Warning: No data found for Aerodynamics Error Analysis plot")
                    return

                # Plot the data
                self.conf_vs_error_plot.plot_data(
                    error_dict=data['error_dict'],
                    confidence=data['confidence'],
                    duration=data['duration'],
                    metadata_list=data['metadata_list'],
                    flight_segments=data['flight_segments'],
                    shots=data['shots'],
                    rallies=data['rallies'],
                    base_folder=data['base_folder'],
                    magnus_sensitivity=data.get('magnus_sensitivity'),
                    drag_sensitivity=data.get('drag_sensitivity'),
                    spin_density=data.get('spin_density'),
                    spin_variance=data.get('spin_variance'),
                    gcs_source=data.get('gcs_source'),
                )

                # Also extract contact residual errors for the additional error type modes
                contact_errors = self.data_processor.extract_contact_residual_errors(
                    self.match_collection, player_type=None, enabled_indices=enabled_indices
                )

                if len(contact_errors.get('velocity_error', [])) > 0:
                    self.conf_vs_error_plot.set_contact_error_data(
                        velocity_error=contact_errors['velocity_error'],
                        spin_error=contact_errors['spin_error'],
                        confidence=contact_errors['confidence'],
                        metadata_list=contact_errors['metadata_list'],
                        shots=contact_errors['shot_list'],
                        rallies=contact_errors['rally_list'],
                        fs_pre_list=contact_errors['fs_pre_list'],
                        fs_post_list=contact_errors['fs_post_list'],
                        max_rmse_opt=contact_errors.get('max_rmse_opt'),
                    )

            except Exception as e:
                print(f"Error updating Aerodynamics Error Analysis plot: {e}")
                import traceback
                traceback.print_exc()

    def _get_table_contact_data(self, enabled_indices=None):
        """Return table contact data, using cache when available.

        Shared by update_table_contact_plot, update_epsilon_vs_vz_plot,
        and update_fitness_box_plot.
        """
        # Determine whether to use extracted_TCM.csv data
        use_tcm = True
        tcm_widget = getattr(self, 'table_contact_plot', None)
        if tcm_widget is not None and hasattr(tcm_widget, 'use_extracted_tcm_checkbox'):
            use_tcm = tcm_widget.use_extracted_tcm_checkbox.isChecked()

        cache_key = f'table_contact_{"tcm" if use_tcm else "hdf5"}'
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        data = self.data_processor.extract_table_contact_data(
            self.match_collection, player_type=None, enabled_indices=enabled_indices,
            use_extracted_tcm=use_tcm,
        )
        self._set_cached(cache_key, data)
        return data

    def _on_tcm_data_source_changed(self, use_extracted_tcm: bool):
        """Handle toggle of 'Use extracted TCM' checkbox in the TCM plot.

        Invalidates relevant caches and re-fetches + re-plots the TCM plot
        (and any other plots sharing the table_contact data).
        """
        # Invalidate both cache variants so the next fetch is fresh
        self._extract_cache.pop('table_contact_tcm', None)
        self._extract_cache.pop('table_contact_hdf5', None)

        enabled_indices = self._get_enabled_indices()
        src_label = "extracted TCM (CSV)" if use_extracted_tcm else "HDF5 (iloc)"
        with self._busy_status(f"Reloading TCM data from {src_label}…"):
            self.update_table_contact_plot(enabled_indices)

    def update_table_contact_plot(self, enabled_indices=None):
        """Update table contact plot with contact data from HDF5"""
        if self.match_collection is not None:
            try:
                contact_data = self._get_table_contact_data(enabled_indices)

                # Check if we have any data
                if len(contact_data['vx_pre']) == 0:
                    print("Warning: No table contact data found")
                    return

                # Plot the contact data (also clears any previous data)
                self.table_contact_plot.plot_data(contact_data)
            except Exception as e:
                print(f"Error updating table contact plot: {e}")

    def update_epsilon_vs_vz_plot(self, enabled_indices=None):
        """Update epsilon vs vz plot with contact data from HDF5"""
        if self.match_collection is not None:
            try:
                contact_data = self._get_table_contact_data(enabled_indices)

                # Check if we have any data
                if len(contact_data['vz_pre']) == 0:
                    print("Warning: No table contact data found for epsilon vs vz plot")
                    return

                # Get base folder for saving plots
                base_folder = str(self.match_collection.base_folder) if self.match_collection.base_folder else None

                # Plot the contact data
                self.epsilon_vs_vz_plot.plot_data(contact_data, base_folder)
            except Exception as e:
                print(f"Error updating epsilon vs vz plot: {e}")
                import traceback
                traceback.print_exc()

    def update_fitness_box_plot(self, enabled_indices=None):
        """Update fitness box plot with Nakashima vs Residual model comparison"""
        if self.match_collection is not None:
            try:
                contact_data = self._get_table_contact_data(enabled_indices)

                if len(contact_data['vz_pre']) == 0:
                    print("Warning: No table contact data found for fitness box plot")
                    return

                base_folder = str(self.match_collection.base_folder) if self.match_collection.base_folder else None
                self.fitness_box_plot.plot_data(contact_data, base_folder)
            except Exception as e:
                print(f"Error updating fitness box plot: {e}")
                import traceback
                traceback.print_exc()

    def update_spin_observation_plot(self, enabled_indices=None):
        """Update spin observation analysis plot"""
        if self.match_collection is not None:
            try:
                cached = self._get_cached('spin_observation')
                if cached is not None:
                    data = cached
                else:
                    data = self.data_processor.extract_spin_observation_data(
                        self.match_collection, enabled_indices
                    )
                    self._set_cached('spin_observation', data)
                if len(data['velocity_mag']) == 0:
                    print("Warning: No spin observation data found (need OPT vel, GT200 spin, and GCS spin)")
                    return
                self.spin_observation_plot.plot_data(
                    velocity_mag=data['velocity_mag'],
                    spin_mag_gt=data['spin_mag_gt'],
                    mean_delta_spin=data['mean_delta_spin'],
                    spin_obs_var=data['spin_obs_var'],
                    confidence=data['confidence'],
                    opt_rmse=data['opt_rmse'],
                    n_gcs_samples=data['n_gcs_samples'],
                    spin_density=data['spin_density'],
                    is_last_shot=data['shot_position'],
                    gcs_source=data['gcs_source'],
                    vx_abs=data['vx_abs'],
                    shot_player=data['shot_player'],
                    mean_x=data['mean_x'],
                    mean_y=data['mean_y'],
                    mean_z=data['mean_z'],
                    contact_type=data['contact_type'],
                    metadata_list=data['metadata_list'],
                    flight_segments=data['flight_segments'],
                    shots=data['shots'],
                    rallies=data['rallies'],
                    base_folder=data['base_folder'],
                )
            except Exception as e:
                print(f"Error updating spin observation plot: {e}")
                import traceback
                traceback.print_exc()

    def update_spin_conversion_plot(self, enabled_indices=None):
        """Update spin conversion (racket contact) analysis plot"""
        if self.match_collection is not None:
            try:
                cached = self._get_cached('spin_conversion')
                if cached is not None:
                    data = cached
                else:
                    data = self.data_processor.extract_racket_contact_data(
                        self.match_collection, enabled_indices
                    )
                    self._set_cached('spin_conversion', data)
                if len(data['vx_pre']) == 0:
                    print("Warning: No racket contact data found")
                    return
                self.spin_conversion_plot.plot_data(data)
            except Exception as e:
                print(f"Error updating spin conversion plot: {e}")
                import traceback
                traceback.print_exc()

    def update_time_series_plot(self, enabled_indices=None):
        """Update time series plot with rally data from current match collection"""
        if self.match_collection is not None:
            self.time_series_plot.set_match_collection(self.match_collection, enabled_indices)

    # ── RCM (Racket Contact Model) update methods ──────────────────

    def _get_rcm_data(self, enabled_indices=None):
        """Return RCM data, using cache when available."""
        cached = self._get_cached('rcm')
        if cached is not None:
            return cached
        data = self.data_processor.extract_rcm_data(
            self.match_collection, enabled_indices
        )
        self._set_cached('rcm', data)
        return data

    def update_rcm_scatter_plot(self, enabled_indices=None):
        """Update the RCM scatter matrix plot."""
        if self.match_collection is not None:
            try:
                data = self._get_rcm_data(enabled_indices)
                if len(data.get('vx_pre', [])) == 0:
                    print("Warning: No RCM data found")
                    return
                self.rcm_scatter_plot.plot_data(data)
            except Exception as e:
                print(f"Error updating RCM scatter plot: {e}")
                import traceback
                traceback.print_exc()

    def update_rcm_boxplot(self, enabled_indices=None):
        """Update the RCM box plots."""
        if self.match_collection is not None:
            try:
                data = self._get_rcm_data(enabled_indices)
                if len(data.get('vx_pre', [])) == 0:
                    print("Warning: No RCM data found")
                    return
                base_folder = str(self.match_collection.base_folder) if self.match_collection.base_folder else None
                self.rcm_boxplot.plot_data(data, base_folder)
            except Exception as e:
                print(f"Error updating RCM box plot: {e}")
                import traceback
                traceback.print_exc()

    def update_rcm_restitution_plot(self, enabled_indices=None):
        """Update the RCM restitution (local-frame post vs pre) plot."""
        if self.match_collection is not None:
            try:
                data = self._get_rcm_data(enabled_indices)
                if len(data.get('vx_pre', [])) == 0:
                    print("Warning: No RCM data found")
                    return
                self.rcm_restitution_plot.plot_data(data)
            except Exception as e:
                print(f"Error updating RCM restitution plot: {e}")
                import traceback
                traceback.print_exc()

    def update_rcm_output_vs_others_plot(self, enabled_indices=None):
        """Update the RCM output vs others (contact loc / racket vel / racket ang vel) scatter plot."""
        if self.match_collection is not None:
            try:
                data = self._get_rcm_data(enabled_indices)
                if len(data.get('vx_pre', [])) == 0:
                    print("Warning: No RCM data found")
                    return
                self.rcm_output_vs_others_plot.plot_data(data)
            except Exception as e:
                print(f"Error updating RCM output vs others plot: {e}")
                import traceback
                traceback.print_exc()

    def update_rcm_histogram_plot(self, enabled_indices=None):
        """Update the RCM histogram distribution plot."""
        if self.match_collection is not None:
            try:
                data = self._get_rcm_data(enabled_indices)
                if len(data.get('vx_pre', [])) == 0:
                    print("Warning: No RCM data found")
                    return
                self.rcm_histogram_plot.plot_data(data)
            except Exception as e:
                print(f"Error updating RCM histogram plot: {e}")
                import traceback
                traceback.print_exc()

    def on_contact_selected(self, contact_data, metadata, shot, rally):
        """Handle selection of a contact point in table contact plot.

        Shows the flight trajectory from the segment before through the segment after the table contact.
        """
        if metadata:
            # Show contact details in status
            info_parts = []
            if metadata.get("match_date"):
                info_parts.append(f"Date: {metadata['match_date']}")
            if metadata.get("game_name"):
                info_parts.append(f"Game: {metadata['game_name']}")
            if metadata.get("rally_id") is not None:
                info_parts.append(f"Rally: {metadata['rally_id']}")
            if metadata.get("contact_type"):
                info_parts.append(f"Type: {metadata['contact_type']}")
            if contact_data:
                info_parts.append(f"vz_pre: {contact_data.get('vz_pre', 0):.2f} m/s")
                info_parts.append(f"vz_post: {contact_data.get('vz_post', 0):.2f} m/s")

            self.status_label.setText(" | ".join(info_parts))

        # ── Cross-highlight across all RCM widgets ───────────────────
        global_idx = contact_data.get('_rcm_global_idx') if contact_data else None
        if global_idx is not None:
            for attr in ('rcm_scatter_plot', 'rcm_contact_location_plot', 'rcm_histogram_plot'):
                widget = getattr(self, attr, None)
                if widget is not None and hasattr(widget, 'highlight_point'):
                    try:
                        widget.highlight_point(global_idx)
                    except Exception:
                        pass

        # Check if right panel should be shown
        if not self.show_right_panel_checkbox.isChecked():
            return

        # Get the flight segments around the contact
        fs_pre = contact_data.get('fs_pre') if contact_data else None
        fs_post = contact_data.get('fs_post') if contact_data else None

        if fs_pre is None or fs_post is None:
            return

        # Build combined data for the two segments (before and after contact)
        import pandas as pd
        try:
            combined_data = pd.concat([fs_pre.data, fs_post.data], ignore_index=False)
        except Exception as e:
            print(f"Error combining segment data: {e}")
            return

        # Show the flight plot panel
        self.flight_plot.show()
        QApplication.processEvents()

        # Update splitter to show both panels
        total = self.splitter.width()
        if total <= 0:
            total = 1000
        half = total // 2

        view_to_index = {
            "Aero - Spin vs Speed": 0,
            "Histograms": 1,
            "Confidence Distribution": 2,
            "GCS Spin Confidence": 3,
            "Rally Analysis": 4,
            "Aero - Drag Coefficient": 5,
            "Aero - Magnus Coefficient": 6,
            "Aero - Error Box Plots": 7,
            "Aero - Error Analysis": 8,
            "TCM - Input vs Output": 9,
            "TCM - Epsilon vs Vz": 10,
            "TCM - Fitness Box Plot": 11,
            "Game Analysis - Spin Observation": 12,
            "RCM - Spin Conversion": 13,
            "Game Analysis - Time Series": 14,
            "RCM - Input vs Output": 15,
            "RCM - Box Plots": 16,
            "RCM - Restitution": 17,
            "Game Analysis - Net Contacts": 18,
            "RCM - Contact Location": 19,
            "RCM - Output vs Others": 20,
            "RCM - Distributions": 21,
        }

        new_sizes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, half]  # 23 elements: 22 plots + flight at index 22
        left_index = view_to_index.get(self.current_left_view, 0)
        new_sizes[left_index] = half
        self.splitter.setSizes(new_sizes)
        QApplication.processEvents()

        # Create a temporary "shot" object with just these two segments for visualization
        # We'll use fs_post as the "main" segment but pass the combined data
        # Update metadata with contact info
        contact_metadata = dict(metadata) if metadata else {}
        contact_metadata['segment_start_time'] = fs_pre.data['time'].iloc[0] if 'time' in fs_pre.data.columns else 0

        # Set view to "Shot" to show the combined pre+post trajectory
        self.flight_plot.view_combo.setCurrentText("Shot")

        # Build rally DataFrame from rally object so the "Rally" view works
        rally_data = None
        if rally is not None:
            all_dfs = [seg.data for s in rally.shots for seg in s.flight_segments]
            if all_dfs:
                rally_data = pd.concat(all_dfs, ignore_index=False)

        # Plot the combined trajectory
        # We pass fs_post as the flight segment (for its structure) but provide combined_data as shot_data
        self.flight_plot.plot_flight_segment(
            fs_post,  # Main segment (post-contact)
            contact_metadata,
            combined_data,  # Combined pre+post data as "shot data"
            rally_data,  # Full rally data for "Rally" view
            shot,  # shot_object - for segment boundaries
            rally  # rally_object
        )

    def update_matches_list(self):
        """Update the matches list widget with loaded matches"""
        self.matches_list.clear()
        if self.match_collection is not None:
            print(f"Updating matches list with {len(self.match_collection.matches)} matches")
            for match in self.match_collection.matches:
                # Extract parent folder name from path
                folder_name = pathlib.Path(match.file).parent.name
                print(f"  Adding match: {folder_name} (file: {match.file})")
                item = QListWidgetItem(folder_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)  # All matches enabled by default
                self.matches_list.addItem(item)
            print(f"  Matches list now has {self.matches_list.count()} items")
        else:
            print("update_matches_list: match_collection is None")

    def on_match_selection_changed(self, item):
        """Handle match checkbox changes — debounce so rapid toggles coalesce."""
        self._selection_debounce.start()

    def _apply_selection_change(self):
        """Actually refresh the current view after match-selection debounce settles."""
        self._invalidate_extract_cache()  # selection changed → stale cache
        enabled_indices = self._get_enabled_indices()

        view = self.current_left_view
        with self._busy_status(f"Updating {view}…"):
            reload_map = {
                "Aero - Spin vs Speed":              lambda: self.update_scatter_plot(enabled_indices),
                "Histograms":                        lambda: self.update_histograms(enabled_indices),
                "Confidence Distribution":            lambda: self.update_confidence_distribution(enabled_indices),
                "GCS Spin Confidence":                lambda: self.update_gcs_spin_confidence(enabled_indices),
                "Rally Analysis":                     lambda: self.update_rally_analysis(enabled_indices),
                "Game Analysis - Time Series":        lambda: self.update_time_series_plot(enabled_indices),
                "Game Analysis - Net Contacts":       lambda: self.update_net_contact_plot(enabled_indices),
                "Game Analysis - Spin Observation":   lambda: self.update_spin_observation_plot(enabled_indices),
                "Aero - Drag Coefficient":            lambda: self.update_drag_coefficient_plot(enabled_indices),
                "Aero - Magnus Coefficient":          lambda: self.update_magnus_coefficient_plot(enabled_indices),
                "Aero - Error Box Plots":             lambda: self.update_aero_error_plot(enabled_indices),
                "Aero - Error Analysis":              lambda: self.update_conf_vs_error_plot(enabled_indices),
                "TCM - Input vs Output":              lambda: self.update_table_contact_plot(enabled_indices),
                "TCM - Epsilon vs Vz":                lambda: self.update_epsilon_vs_vz_plot(enabled_indices),
                "TCM - Fitness Box Plot":             lambda: self.update_fitness_box_plot(enabled_indices),
                "RCM - Spin Conversion":              lambda: self.update_spin_conversion_plot(enabled_indices),
                "RCM - Input vs Output":               lambda: self.update_rcm_scatter_plot(enabled_indices),
                "RCM - Box Plots":                     lambda: self.update_rcm_boxplot(enabled_indices),
                "RCM - Restitution":                   lambda: self.update_rcm_restitution_plot(enabled_indices),
                "RCM - Contact Location":              lambda: self.update_rcm_contact_location_plot(enabled_indices),
                "RCM - Output vs Others":              lambda: self.update_rcm_output_vs_others_plot(enabled_indices),
                "RCM - Distributions":                   lambda: self.update_rcm_histogram_plot(enabled_indices),
            }
            fn = reload_map.get(view)
            if fn is not None:
                fn()

    def _build_status_text(self, stats: dict) -> str:
        """Build status bar text from statistics, including gcs source warnings."""
        text = f"Loaded {stats['matches']} matches, {stats['games']} games, {stats['rallies']} rallies"
        gcs_sources = stats.get('gcs_sources', {})
        fallback_sources = {k: v for k, v in gcs_sources.items() if k not in ('gcs_offline', 'unknown')}
        none_count = gcs_sources.get('none', 0)
        if fallback_sources or none_count:
            parts = []
            for src, count in fallback_sources.items():
                parts.append(f"{count} rallies using {src}")
            if none_count:
                parts.append(f"{none_count} rallies with no gcs")
            text += f"  \u26a0 Spin fallback: {', '.join(parts)}"
        return text

    def load_hdf5_files(self):
        """Open file dialog and load HDF5 files (background thread)."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory Containing HDF5 Files", str(pathlib.Path.home())
        )

        if directory:
            data_folder = pathlib.Path(directory)
            self.status_label.setText(f"Loading files from {data_folder}...")
            self.load_button.setEnabled(False)
            self.rally_analysis_loaded = False  # Reset flag for new data

            self._load_thread = QThread()
            self._load_worker = _DataLoadWorker(
                self.data_processor, data_folder,
                gcs_fallback=getattr(self, '_gcs_fallback', False),
                clear_cache=self.clear_cache_checkbox.isChecked(),
            )
            self._load_worker.moveToThread(self._load_thread)

            self._load_thread.started.connect(self._load_worker.run)
            self._load_worker.progress.connect(self.status_label.setText)
            self._load_worker.finished.connect(self._on_data_loaded)
            self._load_worker.error.connect(
                lambda msg: (
                    QMessageBox.critical(self, "Error Loading Files", f"Failed to load HDF5 files:\n{msg}"),
                    self.status_label.setText("Error loading files"),
                    self.load_button.setEnabled(True),
                )
            )
            self._load_worker.finished.connect(self._load_thread.quit)
            self._load_worker.error.connect(self._load_thread.quit)
            self._load_thread.finished.connect(self._cleanup_load_thread)

            self._load_thread.start()

    def save_as_json(self):
        """Export the currently loaded rallies as a JSON manifest file."""
        if self.match_collection is None:
            QMessageBox.warning(self, "No Data", "Load data first before saving.")
            return

        json_file, _ = QFileDialog.getSaveFileName(
            self, "Save Rally Manifest as JSON",
            str(pathlib.Path.home() / "rally_selection.json"),
            "JSON Files (*.json);;All Files (*)",
        )
        if json_file:
            try:
                self.match_collection.to_json(
                    json_file,
                    relative_to=str(pathlib.Path(json_file).parent),
                )
                self.status_label.setText(f"Saved JSON manifest to {json_file}")
            except Exception as exc:
                QMessageBox.critical(
                    self, "Error Saving JSON",
                    f"Failed to save JSON manifest:\n{exc}",
                )

    def load_from_json(self):
        """Open file dialog for a JSON manifest and load specific rallies (background thread)."""
        json_file, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Rally Manifest",
            str(pathlib.Path.home()),
            "JSON Files (*.json);;All Files (*)",
        )

        if json_file:
            self.status_label.setText(f"Loading rallies from {json_file}...")
            self.load_button.setEnabled(False)
            self.load_json_button.setEnabled(False)
            self.rally_analysis_loaded = False

            self._load_thread = QThread()
            self._load_worker = _JsonLoadWorker(
                self.data_processor, json_file,
                gcs_fallback=getattr(self, '_gcs_fallback', False),
            )
            self._load_worker.moveToThread(self._load_thread)

            self._load_thread.started.connect(self._load_worker.run)
            self._load_worker.progress.connect(self.status_label.setText)
            self._load_worker.finished.connect(self._on_data_loaded)
            self._load_worker.error.connect(
                lambda msg: (
                    QMessageBox.critical(self, "Error Loading JSON", f"Failed to load rallies from JSON:\n{msg}"),
                    self.status_label.setText("Error loading JSON"),
                    self.load_button.setEnabled(True),
                    self.load_json_button.setEnabled(True),
                )
            )
            self._load_worker.finished.connect(self._load_thread.quit)
            self._load_worker.error.connect(self._load_thread.quit)
            self._load_thread.finished.connect(self._cleanup_load_thread)

            self._load_thread.start()

    def on_flight_segment_selected(
        self, flight_segment: FlightSegment, metadata: dict, shot_data, rally_data, shot_object, rally_object
    ):
        """Handle flight segment selection from scatter plot or coefficient plots"""
        # Check if right panel should be shown
        if not self.show_right_panel_checkbox.isChecked():
            return

        # Show the flight plot panel when a point is clicked
        self.flight_plot.show()

        # Get the total available width from the splitter widget itself
        total = self.splitter.width()
        if total <= 0:
            total = 1000

        half = total // 2
        view_to_index = {
            "Aero - Spin vs Speed": 0,
            "Histograms": 1,
            "Confidence Distribution": 2,
            "GCS Spin Confidence": 3,
            "Rally Analysis": 4,
            "Aero - Drag Coefficient": 5,
            "Aero - Magnus Coefficient": 6,
            "Aero - Error Box Plots": 7,
            "Aero - Error Analysis": 8,
            "TCM - Input vs Output": 9,
            "TCM - Epsilon vs Vz": 10,
            "TCM - Fitness Box Plot": 11,
            "Game Analysis - Spin Observation": 12,
            "RCM - Spin Conversion": 13,
            "Game Analysis - Time Series": 14,
            "RCM - Input vs Output": 15,
            "RCM - Box Plots": 16,
            "RCM - Restitution": 17,
            "Game Analysis - Net Contacts": 18,
            "RCM - Contact Location": 19,
            "RCM - Output vs Others": 20,
            "RCM - Distributions": 21,
        }

        new_sizes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, half]  # 23 elements: 22 plots + flight plot at index 22
        left_index = view_to_index.get(self.current_left_view, 0)
        new_sizes[left_index] = half

        self.splitter.setSizes(new_sizes)

        # Plot the data
        try:
            self.flight_plot.plot_flight_segment(flight_segment, metadata, shot_data, rally_data, shot_object, rally_object)
        except Exception as e:
            import traceback
            print(f"ERROR in plot_flight_segment: {e}")
            traceback.print_exc()


def _clear_cache(data_folder: "pathlib.Path | None") -> None:
    """Delete data_plotter pickle cache(s) and exit."""
    cache_name = ".data_plotter_cache.pkl"
    if data_folder is not None:
        # Single folder specified
        cache_file = pathlib.Path(data_folder) / cache_name
        if cache_file.exists():
            size_mb = cache_file.stat().st_size / 1e6
            cache_file.unlink()
            print(f"Deleted {cache_file}  ({size_mb:.1f} MB)")
        else:
            print(f"No cache found at {cache_file}")
    else:
        # Scan common data roots
        search_root = pathlib.Path.home() / "SonyAI" / "Data"
        if not search_root.is_dir():
            search_root = pathlib.Path.home()
        found = list(search_root.rglob(cache_name))
        if not found:
            print(f"No {cache_name} files found under {search_root}")
        for f in found:
            size_mb = f.stat().st_size / 1e6
            f.unlink()
            print(f"Deleted {f}  ({size_mb:.1f} MB)")
    print("Done.")


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Table Tennis Evaluation Data Plotter")
    parser.add_argument(
        "--gcs-fallback", action="store_true", default=False,
        help="Fall back to gcs_filtered / gcs when gcs_offline is not available. "
             "By default, rallies without gcs_offline are excluded.",
    )
    _CLEAR_CACHE_SENTINEL = object()
    parser.add_argument(
        "--clear-cache", metavar="DATA_FOLDER", type=pathlib.Path,
        nargs="?", const=None, default=_CLEAR_CACHE_SENTINEL,
        help="Delete the pickle cache for DATA_FOLDER and exit.  "
             "If DATA_FOLDER is omitted, remove every .data_plotter_cache.pkl "
             "found under ~/SonyAI/Data/.",
    )
    args, remaining = parser.parse_known_args()

    # ── Handle --clear-cache before starting the GUI ──────────────
    if args.clear_cache is not _CLEAR_CACHE_SENTINEL:
        _clear_cache(args.clear_cache)
        return

    app = QApplication(remaining)

    # Set style
    app.setStyle("Fusion")

    # Create and show main window
    window = MainWindow()
    window._gcs_fallback = args.gcs_fallback
    if args.gcs_fallback:
        print("GCS fallback mode enabled: will use gcs_filtered / gcs when gcs_offline is unavailable.")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
