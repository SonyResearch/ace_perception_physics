# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Spin Observation Analysis Plot Widget

Scatter plot comparing per-segment kinematic quantities on the X-axis
(velocity magnitude or spin magnitude) against spin-observation quality
metrics on the Y-axis (mean spin magnitude difference GT vs observation,
or variance of observed spin magnitude).
"""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSlider, QPushButton, QMessageBox,
)
from PySide6.QtCore import Signal, Qt, QTimer

import pathlib
from datetime import datetime

import pyqtgraph as pg
import matplotlib.pyplot as plt

from .base_coefficient_plot import create_range_slider, create_combo_selector, save_plot_as_image


class SpinObservationPlot(QWidget):
    """Scatter plot of kinematic X-axis vs spin observation quality Y-axis.

    X-axis choices:
        - Velocity Magnitude  (OPT at first point, m/s)
        - Spin Magnitude      (GT200 at first point, rad/s)

    Y-axis choices:
        - Mean |ΔSpin Magnitude|  (mean of |‖w_GT200‖ − ‖w_gcs‖| over samples)
        - Spin Magnitude Variance (variance of ‖w_gcs‖ over samples)
    """

    # Signal emitted when a point is clicked (same shape as other scatter plots)
    flight_segment_selected = Signal(object, object, object, object, object, object)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Debounce timer
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self.update_plot)

        # --- Control bar ---
        control_bar = QHBoxLayout()

        # X-axis selector
        xaxis_layout, self.xaxis_combo = create_combo_selector(
            "X Axis:", ["Velocity Magnitude", "Spin Magnitude", "X Position"], self.update_plot
        )
        control_bar.addLayout(xaxis_layout)

        # Y-axis selector
        yaxis_layout, self.yaxis_combo = create_combo_selector(
            "Y Axis:", ["Mean |ΔSpin Magnitude|", "Spin Magnitude Variance"], self.update_plot
        )
        control_bar.addLayout(yaxis_layout)

        # Color-by selector
        color_layout, self.color_combo = create_combo_selector(
            "Color By:", ["Confidence", "OPT RMSE", "Velocity Magnitude", "Spin Magnitude", "GCS Samples", "Spin Density", "|vx|", "X Position", "Y Position", "Z Position", "Shot Position", "GCS Source", "Contact Type"], self.update_plot
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
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(self.update_plot)
        size_layout.addWidget(self.size_slider)
        self.size_label = QLabel("5")
        self.size_label.setMinimumWidth(20)
        size_layout.addWidget(self.size_label)
        control_bar.addLayout(size_layout)

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.setMaximumWidth(80)
        self.save_button.clicked.connect(self.save_plot)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # --- Plot area with colorbar ---
        plot_layout = QHBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.plot_widget, stretch=20)

        self.colorbar_widget = pg.GraphicsLayoutWidget()
        self.colorbar_widget.setFixedWidth(80)
        self.colorbar_widget.setBackground('#1e1e1e')
        plot_layout.addWidget(self.colorbar_widget, stretch=1)

        layout.addLayout(plot_layout)

        # Info / stats labels
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

        # Data storage (set via plot_data)
        self._velocity_mag = None       # per-segment velocity magnitude (m/s)
        self._spin_mag_gt = None        # per-segment GT200 spin magnitude (rad/s)
        self._mean_delta_spin = None    # per-segment mean |‖w_GT‖ − ‖w_curr‖|
        self._spin_obs_var = None       # per-segment variance of ‖w_curr‖
        self._confidence = None
        self._opt_rmse = None           # per-segment OPT RMSE (meters)
        self._n_gcs_samples = None      # per-segment count of non-NaN wx_gcs
        self._spin_density = None       # per-segment spin density (%)
        self._shot_position = None      # per-segment: 0=other, 1=second-to-last, 2=last
        self._gcs_source = None         # per-segment GCS source tag (list of strings)
        self._vx_abs = None             # per-segment |vx_opt| at first point (m/s)
        self._shot_player = None        # per-segment player id (1=robot, 2=human)
        self._mean_x = None             # per-segment mean x_aps (m)
        self._mean_y = None             # per-segment mean y_aps (m)
        self._mean_z = None             # per-segment mean z_aps (m)
        self._contact_type = None       # per-segment trigger event type (str)
        self._metadata_list = None
        self._flight_segments = None
        self._shots = None
        self._rallies = None
        self._base_folder = None

        # Scatter item for click handling
        self._scatter = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot_data(
        self,
        velocity_mag: np.ndarray,
        spin_mag_gt: np.ndarray,
        mean_delta_spin: np.ndarray,
        spin_obs_var: np.ndarray,
        confidence: np.ndarray,
        opt_rmse: np.ndarray,
        n_gcs_samples: np.ndarray,
        spin_density: np.ndarray,
        is_last_shot: np.ndarray,
        gcs_source: list = None,
        vx_abs: np.ndarray = None,
        shot_player: np.ndarray = None,
        mean_x: np.ndarray = None,
        mean_y: np.ndarray = None,
        mean_z: np.ndarray = None,
        contact_type: list = None,
        metadata_list: list = None,
        flight_segments: list = None,
        shots: list = None,
        rallies: list = None,
        base_folder: str = None,
    ):
        self._velocity_mag = velocity_mag
        self._spin_mag_gt = spin_mag_gt
        self._mean_delta_spin = mean_delta_spin
        self._spin_obs_var = spin_obs_var
        self._confidence = confidence
        self._opt_rmse = opt_rmse
        self._n_gcs_samples = n_gcs_samples
        self._spin_density = spin_density
        self._shot_position = is_last_shot
        self._gcs_source = gcs_source if gcs_source is not None else []
        self._vx_abs = vx_abs
        self._shot_player = shot_player
        self._mean_x = mean_x
        self._mean_y = mean_y
        self._mean_z = mean_z
        self._contact_type = contact_type if contact_type is not None else []
        self._metadata_list = metadata_list
        self._flight_segments = flight_segments
        self._shots = shots
        self._rallies = rallies
        self._base_folder = base_folder
        self.update_plot()

    def save_plot(self):
        """Save the current spin observation plot as a PNG image."""
        if self._metadata_list is None or len(self._metadata_list) == 0:
            QMessageBox.warning(self, "No Data", "No plot data to save. Please load data first.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        x_tag = self.xaxis_combo.currentText().replace(" ", "_").lower()
        y_tag = self.yaxis_combo.currentText().replace(" ", "_").replace("|", "").lower()
        filename = f"spin_observation_{x_tag}_{y_tag}_{timestamp}.png"
        output_path = plots_dir / filename

        try:
            save_plot_as_image(self.plot_widget, str(output_path))
            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_filter_changed(self, _=None):
        # Update labels
        self.conf_min_label.setText(str(self.conf_min_slider.value()))
        self.conf_max_label.setText(str(self.conf_max_slider.value()))
        self.rmse_min_label.setText(str(self.rmse_min_slider.value()))
        self.rmse_label.setText(str(self.rmse_slider.value()))
        self._update_timer.start()

    def update_plot(self, _=None):
        self.plot_widget.clear()
        self.colorbar_widget.clear()
        # Remove any existing legend
        if hasattr(self.plot_widget.plotItem, 'legend') and self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.scene().removeItem(self.plot_widget.plotItem.legend)
            self.plot_widget.plotItem.legend = None
        self._scatter = None
        self.size_label.setText(str(self.size_slider.value()))

        if self._velocity_mag is None:
            self.info_label.setText("No data available")
            return

        # --- Read controls ---
        x_choice = self.xaxis_combo.currentText()
        y_choice = self.yaxis_combo.currentText()
        color_by = self.color_combo.currentText()
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0
        rmse_max = self.rmse_slider.value() / 1000.0  # mm -> meters
        point_size = self.size_slider.value()

        # --- Select X data ---
        if x_choice == "Velocity Magnitude":
            x_data = self._velocity_mag
            x_label = "Velocity Magnitude (m/s)"
        elif x_choice == "X Position":
            x_data = self._mean_x if self._mean_x is not None else np.zeros_like(self._velocity_mag)
            x_label = "Mean X Position (m)"
        else:
            x_data = self._spin_mag_gt
            x_label = "Spin Magnitude GT200 (rad/s)"

        # --- Select Y data ---
        if y_choice == "Mean |ΔSpin Magnitude|":
            y_data = self._mean_delta_spin
            y_label = "Mean |‖ω_GT‖ − ‖ω_obs‖| (rad/s)"
        else:
            y_data = self._spin_obs_var
            y_label = "Var(‖ω_obs‖) (rad²/s²)"

        # --- Build filter mask ---
        rmse_min = self.rmse_min_slider.value() / 1000.0  # mm -> meters
        rmse_valid = ~np.isnan(self._opt_rmse)
        rmse_filter = rmse_valid
        if self.rmse_slider.value() < self.rmse_slider.maximum():  # Only apply max if not at slider max
            rmse_filter = rmse_filter & (self._opt_rmse <= rmse_max)
        if self.rmse_min_slider.value() > 0:  # Only apply min if above 0
            rmse_filter = rmse_filter & (self._opt_rmse >= rmse_min)

        valid_mask = (
            ~np.isnan(x_data) &
            ~np.isnan(y_data) &
            ~np.isnan(self._confidence) &
            (self._confidence >= conf_min) &
            (self._confidence <= conf_max) &
            rmse_filter
        )

        if not np.any(valid_mask):
            self.info_label.setText("No data matching current filters")
            return

        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        conf_valid = self._confidence[valid_mask]
        rmse_valid_pts = self._opt_rmse[valid_mask] * 1000  # meters -> mm
        spin_valid = self._spin_mag_gt[valid_mask]
        gcs_samples_valid = self._n_gcs_samples[valid_mask].astype(float)
        spin_density_valid = self._spin_density[valid_mask]
        shot_pos_valid = self._shot_position[valid_mask]
        indices_valid = np.where(valid_mask)[0]
        n_points = len(x_valid)

        # --- Color mapping ---
        cmap = plt.get_cmap('turbo')
        use_categorical = False
        if color_by == "Shot Position":
            use_categorical = True
            # Three-category coloring
            brushes = []
            for i in range(n_points):
                if shot_pos_valid[i] == 2:
                    brushes.append(pg.mkBrush(255, 100, 50, 200))   # orange-red  = last shot
                elif shot_pos_valid[i] == 1:
                    brushes.append(pg.mkBrush(255, 220, 50, 200))   # yellow      = second-to-last
                else:
                    brushes.append(pg.mkBrush(50, 180, 255, 200))   # cyan        = other
            color_label = ""
            c_min, c_max = 0, 1
        elif color_by == "GCS Source":
            use_categorical = True
            # Color by which GCS data source the rally used
            gcs_valid = [self._gcs_source[i] if i < len(self._gcs_source) else 'unknown' for i in indices_valid]
            _gcs_colors = {
                'gcs_offline': pg.mkBrush(50, 200, 50, 200),    # green   = preferred
                'gcs_filtered': pg.mkBrush(255, 160, 50, 200),  # orange  = fallback
                'gcs': pg.mkBrush(255, 80, 80, 200),            # red     = raw / legacy
            }
            _gcs_default = pg.mkBrush(150, 150, 150, 200)       # gray    = unknown/none
            brushes = [_gcs_colors.get(src, _gcs_default) for src in gcs_valid]
            color_label = ""
            c_min, c_max = 0, 1
        elif color_by == "Contact Type":
            use_categorical = True
            ct_valid = [self._contact_type[i] if i < len(self._contact_type) else 'unknown' for i in indices_valid]
            _ct_colors = {
                'shot_p1': pg.mkBrush(220, 50, 50, 200),     # red = robot shot
                'bounce_p1': pg.mkBrush(255, 160, 50, 200),  # orange = robot bounce
                'net': pg.mkBrush(240, 220, 50, 200),        # yellow = net
                'bounce_p2': pg.mkBrush(50, 200, 80, 200),   # green = player bounce
                'shot_p2': pg.mkBrush(80, 180, 255, 200),    # light blue = player shot
                'start': pg.mkBrush(180, 80, 220, 200),      # purple = start
            }
            _ct_default = pg.mkBrush(150, 150, 150, 200)    # gray   = unknown
            brushes = [_ct_colors.get(ct, _ct_default) for ct in ct_valid]
            color_label = ""
            c_min, c_max = 0, 1
        elif color_by == "OPT RMSE":
            color_data = rmse_valid_pts
            color_label = "OPT RMSE (mm)"
            c_min, c_max = float(np.nanmin(rmse_valid_pts)), min(float(np.nanmax(rmse_valid_pts)), self.rmse_slider.value())
        elif color_by == "Velocity Magnitude":
            vel_valid = self._velocity_mag[valid_mask]
            color_data = vel_valid
            color_label = "Velocity Magnitude (m/s)"
            c_min, c_max = float(np.min(vel_valid)), float(np.max(vel_valid))
        elif color_by == "Spin Magnitude":
            color_data = spin_valid
            color_label = "Spin Magnitude (rad/s)"
            c_min, c_max = float(np.min(spin_valid)), float(np.max(spin_valid))
        elif color_by == "GCS Samples":
            color_data = gcs_samples_valid
            color_label = "# GCS Spin Samples"
            c_min, c_max = float(np.min(gcs_samples_valid)), float(np.max(gcs_samples_valid))
        elif color_by == "Spin Density":
            color_data = spin_density_valid
            color_label = "Spin Density (%)"
            c_min, c_max = 0.0, min(100.0, float(np.max(spin_density_valid)) * 1.05) if len(spin_density_valid) > 0 else 100.0
        elif color_by == "|vx|":
            vx_abs_valid = self._vx_abs[valid_mask] if self._vx_abs is not None else np.zeros(n_points)
            color_data = vx_abs_valid
            color_label = "|vx| (m/s)"
            c_min, c_max = float(np.min(vx_abs_valid)), float(np.max(vx_abs_valid)) if len(vx_abs_valid) > 0 else (0, 1)
        elif color_by == "X Position":
            pos_valid = self._mean_x[valid_mask] if self._mean_x is not None else np.zeros(n_points)
            color_data = pos_valid
            color_label = "Mean X (m)"
            c_min, c_max = (float(np.nanmin(pos_valid)), float(np.nanmax(pos_valid))) if len(pos_valid) > 0 else (0, 1)
        elif color_by == "Y Position":
            pos_valid = self._mean_y[valid_mask] if self._mean_y is not None else np.zeros(n_points)
            color_data = pos_valid
            color_label = "Mean Y (m)"
            c_min, c_max = (float(np.nanmin(pos_valid)), float(np.nanmax(pos_valid))) if len(pos_valid) > 0 else (0, 1)
        elif color_by == "Z Position":
            pos_valid = self._mean_z[valid_mask] if self._mean_z is not None else np.zeros(n_points)
            color_data = pos_valid
            color_label = "Mean Z (m)"
            c_min, c_max = (float(np.nanmin(pos_valid)), float(np.nanmax(pos_valid))) if len(pos_valid) > 0 else (0, 1)
        else:  # Confidence
            color_data = conf_valid
            color_label = "Confidence"
            c_min, c_max = conf_min, conf_max

        if not use_categorical:
            brushes = []
            for i in range(n_points):
                val = np.clip(color_data[i], c_min, c_max)
                norm = (val - c_min) / (c_max - c_min) if c_max > c_min else 0.5
                rgba = cmap(norm)
                r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                alpha = int(conf_valid[i] * 204 + 51)
                alpha = int(np.clip(alpha, 51, 255))
                brushes.append(pg.mkBrush(r, g, b, alpha))

        # --- Determine per-point symbol (robot=circle, player=triangle) ---
        player_valid = self._shot_player[valid_mask] if self._shot_player is not None else np.ones(n_points, dtype=int)

        # --- Create scatter ---
        spots = []
        for i in range(n_points):
            sym = 'o' if player_valid[i] == 1 else 't'  # circle=robot, triangle=player
            spots.append({
                'pos': (float(x_valid[i]), float(y_valid[i])),
                'size': point_size,
                'pen': pg.mkPen(color=(50, 50, 50), width=1),
                'brush': brushes[i],
                'symbol': sym,
                'data': int(indices_valid[i]),
            })

        scatter = pg.ScatterPlotItem(spots=spots, hoverable=True, tip=None)
        scatter.sigClicked.connect(self._on_point_clicked)
        self.plot_widget.addItem(scatter)
        self._scatter = scatter

        # --- Auto-scale axes (full data range with padding) ---
        self.plot_widget.enableAutoRange()

        # --- Colorbar / Legend ---
        if use_categorical:
            legend = self.plot_widget.addLegend(offset=(10, 10))
            if color_by == "Shot Position":
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(50,180,255,200), size=8, symbol='o'),
                    "Other Shots"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(255,220,50,200), size=8, symbol='o'),
                    "2nd-to-Last Shot"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(255,100,50,200), size=8, symbol='o'),
                    "Last Shot"
                )
            elif color_by == "GCS Source":
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(50,200,50,200), size=8, symbol='o'),
                    "gcs_offline (preferred)"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(255,160,50,200), size=8, symbol='o'),
                    "gcs_filtered (fallback)"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(255,80,80,200), size=8, symbol='o'),
                    "gcs (raw)"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(150,150,150,200), size=8, symbol='o'),
                    "unknown / none"
                )
            elif color_by == "Contact Type":
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(220,50,50,200), size=8, symbol='o'),
                    "shot_p1 (robot)"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(255,160,50,200), size=8, symbol='o'),
                    "bounce_p1 (robot)"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(240,220,50,200), size=8, symbol='o'),
                    "net"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(50,200,80,200), size=8, symbol='o'),
                    "bounce_p2 (player)"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(80,180,255,200), size=8, symbol='o'),
                    "shot_p2 (player)"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(180,80,220,200), size=8, symbol='o'),
                    "start"
                )
                legend.addItem(
                    pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(150,150,150,200), size=8, symbol='o'),
                    "unknown"
                )
        else:
            colormap = pg.colormap.get('turbo')
            self.colorbar_widget.addItem(pg.ColorBarItem(
                values=(c_min, c_max),
                colorMap=colormap,
                label=color_label,
                limits=(c_min, c_max),
            ))

        # --- Add symbol legend (robot vs player) ---
        if not use_categorical:
            # For continuous color modes, add a small legend for symbols only
            legend = self.plot_widget.addLegend(offset=(10, 10))
        else:
            # For categorical modes, the legend already exists — reuse it
            legend = self.plot_widget.plotItem.legend
        if legend is not None:
            legend.addItem(
                pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(180,180,180,200), size=8, symbol='o'),
                "Robot (P1)"
            )
            legend.addItem(
                pg.ScatterPlotItem([0], [0], pen=pg.mkPen(50,50,50,1), brush=pg.mkBrush(180,180,180,200), size=8, symbol='t'),
                "Player (P2)"
            )

        # --- Labels ---
        self.plot_widget.setTitle(f"{x_choice} vs {y_choice} (colored by {color_by})")
        self.plot_widget.setLabel('left', y_label)
        self.plot_widget.setLabel('bottom', x_label)

        mean_y = np.mean(y_valid)
        median_y = np.median(y_valid)
        corr = np.corrcoef(x_valid, y_valid)[0, 1] if n_points > 1 else 0
        self.info_label.setText(
            f"Showing {n_points} segments | "
            f"Conf: [{int(conf_min * 100)}, {int(conf_max * 100)}]% | "
            f"OPT RMSE: [{self.rmse_min_slider.value()}, {self.rmse_slider.value()}]mm"
        )
        self.stats_label.setText(
            f"n={n_points}, mean_y={mean_y:.3f}, median_y={median_y:.3f}, corr={corr:.3f}"
        )

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _on_point_clicked(self, plot, points):
        if len(points) == 0:
            return
        pt = points[0]
        idx = pt.data()
        if idx is None or self._flight_segments is None:
            return
        if idx < 0 or idx >= len(self._flight_segments):
            return

        fs = self._flight_segments[idx]
        metadata = self._metadata_list[idx] if self._metadata_list else None
        shot = self._shots[idx] if self._shots else None
        rally = self._rallies[idx] if self._rallies else None

        # Build shot data if available
        shot_data = None
        if shot is not None:
            import pandas as pd
            try:
                shot_data = pd.concat([seg.data for seg in shot.flight_segments], ignore_index=False)
            except Exception:
                pass

        # Build rally data if available
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
