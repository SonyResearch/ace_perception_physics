# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Table Contact Plot Widget

Widget for visualizing table contact model: pre/post contact velocity and spin.
Based on contacts_all.html visualization from compare_table_contact_datasets.py
Shows a 6x5 matrix of scatter plots.
"""

from typing import Dict, List, Tuple
import numpy as np
import pathlib
from datetime import datetime

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QMessageBox, QSlider, QGridLayout, QCheckBox
from PySide6.QtCore import Signal, Qt, QTimer

import pyqtgraph as pg
from .base_coefficient_plot import create_range_slider, create_combo_selector


def rotate_to_local_vectorized(vx_pre: np.ndarray, vy_pre: np.ndarray,
                                vectors_x: List[np.ndarray], vectors_y: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Vectorized rotation to local coordinate system where v_y = 0.

    Args:
        vx_pre, vy_pre: Pre-contact velocity x,y components (arrays)
        vectors_x, vectors_y: Lists of x,y components of vectors to rotate

    Returns:
        List of tuples (rotated_x, rotated_y) for each input vector pair
    """
    norm_xy = np.sqrt(vx_pre**2 + vy_pre**2)
    norm_xy = np.where(norm_xy < 1e-9, 1e-9, norm_xy)

    cos_theta = vx_pre / norm_xy  # u_x[0]
    sin_theta = vy_pre / norm_xy  # u_x[1]

    # Rotation: x' = cos*x + sin*y, y' = -sin*x + cos*y
    results = []
    for vx, vy in zip(vectors_x, vectors_y):
        rotated_x = cos_theta * vx + sin_theta * vy
        rotated_y = -sin_theta * vx + cos_theta * vy
        results.append((rotated_x, rotated_y))

    return results


def contact_model_residual_vectorized(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                                       wx: np.ndarray, wy: np.ndarray, wz: np.ndarray,
                                       epsilon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                      np.ndarray, np.ndarray, np.ndarray]:
    """
    Fully-vectorized contact model with residual correction.

    The residual model is: Nakashima base result minus a rotation-dependent
    correction.  The correction for each point is
        R^T @ M_corr @ R @ v   (and similarly for w)
    where R is a 2-D rotation in the XY plane.  By decomposing into
    "rotate to local → apply constant correction matrix → rotate back"
    the entire computation becomes element-wise numpy operations with
    no Python loop (~200× faster than the per-point loop).

    Returns:
        (vx_post, vy_post, vz_post, wx_post, wy_post, wz_post)
    """
    ball_radius = 0.02

    # ── Shared pre-computations (same as Nakashima) ──────────────
    ub_z = -vz
    ub_x = vx - ball_radius * wy
    ub_y = vy + ball_radius * wx
    ub_t = np.sqrt(ub_x**2 + ub_y**2)
    ub_t = np.where(ub_t == 0, 1e-9, ub_t)

    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = np.minimum(ap, 0.4)

    # ── Base Nakashima result (vectorized) ───────────────────────
    vx_nak = (1 - a) * vx + a * ball_radius * wy
    vy_nak = (1 - a) * vy - a * ball_radius * wx
    vz_nak = -epsilon * vz
    wx_nak = -1.5 * a / ball_radius * vy + (1 - 1.5 * a) * wx
    wy_nak = 1.5 * a / ball_radius * vx + (1 - 1.5 * a) * wy
    wz_nak = wz.copy() if isinstance(wz, np.ndarray) else np.full_like(vx, wz)

    # ── Rotation to local coordinates (vectorized) ───────────────
    norm_xy = np.sqrt(vx**2 + vy**2)
    norm_xy = np.where(norm_xy < 1e-9, 1e-9, norm_xy)
    cos_t = vx / norm_xy
    sin_t = vy / norm_xy

    # v_local = R @ v;  w_local = R @ w   (R is XY rotation, z unchanged)
    vlx = cos_t * vx + sin_t * vy
    vly = -sin_t * vx + cos_t * vy
    vlz = vz
    wlx = cos_t * wx + sin_t * wy
    wly = -sin_t * wx + cos_t * wy
    wlz = wz

    # ── Residual correction matrices (constants) ─────────────────
    a_u_corr = np.array([[-0.0108710967, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.0143258878, 0.0, 0.0]])
    b_u_corr = np.array([
        [-0.0000073021, 0.0005743815, 0.0000834545],
        [-0.0005860303, -0.0000306265, -0.0001665864],
        [-0.0000248477, -0.0001321961, -0.0000142774],
    ])
    a_o_corr = np.array([
        [-0.0735163773, 0.0, 0.0],
        [3.0883899621, 0.0, 3.0314342691],
        [-0.0382839144, 0.0, 0.2482747068]
    ])
    b_o_corr = np.array([
        [0.0133803508, 0.0022481343, 0.0110659132],
        [-0.0046491227, 0.0018867774, -0.0055119878],
        [0.0450644266, -0.0034796966, 0.0576889524],
    ])

    # ── Velocity correction: R^T @ (a_u_corr @ v_local + b_u_corr @ w_local)
    cv_xl = a_u_corr[0, 0] * vlx + a_u_corr[0, 1] * vly + a_u_corr[0, 2] * vlz \
          + b_u_corr[0, 0] * wlx + b_u_corr[0, 1] * wly + b_u_corr[0, 2] * wlz
    cv_yl = a_u_corr[1, 0] * vlx + a_u_corr[1, 1] * vly + a_u_corr[1, 2] * vlz \
          + b_u_corr[1, 0] * wlx + b_u_corr[1, 1] * wly + b_u_corr[1, 2] * wlz
    cv_zl = a_u_corr[2, 0] * vlx + a_u_corr[2, 1] * vly + a_u_corr[2, 2] * vlz \
          + b_u_corr[2, 0] * wlx + b_u_corr[2, 1] * wly + b_u_corr[2, 2] * wlz
    # Rotate back: R^T = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    corr_vx = cos_t * cv_xl - sin_t * cv_yl
    corr_vy = sin_t * cv_xl + cos_t * cv_yl
    corr_vz = cv_zl

    # ── Spin correction: R^T @ (a_o_corr @ v_local + b_o_corr @ w_local)
    cw_xl = a_o_corr[0, 0] * vlx + a_o_corr[0, 1] * vly + a_o_corr[0, 2] * vlz \
          + b_o_corr[0, 0] * wlx + b_o_corr[0, 1] * wly + b_o_corr[0, 2] * wlz
    cw_yl = a_o_corr[1, 0] * vlx + a_o_corr[1, 1] * vly + a_o_corr[1, 2] * vlz \
          + b_o_corr[1, 0] * wlx + b_o_corr[1, 1] * wly + b_o_corr[1, 2] * wlz
    cw_zl = a_o_corr[2, 0] * vlx + a_o_corr[2, 1] * vly + a_o_corr[2, 2] * vlz \
          + b_o_corr[2, 0] * wlx + b_o_corr[2, 1] * wly + b_o_corr[2, 2] * wlz
    corr_wx = cos_t * cw_xl - sin_t * cw_yl
    corr_wy = sin_t * cw_xl + cos_t * cw_yl
    corr_wz = cw_zl

    # ── Final: Nakashima base − correction ───────────────────────
    vx_post = vx_nak - corr_vx
    vy_post = vy_nak - corr_vy
    vz_post = vz_nak - corr_vz
    wx_post = wx_nak - corr_wx
    wy_post = wy_nak - corr_wy
    wz_post = wz_nak - corr_wz

    return vx_post, vy_post, vz_post, wx_post, wy_post, wz_post


def contact_model_residual0805_vectorized(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                                           wx: np.ndarray, wy: np.ndarray, wz: np.ndarray,
                                           epsilon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                          np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized contact model with Residual0805 correction (LassoCV, 30K dataset).
    """
    ball_radius = 0.02

    ub_z = -vz
    ub_x = vx - ball_radius * wy
    ub_y = vy + ball_radius * wx
    ub_t = np.sqrt(ub_x**2 + ub_y**2)
    ub_t = np.where(ub_t == 0, 1e-9, ub_t)
    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = np.minimum(ap, 0.4)

    vx_nak = (1 - a) * vx + a * ball_radius * wy
    vy_nak = (1 - a) * vy - a * ball_radius * wx
    vz_nak = -epsilon * vz
    wx_nak = -1.5 * a / ball_radius * vy + (1 - 1.5 * a) * wx
    wy_nak = 1.5 * a / ball_radius * vx + (1 - 1.5 * a) * wy
    wz_nak = wz.copy() if isinstance(wz, np.ndarray) else np.full_like(vx, wz)

    norm_xy = np.sqrt(vx**2 + vy**2)
    norm_xy = np.where(norm_xy < 1e-9, 1e-9, norm_xy)
    cos_t = vx / norm_xy
    sin_t = vy / norm_xy

    vlx = cos_t * vx + sin_t * vy
    vly = -sin_t * vx + cos_t * vy
    vlz = vz
    wlx = cos_t * wx + sin_t * wy
    wly = -sin_t * wx + cos_t * wy
    wlz = wz

    a_u_corr = np.array([[ 0.0,            0.0, 0.0],
                         [ 0.0,            0.0, 0.0033782948],
                         [-0.0234382114,   0.0, -0.0271705996]])
    b_u_corr = np.array([[ 0.0,           0.0003330260,  0.0],
                         [-0.0006939556, -0.0000542802, -0.0002126204],
                         [ 0.0000176814, -0.0000114849, -0.0000082278]])
    a_o_corr = np.array([[-0.6932390739, 0.0, -1.0298409270],
                         [ 0.5111362043, 0.0,  0.0],
                         [ 0.3403263781, 0.0,  0.0]])
    b_o_corr = np.array([[ 0.0645615370, 0.0003501476,  0.0119346861],
                         [ 0.0075608351, 0.0021084738,  0.0038912580],
                         [ 0.0031443158, 0.0029140437,  0.0341134788]])

    cv_xl = a_u_corr[0, 0] * vlx + a_u_corr[0, 1] * vly + a_u_corr[0, 2] * vlz \
          + b_u_corr[0, 0] * wlx + b_u_corr[0, 1] * wly + b_u_corr[0, 2] * wlz
    cv_yl = a_u_corr[1, 0] * vlx + a_u_corr[1, 1] * vly + a_u_corr[1, 2] * vlz \
          + b_u_corr[1, 0] * wlx + b_u_corr[1, 1] * wly + b_u_corr[1, 2] * wlz
    cv_zl = a_u_corr[2, 0] * vlx + a_u_corr[2, 1] * vly + a_u_corr[2, 2] * vlz \
          + b_u_corr[2, 0] * wlx + b_u_corr[2, 1] * wly + b_u_corr[2, 2] * wlz
    corr_vx = cos_t * cv_xl - sin_t * cv_yl
    corr_vy = sin_t * cv_xl + cos_t * cv_yl
    corr_vz = cv_zl

    cw_xl = a_o_corr[0, 0] * vlx + a_o_corr[0, 1] * vly + a_o_corr[0, 2] * vlz \
          + b_o_corr[0, 0] * wlx + b_o_corr[0, 1] * wly + b_o_corr[0, 2] * wlz
    cw_yl = a_o_corr[1, 0] * vlx + a_o_corr[1, 1] * vly + a_o_corr[1, 2] * vlz \
          + b_o_corr[1, 0] * wlx + b_o_corr[1, 1] * wly + b_o_corr[1, 2] * wlz
    cw_zl = a_o_corr[2, 0] * vlx + a_o_corr[2, 1] * vly + a_o_corr[2, 2] * vlz \
          + b_o_corr[2, 0] * wlx + b_o_corr[2, 1] * wly + b_o_corr[2, 2] * wlz
    corr_wx = cos_t * cw_xl - sin_t * cw_yl
    corr_wy = sin_t * cw_xl + cos_t * cw_yl
    corr_wz = cw_zl

    return vx_nak - corr_vx, vy_nak - corr_vy, vz_nak - corr_vz, wx_nak - corr_wx, wy_nak - corr_wy, wz_nak - corr_wz


def contact_model_pysr_vectorized(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                                  wx: np.ndarray, wy: np.ndarray, wz: np.ndarray,
                                  epsilon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                 np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized contact model with PySR symbolic residual correction.
    """
    ball_radius = 0.02

    ub_z = -vz
    ub_x = vx - ball_radius * wy
    ub_y = vy + ball_radius * wx
    ub_t = np.sqrt(ub_x**2 + ub_y**2)
    ub_t = np.where(ub_t == 0, 1e-9, ub_t)
    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = np.minimum(ap, 0.4)

    vx_nak = (1 - a) * vx + a * ball_radius * wy
    vy_nak = (1 - a) * vy - a * ball_radius * wx
    vz_nak = -epsilon * vz
    wx_nak = -1.5 * a / ball_radius * vy + (1 - 1.5 * a) * wx
    wy_nak = 1.5 * a / ball_radius * vx + (1 - 1.5 * a) * wy
    wz_nak = wz.copy() if isinstance(wz, np.ndarray) else np.full_like(vx, wz)

    norm_xy = np.sqrt(vx**2 + vy**2)
    norm_xy = np.where(norm_xy < 1e-9, 1e-9, norm_xy)
    cos_t = vx / norm_xy
    sin_t = vy / norm_xy

    vlx = cos_t * vx + sin_t * vy
    wlx = cos_t * wx + sin_t * wy
    wly = -sin_t * wx + cos_t * wy
    wlz = wz

    # PySR residual expressions (denormalized, real units)
    res_vrx = 0.000331228454808625 * wly - 0.00234328506995007
    res_vry = -2.59237126684253e-6 * wlx * wly + 0.000334357282237056 * wlx + 7.75028081762634e-6 * wly - 0.019965477425329
    res_vz = 0.0353459618428551 - 0.0185605475825958 * vlx
    res_wrx = (1.80864194820441 - 0.0140229379398023 * wly) * np.cos(0.00305392436946761 * wly - 0.393887197174651) - 1.56985386606186
    res_wry = 40.6577706332294 * np.cos(np.sin(0.00305392436946761 * wly - 0.393887197174651)) - 27.5348325673878
    res_wz = 0.0249048361999532 * wlz + 2.75505998061377

    # Rotate residuals back to world frame
    corr_vx = cos_t * res_vrx - sin_t * res_vry
    corr_vy = sin_t * res_vrx + cos_t * res_vry
    corr_vz = res_vz
    corr_wx = cos_t * res_wrx - sin_t * res_wry
    corr_wy = sin_t * res_wrx + cos_t * res_wry
    corr_wz = res_wz

    return vx_nak - corr_vx, vy_nak - corr_vy, vz_nak - corr_vz, wx_nak - corr_wx, wy_nak - corr_wy, wz_nak - corr_wz


def contact_model_0426_vectorized(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                                  wx: np.ndarray, wy: np.ndarray, wz: np.ndarray,
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized table contact model from commit 1757e3ec (2025-04-26).

    Pure Nakashima with velocity-dependent epsilon = vz * 0.02 + 0.98.
    No residual corrections.  Matches :func:`_0426_table_contact` in
    ``table_contacts.py``.
    """
    ball_radius = 0.02

    epsilon = vz * 0.02 + 0.98

    ub_z = -vz
    ub_x = vx - ball_radius * wy
    ub_y = vy + ball_radius * wx
    ub_t = np.sqrt(ub_x**2 + ub_y**2)
    ub_t = np.where(ub_t == 0, 1e-9, ub_t)
    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = np.minimum(ap, 0.4)

    vx_out = (1 - a) * vx + a * ball_radius * wy
    vy_out = (1 - a) * vy - a * ball_radius * wx
    vz_out = -epsilon * vz
    wx_out = -1.5 * a / ball_radius * vy + (1 - 1.5 * a) * wx
    wy_out = 1.5 * a / ball_radius * vx + (1 - 1.5 * a) * wy
    wz_out = wz.copy() if isinstance(wz, np.ndarray) else np.full_like(vx, wz)

    return vx_out, vy_out, vz_out, wx_out, wy_out, wz_out


class TableContactPlot(QWidget):
    """Widget for 6x5 matrix of scatter plots showing table contact data"""

    # Signal for selecting a contact event
    contact_selected = Signal(object, object, object, object)  # (contact_data, metadata, shot, rally)
    # Signal emitted when the "Use extracted TCM" checkbox changes state
    data_source_changed = Signal(bool)  # True = use extracted TCM, False = use HDF5

    # Matrix layout: 6 rows (post values) x 5 columns (pre values)
    POST_LABELS = ['vrx_post', 'vry_post', 'vz_post', 'wrx_post', 'wry_post', 'wz_post']
    PRE_LABELS = ['vrx_pre', 'vz_pre', 'wrx_pre', 'wry_pre', 'wz_pre']

    # Display names for axis labels
    POST_DISPLAY = ['vᵣₓ post', 'vᵣᵧ post', 'vz post', 'wᵣₓ post', 'wᵣᵧ post', 'wz post']
    PRE_DISPLAY = ['vᵣₓ pre', 'vz pre', 'wᵣₓ pre', 'wᵣᵧ pre', 'wz pre']

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Debounce timer for slider updates
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self.update_plot)

        # Top control bar
        control_bar = QHBoxLayout()

        # View type selector
        view_layout, self.view_combo = create_combo_selector(
            "View:",
            ["Data (pseudoGT)", "NakashimaITTF Model", "NakashimaPaper Model", "Residual Model", "Residual0805 Model", "PySR Model", "0426 Model", "Data + NakashimaITTF", "Data + NakashimaPaper", "Data + Residual", "Data + Residual0805", "Data + PySR", "Data + 0426", "Error: NakashimaITTF-Data", "Error: NakashimaPaper-Data", "Error: Residual-Data", "Error: Residual0805-Data", "Error: PySR-Data", "Error: 0426-Data"],
            self.update_plot
        )
        control_bar.addLayout(view_layout)

        # Color by selector
        color_layout, self.color_combo = create_combo_selector(
            "Color By:",
            ["a coefficient", "vz_pre", "Spin Magnitude", "Confidence"],
            self.update_plot
        )
        control_bar.addLayout(color_layout)

        # Velocity range slider
        vel_layout, self.vel_min_slider, self.vel_max_slider, self.vel_min_label, self.vel_max_label = \
            create_range_slider("vz Range (m/s):", -20, 0, -20, 0, 2, self.on_velocity_range_changed, "{}", 1.0, 25)
        control_bar.addLayout(vel_layout)

        # Spin range slider
        spin_layout, self.spin_min_slider, self.spin_max_slider, self.spin_min_label, self.spin_max_label = \
            create_range_slider("Spin Range (rad/s):", 0, 1000, 0, 1000, 100, self.on_spin_range_changed, "{}", 1.0, 35)
        control_bar.addLayout(spin_layout)

        # Confidence range slider (0.0 to 1.0, displayed as percentage)
        self._conf_layout, self.conf_min_slider, self.conf_max_slider, self.conf_min_label, self.conf_max_label = \
            create_range_slider("Confidence (%):", 0, 100, 50, 100, 10, self.on_confidence_range_changed, "{}", 1.0, 25)
        control_bar.addLayout(self._conf_layout)

        # Error filter (max RMSE vs OPT in mm)
        self._error_layout, self.error_min_slider, self.error_max_slider, self.error_min_label, self.error_max_label = \
            create_range_slider("Error vs OPT (mm):", 0, 100, 0, 100, 10, self.on_error_range_changed, "{:.0f}", 1.0, 35)
        control_bar.addLayout(self._error_layout)

        # Physics sanity-check filters
        self.vz_pre_checkbox = QCheckBox("vz_pre < 0 (falling)")
        self.vz_pre_checkbox.setChecked(True)
        self.vz_pre_checkbox.stateChanged.connect(self._update_timer.start)
        control_bar.addWidget(self.vz_pre_checkbox)

        self.vz_post_checkbox = QCheckBox("vz_post > 0 (rising)")
        self.vz_post_checkbox.setChecked(True)
        self.vz_post_checkbox.stateChanged.connect(self._update_timer.start)
        control_bar.addWidget(self.vz_post_checkbox)

        # Data source toggle: extracted_TCM.csv vs HDF5 iloc fallback
        self.use_extracted_tcm_checkbox = QCheckBox("Use extracted TCM")
        self.use_extracted_tcm_checkbox.setChecked(True)
        self.use_extracted_tcm_checkbox.setToolTip(
            "When checked, use pre/post values from extracted_TCM.csv (ODE endpoints).\n"
            "When unchecked, use iloc-based extraction from HDF5 aerodynamics data."
        )
        self.use_extracted_tcm_checkbox.stateChanged.connect(
            lambda state: self.data_source_changed.emit(bool(state))
        )
        control_bar.addWidget(self.use_extracted_tcm_checkbox)

        # Point size slider
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Point Size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(2)
        self.size_slider.setMaximum(15)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(self._update_timer.start)
        size_layout.addWidget(self.size_slider)
        self.size_label = QLabel("5")
        self.size_label.setMinimumWidth(20)
        size_layout.addWidget(self.size_label)
        control_bar.addLayout(size_layout)

        # Save button
        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setMaximumWidth(100)
        control_bar.addWidget(self.save_button)

        layout.addLayout(control_bar)

        # Info label
        self.info_label = QLabel("Showing table contact matrix: post (rows) vs pre (columns)")
        layout.addWidget(self.info_label)

        # Create grid layout for matrix of plots
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.grid_container, stretch=1)

        self.setLayout(layout)

        # Store data references
        self.contact_data = None
        self.plot_items = {}  # (row, col) -> PlotWidget
        self.scatter_items = {}  # (row, col) -> ScatterPlotItem
        self.stat_lines = {}  # (row, col) -> list of InfiniteLine items (median, ±1σ)
        self.stat_text_items = {}  # (row, col) -> pg.TextItem for median/std annotation

        # Initialize plot matrix
        self._create_plot_matrix()

    def _create_plot_matrix(self):
        """Create the 6x5 matrix of plot widgets"""
        self.plot_items = {}
        self.scatter_items = {}

        # Add column labels at top row
        for col, pre_label in enumerate(self.PRE_DISPLAY):
            label = QLabel(pre_label)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; font-size: 10pt;")
            self.grid_layout.addWidget(label, 0, col + 1)

        # Create plots
        for row, post_label in enumerate(self.POST_DISPLAY):
            # Row label
            label = QLabel(post_label)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; font-size: 10pt;")
            self.grid_layout.addWidget(label, row + 1, 0)

            for col in range(len(self.PRE_LABELS)):
                # Create individual PlotWidget for each cell
                plot = pg.PlotWidget()
                plot.setDefaultPadding(0.02)
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.getAxis('left').enableAutoSIPrefix(False)
                plot.getAxis('bottom').enableAutoSIPrefix(False)

                # Set axis labels on edge plots only
                if row == len(self.POST_LABELS) - 1:
                    plot.setLabel('bottom', self.PRE_DISPLAY[col])
                if col == 0:
                    plot.setLabel('left', self.POST_DISPLAY[row])

                # Enable auto-range
                plot.enableAutoRange(axis='x', enable=True)
                plot.enableAutoRange(axis='y', enable=True)

                self.plot_items[(row, col)] = plot
                self.grid_layout.addWidget(plot, row + 1, col + 1)

                # Create scatter item
                scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None))
                scatter.sigClicked.connect(self.on_point_clicked)
                plot.addItem(scatter)
                self.scatter_items[(row, col)] = scatter

        # Store filtered indices for click handling
        self.current_filtered_indices = None

        # Link axes after all plots are created
        # Link X axes for plots in the same column
        for col in range(len(self.PRE_LABELS)):
            first_plot = self.plot_items[(0, col)]
            for row in range(1, len(self.POST_LABELS)):
                self.plot_items[(row, col)].setXLink(first_plot)

        # Link Y axes for plots in the same row
        for row in range(len(self.POST_LABELS)):
            first_plot = self.plot_items[(row, 0)]
            for col in range(1, len(self.PRE_LABELS)):
                self.plot_items[(row, col)].setYLink(first_plot)

    def on_velocity_range_changed(self):
        """Handle velocity range slider changes"""
        vel_min = self.vel_min_slider.value()
        vel_max = self.vel_max_slider.value()

        if vel_min > vel_max:
            if self.sender() == self.vel_min_slider:
                self.vel_min_slider.setValue(vel_max)
                vel_min = vel_max
            else:
                self.vel_max_slider.setValue(vel_min)
                vel_max = vel_min

        self.vel_min_label.setText(str(vel_min))
        self.vel_max_label.setText(str(vel_max))
        self._update_timer.start()

    def on_spin_range_changed(self):
        """Handle spin range slider changes"""
        spin_min = round(self.spin_min_slider.value() / 10) * 10
        spin_max = round(self.spin_max_slider.value() / 10) * 10

        self.spin_min_slider.blockSignals(True)
        self.spin_max_slider.blockSignals(True)
        self.spin_min_slider.setValue(spin_min)
        self.spin_max_slider.setValue(spin_max)
        self.spin_min_slider.blockSignals(False)
        self.spin_max_slider.blockSignals(False)

        if spin_min > spin_max:
            if self.sender() == self.spin_min_slider:
                self.spin_min_slider.setValue(spin_max)
                spin_min = spin_max
            else:
                self.spin_max_slider.setValue(spin_min)
                spin_max = spin_min

        self.spin_min_label.setText(str(spin_min))
        self.spin_max_label.setText(str(spin_max))
        self._update_timer.start()

    def on_confidence_range_changed(self):
        """Handle confidence range slider changes"""
        conf_min = self.conf_min_slider.value()
        conf_max = self.conf_max_slider.value()

        if conf_min > conf_max:
            if self.sender() == self.conf_min_slider:
                self.conf_min_slider.setValue(conf_max)
                conf_min = conf_max
            else:
                self.conf_max_slider.setValue(conf_min)
                conf_max = conf_min

        self.conf_min_label.setText(str(conf_min))
        self.conf_max_label.setText(str(conf_max))
        self._update_timer.start()

    def on_error_range_changed(self):
        """Handle error (max RMSE vs OPT) range slider changes"""
        error_min = self.error_min_slider.value()
        error_max = self.error_max_slider.value()

        if error_min > error_max:
            if self.sender() == self.error_min_slider:
                self.error_min_slider.setValue(error_max)
                error_min = error_max
            else:
                self.error_max_slider.setValue(error_min)
                error_max = error_min

        self.error_min_label.setText(f"{error_min:.0f}")
        self.error_max_label.setText(f"{error_max:.0f}")
        self._update_timer.start()

    def update_plot(self):
        """Update all plots in the matrix based on current settings.

        Uses pre-computed physics model outputs and rotated coordinates from
        plot_data() — only filtering and rendering happens here.
        """
        import matplotlib.pyplot as plt

        # Update size label
        self.size_label.setText(str(self.size_slider.value()))

        # Clear all scatter items
        for scatter in self.scatter_items.values():
            scatter.clear()

        # Clear existing stat lines (median / ±1σ) and text labels
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

        if self.contact_data is None or len(self.contact_data.get('vx_pre', [])) == 0 or self._precomputed is None:
            self.info_label.setText("No contact data available")
            return

        data = self.contact_data
        pc = self._precomputed

        # Apply filters
        vz_min = self.vel_min_slider.value()
        vz_max = self.vel_max_slider.value()
        spin_min = self.spin_min_slider.value()
        spin_max = self.spin_max_slider.value()
        conf_min = self.conf_min_slider.value() / 100.0
        conf_max = self.conf_max_slider.value() / 100.0
        error_min = self.error_min_slider.value() / 1000.0
        error_max = self.error_max_slider.value() / 1000.0

        # Use pre-computed spin magnitude
        spin_mag_all = pc['spin_mag_all']

        # Get confidence data
        confidence = data.get('confidence', np.ones(len(data['vx_pre'])))
        max_rmse_opt = data.get('max_rmse_opt', np.full(len(data['vx_pre']), np.nan))

        # Build masks (these are fast numpy boolean ops)
        if self.error_max_slider.value() >= self.error_max_slider.maximum():
            error_mask = (max_rmse_opt >= error_min) | np.isnan(max_rmse_opt)
        else:
            error_mask = ((max_rmse_opt >= error_min) & (max_rmse_opt <= error_max)) | np.isnan(max_rmse_opt)

        if vz_min <= -20:
            vz_mask = (data['vz_pre'] <= vz_max)
        else:
            vz_mask = (data['vz_pre'] >= vz_min) & (data['vz_pre'] <= vz_max)

        if spin_max >= 1000:
            spin_mask = (spin_mag_all >= spin_min)
        else:
            spin_mask = (spin_mag_all >= spin_min) & (spin_mag_all <= spin_max)

        if conf_max >= 1.0:
            conf_mask = (confidence >= conf_min)
        else:
            conf_mask = (confidence >= conf_min) & (confidence <= conf_max)

        # Physics sanity-check filters
        if self.vz_pre_checkbox.isChecked():
            vz_pre_sanity = data['vz_pre'] < 0
        else:
            vz_pre_sanity = np.ones(len(data['vz_pre']), dtype=bool)

        if self.vz_post_checkbox.isChecked():
            vz_post_sanity = data['vz_post'] > 0
        else:
            vz_post_sanity = np.ones(len(data['vz_post']), dtype=bool)

        mask = vz_mask & spin_mask & conf_mask & error_mask & vz_pre_sanity & vz_post_sanity

        if not np.any(mask):
            self.info_label.setText("No data points in selected range")
            return

        filtered_indices = np.where(mask)[0]
        n_points = len(filtered_indices)

        # Slice pre-computed arrays with mask (fast numpy indexing, no recomputation)
        pre_data = {k: v[mask] for k, v in pc['pre_data'].items()}
        post_data_gt = {k: v[mask] for k, v in pc['post_data_gt'].items()}
        post_data_ittf = {k: v[mask] for k, v in pc['post_data_ittf'].items()} if pc['post_data_ittf'] is not None else None
        post_data_paper = {k: v[mask] for k, v in pc['post_data_paper'].items()} if pc['post_data_paper'] is not None else None
        post_data_residual = {k: v[mask] for k, v in pc['post_data_residual'].items()}
        post_data_residual0805 = {k: v[mask] for k, v in pc['post_data_residual0805'].items()}
        post_data_pysr = {k: v[mask] for k, v in pc['post_data_pysr'].items()}
        post_data_0426 = {k: v[mask] for k, v in pc['post_data_0426'].items()}
        a_coefficients = pc['a_coefficients'][mask]
        spin_magnitudes = spin_mag_all[mask]

        # Get color data
        view_type = self.view_combo.currentText()
        color_by = self.color_combo.currentText()
        if color_by == "a coefficient":
            color_data = a_coefficients
            color_min, color_max = 0.0, 0.4
        elif color_by == "vz_pre":
            color_data = pre_data['vz_pre']
            color_min, color_max = -15.0, 0.0
        elif color_by == "Confidence":
            color_data = confidence[mask]
            color_min, color_max = 0.0, 1.0
        else:  # Spin Magnitude
            color_data = spin_magnitudes
            color_min, color_max = 0.0, 800.0

        # Compute colors vectorized
        cmap = plt.get_cmap('turbo')
        color_data_clipped = np.clip(color_data, color_min, color_max)
        norm_vals = (color_data_clipped - color_min) / (color_max - color_min) if color_max > color_min else np.full(n_points, 0.5)
        rgba_colors = cmap(norm_vals)
        rgb_data = (rgba_colors[:, :3] * 255).astype(np.int32)

        point_size = self.size_slider.value()

        # Plot data based on view type - using efficient array-based setData
        for row, post_key in enumerate(self.POST_LABELS):
            for col, pre_key in enumerate(self.PRE_LABELS):
                scatter = self.scatter_items[(row, col)]
                x_data = pre_data[pre_key]

                if view_type == "Data (pseudoGT)":
                    y_data = post_data_gt[post_key]
                    scatter.setData(
                        x=x_data, y=y_data,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "NakashimaITTF Model":
                    if post_data_ittf is not None:
                        y_data = post_data_ittf[post_key]
                    else:
                        y_data = post_data_gt[post_key]
                    scatter.setData(
                        x=x_data, y=y_data,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "NakashimaPaper Model":
                    if post_data_paper is not None:
                        y_data = post_data_paper[post_key]
                    else:
                        y_data = post_data_gt[post_key]
                    scatter.setData(
                        x=x_data, y=y_data,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "Residual Model":
                    y_data = post_data_residual[post_key]
                    scatter.setData(
                        x=x_data, y=y_data,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "Residual0805 Model":
                    y_data = post_data_residual0805[post_key]
                    scatter.setData(
                        x=x_data, y=y_data,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "PySR Model":
                    y_data = post_data_pysr[post_key]
                    scatter.setData(
                        x=x_data, y=y_data,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "0426 Model":
                    y_data = post_data_0426[post_key]
                    scatter.setData(
                        x=x_data, y=y_data,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "Data + NakashimaITTF":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_ittf[post_key] if post_data_ittf is not None else y_data_gt
                    x_combined = np.concatenate([x_data, x_data])
                    y_combined = np.concatenate([y_data_gt, y_data_model])
                    sizes = np.concatenate([np.full(n_points, point_size), np.full(n_points, max(point_size - 1, 2))])
                    symbols = ['o'] * n_points + ['x'] * n_points
                    brushes = ([pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)] +
                              [pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 100) for i in range(n_points)])
                    pens = ([pg.mkPen(color=(50, 50, 50), width=0.5)] * n_points +
                           [pg.mkPen(color=(100, 100, 100), width=1)] * n_points)
                    data_indices = np.concatenate([filtered_indices, filtered_indices])
                    scatter.setData(x=x_combined, y=y_combined, size=sizes, symbol=symbols, brush=brushes, pen=pens, data=data_indices)

                elif view_type == "Data + NakashimaPaper":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_paper[post_key] if post_data_paper is not None else y_data_gt
                    x_combined = np.concatenate([x_data, x_data])
                    y_combined = np.concatenate([y_data_gt, y_data_model])
                    sizes = np.concatenate([np.full(n_points, point_size), np.full(n_points, max(point_size - 1, 2))])
                    symbols = ['o'] * n_points + ['x'] * n_points
                    brushes = ([pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)] +
                              [pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 100) for i in range(n_points)])
                    pens = ([pg.mkPen(color=(50, 50, 50), width=0.5)] * n_points +
                           [pg.mkPen(color=(100, 100, 100), width=1)] * n_points)
                    data_indices = np.concatenate([filtered_indices, filtered_indices])
                    scatter.setData(x=x_combined, y=y_combined, size=sizes, symbol=symbols, brush=brushes, pen=pens, data=data_indices)

                elif view_type == "Data + Residual":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_residual[post_key]
                    x_combined = np.concatenate([x_data, x_data])
                    y_combined = np.concatenate([y_data_gt, y_data_model])
                    sizes = np.concatenate([np.full(n_points, point_size), np.full(n_points, max(point_size - 1, 2))])
                    symbols = ['o'] * n_points + ['x'] * n_points
                    brushes = ([pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)] +
                              [pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 100) for i in range(n_points)])
                    pens = ([pg.mkPen(color=(50, 50, 50), width=0.5)] * n_points +
                           [pg.mkPen(color=(100, 100, 100), width=1)] * n_points)
                    data_indices = np.concatenate([filtered_indices, filtered_indices])
                    scatter.setData(x=x_combined, y=y_combined, size=sizes, symbol=symbols, brush=brushes, pen=pens, data=data_indices)

                elif view_type == "Data + Residual0805":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_residual0805[post_key]
                    x_combined = np.concatenate([x_data, x_data])
                    y_combined = np.concatenate([y_data_gt, y_data_model])
                    sizes = np.concatenate([np.full(n_points, point_size), np.full(n_points, max(point_size - 1, 2))])
                    symbols = ['o'] * n_points + ['x'] * n_points
                    brushes = ([pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)] +
                              [pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 100) for i in range(n_points)])
                    pens = ([pg.mkPen(color=(50, 50, 50), width=0.5)] * n_points +
                           [pg.mkPen(color=(100, 100, 100), width=1)] * n_points)
                    data_indices = np.concatenate([filtered_indices, filtered_indices])
                    scatter.setData(x=x_combined, y=y_combined, size=sizes, symbol=symbols, brush=brushes, pen=pens, data=data_indices)

                elif view_type == "Data + PySR":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_pysr[post_key]
                    x_combined = np.concatenate([x_data, x_data])
                    y_combined = np.concatenate([y_data_gt, y_data_model])
                    sizes = np.concatenate([np.full(n_points, point_size), np.full(n_points, max(point_size - 1, 2))])
                    symbols = ['o'] * n_points + ['x'] * n_points
                    brushes = ([pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)] +
                              [pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 100) for i in range(n_points)])
                    pens = ([pg.mkPen(color=(50, 50, 50), width=0.5)] * n_points +
                           [pg.mkPen(color=(100, 100, 100), width=1)] * n_points)
                    data_indices = np.concatenate([filtered_indices, filtered_indices])
                    scatter.setData(x=x_combined, y=y_combined, size=sizes, symbol=symbols, brush=brushes, pen=pens, data=data_indices)

                elif view_type == "Data + 0426":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_0426[post_key]
                    x_combined = np.concatenate([x_data, x_data])
                    y_combined = np.concatenate([y_data_gt, y_data_model])
                    sizes = np.concatenate([np.full(n_points, point_size), np.full(n_points, max(point_size - 1, 2))])
                    symbols = ['o'] * n_points + ['x'] * n_points
                    brushes = ([pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)] +
                              [pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 100) for i in range(n_points)])
                    pens = ([pg.mkPen(color=(50, 50, 50), width=0.5)] * n_points +
                           [pg.mkPen(color=(100, 100, 100), width=1)] * n_points)
                    data_indices = np.concatenate([filtered_indices, filtered_indices])
                    scatter.setData(x=x_combined, y=y_combined, size=sizes, symbol=symbols, brush=brushes, pen=pens, data=data_indices)

                elif view_type == "Error: NakashimaITTF-Data":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_ittf[post_key] if post_data_ittf is not None else y_data_gt
                    y_error = y_data_model - y_data_gt
                    scatter.setData(
                        x=x_data, y=y_error,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "Error: NakashimaPaper-Data":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_paper[post_key] if post_data_paper is not None else y_data_gt
                    y_error = y_data_model - y_data_gt
                    scatter.setData(
                        x=x_data, y=y_error,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "Error: Residual-Data":
                    y_data_gt = post_data_gt[post_key]
                    y_data_model = post_data_residual[post_key]
                    y_error = y_data_model - y_data_gt
                    scatter.setData(
                        x=x_data, y=y_error,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "Error: Residual0805-Data":
                    y_data_gt = post_data_gt[post_key]
                    y_error = post_data_residual0805[post_key] - y_data_gt
                    scatter.setData(
                        x=x_data, y=y_error,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                elif view_type == "Error: PySR-Data":
                    y_data_gt = post_data_gt[post_key]
                    y_error = post_data_pysr[post_key] - y_data_gt
                    scatter.setData(
                        x=x_data, y=y_error,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                else:  # Error: 0426-Data
                    y_data_gt = post_data_gt[post_key]
                    y_error = post_data_0426[post_key] - y_data_gt
                    scatter.setData(
                        x=x_data, y=y_error,
                        size=point_size,
                        pen=pg.mkPen(color=(50, 50, 50), width=0.5),
                        brush=[pg.mkBrush(int(rgb_data[i, 0]), int(rgb_data[i, 1]), int(rgb_data[i, 2]), 180) for i in range(n_points)],
                        data=filtered_indices
                    )

                # ── Add median and ±1σ horizontal lines (error views only) ──
                if view_type.startswith("Error:"):
                    if view_type.startswith("Error: NakashimaITTF"):
                        _model = post_data_ittf if post_data_ittf is not None else post_data_gt
                        _y_stat = _model[post_key] - post_data_gt[post_key]
                    elif view_type.startswith("Error: NakashimaPaper"):
                        _model = post_data_paper if post_data_paper is not None else post_data_gt
                        _y_stat = _model[post_key] - post_data_gt[post_key]
                    elif view_type.startswith("Error: Residual0805"):
                        _y_stat = post_data_residual0805[post_key] - post_data_gt[post_key]
                    elif view_type.startswith("Error: PySR"):
                        _y_stat = post_data_pysr[post_key] - post_data_gt[post_key]
                    elif view_type.startswith("Error: 0426"):
                        _y_stat = post_data_0426[post_key] - post_data_gt[post_key]
                    else:
                        _y_stat = post_data_residual[post_key] - post_data_gt[post_key]

                    if len(_y_stat) > 0:
                        _med = float(np.nanmedian(_y_stat))
                        _std = float(np.nanstd(_y_stat))
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

                        # Show numeric labels next to each line (first column only)
                        if col == 0:
                            vb = pw.getViewBox()
                            vr = vb.viewRange()
                            _x_right = vr[0][1]  # right edge of view
                            _texts = []
                            for _label, _ypos in [
                                (f'med={_med:.4g}', _med),
                                (f'+σ={_med + _std:.4g}', _med + _std),
                                (f'−σ={_med - _std:.4g}', _med - _std),
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

        self.info_label.setText(f"Showing {n_points} contacts | vz: [{vz_min}, {vz_max}] m/s | spin: [{spin_min}, {spin_max}] rad/s | conf: [{int(conf_min*100)}, {int(conf_max*100)}]% | View: {view_type}")

        # Store filtered indices for click handling
        self.current_filtered_indices = filtered_indices

    def on_point_clicked(self, plot_item, points):
        """Handle click on scatter plot point"""
        if len(points) == 0:
            return

        point = points[0]

        # Get the index from the point's data attribute
        point_data = point.data()
        if point_data is None:
            return

        # point_data is the original index in the contact_data arrays
        idx = int(point_data)

        if self.contact_data is None:
            return

        # Get the metadata, shot, and rally for this contact
        metadata_list = self.contact_data.get('metadata_list', [])
        shot_list = self.contact_data.get('shot_list', [])
        rally_list = self.contact_data.get('rally_list', [])
        fs_pre_list = self.contact_data.get('fs_pre_list', [])
        fs_post_list = self.contact_data.get('fs_post_list', [])

        if idx >= len(metadata_list):
            return

        metadata = metadata_list[idx]
        shot = shot_list[idx] if idx < len(shot_list) else None
        rally = rally_list[idx] if idx < len(rally_list) else None
        fs_pre = fs_pre_list[idx] if idx < len(fs_pre_list) else None
        fs_post = fs_post_list[idx] if idx < len(fs_post_list) else None

        # Build contact data for this single point
        contact_point_data = {
            'vx_pre': self.contact_data['vx_pre'][idx],
            'vy_pre': self.contact_data['vy_pre'][idx],
            'vz_pre': self.contact_data['vz_pre'][idx],
            'wx_pre': self.contact_data['wx_pre'][idx],
            'wy_pre': self.contact_data['wy_pre'][idx],
            'wz_pre': self.contact_data['wz_pre'][idx],
            'vx_post': self.contact_data['vx_post'][idx],
            'vy_post': self.contact_data['vy_post'][idx],
            'vz_post': self.contact_data['vz_post'][idx],
            'wx_post': self.contact_data['wx_post'][idx],
            'wy_post': self.contact_data['wy_post'][idx],
            'wz_post': self.contact_data['wz_post'][idx],
            'fs_pre': fs_pre,
            'fs_post': fs_post,
        }

        # Emit signal with contact data, metadata, shot, and rally
        self.contact_selected.emit(contact_point_data, metadata, shot, rally)

    def plot_data(self, contact_data: Dict[str, np.ndarray]):
        """
        Plot contact data.

        Args:
            contact_data: Dictionary with keys:
                - vx_pre, vy_pre, vz_pre: Pre-contact velocity
                - wx_pre, wy_pre, wz_pre: Pre-contact spin
                - vx_post, vy_post, vz_post: Post-contact velocity (pseudoGT)
                - wx_post, wy_post, wz_post: Post-contact spin (pseudoGT)
                - metadata_list: List of metadata dicts
                - shot_list, rally_list: Optional references
        """
        self.contact_data = contact_data

        # ── Pre-compute all data-dependent quantities once ──
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

            # Spin magnitude (for filtering)
            spin_mag_all = np.sqrt(wx_pre**2 + wy_pre**2 + wz_pre**2)

            # Epsilon (data-dependent, not slider-dependent)
            epsilon = vz_pre * 0.02 + 0.98

            # a-coefficients
            ball_radius = 0.02
            ub_x = vx_pre - ball_radius * wy_pre
            ub_y = vy_pre + ball_radius * wx_pre
            ub_t = np.sqrt(ub_x**2 + ub_y**2)
            ub_t = np.where(ub_t < 1e-9, 1e-9, ub_t)
            ub_z = -vz_pre
            ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
            a_coefficients = np.minimum(ap, 2.0 / 5.0)

            # Rotate to local coordinates (GT data)
            rotated_pre = rotate_to_local_vectorized(vx_pre, vy_pre, [vx_pre, wx_pre], [vy_pre, wy_pre])
            vrx_pre, vry_pre = rotated_pre[0]
            wrx_pre, wry_pre = rotated_pre[1]

            rotated_post = rotate_to_local_vectorized(vx_pre, vy_pre, [vx_post, wx_post], [vy_post, wy_post])
            vrx_post, vry_post = rotated_post[0]
            wrx_post, wry_post = rotated_post[1]

            # Pre-data dicts (data-only, not slider-dependent)
            pre_data = {
                'vrx_pre': vrx_pre, 'vz_pre': vz_pre,
                'wrx_pre': wrx_pre, 'wry_pre': wry_pre, 'wz_pre': wz_pre,
            }
            post_data_gt = {
                'vrx_post': vrx_post, 'vry_post': vry_post, 'vz_post': vz_post,
                'wrx_post': wrx_post, 'wry_post': wry_post, 'wz_post': wz_post,
            }

            # Pre-compute Residual model (expensive, computed ONCE here)
            # Epsilon for residual model uses velocity-dependent formula
            epsilon = vz_pre * 0.02 + 0.98

            vx_res, vy_res, vz_res, wx_res, wy_res, wz_res = contact_model_residual_vectorized(
                vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre, epsilon
            )
            rotated_res = rotate_to_local_vectorized(vx_pre, vy_pre, [vx_res, wx_res], [vy_res, wy_res])
            post_data_residual = {
                'vrx_post': rotated_res[0][0], 'vry_post': rotated_res[0][1],
                'vz_post': vz_res,
                'wrx_post': rotated_res[1][0], 'wry_post': rotated_res[1][1],
                'wz_post': wz_res,
            }

            # Pre-compute Residual0805 model
            vx_r05, vy_r05, vz_r05, wx_r05, wy_r05, wz_r05 = contact_model_residual0805_vectorized(
                vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre, epsilon
            )
            rotated_r05 = rotate_to_local_vectorized(vx_pre, vy_pre, [vx_r05, wx_r05], [vy_r05, wy_r05])
            post_data_residual0805 = {
                'vrx_post': rotated_r05[0][0], 'vry_post': rotated_r05[0][1],
                'vz_post': vz_r05,
                'wrx_post': rotated_r05[1][0], 'wry_post': rotated_r05[1][1],
                'wz_post': wz_r05,
            }

            # Pre-compute PySR model
            vx_psr, vy_psr, vz_psr, wx_psr, wy_psr, wz_psr = contact_model_pysr_vectorized(
                vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre, epsilon
            )
            rotated_psr = rotate_to_local_vectorized(vx_pre, vy_pre, [vx_psr, wx_psr], [vy_psr, wy_psr])
            post_data_pysr = {
                'vrx_post': rotated_psr[0][0], 'vry_post': rotated_psr[0][1],
                'vz_post': vz_psr,
                'wrx_post': rotated_psr[1][0], 'wry_post': rotated_psr[1][1],
                'wz_post': wz_psr,
            }

            # Pre-compute 0426 model (velocity-dependent epsilon)
            vx_0426, vy_0426, vz_0426, wx_0426, wy_0426, wz_0426 = contact_model_0426_vectorized(
                vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre
            )
            rotated_0426 = rotate_to_local_vectorized(vx_pre, vy_pre, [vx_0426, wx_0426], [vy_0426, wy_0426])
            post_data_0426 = {
                'vrx_post': rotated_0426[0][0], 'vry_post': rotated_0426[0][1],
                'vz_post': vz_0426,
                'wrx_post': rotated_0426[1][0], 'wry_post': rotated_0426[1][1],
                'wz_post': wz_0426,
            }

            # HDF5-based NakashimaITTF and NakashimaPaper predictions
            post_data_ittf = None
            post_data_paper = None
            for model_suffix, attr_name in [('ittf', 'post_data_ittf'), ('paper', 'post_data_paper')]:
                vx_m = contact_data.get(f'vx_post_{model_suffix}')
                if vx_m is not None and len(vx_m) == len(vx_pre):
                    vy_m = contact_data[f'vy_post_{model_suffix}']
                    vz_m = contact_data[f'vz_post_{model_suffix}']
                    wx_m = contact_data[f'wx_post_{model_suffix}']
                    wy_m = contact_data[f'wy_post_{model_suffix}']
                    wz_m = contact_data[f'wz_post_{model_suffix}']
                    rotated_m = rotate_to_local_vectorized(vx_pre, vy_pre, [vx_m, wx_m], [vy_m, wy_m])
                    d = {
                        'vrx_post': rotated_m[0][0], 'vry_post': rotated_m[0][1],
                        'vz_post': vz_m,
                        'wrx_post': rotated_m[1][0], 'wry_post': rotated_m[1][1],
                        'wz_post': wz_m,
                    }
                    if attr_name == 'post_data_ittf':
                        post_data_ittf = d
                    else:
                        post_data_paper = d

            self._precomputed = {
                'spin_mag_all': spin_mag_all,
                'a_coefficients': a_coefficients,
                'pre_data': pre_data,
                'post_data_gt': post_data_gt,
                'post_data_ittf': post_data_ittf,
                'post_data_paper': post_data_paper,
                'post_data_residual': post_data_residual,
                'post_data_residual0805': post_data_residual0805,
                'post_data_pysr': post_data_pysr,
                'post_data_0426': post_data_0426,
            }

        self.update_plot()

    def save_plot(self):
        """Save the current plot matrix as a PNG image"""
        if self.contact_data is None:
            QMessageBox.warning(self, "No Data", "No plot data to save.")
            return

        plots_dir = pathlib.Path(getattr(self, 'plots_folder', None) or '.')
        plots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"table_contact_matrix_{timestamp}.png"
        output_path = plots_dir / filename

        try:
            # Grab the grid container as a screenshot
            pixmap = self.grid_container.grab()
            pixmap.save(str(output_path), "PNG")
            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
