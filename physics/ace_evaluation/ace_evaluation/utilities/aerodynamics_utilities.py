# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""Aerodynamics utility functions for magnus coefficient estimation."""
import numpy as np


def get_magnus_estimate_v2(v, w):
    """
    [V2] Data-fitted Magnus coefficient estimate using piecewise linear + quadratic model.

    For each velocity, the model is:
    - Linear region (w <= w_break): C_M = m1 * w + c1
    - Quadratic region (w > w_break): C_M = a * w^2 + b * w + c

    Parameters are interpolated between reference velocities (2.0, 3.5, 7.5, 10.5, 13.5, 17.0 m/s).
    For v < 2.0, linearly extrapolate using the 2.0→3.5 slope.

    Fitted from user-provided data points:
    - v=2.0:  (10, 0.08), (150, 0.08), (300, 0.055), (450, 0.025)
    - v=3.5:  (100, 0.2), (200, 0.09), (300, 0.08), (500, 0.05)
    - v=7.5:  (150, 0.25), (350, 0.09), (500, 0.095), (750, 0.075)
    - v=10.5: (250, 0.21), (440, 0.085), (600, 0.1), (800, 0.095)
    - v=13.5: (300, 0.215), (550, 0.075), (750, 0.105), (900, 0.095)
    - v=17.0: (360, 0.21), (650, 0.08), (750, 0.095), (900, 0.11)

    Args:
        v: velocity in m/s (scalar or array)
        w: angular velocity in rad/s (scalar or array)

    Returns:
        C_M: magnus coefficient (scalar or array matching input shape)
    """
    # Convert to numpy arrays
    v_arr = np.asarray(v, dtype=float)
    w_arr = np.asarray(w, dtype=float)

    # Track scalar input
    scalar_input = (v_arr.ndim == 0) and (w_arr.ndim == 0)
    if scalar_input:
        v_arr = np.array([v_arr])
        w_arr = np.array([w_arr])

    # Broadcast arrays to same shape
    v_arr, w_arr = np.broadcast_arrays(v_arr, w_arr)
    original_shape = v_arr.shape
    v_flat = v_arr.ravel()
    w_flat = w_arr.ravel()

    # Reference velocities and their fitted parameters
    v_refs = np.array([2.0, 3.5, 7.5, 10.5, 13.5, 17.0])

    # Linear region parameters (m1, c1)
    m1_refs = np.array([0.0, -0.0011, -0.0008, -0.000658, -0.00056, -0.000448])
    c1_refs = np.array([0.08, 0.31, 0.37, 0.375, 0.383, 0.371])

    # Break point (second data point for each velocity)
    w_break_refs = np.array([150.0, 200.0, 350.0, 440.0, 550.0, 650.0])

    # Quadratic region parameters (a, b, c)
    def solve_quadratic_coeffs(pts):
        """Solve for a, b, c given 3 points [(w1,cm1), (w2,cm2), (w3,cm3)]"""
        w = np.array([p[0] for p in pts])
        cm = np.array([p[1] for p in pts])
        A = np.column_stack([w**2, w, np.ones_like(w)])
        return np.linalg.solve(A, cm)

    # Data points for quadratic region (last 3 points for each velocity)
    quad_pts = {
        2.0:  [(150, 0.08), (300, 0.055), (450, 0.025)],
        3.5:  [(200, 0.09), (300, 0.08), (500, 0.05)],
        7.5:  [(350, 0.09), (500, 0.095), (750, 0.075)],
        10.5: [(440, 0.085), (600, 0.1), (800, 0.095)],
        13.5: [(550, 0.075), (750, 0.105), (900, 0.095)],
        17.0: [(650, 0.08), (750, 0.095), (900, 0.11)],
    }

    # Compute coefficients at reference velocities
    coeffs_20 = solve_quadratic_coeffs(quad_pts[2.0])
    coeffs_35 = solve_quadratic_coeffs(quad_pts[3.5])
    coeffs_75 = solve_quadratic_coeffs(quad_pts[7.5])
    coeffs_105 = solve_quadratic_coeffs(quad_pts[10.5])
    coeffs_135 = solve_quadratic_coeffs(quad_pts[13.5])
    coeffs_170 = solve_quadratic_coeffs(quad_pts[17.0])

    a_refs = np.array([coeffs_20[0], coeffs_35[0], coeffs_75[0], coeffs_105[0], coeffs_135[0], coeffs_170[0]])
    b_refs = np.array([coeffs_20[1], coeffs_35[1], coeffs_75[1], coeffs_105[1], coeffs_135[1], coeffs_170[1]])
    c_refs = np.array([coeffs_20[2], coeffs_35[2], coeffs_75[2], coeffs_105[2], coeffs_135[2], coeffs_170[2]])

    def interp_extrap_low(v, v_refs, y_refs, extrap_low=True):
        """Interpolate with linear extrapolation for v < v_refs[0] only."""
        result = np.interp(v, v_refs, y_refs)
        mask_low = v < v_refs[0]
        if np.any(mask_low):
            if extrap_low:
                slope_low = (y_refs[1] - y_refs[0]) / (v_refs[1] - v_refs[0])
                result[mask_low] = y_refs[0] + slope_low * (v[mask_low] - v_refs[0])
        return result

    # Linear region parameters: keep constant for v < 2.0
    m1 = interp_extrap_low(v_flat, v_refs, m1_refs, extrap_low=False)
    c1 = interp_extrap_low(v_flat, v_refs, c1_refs, extrap_low=False)
    w_break = interp_extrap_low(v_flat, v_refs, w_break_refs, extrap_low=False)

    # Quadratic region parameters: extrapolate for v < 2.0
    a = interp_extrap_low(v_flat, v_refs, a_refs, extrap_low=True)
    b = interp_extrap_low(v_flat, v_refs, b_refs, extrap_low=True)
    c = interp_extrap_low(v_flat, v_refs, c_refs, extrap_low=True)

    # Ensure quadratic coefficient 'a' is always negative (concave down)
    a = np.minimum(a, -1e-10)

    # Ensure w_break stays positive
    w_break = np.maximum(w_break, 0.0)

    # Compute C_M using piecewise function
    cm_linear = m1 * w_flat + c1
    cm_quadratic = a * w_flat**2 + b * w_flat + c
    cm = np.where(w_flat <= w_break, cm_linear, cm_quadratic)

    # Ensure non-negative
    cm = np.maximum(cm, 0.0)

    # Reshape to original shape
    cm = cm.reshape(original_shape)

    # Return scalar if input was scalar
    if scalar_input:
        return float(cm.ravel()[0])
    return cm
