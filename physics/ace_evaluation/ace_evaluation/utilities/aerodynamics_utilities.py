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

    # Reference velocities and their fitted parameters (matching C++ PhysicsParameters.hpp exactly)
    v_refs = np.array([2.0, 3.5, 7.5, 10.5, 13.5, 17.0])

    m1_refs = np.array([0.0, -0.0011, -0.000775, -0.000658, -0.00056, -0.000448])
    c1_refs = np.array([0.08, 0.31, 0.366, 0.375, 0.383, 0.371])
    w_break_refs = np.array([150.0, 200.0, 350.0, 440.0, 550.0, 650.0])

    a_refs = np.array([
        -1.8518518518518517e-07,
        -1.6666666666666665e-07,
        -2.0000000000000002e-07,
        -2.6041666666666690e-07,
        -3.5714285714285724e-07,
        -1.0000000000000002e-07
    ])
    b_refs = np.array([
        -1.2962962962962976e-04,
        -3.3333333333333576e-05,
        1.7000000000000013e-04,
        3.6458333333333426e-04,
        5.3571428571428634e-04,
        2.3000000000000009e-04
    ])
    c_refs = np.array([
        9.8333333333333356e-02,
        0.1,
        5.8749999999999969e-02,
        -2.2500000000000186e-02,
        -8.9285714285714691e-02,
        -3.7500000000000061e-02
    ])

    def interp(x, x_refs, y_refs, extrap_low=False):
        result = np.zeros_like(x)
        
        # Below first reference
        mask_low = x < x_refs[0]
        if extrap_low:
            slope = (y_refs[1] - y_refs[0]) / (x_refs[1] - x_refs[0])
            result[mask_low] = y_refs[0] + slope * (x[mask_low] - x_refs[0])
        else:
            result[mask_low] = y_refs[0]
            
        # Above last reference
        mask_high = x >= x_refs[-1]
        result[mask_high] = y_refs[-1]
        
        # In between
        for i in range(len(x_refs) - 1):
            mask_mid = (x >= x_refs[i]) & (x < x_refs[i + 1])
            alpha = (x[mask_mid] - x_refs[i]) / (x_refs[i + 1] - x_refs[i])
            result[mask_mid] = y_refs[i] + alpha * (y_refs[i + 1] - y_refs[i])
            
        return result

    m1 = interp(v_flat, v_refs, m1_refs, extrap_low=False)
    c1 = interp(v_flat, v_refs, c1_refs, extrap_low=False)
    w_break = interp(v_flat, v_refs, w_break_refs, extrap_low=False)

    a = interp(v_flat, v_refs, a_refs, extrap_low=True)
    b = interp(v_flat, v_refs, b_refs, extrap_low=True)
    c = interp(v_flat, v_refs, c_refs, extrap_low=True)

    a = np.minimum(a, -1e-10)
    w_break = np.maximum(w_break, 0.0)

    cm_linear = m1 * w_flat + c1
    cm_quadratic = a * w_flat**2 + b * w_flat + c
    cm = np.where(w_flat <= w_break, cm_linear, cm_quadratic)
    cm = np.maximum(cm, 0.0)

    cm = cm.reshape(original_shape)
    if scalar_input:
        return float(cm.ravel()[0])
    return cm
