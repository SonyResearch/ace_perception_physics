# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Aerodynamic Coefficient Estimates

Local copies of drag and magnus coefficient estimation functions.
These can be modified without affecting the original implementations
in ace_evaluation.plotting.plot_aerodynamics.

This module provides two versions of drag estimation:
1. get_drag_estimate_v1: Original hand-designed piecewise linear function
2. get_drag_estimate: Alias to the currently active version (can be switched)
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from typing import Optional
from pathlib import Path

# Global model cache for fitted drag model
_fitted_drag_model = None
_use_fitted_model = True  # Default to V2 (data-fitted model)


def _get_drag_estimate_v1(v, S):
    """
    [V1] Original hand-designed piecewise linear drag estimate.

    Estimate drag coefficient C_D as a function of velocity v and spin parameter S = r*w/v.

    The piecewise linear function is modulated by velocity:
    - At v=2.5 m/s: constant at 0.55 everywhere
    - At v=7.5 m/s: reference curve with 5 segments
    - At v=12.5 m/s: shifted curve with different breakpoints and values
    - For v > 12.5 m/s: uses the v=12.5 curve (no extrapolation)
    - Linearly interpolated between reference velocities

    Args:
        v: velocity in m/s (scalar or array)
        S: spin parameter S = r*w/v (scalar or array)

    Returns:
        C_D: drag coefficient (scalar or array matching input shape)
    """
    # Convert pandas Series to numpy arrays if needed
    v_arr = np.asarray(v)
    S_arr = np.asarray(S)

    # Store original shape for output
    original_shape = None
    scalar_input = (v_arr.ndim == 0) and (S_arr.ndim == 0)

    # Broadcast scalar to array if needed
    if v_arr.ndim == 0 and S_arr.ndim > 0:
        v_arr = np.full_like(S_arr, float(v_arr), dtype=float)
        original_shape = S_arr.shape
    elif S_arr.ndim == 0 and v_arr.ndim > 0:
        S_arr = np.full_like(v_arr, float(S_arr), dtype=float)
        original_shape = v_arr.shape
    elif v_arr.ndim == 0 and S_arr.ndim == 0:
        v_arr = np.array([float(v_arr)])
        S_arr = np.array([float(S_arr)])
        original_shape = 'scalar'
    else:
        original_shape = v_arr.shape

    # Flatten arrays for processing
    v_flat = v_arr.flatten()
    S_flat = S_arr.flatten()

    # Helper function for linear interpolation
    def lerp(x, x0, x1, y0, y1):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    # Define reference velocities and their corresponding curves
    v_ref_verylow = 2.5
    v_ref_low = 7.5
    v_ref_mid = 12.5
    v_ref_high = 17.5

    # Reference curve at v=2.5 m/s (constant)
    S_breaks_verylow = np.array([0.0, 0.3, 0.7, 0.95, 1.5, 2.0])
    CD_values_verylow = np.array([0.55, 0.55, 0.55, 0.55, 0.55, 0.55])

    # Reference curve at v=7.5 m/s
    S_breaks_low = np.array([0.0, 0.4, 0.75, 1.1, 1.3, 2.0])
    CD_values_low = np.array([0.49, 0.49, 0.55, 0.48, 0.53, 0.53])

    # Reference curve at v=12.5 m/s
    S_breaks_mid = np.array([0.0, 0.4, 0.62, 0.95, 1.3, 2.0])
    CD_values_mid = np.array([0.47, 0.47, 0.53, 0.41, 0.48, 0.48])

    # Reference curve at v=17.5 m/s
    S_breaks_high = np.array([0.0, 0.4, 0.5, 0.84, 1.2, 2.0])
    CD_values_high = np.array([0.47, 0.47, 0.51, 0.37, 0.45, 0.45])

    # Function to evaluate piecewise linear CD(S) for a given set of breakpoints
    def eval_cd_curve(S_val, S_breaks, CD_values):
        # S <= first break: constant at first value
        if S_val <= S_breaks[1]:
            return CD_values[0]
        # S >= last break: constant at last value
        elif S_val >= S_breaks[4]:
            return CD_values[5]
        # In between: find which segment and interpolate
        else:
            for i in range(1, 5):
                if S_breaks[i] <= S_val <= S_breaks[i+1]:
                    return lerp(S_val, S_breaks[i], S_breaks[i+1], CD_values[i], CD_values[i+1])
        return CD_values[5]

    # Vectorized evaluation
    cd = np.zeros_like(S_flat, dtype=float)

    for i in range(len(v_flat)):
        v_val = v_flat[i]
        S_val = S_flat[i]

        # Determine which velocity regime and interpolate accordingly
        if v_val <= v_ref_verylow:
            # v <= 2.5: use constant curve
            S_breaks = S_breaks_verylow.copy()
            CD_values = CD_values_verylow.copy()
        elif v_val <= v_ref_low:
            # 2.5 < v <= 7.5: interpolate between verylow and low
            alpha = (v_val - v_ref_verylow) / (v_ref_low - v_ref_verylow)
            S_breaks = (1 - alpha) * S_breaks_verylow + alpha * S_breaks_low
            CD_values = (1 - alpha) * CD_values_verylow + alpha * CD_values_low
        elif v_val <= v_ref_mid:
            # 7.5 < v <= 12.5: interpolate between low and mid
            alpha = (v_val - v_ref_low) / (v_ref_mid - v_ref_low)
            S_breaks = (1 - alpha) * S_breaks_low + alpha * S_breaks_mid
            CD_values = (1 - alpha) * CD_values_low + alpha * CD_values_mid
        elif v_val <= v_ref_high:
            # 12.5 < v <= 17.5: interpolate between mid and high
            alpha = (v_val - v_ref_mid) / (v_ref_high - v_ref_mid)
            S_breaks = (1 - alpha) * S_breaks_mid + alpha * S_breaks_high
            CD_values = (1 - alpha) * CD_values_mid + alpha * CD_values_high
        else:
            # v > 17.5: linearly extrapolate using slope from 12.5→17.5
            alpha = (v_val - v_ref_mid) / (v_ref_high - v_ref_mid)
            S_breaks = (1 - alpha) * S_breaks_mid + alpha * S_breaks_high
            CD_values = (1 - alpha) * CD_values_mid + alpha * CD_values_high

        # Evaluate CD for this velocity and spin parameter
        cd[i] = eval_cd_curve(S_val, S_breaks, CD_values)

    # Return result in original shape
    if original_shape == 'scalar':
        return cd[0]
    elif original_shape is not None:
        return cd.reshape(original_shape)
    return cd


def get_drag_estimate_v1(v, S):
    """Alias for _get_drag_estimate_v1 for external use."""
    return _get_drag_estimate_v1(v, S)


def get_drag_estimate_smoothstep(v, S):
    """
    [Smoothstep] Piecewise drag estimate with smoothstep (Hermite) interpolation.

    Same reference points as V1 but uses smoothstep instead of linear interpolation
    for smooth C1 transitions at breakpoints.

    Args:
        v: velocity in m/s (scalar or array)
        S: spin parameter S = r*w/v (scalar or array)

    Returns:
        C_D: drag coefficient (scalar or array matching input shape)
    """
    # Convert pandas Series to numpy arrays if needed
    v_arr = np.asarray(v)
    S_arr = np.asarray(S)

    # Store original shape for output
    original_shape = None
    scalar_input = (v_arr.ndim == 0) and (S_arr.ndim == 0)

    # Broadcast scalar to array if needed
    if v_arr.ndim == 0 and S_arr.ndim > 0:
        v_arr = np.full_like(S_arr, float(v_arr), dtype=float)
        original_shape = S_arr.shape
    elif S_arr.ndim == 0 and v_arr.ndim > 0:
        S_arr = np.full_like(v_arr, float(S_arr), dtype=float)
        original_shape = v_arr.shape
    elif v_arr.ndim == 0 and S_arr.ndim == 0:
        v_arr = np.array([float(v_arr)])
        S_arr = np.array([float(S_arr)])
        original_shape = 'scalar'
    else:
        original_shape = v_arr.shape

    # Flatten arrays for processing
    v_flat = v_arr.flatten()
    S_flat = S_arr.flatten()

    # Smoothstep interpolation (Hermite S-curve)
    def smoothstep(x, x0, x1, y0, y1):
        t = np.clip((x - x0) / (x1 - x0), 0, 1)
        t = t * t * (3 - 2 * t)  # Hermite smoothstep: 3t^2 - 2t^3
        return y0 + (y1 - y0) * t

    # Define reference velocities and their corresponding curves (same as V1)
    v_ref_verylow = 2.5
    v_ref_low = 7.5
    v_ref_mid = 12.5
    v_ref_high = 17.5

    # Reference curve at v=2.5 m/s (constant)
    S_breaks_verylow = np.array([0.0, 0.3, 0.7, 0.95, 1.5, 2.0])
    CD_values_verylow = np.array([0.55, 0.55, 0.55, 0.55, 0.55, 0.55])

    # Reference curve at v=7.5 m/s
    S_breaks_low = np.array([0.0, 0.4, 0.75, 1.1, 1.3, 2.0])
    CD_values_low = np.array([0.49, 0.49, 0.55, 0.48, 0.53, 0.53])

    # Reference curve at v=12.5 m/s
    S_breaks_mid = np.array([0.0, 0.4, 0.62, 0.95, 1.3, 2.0])
    CD_values_mid = np.array([0.47, 0.47, 0.53, 0.41, 0.48, 0.48])

    # Reference curve at v=17.5 m/s
    S_breaks_high = np.array([0.0, 0.4, 0.5, 0.84, 1.2, 2.0])
    CD_values_high = np.array([0.47, 0.47, 0.51, 0.37, 0.45, 0.45])

    # Function to evaluate piecewise smoothstep CD(S) for a given set of breakpoints
    def eval_cd_curve(S_val, S_breaks, CD_values):
        # S <= first break: constant at first value
        if S_val <= S_breaks[1]:
            return CD_values[0]
        # S >= last break: constant at last value
        elif S_val >= S_breaks[4]:
            return CD_values[5]
        # In between: find which segment and smoothstep
        else:
            for i in range(1, 5):
                if S_breaks[i] <= S_val <= S_breaks[i+1]:
                    return smoothstep(S_val, S_breaks[i], S_breaks[i+1], CD_values[i], CD_values[i+1])
        return CD_values[5]

    # Vectorized evaluation
    cd = np.zeros_like(S_flat, dtype=float)

    for i in range(len(v_flat)):
        v_val = v_flat[i]
        S_val = S_flat[i]

        # Determine which velocity regime and interpolate accordingly
        if v_val <= v_ref_verylow:
            S_breaks = S_breaks_verylow.copy()
            CD_values = CD_values_verylow.copy()
        elif v_val <= v_ref_low:
            alpha = (v_val - v_ref_verylow) / (v_ref_low - v_ref_verylow)
            S_breaks = (1 - alpha) * S_breaks_verylow + alpha * S_breaks_low
            CD_values = (1 - alpha) * CD_values_verylow + alpha * CD_values_low
        elif v_val <= v_ref_mid:
            alpha = (v_val - v_ref_low) / (v_ref_mid - v_ref_low)
            S_breaks = (1 - alpha) * S_breaks_low + alpha * S_breaks_mid
            CD_values = (1 - alpha) * CD_values_low + alpha * CD_values_mid
        elif v_val <= v_ref_high:
            alpha = (v_val - v_ref_mid) / (v_ref_high - v_ref_mid)
            S_breaks = (1 - alpha) * S_breaks_mid + alpha * S_breaks_high
            CD_values = (1 - alpha) * CD_values_mid + alpha * CD_values_high
        else:
            # v > 17.5: linearly extrapolate using slope from 12.5→17.5
            alpha = (v_val - v_ref_mid) / (v_ref_high - v_ref_mid)
            S_breaks = (1 - alpha) * S_breaks_mid + alpha * S_breaks_high
            CD_values = (1 - alpha) * CD_values_mid + alpha * CD_values_high

        # Evaluate CD for this velocity and spin parameter
        cd[i] = eval_cd_curve(S_val, S_breaks, CD_values)

    # Return result in original shape
    if original_shape == 'scalar':
        return cd[0]
    elif original_shape is not None:
        return cd.reshape(original_shape)
    return cd


def get_drag_estimate_spline(v, S):
    """
    [Cubic Spline] Piecewise drag estimate with cubic spline interpolation.

    Same reference points as V1 but uses cubic spline for C2 smooth curves.

    Args:
        v: velocity in m/s (scalar or array)
        S: spin parameter S = r*w/v (scalar or array)

    Returns:
        C_D: drag coefficient (scalar or array matching input shape)
    """
    from scipy.interpolate import CubicSpline

    # Convert pandas Series to numpy arrays if needed
    v_arr = np.asarray(v)
    S_arr = np.asarray(S)

    # Store original shape for output
    original_shape = None
    scalar_input = (v_arr.ndim == 0) and (S_arr.ndim == 0)

    # Broadcast scalar to array if needed
    if v_arr.ndim == 0 and S_arr.ndim > 0:
        v_arr = np.full_like(S_arr, float(v_arr), dtype=float)
        original_shape = S_arr.shape
    elif S_arr.ndim == 0 and v_arr.ndim > 0:
        S_arr = np.full_like(v_arr, float(S_arr), dtype=float)
        original_shape = v_arr.shape
    elif v_arr.ndim == 0 and S_arr.ndim == 0:
        v_arr = np.array([float(v_arr)])
        S_arr = np.array([float(S_arr)])
        original_shape = 'scalar'
    else:
        original_shape = v_arr.shape

    # Flatten arrays for processing
    v_flat = v_arr.flatten()
    S_flat = S_arr.flatten()

    # Define reference velocities and their corresponding curves (same as V1)
    v_ref_verylow = 2.5
    v_ref_low = 7.5
    v_ref_mid = 12.5
    v_ref_high = 17.5

    # Reference curve at v=2.5 m/s (constant)
    S_breaks_verylow = np.array([0.0, 0.3, 0.7, 0.95, 1.5, 2.0])
    CD_values_verylow = np.array([0.55, 0.55, 0.55, 0.55, 0.55, 0.55])

    # Reference curve at v=7.5 m/s
    S_breaks_low = np.array([0.0, 0.4, 0.75, 1.1, 1.3, 2.0])
    CD_values_low = np.array([0.49, 0.49, 0.55, 0.48, 0.53, 0.53])

    # Reference curve at v=12.5 m/s
    S_breaks_mid = np.array([0.0, 0.4, 0.62, 0.95, 1.3, 2.0])
    CD_values_mid = np.array([0.47, 0.47, 0.53, 0.41, 0.48, 0.48])

    # Reference curve at v=17.5 m/s
    S_breaks_high = np.array([0.0, 0.4, 0.5, 0.84, 1.2, 2.0])
    CD_values_high = np.array([0.47, 0.47, 0.51, 0.37, 0.45, 0.45])

    # Function to evaluate cubic spline CD(S) for a given set of breakpoints
    def eval_cd_curve(S_val, S_breaks, CD_values):
        # Create cubic spline with clamped boundary conditions (zero derivative at ends)
        spline = CubicSpline(S_breaks, CD_values, bc_type='clamped')
        # Clip S to valid range to avoid extrapolation
        S_clipped = np.clip(S_val, S_breaks[0], S_breaks[-1])
        return float(spline(S_clipped))

    # Vectorized evaluation
    cd = np.zeros_like(S_flat, dtype=float)

    for i in range(len(v_flat)):
        v_val = v_flat[i]
        S_val = S_flat[i]

        # Determine which velocity regime and interpolate accordingly
        if v_val <= v_ref_verylow:
            S_breaks = S_breaks_verylow.copy()
            CD_values = CD_values_verylow.copy()
        elif v_val <= v_ref_low:
            alpha = (v_val - v_ref_verylow) / (v_ref_low - v_ref_verylow)
            S_breaks = (1 - alpha) * S_breaks_verylow + alpha * S_breaks_low
            CD_values = (1 - alpha) * CD_values_verylow + alpha * CD_values_low
        elif v_val <= v_ref_mid:
            alpha = (v_val - v_ref_low) / (v_ref_mid - v_ref_low)
            S_breaks = (1 - alpha) * S_breaks_low + alpha * S_breaks_mid
            CD_values = (1 - alpha) * CD_values_low + alpha * CD_values_mid
        elif v_val <= v_ref_high:
            alpha = (v_val - v_ref_mid) / (v_ref_high - v_ref_mid)
            S_breaks = (1 - alpha) * S_breaks_mid + alpha * S_breaks_high
            CD_values = (1 - alpha) * CD_values_mid + alpha * CD_values_high
        else:
            # v > 17.5: linearly extrapolate using slope from 12.5→17.5
            alpha = (v_val - v_ref_mid) / (v_ref_high - v_ref_mid)
            S_breaks = (1 - alpha) * S_breaks_mid + alpha * S_breaks_high
            CD_values = (1 - alpha) * CD_values_mid + alpha * CD_values_high

        # Evaluate CD for this velocity and spin parameter
        cd[i] = eval_cd_curve(S_val, S_breaks, CD_values)

    # Return result in original shape
    if original_shape == 'scalar':
        return cd[0]
    elif original_shape is not None:
        return cd.reshape(original_shape)
    return cd


def get_magnus_estimate(v, w):
    """
    Estimate Magnus coefficient C_M(v, w) using piecewise linear model.

    This matches the C++ implementation in PhysicsParameters.hpp::GetMagnusCoefficient.

    The model uses:
    - Two slopes: slope1 (for w < w_break) and slope2 (for w >= w_break)
    - Both slopes and w_break are interpolated based on velocity
    - Constrained to pass through C_M(v, 900) = 0.12

    Args:
        v: velocity in m/s (scalar or array)
        w: angular velocity in rad/s (scalar or array)

    Returns:
        C_M: magnus coefficient (scalar or array matching input shape)
    """
    # Convert pandas Series to numpy arrays if needed
    v_arr = np.asarray(v)
    w_arr = np.asarray(w)

    # Ensure we have arrays for vectorized operations
    scalar_input = (v_arr.ndim == 0) and (w_arr.ndim == 0)
    if scalar_input:
        v_arr = np.array([v_arr])
        w_arr = np.array([w_arr])

    # Helper: interpolate parameter between (x0, y0) and (x1, y1)
    def lerp(x, x0, x1, y0, y1):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    # Base definitions at the special velocity points
    s_low = 0.5
    s_mid1 = 5.5
    s_mid2 = 18.5

    # Slopes for w < w_break (interpolated by velocity)
    slope1_low = 0.00005    # at speed = 0.5 → simple linear
    slope1_mid1 = -0.0010   # at speed = 5.5
    slope1_mid2 = -0.00053  # at speed = 18.5

    # Slope for w > w_break (constant)
    slope2 = 0.00005

    # Break positions (interpolated by velocity)
    w_break_low = 0      # at speed 0.5 (no break, but keep structure)
    w_break_mid1 = 245   # at speed 5.5
    w_break_mid2 = 550   # at speed 18.5

    # Initialize arrays for parameters
    slope1 = np.zeros_like(v_arr, dtype=float)
    w_break = np.zeros_like(v_arr, dtype=float)

    # Vectorized conditional logic for velocity regimes
    mask_low = v_arr <= s_low
    mask_mid1 = (v_arr > s_low) & (v_arr <= s_mid1)
    mask_mid2 = (v_arr > s_mid1) & (v_arr <= s_mid2)
    mask_high = v_arr > s_mid2

    # Apply conditions for each velocity regime
    slope1[mask_low] = slope1_low
    w_break[mask_low] = w_break_low

    slope1[mask_mid1] = lerp(v_arr[mask_mid1], s_low, s_mid1, slope1_low, slope1_mid1)
    w_break[mask_mid1] = lerp(v_arr[mask_mid1], s_low, s_mid1, w_break_low, w_break_mid1)

    slope1[mask_mid2] = lerp(v_arr[mask_mid2], s_mid1, s_mid2, slope1_mid1, slope1_mid2)
    w_break[mask_mid2] = lerp(v_arr[mask_mid2], s_mid1, s_mid2, w_break_mid1, w_break_mid2)

    # v > s_mid2: extrapolate using same formula as mid2 regime
    slope1[mask_high] = lerp(v_arr[mask_high], s_mid1, s_mid2, slope1_mid1, slope1_mid2)
    w_break[mask_high] = lerp(v_arr[mask_high], s_mid1, s_mid2, w_break_mid1, w_break_mid2)

    # Enforce constraint: C_M(v, 900) = 0.12
    w_target = 900
    cm_target = 0.12

    # Compute intercept C0 such that constraint holds (vectorized)
    C0 = np.where(
        w_target <= w_break,
        cm_target - slope1 * w_target,
        cm_target - (slope1 * w_break + slope2 * (w_target - w_break))
    )

    # Piecewise function (vectorized)
    cm = np.where(
        w_arr <= w_break,
        C0 + slope1 * w_arr,
        C0 + slope1 * w_break + slope2 * (w_arr - w_break)
    )

    # Return scalar if input was scalar
    if scalar_input:
        return cm[0]
    return cm


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
    # Data points per velocity:
    #   v=2.0:  (10, 0.08), (150, 0.08), (300, 0.055), (450, 0.025)
    #   v=3.5:  (100, 0.2), (200, 0.09), (300, 0.08), (500, 0.05)
    #   v=7.5:  (150, 0.25), (350, 0.09), (500, 0.095), (750, 0.075)
    #   v=10.5: (250, 0.21), (440, 0.085), (600, 0.1), (800, 0.095)
    #   v=13.5: (300, 0.215), (550, 0.075), (750, 0.105), (900, 0.095)
    #   v=17:   (360, 0.21), (650, 0.08), (750, 0.095), (900, 0.11)
    v_refs = np.array([2.0, 3.5, 7.5, 10.5, 13.5, 17.0])

    # Linear region parameters (m1, c1)
    # Fitted from first 2 points for each velocity: m1 = (cm2-cm1)/(w2-w1), c1 = cm1 - m1*w1
    # v=2.0:  m1 = (0.08-0.08)/(150-10) = 0.0,       c1 = 0.08 - 0.0*10 = 0.08
    # v=3.5:  m1 = (0.09-0.2)/(200-100) = -0.0011,   c1 = 0.2 - (-0.0011)*100 = 0.31
    # v=7.5:  m1 = (0.09-0.25)/(350-150) = -0.0008, c1 = 0.25 - (-0.0008)*150 = 0.37
    # v=10.5: m1 = (0.085-0.21)/(440-250) = -0.000658, c1 = 0.21 - (-0.000658)*250 = 0.375
    # v=13.5: m1 = (0.075-0.215)/(550-300) = -0.00056, c1 = 0.215 - (-0.00056)*300 = 0.383
    # v=17:   m1 = (0.08-0.21)/(650-360) = -0.000448, c1 = 0.21 - (-0.000448)*360 = 0.371
    m1_refs = np.array([0.0, -0.0011, -0.0008, -0.000658, -0.00056, -0.000448])
    c1_refs = np.array([0.08, 0.31, 0.37, 0.375, 0.383, 0.371])

    # Break point (second data point for each velocity)
    w_break_refs = np.array([150.0, 200.0, 350.0, 440.0, 550.0, 650.0])

    # Quadratic region parameters (a, b, c)
    # C_M = a*w^2 + b*w + c for w > w_break
    # Computed dynamically to ensure exact fit through data points
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

    # Interpolate parameters based on velocity
    # For v < v_refs[0] (2.0): linear params (m1, c1, w_break) stay constant, quadratic params extrapolate
    # For v > v_refs[-1] (17.0): saturate at v=17.0 values (no extrapolation)
    def interp_extrap_low(v, v_refs, y_refs, extrap_low=True):
        """Interpolate with linear extrapolation for v < v_refs[0] only.

        Args:
            extrap_low: If False, clamp to y_refs[0] for v < v_refs[0] instead of extrapolating.
        """
        result = np.interp(v, v_refs, y_refs)
        # Extrapolate below (only if extrap_low is True)
        mask_low = v < v_refs[0]
        if np.any(mask_low):
            if extrap_low:
                slope_low = (y_refs[1] - y_refs[0]) / (v_refs[1] - v_refs[0])
                result[mask_low] = y_refs[0] + slope_low * (v[mask_low] - v_refs[0])
            # else: keep the clamped value from np.interp (y_refs[0])
        # For v > v_refs[-1], np.interp already clamps to y_refs[-1]
        return result

    # Linear region parameters: keep constant for v < 2.0 (no extrapolation)
    # For v > 17.0, use np.interp clamping (keeps v=17.0 values)
    m1 = interp_extrap_low(v_flat, v_refs, m1_refs, extrap_low=False)
    c1 = interp_extrap_low(v_flat, v_refs, c1_refs, extrap_low=False)
    w_break = interp_extrap_low(v_flat, v_refs, w_break_refs, extrap_low=False)

    # Quadratic region parameters:
    # - For v < 2.0: extrapolate coefficients linearly
    # - For v > 17.0: saturate at v=17.0 values (np.interp clamps automatically)
    a = interp_extrap_low(v_flat, v_refs, a_refs, extrap_low=True)
    b = interp_extrap_low(v_flat, v_refs, b_refs, extrap_low=True)
    c = interp_extrap_low(v_flat, v_refs, c_refs, extrap_low=True)

    # Ensure quadratic coefficient 'a' is always negative (concave down)
    # This prevents extrapolation from changing the concavity of the function
    a = np.minimum(a, -1e-10)

    # Ensure w_break stays positive (safety for edge cases)
    w_break = np.maximum(w_break, 0.0)

    # Compute C_M using piecewise function
    # Linear region: C_M = m1 * w + c1
    cm_linear = m1 * w_flat + c1

    # Quadratic region: C_M = a * w^2 + b * w + c
    cm_quadratic = a * w_flat**2 + b * w_flat + c

    # Select based on w_break
    cm = np.where(w_flat <= w_break, cm_linear, cm_quadratic)

    # Ensure non-negative
    cm = np.maximum(cm, 0.0)

    # Reshape to original shape
    cm = cm.reshape(original_shape)

    # Return scalar if input was scalar
    if scalar_input:
        return float(cm.ravel()[0])
    return cm


# Alias for the original implementation
get_magnus_estimate_v1 = get_magnus_estimate


# =============================================================================
# V2: Data-fitted model using lookup table with bilinear interpolation
# =============================================================================
# Fitted model statistics:
# - Raw data points: 54178
# - Valid data points: 52576
# - RMSE: 0.018533
# - R²: 0.849184
# - Domain: v_eff_drag [0, 30] m/s, spin_ratio S [0, 2]
# - Grid: 31 x 21 points (1.0 m/s and 0.1 S spacing)

# Lookup table grid points
_V_GRID_V2 = np.array([
    0.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000,
    8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000,
    16.000000, 17.000000, 18.000000, 19.000000, 20.000000, 21.000000, 22.000000, 23.000000,
    24.000000, 25.000000, 26.000000, 27.000000, 28.000000, 29.000000, 30.000000
])

_S_GRID_V2 = np.array([
    0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
    0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000,
    1.600000, 1.700000, 1.800000, 1.900000, 2.000000
])

# Precomputed C_D values: shape (31, 21)
# _CD_TABLE_V2[i, j] = C_D(v=_V_GRID_V2[i], S=_S_GRID_V2[j])
_CD_TABLE_V2 = np.array([
    [0.561302, 0.563949, 0.565474, 0.565429, 0.563661, 0.560743, 0.558770, 0.558433, 0.558603, 0.558410, 0.557794, 0.556987, 0.555618, 0.553859, 0.552938, 0.553022, 0.553333, 0.553505, 0.553385, 0.552787, 0.551808],
    [0.557832, 0.561741, 0.563878, 0.564375, 0.563119, 0.560324, 0.559009, 0.559865, 0.560941, 0.561155, 0.560573, 0.559776, 0.558147, 0.555861, 0.554966, 0.555493, 0.556055, 0.556300, 0.556151, 0.555253, 0.553769],
    [0.552013, 0.556221, 0.558369, 0.559189, 0.559136, 0.557982, 0.558326, 0.560489, 0.562420, 0.563105, 0.562634, 0.561568, 0.559581, 0.557287, 0.556781, 0.557812, 0.558673, 0.558981, 0.558849, 0.557927, 0.555811],
    [0.542375, 0.544888, 0.546423, 0.547961, 0.550255, 0.552550, 0.555786, 0.559494, 0.562083, 0.562991, 0.562462, 0.560815, 0.558489, 0.556819, 0.557225, 0.559016, 0.560418, 0.560955, 0.560928, 0.560133, 0.557293],
    [0.529495, 0.528958, 0.529436, 0.532354, 0.537783, 0.544144, 0.550907, 0.556485, 0.559376, 0.559727, 0.558375, 0.556034, 0.553720, 0.552944, 0.554669, 0.557844, 0.560377, 0.561517, 0.561761, 0.561056, 0.557435],
    [0.515800, 0.512347, 0.511693, 0.515853, 0.524138, 0.534202, 0.544009, 0.550951, 0.553327, 0.551951, 0.548930, 0.545949, 0.544157, 0.544645, 0.547960, 0.552967, 0.557198, 0.559266, 0.559934, 0.559256, 0.554965],
    [0.504365, 0.498907, 0.497751, 0.502750, 0.513046, 0.525649, 0.537012, 0.543867, 0.544297, 0.539823, 0.534449, 0.530995, 0.530109, 0.532165, 0.537177, 0.543961, 0.549920, 0.552982, 0.553978, 0.552996, 0.548410],
    [0.496285, 0.490168, 0.488814, 0.494342, 0.505854, 0.519700, 0.531258, 0.536514, 0.533453, 0.524738, 0.516544, 0.512809, 0.513273, 0.517172, 0.523912, 0.532185, 0.539422, 0.543289, 0.544347, 0.542925, 0.538559],
    [0.491079, 0.485194, 0.483783, 0.489707, 0.501726, 0.515748, 0.526420, 0.528855, 0.521154, 0.507670, 0.496865, 0.493358, 0.495713, 0.501775, 0.510248, 0.519589, 0.527478, 0.531794, 0.532661, 0.530796, 0.526864],
    [0.486748, 0.481514, 0.481044, 0.487434, 0.499434, 0.512938, 0.521790, 0.520433, 0.507574, 0.489705, 0.477270, 0.474757, 0.479319, 0.487595, 0.497573, 0.507410, 0.515317, 0.519581, 0.519960, 0.517587, 0.514088],
    [0.483234, 0.478475, 0.479183, 0.486153, 0.497983, 0.510411, 0.516744, 0.511070, 0.493238, 0.472163, 0.459411, 0.458552, 0.465353, 0.475499, 0.486336, 0.496014, 0.503206, 0.506833, 0.506736, 0.504052, 0.501091],
    [0.480517, 0.476049, 0.477657, 0.485229, 0.496822, 0.507705, 0.511080, 0.500994, 0.478890, 0.456048, 0.444148, 0.445273, 0.454040, 0.465384, 0.476293, 0.485299, 0.491487, 0.494280, 0.493840, 0.491106, 0.488676],
    [0.478907, 0.474615, 0.476926, 0.484796, 0.495807, 0.504857, 0.505024, 0.490708, 0.465308, 0.442003, 0.431739, 0.434803, 0.444899, 0.456551, 0.467127, 0.475433, 0.480784, 0.482995, 0.482290, 0.479707, 0.477905],
    [0.479567, 0.475140, 0.477350, 0.484895, 0.494956, 0.502052, 0.498925, 0.480868, 0.453218, 0.430321, 0.421940, 0.426617, 0.437294, 0.448581, 0.458666, 0.466503, 0.471394, 0.473402, 0.472763, 0.470594, 0.469660],
    [0.481749, 0.477358, 0.478726, 0.485242, 0.494085, 0.499204, 0.492985, 0.472000, 0.443048, 0.420920, 0.414259, 0.420040, 0.430744, 0.441433, 0.450974, 0.458560, 0.463509, 0.465779, 0.465648, 0.464189, 0.463992],
    [0.484653, 0.480224, 0.480421, 0.485565, 0.492864, 0.495866, 0.487073, 0.464194, 0.434869, 0.413681, 0.408331, 0.414552, 0.424873, 0.435058, 0.444303, 0.451911, 0.457278, 0.460264, 0.461114, 0.460757, 0.461624],
    [0.487530, 0.482573, 0.481466, 0.484920, 0.490196, 0.491161, 0.480684, 0.457206, 0.428723, 0.408786, 0.404110, 0.409969, 0.419610, 0.429523, 0.438770, 0.446677, 0.452746, 0.456820, 0.459100, 0.460282, 0.462082],
    [0.489504, 0.483641, 0.480709, 0.481653, 0.484391, 0.483728, 0.472829, 0.450402, 0.424443, 0.406463, 0.401783, 0.406574, 0.415360, 0.425069, 0.434458, 0.442849, 0.449817, 0.455229, 0.459268, 0.462483, 0.465967],
    [0.490603, 0.483190, 0.477592, 0.474887, 0.474217, 0.472219, 0.462255, 0.442672, 0.420889, 0.405652, 0.400934, 0.404532, 0.412421, 0.421841, 0.431390, 0.440335, 0.448282, 0.455118, 0.461007, 0.466370, 0.471584],
    [0.490916, 0.481451, 0.472555, 0.465199, 0.459581, 0.456117, 0.448756, 0.433625, 0.416885, 0.404864, 0.400750, 0.403609, 0.410734, 0.419829, 0.429501, 0.438980, 0.447868, 0.456044, 0.463603, 0.470716, 0.477009],
    [0.490522, 0.478985, 0.466721, 0.454478, 0.442559, 0.437211, 0.433964, 0.424158, 0.412461, 0.403772, 0.400813, 0.403479, 0.410096, 0.418907, 0.428657, 0.438586, 0.448284, 0.457587, 0.466473, 0.474827, 0.481843],
    [0.490000, 0.476536, 0.461470, 0.445549, 0.429836, 0.422625, 0.421222, 0.415693, 0.408263, 0.402616, 0.401053, 0.403925, 0.410294, 0.418895, 0.428682, 0.438945, 0.449265, 0.459415, 0.469239, 0.478396, 0.485905],
    [0.489557, 0.474661, 0.457807, 0.440216, 0.424313, 0.415710, 0.412938, 0.409449, 0.404999, 0.401791, 0.401572, 0.404852, 0.411157, 0.419617, 0.429397, 0.439865, 0.450595, 0.461298, 0.471697, 0.481295, 0.489039],
    [0.489305, 0.473647, 0.456011, 0.438142, 0.422751, 0.413226, 0.408841, 0.405846, 0.403121, 0.401633, 0.402516, 0.406235, 0.412567, 0.420920, 0.430643, 0.441182, 0.452114, 0.463098, 0.473765, 0.483544, 0.491457],
    [0.489554, 0.473554, 0.455838, 0.438340, 0.423474, 0.413398, 0.407827, 0.404659, 0.402753, 0.402330, 0.403997, 0.408074, 0.414441, 0.422683, 0.432289, 0.442773, 0.453719, 0.464748, 0.475435, 0.485184, 0.493082],
    [0.490088, 0.474203, 0.456868, 0.440008, 0.425650, 0.415302, 0.408949, 0.405413, 0.403784, 0.403923, 0.406065, 0.410364, 0.416719, 0.424815, 0.434234, 0.444547, 0.455347, 0.466232, 0.476749, 0.486318, 0.494177],
    [0.490855, 0.475396, 0.458705, 0.442597, 0.428764, 0.418360, 0.411529, 0.407627, 0.405990, 0.406350, 0.408709, 0.413083, 0.419351, 0.427245, 0.436408, 0.446449, 0.456974, 0.467571, 0.477778, 0.487047, 0.494723],
    [0.491920, 0.476969, 0.461045, 0.445741, 0.432469, 0.422173, 0.415091, 0.410893, 0.409123, 0.409489, 0.411873, 0.416190, 0.422288, 0.429917, 0.438757, 0.448445, 0.458596, 0.468803, 0.478613, 0.487517, 0.494988],
    [0.493005, 0.478741, 0.463666, 0.449194, 0.436524, 0.426460, 0.419294, 0.414890, 0.412945, 0.413198, 0.415475, 0.419630, 0.425482, 0.432788, 0.441247, 0.450516, 0.460225, 0.469977, 0.479341, 0.487863, 0.495123],
    [0.494157, 0.480618, 0.466422, 0.452795, 0.440759, 0.431020, 0.423896, 0.419370, 0.417256, 0.417337, 0.419424, 0.423340, 0.428886, 0.435819, 0.443850, 0.452654, 0.461876, 0.471137, 0.480036, 0.488175, 0.495213],
    [0.495398, 0.482552, 0.469229, 0.456441, 0.445060, 0.435712, 0.428721, 0.424149, 0.421892, 0.421780, 0.423631, 0.427259, 0.432455, 0.438977, 0.446548, 0.454856, 0.463565, 0.472318, 0.480749, 0.488508, 0.495312]
])


def get_drag_estimate_v2(v, S):
    """
    [V2] Data-fitted drag coefficient estimate using bilinear interpolation.

    Fitted from 54178 data points with filters:
    - confidence >= 0.5
    - duration >= 0.15s
    - C_D = 0.55 constant for v <= 4 m/s

    Args:
        v: velocity (v_eff_drag) in m/s (scalar or array)
        S: spin ratio S = r*w/v (scalar or array)

    Returns:
        C_D: drag coefficient (scalar or array matching input shape)
    """
    # Handle input types
    v_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
    S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))

    # Track if input was scalar
    scalar_input = (np.ndim(v) == 0) and (np.ndim(S) == 0)

    # Broadcast arrays
    v_arr, S_arr = np.broadcast_arrays(v_arr, S_arr)
    original_shape = v_arr.shape
    v_flat = v_arr.ravel()
    S_flat = S_arr.ravel()

    # Clamp to valid domain
    v_clamped = np.clip(v_flat, _V_GRID_V2[0], _V_GRID_V2[-1])
    S_clamped = np.clip(S_flat, _S_GRID_V2[0], _S_GRID_V2[-1])

    # Find grid indices and interpolation weights
    # For v
    v_idx_float = (v_clamped - _V_GRID_V2[0]) / (_V_GRID_V2[-1] - _V_GRID_V2[0]) * (len(_V_GRID_V2) - 1)
    v_idx_lo = np.clip(np.floor(v_idx_float).astype(int), 0, len(_V_GRID_V2) - 2)
    v_idx_hi = v_idx_lo + 1
    v_frac = v_idx_float - v_idx_lo

    # For S
    S_idx_float = (S_clamped - _S_GRID_V2[0]) / (_S_GRID_V2[-1] - _S_GRID_V2[0]) * (len(_S_GRID_V2) - 1)
    S_idx_lo = np.clip(np.floor(S_idx_float).astype(int), 0, len(_S_GRID_V2) - 2)
    S_idx_hi = S_idx_lo + 1
    S_frac = S_idx_float - S_idx_lo

    # Bilinear interpolation
    C00 = _CD_TABLE_V2[v_idx_lo, S_idx_lo]
    C10 = _CD_TABLE_V2[v_idx_hi, S_idx_lo]
    C01 = _CD_TABLE_V2[v_idx_lo, S_idx_hi]
    C11 = _CD_TABLE_V2[v_idx_hi, S_idx_hi]

    cd = ((1 - v_frac) * (1 - S_frac) * C00 +
          v_frac * (1 - S_frac) * C10 +
          (1 - v_frac) * S_frac * C01 +
          v_frac * S_frac * C11)

    # Reshape to original shape
    cd = cd.reshape(original_shape)

    # Return scalar if input was scalar
    if scalar_input:
        return float(cd.ravel()[0])
    return cd


def use_fitted_model(enable: bool = True):
    """Enable or disable the fitted model (falls back to v1 if disabled)."""
    global _use_fitted_model
    _use_fitted_model = enable
    return _use_fitted_model


# =============================================================================
# Active function selection
# =============================================================================
def get_drag_estimate(v, S):
    """
    Estimate drag coefficient C_D as a function of velocity and spin ratio.

    This is the main entry point that delegates to either:
    - v1: Original hand-designed piecewise linear function
    - v2: Data-fitted lookup table model (default)

    To switch to v1:
        from data_plotter.plot_widgets.aero_estimates import use_fitted_model
        use_fitted_model(False)

    To switch back to v2:
        use_fitted_model(True)

    Args:
        v: velocity (v_eff_drag) in m/s (scalar or array)
        S: spin ratio S = r*w/v (scalar or array)

    Returns:
        C_D: drag coefficient (scalar or array matching input shape)
    """
    if _use_fitted_model:
        return get_drag_estimate_v2(v, S)
    else:
        return get_drag_estimate_v1(v, S)
