# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Pure-Python RK4 aerodynamics integrator.

Reproduces the behaviour of the C++ ``physics_layer_pybind.predict_ball_state``
using a standard 4th-order Runge-Kutta scheme with velocity/spin-dependent
drag and magnus coefficients.

Two aerodynamics model variants are provided:
- "0226" (new model):  piecewise drag + piecewise-quadratic magnus from aero_estimates
- "0426" (old linear): Cd = const, Cm = linear function of vel/spin ratio

The coefficient functions use the **exact same constants** as the C++ implementation
in PhysicsParameters.hpp to ensure bit-level agreement (within floating-point
rounding of double vs float operations).
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ─── Ball state container ─────────────────────────────────────────────────────


@dataclass
class BallState:
    """Minimal ball state matching the C++ BallStateT interface."""
    timestamp: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "BallState":
        return BallState(
            timestamp=self.timestamp,
            position=self.position.copy(),
            linear_velocity=self.linear_velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )


# ─── Physics parameters ──────────────────────────────────────────────────────


@dataclass
class PhysicsParams:
    """Parameters used by the integrator."""
    ball_radius: float = 0.02
    air_density: float = 1.204
    ball_density: float = 81.2
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))

    @property
    def half_area_to_volume_ratio(self) -> float:
        return 3.0 / (8.0 * self.ball_radius)

    @property
    def density_ratio(self) -> float:
        return self.air_density / self.ball_density


def make_physics_params(params_dict: dict) -> PhysicsParams:
    """Create PhysicsParams from the dict returned by ``get_params()``."""
    return PhysicsParams(
        ball_radius=params_dict.get("ball_radius", 0.02),
        air_density=params_dict.get("air_density", 1.204),
        ball_density=params_dict.get("ball_density", 81.2),
        gravity=np.array([0.0, 0.0, params_dict.get("gravity", -9.81)]),
    )


# ─── Coefficient functions (matching C++ PhysicsParameters.hpp exactly) ───────


def get_drag_coefficient_new(v: float, w: float, ball_radius: float = 0.02) -> float:
    """Piecewise linear drag estimate matching C++ GetDragCoefficientNew exactly.

    Uses the spin-velocity ratio S = w * r / v.
    """
    if v <= 0:
        return 0.55  # Fallback for zero velocity

    S = w * ball_radius / v

    # Reference velocities
    V_REF_VERYLOW = 2.5
    V_REF_LOW = 7.5
    V_REF_MID = 12.5
    V_REF_HIGH = 17.5

    # Reference curves (same constants as C++)
    S_BREAKS_VERYLOW = np.array([0.0, 0.3, 0.7, 0.95, 1.5, 2.0])
    CD_VALUES_VERYLOW = np.array([0.55, 0.55, 0.55, 0.55, 0.55, 0.55])

    S_BREAKS_LOW = np.array([0.0, 0.4, 0.75, 1.1, 1.3, 2.0])
    CD_VALUES_LOW = np.array([0.49, 0.49, 0.55, 0.48, 0.53, 0.53])

    S_BREAKS_MID = np.array([0.0, 0.4, 0.62, 0.95, 1.3, 2.0])
    CD_VALUES_MID = np.array([0.47, 0.47, 0.53, 0.41, 0.48, 0.48])

    S_BREAKS_HIGH = np.array([0.0, 0.4, 0.5, 0.84, 1.2, 2.0])
    CD_VALUES_HIGH = np.array([0.47, 0.47, 0.51, 0.37, 0.45, 0.45])

    def lerp(x, x0, x1, y0, y1):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def eval_cd_curve(s_val, s_breaks, cd_values):
        if s_val <= s_breaks[1]:
            return cd_values[0]
        if s_val >= s_breaks[4]:
            return cd_values[5]
        for i in range(1, 5):
            if s_breaks[i] <= s_val <= s_breaks[i + 1]:
                return lerp(s_val, s_breaks[i], s_breaks[i + 1], cd_values[i], cd_values[i + 1])
        return cd_values[5]

    if v <= V_REF_VERYLOW:
        return eval_cd_curve(S, S_BREAKS_VERYLOW, CD_VALUES_VERYLOW)

    if v <= V_REF_LOW:
        alpha = (v - V_REF_VERYLOW) / (V_REF_LOW - V_REF_VERYLOW)
    elif v <= V_REF_MID:
        alpha = (v - V_REF_LOW) / (V_REF_MID - V_REF_LOW)
        S_BREAKS_VERYLOW = S_BREAKS_LOW
        CD_VALUES_VERYLOW = CD_VALUES_LOW
        S_BREAKS_LOW = S_BREAKS_MID
        CD_VALUES_LOW = CD_VALUES_MID
    elif v <= V_REF_HIGH:
        alpha = (v - V_REF_MID) / (V_REF_HIGH - V_REF_MID)
        S_BREAKS_VERYLOW = S_BREAKS_MID
        CD_VALUES_VERYLOW = CD_VALUES_MID
        S_BREAKS_LOW = S_BREAKS_HIGH
        CD_VALUES_LOW = CD_VALUES_HIGH
    else:
        # v > 17.5: extrapolate using mid→high slope
        alpha = (v - V_REF_MID) / (V_REF_HIGH - V_REF_MID)
        S_BREAKS_VERYLOW = S_BREAKS_MID
        CD_VALUES_VERYLOW = CD_VALUES_MID
        S_BREAKS_LOW = S_BREAKS_HIGH
        CD_VALUES_LOW = CD_VALUES_HIGH

    one_minus_alpha = 1.0 - alpha
    s_breaks = one_minus_alpha * S_BREAKS_VERYLOW + alpha * S_BREAKS_LOW
    cd_values = one_minus_alpha * CD_VALUES_VERYLOW + alpha * CD_VALUES_LOW
    return eval_cd_curve(S, s_breaks, cd_values)


def get_magnus_coefficient_new(v: float, w: float) -> float:
    """Piecewise linear+quadratic Magnus estimate matching C++ GetMagnusCoefficientNew exactly.

    Uses the **same hardcoded constants** as the C++ header.
    """
    # Reference velocities
    V_REFS = np.array([2.0, 3.5, 7.5, 10.5, 13.5, 17.0])

    # Linear region: C_M = m1*w + c1
    M1_REFS = np.array([0.0, -0.0011, -0.000775, -0.000658, -0.00056, -0.000448])
    C1_REFS = np.array([0.08, 0.31, 0.366, 0.375, 0.383, 0.371])

    # Break point (w where we switch from linear to quadratic)
    W_BREAK_REFS = np.array([150.0, 200.0, 350.0, 440.0, 550.0, 650.0])

    # Quadratic region: C_M = a*w^2 + b*w + c (pre-computed, matching C++)
    # v=2.0:  pts: (150, 0.08), (300, 0.055), (450, 0.025)
    A_REFS = np.array([
        -1.8518518518518517e-07,  # v=2.0
        -1.6666666666666665e-07,  # v=3.5
        -2.0000000000000002e-07,  # v=7.5  (pts: 350,0.095; 500,0.095; 750,0.075)
        -2.6041666666666690e-07,  # v=10.5
        -3.5714285714285724e-07,  # v=13.5
        -1.0000000000000002e-07,  # v=17.0
    ])
    B_REFS = np.array([
        -1.2962962962962976e-04,  # v=2.0
        -3.3333333333333576e-05,  # v=3.5
         1.7000000000000013e-04,  # v=7.5
         3.6458333333333426e-04,  # v=10.5
         5.3571428571428634e-04,  # v=13.5
         2.3000000000000009e-04,  # v=17.0
    ])
    C_REFS = np.array([
         9.8333333333333356e-02,  # v=2.0
         0.1,                     # v=3.5
         5.8749999999999969e-02,  # v=7.5
        -2.2500000000000186e-02,  # v=10.5
        -8.9285714285714691e-02,  # v=13.5
        -3.7500000000000061e-02,  # v=17.0
    ])

    def interp_param(x, x_refs, y_refs, extrap_low):
        """Interpolate matching C++ logic exactly."""
        n = len(x_refs)
        if x < x_refs[0]:
            if extrap_low:
                slope = (y_refs[1] - y_refs[0]) / (x_refs[1] - x_refs[0])
                return y_refs[0] + slope * (x - x_refs[0])
            return y_refs[0]
        if x >= x_refs[n - 1]:
            return y_refs[n - 1]
        for i in range(n - 1):
            if x_refs[i] <= x < x_refs[i + 1]:
                alpha = (x - x_refs[i]) / (x_refs[i + 1] - x_refs[i])
                return y_refs[i] + alpha * (y_refs[i + 1] - y_refs[i])
        return y_refs[n - 1]

    # Interpolate parameters (linear params don't extrapolate low, quadratic do)
    m1 = interp_param(v, V_REFS, M1_REFS, extrap_low=False)
    c1 = interp_param(v, V_REFS, C1_REFS, extrap_low=False)
    w_break = interp_param(v, V_REFS, W_BREAK_REFS, extrap_low=False)

    a = interp_param(v, V_REFS, A_REFS, extrap_low=True)
    b = interp_param(v, V_REFS, B_REFS, extrap_low=True)
    c = interp_param(v, V_REFS, C_REFS, extrap_low=True)

    # Ensure 'a' is negative (concave down)
    a = min(a, -1e-10)
    # Ensure w_break is positive
    w_break = max(w_break, 0.0)

    # Evaluate
    if w <= w_break:
        cm = m1 * w + c1
    else:
        cm = a * w * w + b * w + c

    return max(cm, 0.0)


# ─── Acceleration functions ───────────────────────────────────────────────────


def _acceleration_new_model(
    vel: np.ndarray,
    ang_vel: np.ndarray,
    pp: PhysicsParams,
    get_cd: Callable[[float, float], float],
    get_cm: Callable[[float, float], float],
) -> np.ndarray:
    """Acceleration using velocity-dependent Cd(v,w) and Cm(v,w).

    Reproduces C++ CalculateAcceleration with test_new_model=true.
    """
    v_norm = np.linalg.norm(vel)
    w_norm = np.linalg.norm(ang_vel)

    cd = get_cd(v_norm, w_norm)
    cm = get_cm(v_norm, w_norm)

    drag = -pp.half_area_to_volume_ratio * cd * pp.density_ratio * vel * v_norm
    magnus = -cm * pp.density_ratio * np.cross(vel, ang_vel)
    gravity = (1.0 - pp.density_ratio) * pp.gravity

    return drag + magnus + gravity


def _acceleration_old_model(
    vel: np.ndarray,
    ang_vel: np.ndarray,
    pp: PhysicsParams,
    drag_coeff: float,
    drag_coeff_linear: float,
    magnus_coeff: float,
    magnus_coeff_linear: float,
) -> np.ndarray:
    """Acceleration using the old linear Cd/Cm model.

    Reproduces C++ CalculateAcceleration with test_new_model=false.

    Cd = drag_coeff_linear * vel_spin_ratio + drag_coeff
    Cm = min(magnus_coeff_linear * vel_spin_ratio + magnus_coeff, 1.0)
    where vel_spin_ratio = |v| / (|ω| * r)   (or 1 if ω=0)
    """
    v_norm = np.linalg.norm(vel)
    w_norm = np.linalg.norm(ang_vel)

    if w_norm == 0.0:
        vel_spin_ratio = 1.0
    else:
        vel_spin_ratio = v_norm / (w_norm * pp.ball_radius)

    cd = drag_coeff_linear * vel_spin_ratio + drag_coeff
    cm = min(magnus_coeff_linear * vel_spin_ratio + magnus_coeff, 1.0)

    drag = -pp.half_area_to_volume_ratio * cd * pp.density_ratio * vel * v_norm
    magnus = -cm * pp.density_ratio * np.cross(vel, ang_vel)
    gravity = (1.0 - pp.density_ratio) * pp.gravity

    return drag + magnus + gravity


# ─── RK4 stepper ─────────────────────────────────────────────────────────────


def _rk4_step(
    state: BallState,
    dt: float,
    accel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> BallState:
    """Single RK4 step. Angular velocity is constant (not integrated).

    Matches the C++ StateIntegratorsT::RungeKutta4th exactly.
    """
    pos = state.position
    vel = state.linear_velocity
    ang = state.angular_velocity  # constant throughout

    # k1
    k1_vel = vel
    k1_acc = accel_fn(vel, ang)

    # k2
    vel2 = vel + (dt / 2.0) * k1_acc
    k2_vel = vel2
    k2_acc = accel_fn(vel2, ang)

    # k3
    vel3 = vel + (dt / 2.0) * k2_acc
    k3_vel = vel3
    k3_acc = accel_fn(vel3, ang)

    # k4
    vel4 = vel + dt * k3_acc
    k4_vel = vel4
    k4_acc = accel_fn(vel4, ang)

    # integrate
    new_pos = pos + (dt / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel)
    new_vel = vel + (dt / 6.0) * (k1_acc + 2.0 * k2_acc + 2.0 * k3_acc + k4_acc)

    return BallState(
        timestamp=state.timestamp + dt,
        position=new_pos,
        linear_velocity=new_vel,
        angular_velocity=ang.copy(),
    )


# ─── High-level integration functions ─────────────────────────────────────────


def predict_ball_state_new_model(
    state: BallState,
    dt: float,
    pp: PhysicsParams,
    get_cd: Callable[[float, float], float],
    get_cm: Callable[[float, float], float],
) -> BallState:
    """One RK4 step using the new (piecewise) Cd/Cm model.

    Drop-in replacement for:
        physics_layer.predict_ball_state(state, dt, physics_0226, False)
    when test_new_model=True.
    """
    def accel_fn(vel, ang):
        return _acceleration_new_model(vel, ang, pp, get_cd, get_cm)

    return _rk4_step(state, dt, accel_fn)


def predict_ball_state_old_model(
    state: BallState,
    dt: float,
    pp: PhysicsParams,
    params_dict: dict,
) -> BallState:
    """One RK4 step using the old (linear) Cd/Cm model.

    Drop-in replacement for:
        physics_layer.predict_ball_state(state, dt, physics_0426, False)
    when test_new_model=False.
    """
    drag_coeff = params_dict.get("drag_coeff", 0.55)
    drag_coeff_linear = params_dict.get("drag_coeff_linear", 0.0)
    magnus_coeff = params_dict.get("magnus_coeff", -0.001)
    magnus_coeff_linear = params_dict.get("magnus_coeff_linear", 0.1)

    def accel_fn(vel, ang):
        return _acceleration_old_model(
            vel, ang, pp,
            drag_coeff, drag_coeff_linear,
            magnus_coeff, magnus_coeff_linear,
        )

    return _rk4_step(state, dt, accel_fn)
