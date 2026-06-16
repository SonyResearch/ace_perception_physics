# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""racket contact models"""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class SimpleContact:
    """Lightweight contact descriptor for racket-contact models.

    Holds the fields the Python racket-contact models read from
    a contact object (orientation, velocities, etc.).

    Use this when calling any of the contact models in this module
    (``nakashima``, ``parametric_7p``, ``rcm_tangential``, …).
    """

    orientation: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    friction_coefficient: float = 0.5
    restitution_coefficient: float = 0.74
    vel_elasticity_coefficient: float = 0.6
    spin_elasticity_coefficient: float = 1.75


def nakashima(b_vel, b_spin, contact, params):
    """Original Nakashima matrix model (baseline).

    This is the reference implementation.  All other models in this module
    are variants of this one.  Key characteristics:

    - **1 free parameter**: tangential impulse coefficient ``params[0]``
      (converted to ``kp = params[0] / m``).
    - **Constant COR**: ``eps = 0.7`` (hard-coded).
    - **Full grip assumed**: kp enters both velocity and spin matrices
      with the standard Nakashima coupling (``kappa = 1.5`` implicitly
      via the ``1.5 * kp`` terms).
    - **No spin-z damping**: ``b_o[2,2] = 1`` (normal spin is preserved).
    - **No additive offset**: output is purely ``a_u @ v_rel + b_u @ w + v_racket``.

    Matrix structure (in contact frame [t1, t2, n]):
        a_u = diag(1-kp, 1-kp, -eps)
        b_u: off-diagonal ±kp·r coupling spin → velocity
        a_o: off-diagonal ±1.5·kp/r coupling velocity → spin
        b_o = diag(1-1.5·kp, 1-1.5·kp, 1)
    """
    rotate_system = np.array([
        [0., -1., 0.],
        [0., 0., -1.],
        [1., 0., 0.]
    ])

    rotation = R.from_quat(contact.orientation)
    racket_orientation = rotation.as_matrix()

    r = 0.02
    rho = 81.2
    m = 4./3. * np.pi * r * r * r * rho
    kp = params[0] / m
    eps = 0.81
    # eps = 0.7

    a_u = np.array([
        [1. - kp, 0., 0.],
        [0., 1. - kp, 0.],
        [0., 0., -eps],
    ])

    b_u = np.array([
        [0., kp * r, 0.],
        [-kp * r, 0., 0.],
        [0., 0., 0.],
    ])

    a_o = np.array([
        [0., -1.5 * kp / r, 0.],
        [1.5 * kp / r, 0., 0.],
        [0., 0., 0.],
    ])

    b_o = np.array([
        [1. - 1.5 * kp, 0., 0.],
        [0., 1. - 1.5 * kp, 0.],
        [0., 0., 1], #.8],
    ])

    a_u = racket_orientation @ rotate_system.T @ a_u @ rotate_system @ racket_orientation.T
    b_u = racket_orientation @ rotate_system.T @ b_u @ rotate_system @ racket_orientation.T
    a_o = racket_orientation @ rotate_system.T @ a_o @ rotate_system @ racket_orientation.T
    b_o = racket_orientation @ rotate_system.T @ b_o @ rotate_system @ racket_orientation.T

    lin_vel = a_u @ (b_vel - contact.linear_velocity) + b_u @ b_spin + contact.linear_velocity
    ang_vel = a_o @ (b_vel - contact.linear_velocity) + b_o @ b_spin

    return tuple([np.array(lin_vel, dtype=np.float32), np.array(ang_vel, dtype=np.float32)])


# ── Parametric contact model (from parametric_model_fitting.ipynb) ──
# kappa (inertia ratio) is fixed at the hollow-sphere value and NOT optimised.
_PARAMETRIC_7P_KAPPA = 1.5

# Joint-fit parameters on Tournament_Feb_2026 data (6 free params):
_PARAMETRIC_7P_DEFAULTS = {
    "e0":     0.8779,   # base COR
    "e1":    -0.0197,   # COR speed correction
    "et0":    0.8186,   # base tangential restitution
    "et1":   -0.0098,   # tangential restitution slope
    "c_spin": 0.1950,   # normal-spin damping
    "alpha":  0.0071,   # normal–tangential cross-coupling
}

PARAMETRIC_7P_NEW = {
    "e0":     0.910751,   # base COR
    "e1":    -0.021818,   # COR speed correction
    "et0":    0.792023,   # base tangential restitution
    "et1":   -0.007413,   # tangential restitution slope
    "c_spin":  0.214657,   # normal-spin damping
    "alpha":  0.003413,   # normal–tangential cross-coupling
}

def parametric_7p(b_vel, b_spin, contact, params):
    """Parametric contact model (6 free parameters, kappa fixed).

    Differences from ``nakashima`` (baseline):

    1. **Not in matrix form** — works with explicit impulse calculations
       in the contact frame rather than the a_u/b_u/a_o/b_o matrices.
    2. **Speed-dependent COR** (linear):
       ``e(|vn|) = clip(e0 + e1*|vn|, 0, 1)``  (2 free params)
       instead of constant ``eps = 0.7``.
    3. **Slip-speed-dependent tangential restitution** (new concept):
       ``e_t(|u|) = clip(et0 + et1*|u|, 0, 1.5)``  (2 free params).
       The baseline uses a single ``kp`` for tangential impulse;
       here the tangential impulse is
       ``jt = (1 + e_t) * |u| / (1 + kappa)``.
    4. **kappa fixed at 1.5** (hollow-sphere value) rather than
       being implicitly derived from ``params[0] / m``.
    5. **Normal-spin damping** ``c_spin``:
       ``wn_post = (1 - c_spin) * wn``.  The baseline preserves
       normal spin (``b_o[2,2] = 1``).
    6. **Normal–tangential cross-coupling** ``alpha``:
       ``vt_post += alpha * vn * vt`` — the normal impact speed
       modulates tangential restitution.  Not present in the baseline.

    Parameters: 6 free — ``[e0, e1, et0, et1, c_spin, alpha]``.

    Parameters
    ----------
    b_vel   : (3,) ball velocity in world frame
    b_spin  : (3,) ball angular velocity in world frame
    contact : BallContact with orientation, linear_velocity, etc.
    params  : list of 6 floats [e0, e1, et0, et1, c_spin, alpha]
              or ``None`` to use defaults from parametric_model_fitting.ipynb
    """
    if params is None:
        d = _PARAMETRIC_7P_DEFAULTS
        params = [d["e0"], d["e1"], d["et0"], d["et1"],
                  d["c_spin"], d["alpha"]]

    e0, e1, et0, et1, c_spin, alpha = params
    kappa = _PARAMETRIC_7P_KAPPA
    r = 0.02  # ball radius

    rotate_system = np.array([
        [0., -1., 0.],
        [0., 0., -1.],
        [1., 0., 0.]
    ])

    rotation = R.from_quat(contact.orientation)
    racket_orientation = rotation.as_matrix()

    # Project to contact frame  [tangent1, tangent2, normal]
    M = rotate_system @ racket_orientation.T
    v_rel = b_vel - contact.linear_velocity
    v_comp = M @ v_rel
    w_comp = M @ b_spin

    vt1, vt2, vn = v_comp[0], v_comp[1], v_comp[2]
    wt1, wt2, wn = w_comp[0], w_comp[1], w_comp[2]

    # ── Normal direction: speed-dependent COR ──
    e = np.clip(e0 + e1 * np.abs(vn), 0.0, 1.0)
    vn_post = -e * vn

    # ── Tangential direction: gripping with slip-speed restitution ──
    u1 = vt1 - r * wt2
    u2 = vt2 + r * wt1
    u_mag = np.sqrt(u1**2 + u2**2 + 1e-12)
    s1 = u1 / u_mag
    s2 = u2 / u_mag

    e_t = np.clip(et0 + et1 * u_mag, 0.0, 1.5)
    jt = (1.0 + e_t) * u_mag / (1.0 + kappa)

    vt1_post = vt1 - jt * s1 + alpha * vn * vt1
    vt2_post = vt2 - jt * s2 + alpha * vn * vt2

    # ── Spin updates ──
    wt1_post = wt1 - kappa * jt * s2 / r
    wt2_post = wt2 + kappa * jt * s1 / r
    wn_post  = (1.0 - c_spin) * wn

    # ── Transform back to world frame ──
    M_inv = racket_orientation @ rotate_system.T
    lin_vel = M_inv @ np.array([vt1_post, vt2_post, vn_post]) + contact.linear_velocity
    ang_vel = M_inv @ np.array([wt1_post, wt2_post, wn_post])

    return tuple([np.array(lin_vel, dtype=np.float32), np.array(ang_vel, dtype=np.float32)])


# ── rcm_tangential: like parametric_7p but without the alpha cross-coupling ──

_RCM_TANGENTIAL_DEFAULTS = {
    "e0":     0.8779,   # base COR
    "e1":    -0.0197,   # COR speed correction
    "et0":    0.8186,   # base tangential restitution
    "et1":   -0.0098,   # tangential restitution slope
    "c_spin": 0.1950,   # normal-spin damping
}

RCM_TANGENTIAL_NEW = {
    "e0":     0.915520,   # base COR
    "e1":    -0.022830,   # COR speed correction
    "et0":    0.810542,   # base tangential restitution
    "et1":   -0.008380,   # tangential restitution slope
    "c_spin":  0.217442,   # normal-spin damping
}


def rcm_tangential(b_vel, b_spin, contact, params):
    """Tangential-only parametric model in Nakashima matrix form (5 free params).

    Differences from ``nakashima`` (baseline):

    1. **Back in matrix form** (like the baseline) but with
       **state-dependent** matrix entries instead of constant ones.
    2. **Speed-dependent COR** (linear, clipped):
       ``eps = clip(e0 + e1*|vn|, 0, 1)``  (2 free params)
       instead of constant ``eps = 0.7``.
    3. **State-dependent tangential grip** ``kp``:
       ``kp = min((1+e_t)/(1+kappa), 1)``
       where ``e_t = clip(et0 + et1*|u|, 0, 1.5)``.
       This replaces the single ``kp = params[0]/m`` of the baseline.
       The ``min(..., 1)`` clamp prevents super-grip (kp > 1).
    4. **kappa fixed at 1.5** (hollow sphere) and enters the matrices
       explicitly (``kappa * kp`` in ``a_o``, ``b_o``).
       The baseline uses ``1.5 * kp`` hard-coded.
    5. **Normal-spin damping** ``c_spin``:
       ``b_o[2,2] = 1 - c_spin``.  The baseline has ``b_o[2,2] = 1``.
    6. **No alpha cross-coupling** — unlike ``parametric_7p``, there
       is no ``alpha * vn * vt`` correction.
    7. **No additive offset** — unlike the piecewise variants, the
       output is purely matrix-based.

    Parameters: 5 free — ``[e0, e1, et0, et1, c_spin]``.

    Parameters
    ----------
    b_vel   : (3,) ball velocity in world frame
    b_spin  : (3,) ball angular velocity in world frame
    contact : BallContact with orientation, linear_velocity, etc.
    params  : list of 5 floats [e0, e1, et0, et1, c_spin]
              or ``None`` to use defaults
    """
    if params is None:
        d = _RCM_TANGENTIAL_DEFAULTS
        params = [d["e0"], d["e1"], d["et0"], d["et1"],
                  d["c_spin"]]

    e0, e1, et0, et1, c_spin = params
    kappa = _PARAMETRIC_7P_KAPPA
    r = 0.02  # ball radius

    rotate_system = np.array([
        [0., -1., 0.],
        [0., 0., -1.],
        [1., 0., 0.]
    ])

    rotation = R.from_quat(contact.orientation)
    racket_orientation = rotation.as_matrix()

    # ── Pre-compute state-dependent coefficients in contact frame ──
    v_relative = b_vel - contact.linear_velocity
    w_relative = b_spin - contact.angular_velocity
    v_comp = rotate_system @ racket_orientation.T @ v_relative
    w_comp = rotate_system @ racket_orientation.T @ w_relative

    vt1, vt2, vn = v_comp[0], v_comp[1], v_comp[2]
    wt1, wt2 = w_comp[0], w_comp[1]

    # Speed-dependent COR (normal direction)
    eps = np.clip(e0 + e1 * np.abs(vn), 0.0, 1.0)

    # Slip speed and tangential restitution
    u1 = vt1 - r * wt2
    u2 = vt2 + r * wt1
    u_mag = np.sqrt(u1**2 + u2**2 + 1e-12)
    e_t = np.clip(et0 + et1 * u_mag, 0.0, 1.5)

    # Effective tangential grip fraction (clamped to full grip)
    kp = (1.0 + e_t) / (1.0 + kappa)

    # ── Build Nakashima-style matrices ──
    a_u = np.array([
        [1. - kp, 0., 0.],
        [0., 1. - kp, 0.],
        [0., 0., -eps],
    ])

    b_u = np.array([
        [0., kp * r, 0.],
        [-kp * r, 0., 0.],
        [0., 0., 0.],
    ])

    a_o = np.array([
        [0., -kappa * kp / r, 0.],
        [kappa * kp / r, 0., 0.],
        [0., 0., 0.],
    ])

    b_o = np.array([
        [1. - kappa * kp, 0., 0.],
        [0., 1. - kappa * kp, 0.],
        [0., 0., 1. - c_spin],
    ])

    # ── Rotate matrices to world frame ──
    a_u = racket_orientation @ rotate_system.T @ a_u @ rotate_system @ racket_orientation.T
    b_u = racket_orientation @ rotate_system.T @ b_u @ rotate_system @ racket_orientation.T
    a_o = racket_orientation @ rotate_system.T @ a_o @ rotate_system @ racket_orientation.T
    b_o = racket_orientation @ rotate_system.T @ b_o @ rotate_system @ racket_orientation.T

    lin_vel = a_u @ v_relative + b_u @ w_relative + contact.linear_velocity
    ang_vel = a_o @ v_relative + b_o @ w_relative + contact.angular_velocity

    return tuple([np.array(lin_vel, dtype=np.float32), np.array(ang_vel, dtype=np.float32)])
