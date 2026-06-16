# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Racket contact extraction and HDF5 storage.

Extracts racket-ball contact events from trajectory data using physics-based
analysis and optimisation.  Results are stored in the HDF5 file under
``ground_truth_200/racket_contacts``.

Optionally writes ``extracted.csv`` for backward compatibility.

This module adapts the logic from ``extract_racket_contacts.py`` to work
with the MatchCollection data structure.
"""

import pathlib
import warnings

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.stats import mode

from ace_evaluation.utilities import math_utilities
from ace_evaluation.utilities.physics_params import get_params
# NOTE: get_magnus_estimate_v2 has a minor data discrepancy vs the C++
# GetMagnusCoefficientNew at v=7.5: Python uses (350, 0.09) while C++ uses
# (350, 0.095), which cascades into different m1/c1/quadratic coefficients
# for that reference velocity.  All other reference velocities match exactly.
from ace_evaluation.utilities.aerodynamics_utilities import get_magnus_estimate_v2
from ace_evaluation import racket_contacts
from ace_evaluation.update_hdf5.column_mapping import rename_to_flat

from ace_evaluation.utilities.data_classes import (
    MatchCollection,
    RacketContactEvent,
)

from ace_evaluation.utilities.aerodynamics_integrator import (
    BallState,
    make_physics_params,
    predict_ball_state_new_model,
    predict_ball_state_old_model,
    get_drag_coefficient_new,
    get_magnus_coefficient_new,
)
from ace_evaluation.racket_contacts import SimpleContact


# ── Standalone replacements for proprietary modules ───────────────────────

class RacketContactLocation:
    """Minimal replacement for synapse.core.RacketContactLocation."""
    NONE = 0
    RED = 1
    BLACK = 2
    EDGE = 3


def compute_racket_contact_quat_v(b_vel, r_vel, r_quat):
    """Choose racket face so that the surface normal opposes the relative velocity.

    If the relative velocity has a positive dot product with the racket's
    x-axis (the normal), flip the quaternion 180° around the local z-axis
    so that the contact normal faces the ball's approach direction.

    Replacement for physics_layer.serve_racket_residual_model.compute_racket_contact_quat_v.
    """
    v_relative = b_vel - r_vel
    rot_mat = math_utilities.q_to_rot_mat(r_quat, "xyzw")
    if np.dot(v_relative, rot_mat[:, 0]) > 0:
        r_quat = math_utilities.quat_quat_rot(
            r_quat, np.array([0.0, 0.0, 1.0, 0.0]), q_format="xyzw"
        )
    return r_quat

# ── Physical constants ────────────────────────────────────────────────────
GRAVITY = 9.81
BALL_RADIUS = 0.02
NET_HEIGHT = 0.1525
DT = 1.0 / 200.0


# ── ONNX RCM helpers (ported from rcm.py) ────────────────────────────────

_ROTATION_NEG = np.array([1, -1, 1, -1], dtype=np.float32)
_RCM_OUTPUTS = ["Velocity_Output_Att", "Spin_Output_Att"]


def _rotate_to_surface(ori):
    """Rotate quaternion for BLACK side contact (always positive x)."""
    out = np.empty(4, dtype=np.float32)
    out.reshape(2, 2)[:] = ori.reshape(2, 2)[:, ::-1]
    out *= _ROTATION_NEG
    return out


def _init_onnx_rcm(onnx_file):
    """Initialise an ONNX RCM session and return pre-allocated buffers.

    Returns (session, input_dict, velxyz, spinxyz, posyz, velocity_out, angular_velocity_out, position_out).
    """
    data = np.zeros(17, dtype=np.float32)
    outputs_buf = data[:9]
    inputs_buf = data[9:]

    position_out = outputs_buf[:3]
    velocity_out = outputs_buf[3:6]
    angular_velocity_out = outputs_buf[6:]

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opt.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(onnx_file, opt, providers=["CPUExecutionProvider"])

    vel_shape = next(x.shape for x in session.get_outputs() if x.name == _RCM_OUTPUTS[0])
    spin_shape = next(x.shape for x in session.get_outputs() if x.name == _RCM_OUTPUTS[1])
    _ort_outputs = [
        ort.OrtValue.ortvalue_from_numpy(velocity_out.reshape(vel_shape)),
        ort.OrtValue.ortvalue_from_numpy(angular_velocity_out.reshape(spin_shape)),
    ]

    velxyz = inputs_buf[:3]
    spinxyz = inputs_buf[3:6]
    posyz = inputs_buf[6:]
    input_dict = {"Ball_Pre_State": ort.OrtValue.ortvalue_from_numpy(
        inputs_buf.reshape(session.get_inputs()[0].shape)
    )}

    return session, input_dict, velxyz, spinxyz, posyz, velocity_out, angular_velocity_out, position_out


def _determine_contact_face(ball_pos, racket_pos, racket_quat):
    """Determine which racket face the ball contacts (RED or BLACK).

    Returns the appropriate ``RacketContactLocation``.
    """
    ball_local = R.from_quat(racket_quat).inv().apply(ball_pos - racket_pos)
    return RacketContactLocation.RED if ball_local[0] > 0 else RacketContactLocation.BLACK


def _run_onnx_rcm(session, input_dict, velxyz, spinxyz, posyz,
                   _velocity_out, _angular_velocity_out,
                   ball_pos, ball_vel, ball_spin,
                   racket_pos, racket_quat, racket_vel,
                   racket_angular_vel=None):
    """Run the ONNX RCM and return (post_velocity, post_spin) in world frame.

    When *racket_angular_vel* is provided, uses the RcmPlus formulation
    (subtracts angular velocity contributions before the model and adds
    them back afterwards).  Otherwise uses the OldRcm formulation.
    """
    location = _determine_contact_face(ball_pos, racket_pos, racket_quat)
    racket_ball = ball_pos - racket_pos

    if racket_angular_vel is not None:
        # RcmPlus: account for racket angular velocity
        r_omega = np.cross(racket_angular_vel, racket_ball)
        velxyz[:] = ball_vel - racket_vel - r_omega
        spinxyz[:] = ball_spin - racket_angular_vel
    else:
        # OldRcm: ignore racket angular velocity
        r_omega = None
        velxyz[:] = ball_vel - racket_vel
        spinxyz[:] = ball_spin

    orientation = _rotate_to_surface(racket_quat) if location == RacketContactLocation.BLACK else racket_quat.copy()
    rotation = R.from_quat(orientation)
    inv_rotation = rotation.inv()

    velxyz[:] = inv_rotation.apply(velxyz)
    spinxyz[:] = inv_rotation.apply(spinxyz)
    posyz[:] = inv_rotation.apply(racket_ball)[1:]

    outputs = session.run(_RCM_OUTPUTS, input_dict)

    post_vel = rotation.apply(outputs[0]).flatten() + racket_vel
    post_spin = rotation.apply(outputs[1]).flatten()

    if racket_angular_vel is not None:
        post_vel += r_omega
        post_spin += racket_angular_vel

    return post_vel.astype(np.float64), post_spin.astype(np.float64)


# ── Old (0426) 6-input ONNX RCM (commit 1757e3ec) ───────────────────────

_RCM_0426_OUTPUTS = ["Velocity_Output", "Spin_Output"]


def _init_onnx_rcm_0426(onnx_file):
    """Initialise the old 6-input ONNX RCM session (commit 1757e3ec).

    The old model takes [vx, vy, vz, wx, wy, wz] in racket frame (no position)
    and returns post-contact velocity and spin in racket frame.
    """
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opt.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(onnx_file, opt, providers=["CPUExecutionProvider"])
    return session


def _run_onnx_rcm_0426(session, ball_vel, ball_spin, _racket_pos, racket_quat, racket_vel):
    """Run the old 6-input ONNX RCM (commit 1757e3ec).

    Matches the C++ ResolveContactRacket with use_compact_residual_model=True
    at that commit: transforms ball state into racket frame (no face detection,
    no position input), runs inference, transforms back.
    """
    racket_rotation = R.from_quat(racket_quat)
    inv_rotation = racket_rotation.inv()

    # Ball velocity relative to racket, in racket frame
    ball_racket_velocity = inv_rotation.apply(ball_vel - racket_vel).astype(np.float32)
    # Ball spin in racket frame
    ball_racket_spin = inv_rotation.apply(ball_spin).astype(np.float32)

    # Build 6-element input: [vx, vy, vz, wx, wy, wz]
    input_data = np.concatenate([ball_racket_velocity, ball_racket_spin]).reshape(1, 6)

    outputs = session.run(_RCM_0426_OUTPUTS, {"Ball_Pre_State": input_data})

    # outputs[0] = velocity in racket frame, outputs[1] = spin in racket frame
    post_vel_racket = outputs[0].flatten()
    post_spin_racket = outputs[1].flatten()

    # Transform back to world frame
    post_vel = racket_rotation.apply(post_vel_racket) + racket_vel
    post_spin = racket_rotation.apply(post_spin_racket)

    return post_vel.astype(np.float64), post_spin.astype(np.float64)


# ── Aerodynamics ODE (racket-contact variant) ────────────────────────────

def _aerodynamics_diff(_, s, const_params):
    """Aerodynamics differential equations.

    ``const_params``:
        [wx, wy, wz, rho_air, rho_ball, radius, c_drag]
    """
    _, _, _, vx, vy, vz = s
    wx, wy, wz, rho_air, rho_ball, radius, c_drag = const_params
    vmag = np.sqrt(vx * vx + vy * vy + vz * vz)
    wmag = np.sqrt(wx * wx + wy * wy + wz * wz)
    kd = c_drag * 3.0 / (8.0 * radius) * rho_air / rho_ball
    c_magnus = get_magnus_estimate_v2(vmag, wmag)
    km = c_magnus * rho_air / rho_ball
    buoyancy_factor = 1.0 - rho_air / rho_ball
    dsdt = [
        vx, vy, vz,
        -kd * vmag * vx - km * (vy * wz - vz * wy),
        -kd * vmag * vy - km * (vz * wx - vx * wz),
        -kd * vmag * vz - km * (vx * wy - vy * wx) - buoyancy_factor * GRAVITY,
    ]
    return dsdt


def _compute_fitness_max(ref_pos, sim_pos, dt):
    """Maximum position error between matched time points.

    NOTE: max-error is non-smooth (kinks where argmax switches between
    points), which is suboptimal for gradient-based L-BFGS-B.  MSE would
    be a better choice but this works well enough in practice.
    """
    pos_diff_max = 0.0
    for sample_ref in ref_pos:
        for sample_sim in sim_pos:
            if np.abs(sample_ref[0] - sample_sim[0]) < dt * 0.5:
                pos_error = np.linalg.norm(sample_ref[1:4] - sample_sim[1:4])
                pos_diff_max = max(pos_diff_max, pos_error)
    return pos_diff_max


def _solve_aerodynamics_vel(params, input_data, tstart, tend, dt, const_params):
    """Find optimal initial velocity by simulating and comparing to observations."""
    try:
        t_eval = np.arange(tstart, tend, dt)
        initial_state = [
            input_data.iloc[0].ball_position_x,
            input_data.iloc[0].ball_position_y,
            input_data.iloc[0].ball_position_z,
            params[0], params[1], params[2],
        ]
        result = solve_ivp(
            _aerodynamics_diff, [tstart, tend], initial_state,
            t_eval=t_eval[:-1], method="RK45", args=(np.array(const_params),),
        )
        if not result.success:
            return 1e6
        ref_pos = np.column_stack([
            input_data["time"].to_numpy(),
            input_data["ball_position_x"].to_numpy(),
            input_data["ball_position_y"].to_numpy(),
            input_data["ball_position_z"].to_numpy(),
        ])
        sim_pos = np.column_stack([result.t, result.y[0], result.y[1], result.y[2]])
        return _compute_fitness_max(ref_pos, sim_pos, dt)
    except Exception:  # pylint: disable=broad-exception-caught
        return 1e6


# ── Helper functions (ported from extract_racket_contacts.py) ─────────────

def _update_orientation_exponential(q, omega, dt):
    """Integrate quaternion with angular velocity."""
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-10:
        return q
    axis = omega / np.linalg.norm(omega)
    sin_half = np.sin(angle * 0.5)
    cos_half = np.cos(angle * 0.5)
    rotation_q = np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half])
    q_new = math_utilities.quaternion_multiply(rotation_q, q)
    return math_utilities.quaternion_normalize(q_new)


def _bisect_contact_time(x_ball, v_ball, x_racket, v_racket, q_racket, w_racket,
                         radius, tol, t_guess, _t_interval, max_steps=20):
    """Bisection search for exact contact time when ball touches racket surface."""
    t_contact = t_guess
    t_offset = t_guess * 0.5
    x_ball_tmp = x_ball
    x_racket_tmp = x_racket
    q_racket_tmp = q_racket
    d_local = np.zeros(3)

    for _ in range(max_steps):
        x_ball_tmp = x_ball + t_contact * v_ball
        x_racket_tmp = x_racket + t_contact * v_racket
        q_racket_tmp = _update_orientation_exponential(q_racket, w_racket, t_contact)
        racket_orientation = R.from_quat(q_racket_tmp)
        d_local = racket_orientation.as_matrix().T @ (x_ball_tmp - x_racket_tmp)
        if radius - tol <= d_local[0] <= radius + tol:
            break
        elif d_local[0] < radius - tol:
            t_contact -= t_offset
        else:
            t_contact += t_offset
        t_offset *= 0.5

    return t_contact, x_ball_tmp, x_racket_tmp, q_racket_tmp, d_local


def _estimate_quantities_at_impact(sim_data, contact_idx, shot, shot_contact_index, racket_angular_vel):
    """Estimate ball and racket states at impact using bisection search."""
    tol = 0.001
    radius = 0.027

    t_interval = (shot.loc[shot_contact_index + 2]['time'] - shot.loc[shot_contact_index - 2]['time'])
    t_guess = 0.5 * t_interval

    x_racket_pre = np.array([
        shot.loc[shot_contact_index - 2]['racket_1_position_x'],
        shot.loc[shot_contact_index - 2]['racket_1_position_y'],
        shot.loc[shot_contact_index - 2]['racket_1_position_z'],
    ])
    q_racket_pre = np.array([
        shot.loc[shot_contact_index - 2]['racket_1_orientation_x'],
        shot.loc[shot_contact_index - 2]['racket_1_orientation_y'],
        shot.loc[shot_contact_index - 2]['racket_1_orientation_z'],
        shot.loc[shot_contact_index - 2]['racket_1_orientation_w'],
    ])
    x_racket_post = np.array([
        shot.loc[shot_contact_index + 2]['racket_1_position_x'],
        shot.loc[shot_contact_index + 2]['racket_1_position_y'],
        shot.loc[shot_contact_index + 2]['racket_1_position_z'],
    ])
    q_racket_post = np.array([
        shot.loc[shot_contact_index + 2]['racket_1_orientation_x'],
        shot.loc[shot_contact_index + 2]['racket_1_orientation_y'],
        shot.loc[shot_contact_index + 2]['racket_1_orientation_z'],
        shot.loc[shot_contact_index + 2]['racket_1_orientation_w'],
    ])

    dt_racket = shot.loc[shot_contact_index - 1]['time'] - shot.loc[shot_contact_index - 2]['time']
    if dt_racket == 0:
        print(f"invalid time interval {shot.loc[shot_contact_index-1]['time']} - {shot.loc[shot_contact_index-2]['time']}")

    v_racket_pre = np.array([
        (shot.loc[shot_contact_index - 1][f'racket_1_position_{c}'] - shot.loc[shot_contact_index - 2][f'racket_1_position_{c}']) / dt_racket
        for c in ('x', 'y', 'z')
    ])
    dt_racket_post = shot.loc[shot_contact_index + 2]['time'] - shot.loc[shot_contact_index + 1]['time']
    v_racket_post = np.array([
        (shot.loc[shot_contact_index + 2][f'racket_1_position_{c}'] - shot.loc[shot_contact_index + 1][f'racket_1_position_{c}']) / dt_racket_post
        for c in ('x', 'y', 'z')
    ])
    w_racket = np.array(racket_angular_vel)

    x_pre = np.array([sim_data.iloc[contact_idx - 2][k] for k in ['x', 'y', 'z']])
    v_pre = np.array([sim_data.iloc[contact_idx - 2][k] for k in ['vx', 'vy', 'vz']])
    x_post = np.array([sim_data.iloc[contact_idx + 2][k] for k in ['x', 'y', 'z']])
    v_post = np.array([sim_data.iloc[contact_idx + 2][k] for k in ['vx', 'vy', 'vz']])

    q_racket_pre = compute_racket_contact_quat_v(v_pre, v_racket_pre, q_racket_pre)
    # Negate post velocities so the normal faces the departure side (where the
    # ball actually is), matching the backward-in-time bisection perspective.
    q_racket_post = compute_racket_contact_quat_v(-v_post, -v_racket_post, q_racket_post)

    # Pre-contact bisection
    t_contact_pre, x_pre_refined, x_racket_pre_refined, q_racket_pre_refined, d_pre_local = \
        _bisect_contact_time(x_pre, v_pre, x_racket_pre, v_racket_pre, q_racket_pre, w_racket, radius, tol, t_guess, t_interval)
    r_pre = np.array([0, d_pre_local[1], d_pre_local[2]])
    racket_orientation_pre = R.from_quat(q_racket_pre_refined)
    v_racket_pre_refined = v_racket_pre * ((t_interval - t_contact_pre) / t_interval) + v_racket_post * (t_contact_pre / t_interval)
    v_racket_w_angvel_pre = v_racket_pre_refined + np.cross(w_racket, racket_orientation_pre.as_matrix() @ r_pre)

    # Post-contact bisection (backward in time: negate velocities)
    t_contact_post, x_post_refined, x_racket_post_refined, q_racket_post_refined, d_post_local = \
        _bisect_contact_time(x_post, -v_post, x_racket_post, -v_racket_post, q_racket_post, -w_racket, radius, tol, t_guess, t_interval)
    if not (radius - tol <= d_post_local[0] <= radius + tol):
        print(f"  WARNING: post-contact bisection did not converge: d_local[0]={d_post_local[0]:.6f}, expected ~{radius}")
    r_post = np.array([0, d_post_local[1], d_post_local[2]])
    racket_orientation_post = R.from_quat(q_racket_post_refined)
    v_racket_post_refined_val = v_racket_post * ((t_interval - t_contact_post) / t_interval) + v_racket_pre * (t_contact_post / t_interval)
    v_racket_w_angvel_post = v_racket_post_refined_val + np.cross(w_racket, racket_orientation_post.as_matrix() @ r_post)

    return (
        t_contact_pre, x_pre_refined, x_racket_pre_refined, v_racket_pre_refined, q_racket_pre_refined, d_pre_local, v_racket_w_angvel_pre,
        t_contact_post, x_post_refined, x_racket_post_refined, v_racket_post_refined_val, q_racket_post_refined, d_post_local, v_racket_w_angvel_post,
    )


def _refine_racket_contacts(rally_df):
    """Detect racket contact events using acceleration-based detection."""
    t = rally_df['time'].values
    dt = np.gradient(t)
    x = rally_df['ball_position_x'].values
    vx = np.gradient(x) / dt
    ax = np.gradient(vx) / dt

    # NOTE: np.greater finds only positive acceleration peaks.  This works for
    # shot_p1 (ball travels negative→positive x) but would miss contacts with
    # a negative ax spike.  Inherited from the generic contact-detection code;
    # acceptable here since we only process shot_p1 events.
    maxima = argrelextrema(ax, np.greater, order=50)[0]
    candidate_indices = []
    for i in maxima:
        if abs(ax[i]) > 100 and abs(x[i]) > 0.1 and t[i] < t[len(t) - 5]:
            candidate_indices.append(i)

    clustered_indices = []
    if candidate_indices:
        current_cluster = [candidate_indices[0]]
        for idx in candidate_indices[1:]:
            if t[idx] - t[current_cluster[-1]] <= 0.1:
                current_cluster.append(idx)
            else:
                best = max(current_cluster, key=lambda j: abs(ax[j]))
                clustered_indices.append(best)
                current_cluster = [idx]
        best = max(current_cluster, key=lambda j: abs(ax[j]))
        clustered_indices.append(best)

    contact_times = []
    for idx in clustered_indices:
        pre_window = slice(max(0, idx - 5), idx - 1)
        post_window = slice(idx + 2, min(len(vx), idx + 6))
        vx_pre = np.mean(vx[pre_window])
        vx_post = np.mean(vx[post_window])
        x_pre = x[idx - 1]
        x_post = x[idx + 1]
        t_diff = t[idx + 1] - t[idx - 1]
        denom = vx_pre - vx_post
        if abs(denom) > 1e-9:
            t_contact = t[idx - 1] + (x_post - x_pre - vx_post * t_diff) / denom
            contact_times.append(t_contact)

    if len(contact_times) == 0:
        return None
    if len(contact_times) > 1:
        print(f"Warning: detected {len(contact_times)} contacts, expected 1. Times: {contact_times}")
    return contact_times[0]


def _is_ball_toss(events, event_idx, shot_pre, vel_threshold=1.0, x_behind_table=1.37):
    """Detect whether a racket contact event is a ball toss (serve toss)."""
    if event_idx == 0:
        return False
    prev_event = events[event_idx - 1]
    if "start" not in prev_event.type:
        return False
    if len(shot_pre) < 2:
        return False
    dt = shot_pre["time"].diff()
    vx = shot_pre["ball_position_x"].diff() / dt
    vy = shot_pre["ball_position_y"].diff() / dt
    mean_vx = vx.dropna().mean()
    mean_vy = vy.dropna().mean()
    if abs(mean_vx) > vel_threshold or abs(mean_vy) > vel_threshold:
        return False
    mean_x = shot_pre["ball_position_x"].mean()
    return abs(mean_x) >= x_behind_table


def _estimate_spin(trajectory):
    """Estimate ball spin using mode of gt200 spin columns.

    Matches the original estimate_spin(): calls calculate_spin (which
    drops the last row via [:-1]), removes NaN rows, then returns the
    mode of each component.
    """
    spin = math_utilities.calculate_spin(trajectory)  # applies [:-1]
    spin = spin[~np.isnan(spin).any(axis=1)]
    if len(spin) == 0:
        return 0.0, 0.0, 0.0

    wx_mode = mode(spin[:, 0], keepdims=True).mode[0]
    wy_mode = mode(spin[:, 1], keepdims=True).mode[0]
    wz_mode = mode(spin[:, 2], keepdims=True).mode[0]
    return float(wx_mode), float(wy_mode), float(wz_mode)


def _sim_python(start, end, pos, const_params, optimization_result, t_eval):
    """Simulate trajectory with optimised initial velocity."""
    result = solve_ivp(
        _aerodynamics_diff,
        [start, end],
        (pos[0], pos[1], pos[2], optimization_result.x[0], optimization_result.x[1], optimization_result.x[2]),
        t_eval=t_eval, method="RK45", args=(np.array(const_params),),
    )
    sim_data = pd.DataFrame({
        "t": result.t, "x": result.y[0], "y": result.y[1], "z": result.y[2],
        "vx": result.y[3], "vy": result.y[4], "vz": result.y[5],
        "wx": [const_params[0]] * len(result.t),
        "wy": [const_params[1]] * len(result.t),
        "wz": [const_params[2]] * len(result.t),
    })
    return sim_data


def _get_ball_type(sequence_name):
    """Determine ball type from sequence name."""
    seq_lower = str(sequence_name).lower()
    for patterns, ball_type in [
        (["logo-", "_logo_", "-l_", "logo"], "logo"),
        (["_dhs_", "dhs"], "dhs"),
        (["tihbar", "tibhar"], "tibhar"),
        (["_ll_", "ladylogo"], "ladylogo"),
    ]:
        if any(p in seq_lower for p in patterns):
            return ball_type
    return "ladybug"


# ── Polynomial-fit contact refinement (Method B) ─────────────────────────

RACKET_RADIUS = 0.075
RACKET_THICKNESS = 0.014

_POLYFIT_PRE_WINDOW = (-115, -15)   # ms relative to event label
_POLYFIT_POST_WINDOW = (15, 115)    # ms relative to event label


class _PolyMultiAxis:
    """Multi-axis polynomial fitted to trajectory data."""

    def __init__(self, coeffs):
        self.coeffs = np.array(coeffs)

    def __call__(self, t):
        return np.stack([np.polyval(self.coeffs[i], t) for i in range(len(self.coeffs))], axis=-1)

    def derivative(self, order=1):
        return _PolyMultiAxis([np.polyder(self.coeffs[i], order) for i in range(len(self.coeffs))])


def _fit_multiaxis_poly(x, y, deg=2):
    """Fit degree-deg polynomial per axis.  Returns (poly, mae_per_axis)."""
    valid = ~np.isnan(y).any(axis=1)
    x_v, y_v = x[valid], y[valid]
    if len(x_v) < deg + 1:
        raise ValueError(f"Not enough valid points ({len(x_v)}) for degree {deg}")
    coeffs = [np.polyfit(x_v, y_v[:, i], deg) for i in range(y_v.shape[1])]
    mae = np.array([np.mean(np.abs(np.polyval(coeffs[i], x_v) - y_v[:, i])) for i in range(y_v.shape[1])])
    return _PolyMultiAxis(coeffs), mae


def _polyfit_contact_instant(pre_poly, post_poly):
    """Find contact time minimizing squared distance between pre/post polynomials.

    Returns (t_contact_ms, ball_pos_contact, ball_pos_diff).
    """
    diff_coeffs = pre_poly.coeffs - post_poly.coeffs
    a = diff_coeffs[:, 0]
    b = diff_coeffs[:, 1]
    c = diff_coeffs[:, 2]

    A = np.dot(a, a)
    B = 2 * np.dot(a, b)
    C = np.dot(b, b) + 2 * np.dot(a, c)
    D = 2 * np.dot(b, c)

    roots = np.roots([4 * A, 3 * B, 2 * C, D])
    t_candidates = roots.real
    f_vals = np.array([np.sum((a * t ** 2 + b * t + c) ** 2) for t in t_candidates])
    t_contact = t_candidates[np.argmin(f_vals)]

    pos_pre = pre_poly(np.array([t_contact]))[0]
    pos_post = post_poly(np.array([t_contact]))[0]
    return float(t_contact), 0.5 * (pos_pre + pos_post), pos_post - pos_pre


def _detect_racket_contact_cylinder(ball_pos, racket_pos, racket_ori_quat,
                                     racket_radius=RACKET_RADIUS,
                                     racket_thickness=RACKET_THICKNESS,
                                     ball_radius=BALL_RADIUS):
    """Cylinder-sphere collision.  Returns (idx, face, ball_local) or (None, None, None)."""
    rot = R.from_quat(racket_ori_quat)
    ball_local = rot.inv().apply(ball_pos - racket_pos)
    radial = np.sqrt(ball_local[:, 1] ** 2 + ball_local[:, 2] ** 2)
    axial = np.abs(ball_local[:, 0])
    contact = (radial <= racket_radius) & (axial <= racket_thickness / 2 + ball_radius)
    idxs = np.flatnonzero(contact)
    if len(idxs) == 0:
        return None, None, None
    idx = idxs[0]
    face = 1 if ball_local[idx, 0] > 0 else 2
    return idx, face, ball_local[idx]


def _polyfit_refine_contact(shot, event_timestamp):
    """Run polynomial-fit refinement (Method B) on a shot DataFrame.

    Fits quadratic polynomials to pre/post ball trajectories around the
    event label, finds the contact instant via quartic minimisation,
    and detects racket contact via cylinder-sphere collision with
    upsampled racket trajectory.

    Parameters
    ----------
    shot : pd.DataFrame
        Rally data with columns ``time``, ``ball_position_{x,y,z}``,
        ``racket_1_position_{x,y,z}``, ``racket_1_orientation_{x,y,z,w}``.
    event_timestamp : float
        Shot label timestamp in seconds.

    Returns
    -------
    dict or None
        Polyfit-refined contact quantities, or *None* if refinement fails.
    """
    times_ms = (shot["time"].values * 1000).astype(int)
    event_ms = int(event_timestamp * 1000)

    pre_mask = (times_ms >= event_ms + _POLYFIT_PRE_WINDOW[0]) & (times_ms < event_ms + _POLYFIT_PRE_WINDOW[1])
    post_mask = (times_ms >= event_ms + _POLYFIT_POST_WINDOW[0]) & (times_ms < event_ms + _POLYFIT_POST_WINDOW[1])

    ball_cols = ["ball_position_x", "ball_position_y", "ball_position_z"]
    pre_times = times_ms[pre_mask].astype(float)
    pre_pos = shot[ball_cols].values[pre_mask]
    post_times = times_ms[post_mask].astype(float)
    post_pos = shot[ball_cols].values[post_mask]

    if len(pre_times) < 3 or len(post_times) < 3:
        return None

    # 1. Polynomial fitting
    try:
        pre_poly, pre_mae = _fit_multiaxis_poly(pre_times, pre_pos)
        post_poly, post_mae = _fit_multiaxis_poly(post_times, post_pos)
    except (ValueError, np.linalg.LinAlgError):
        return None

    # 2. Contact instant (quartic minimisation)
    t_contact_ms, ball_pos, ball_pos_diff = _polyfit_contact_instant(pre_poly, post_poly)

    # 3. Ball velocity from polynomial derivative (m/ms → m/s)
    ball_vel_pre = 1000.0 * pre_poly.derivative()(np.array([t_contact_ms]))[0]
    ball_vel_post = 1000.0 * post_poly.derivative()(np.array([t_contact_ms]))[0]

    # 4. Racket collision detection (±250 ms window)
    racket_mask = (times_ms >= event_ms - 250) & (times_ms < event_ms + 250)
    r_times = times_ms[racket_mask]
    r_pos = shot[["racket_1_position_x", "racket_1_position_y", "racket_1_position_z"]].values[racket_mask]
    r_ori = shot[["racket_1_orientation_x", "racket_1_orientation_y", "racket_1_orientation_z",
                  "racket_1_orientation_w"]].values[racket_mask]

    if len(r_times) < 5 or np.isnan(r_ori).any() or np.isnan(r_pos).any():
        return None

    ball_pos_tiled = np.tile(ball_pos, (len(r_times), 1))
    contact_idx, _contact_face, ball_local = _detect_racket_contact_cylinder(ball_pos_tiled, r_pos, r_ori)
    if contact_idx is None or contact_idx == 0:
        return None

    # 5. Racket velocity at coarse contact (backward difference, 200 Hz)
    dt_r = (r_times[contact_idx] - r_times[contact_idx - 1]) / 1000.0
    if dt_r <= 0:
        return None
    racket_vel = (r_pos[contact_idx] - r_pos[contact_idx - 1]) / dt_r
    r_rel = R.from_quat(r_ori[contact_idx]) * R.from_quat(r_ori[contact_idx - 1]).inv()
    racket_angvel = r_rel.as_rotvec() / dt_r

    racket_pos_contact = r_pos[contact_idx]
    racket_ori_contact = r_ori[contact_idx]
    ball_local_final = ball_local

    # 6. Upsample racket for refined collision (±5 ms around coarse hit, 0.01 ms step)
    t_coarse = r_times[contact_idx]
    up_mask = (r_times >= t_coarse - 10) & (r_times <= t_coarse + 10)
    up_times = r_times[up_mask].astype(float)
    up_pos = r_pos[up_mask]
    up_ori = r_ori[up_mask]

    if len(up_times) >= 3:
        t_fine = np.arange(t_coarse - 5, t_coarse + 5, 0.01)
        try:
            fine_pos = interp1d(up_times, up_pos, axis=0, bounds_error=True)(t_fine)
            fine_ori = Slerp(up_times, R.from_quat(up_ori))(t_fine).as_quat()
            fine_ball = np.tile(ball_pos, (len(t_fine), 1))
            fine_idx, _fine_face, fine_local = _detect_racket_contact_cylinder(fine_ball, fine_pos, fine_ori)
            if fine_idx is not None:
                racket_pos_contact = fine_pos[fine_idx]
                racket_ori_contact = fine_ori[fine_idx]
                ball_local_final = fine_local
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    # 7. Racket velocity at ball contact point (including angular velocity contribution)
    r_contact_local = np.array([0.0, ball_local_final[1], ball_local_final[2]])
    r_contact_world = R.from_quat(racket_ori_contact).apply(r_contact_local)
    racket_vel_at_ball = racket_vel + np.cross(racket_angvel, r_contact_world)

    return {
        "x_polyfit": float(ball_pos[0]),
        "y_polyfit": float(ball_pos[1]),
        "z_polyfit": float(ball_pos[2]),
        "vx_polyfit_pre": float(ball_vel_pre[0]),
        "vy_polyfit_pre": float(ball_vel_pre[1]),
        "vz_polyfit_pre": float(ball_vel_pre[2]),
        "vx_polyfit_post": float(ball_vel_post[0]),
        "vy_polyfit_post": float(ball_vel_post[1]),
        "vz_polyfit_post": float(ball_vel_post[2]),
        "x_racket_polyfit": float(racket_pos_contact[0]),
        "y_racket_polyfit": float(racket_pos_contact[1]),
        "z_racket_polyfit": float(racket_pos_contact[2]),
        "qx_racket_polyfit": float(racket_ori_contact[0]),
        "qy_racket_polyfit": float(racket_ori_contact[1]),
        "qz_racket_polyfit": float(racket_ori_contact[2]),
        "qw_racket_polyfit": float(racket_ori_contact[3]),
        "vx_racket_polyfit": float(racket_vel[0]),
        "vy_racket_polyfit": float(racket_vel[1]),
        "vz_racket_polyfit": float(racket_vel[2]),
        "vx_racket_w_angvel_polyfit": float(racket_vel_at_ball[0]),
        "vy_racket_w_angvel_polyfit": float(racket_vel_at_ball[1]),
        "vz_racket_w_angvel_polyfit": float(racket_vel_at_ball[2]),
        "wx_racket_polyfit": float(racket_angvel[0]),
        "wy_racket_polyfit": float(racket_angvel[1]),
        "wz_racket_polyfit": float(racket_angvel[2]),
        "polyfit_mae_pre": float(np.linalg.norm(pre_mae)),
        "polyfit_mae_post": float(np.linalg.norm(post_mae)),
        "polyfit_pos_diff": float(np.linalg.norm(ball_pos_diff)),
    }


# ── All CSV column names (matching extracted.csv schema) ──────────────────

RACKET_CONTACT_COLUMNS = [
    "sequence_name", "log", "event_timestamp", "last_modified", "ball_type", "date", "court",
    "usage_type",
    "x_pre", "y_pre", "z_pre",
    "x_pre_refined", "y_pre_refined", "z_pre_refined",
    "x_post_refined", "y_post_refined", "z_post_refined",
    "vx_pre", "vy_pre", "vz_pre",
    "vx_post", "vy_post", "vz_post",
    "wx_pre", "wy_pre", "wz_pre",
    "wx_pre_opt", "wy_pre_opt", "wz_pre_opt",
    "wx_post", "wy_post", "wz_post",
    "wx_post_opt", "wy_post_opt", "wz_post_opt",
    "qx_racket", "qy_racket", "qz_racket", "qw_racket",
    "qx_racket_pre_refined", "qy_racket_pre_refined", "qz_racket_pre_refined", "qw_racket_pre_refined",
    "qx_racket_post_refined", "qy_racket_post_refined", "qz_racket_post_refined", "qw_racket_post_refined",
    "x_racket", "y_racket", "z_racket",
    "x_racket_pre_refined", "y_racket_pre_refined", "z_racket_pre_refined",
    "x_racket_post_refined", "y_racket_post_refined", "z_racket_post_refined",
    "vx_racket", "vy_racket", "vz_racket",
    "vx_racket_pre_refined", "vy_racket_pre_refined", "vz_racket_pre_refined",
    "vx_racket_post_refined", "vy_racket_post_refined", "vz_racket_post_refined",
    "vx_racket_w_angvel_pre", "vy_racket_w_angvel_pre", "vz_racket_w_angvel_pre",
    "vx_racket_w_angvel_post", "vy_racket_w_angvel_post", "vz_racket_w_angvel_post",
    "wx_racket", "wy_racket", "wz_racket",
    "dt_racket",
    "fitness_pre", "fitness_post",
    "pre_duration", "post_duration",
    "t_contact_pre_offset", "t_contact_post_offset",
    "dx_pre", "dy_pre", "dz_pre",
    "dx_post", "dy_post", "dz_post",
    "vx_post_Nakashima_default", "vy_post_Nakashima_default", "vz_post_Nakashima_default",
    "wx_post_Nakashima_default", "wy_post_Nakashima_default", "wz_post_Nakashima_default",
    "vx_post_Nakashima_cpp", "vy_post_Nakashima_cpp", "vz_post_Nakashima_cpp",
    "wx_post_Nakashima_cpp", "wy_post_Nakashima_cpp", "wz_post_Nakashima_cpp",
    "vx_post_Parametric_7p", "vy_post_Parametric_7p", "vz_post_Parametric_7p",
    "wx_post_Parametric_7p", "wy_post_Parametric_7p", "wz_post_Parametric_7p",
    "vx_post_RCM_tangential", "vy_post_RCM_tangential", "vz_post_RCM_tangential",
    "wx_post_RCM_tangential", "wy_post_RCM_tangential", "wz_post_RCM_tangential",
    "vx_post_cpp_p7p", "vy_post_cpp_p7p", "vz_post_cpp_p7p",
    "wx_post_cpp_p7p", "wy_post_cpp_p7p", "wz_post_cpp_p7p",
    "vx_post_cpp_tangential", "vy_post_cpp_tangential", "vz_post_cpp_tangential",
    "wx_post_cpp_tangential", "wy_post_cpp_tangential", "wz_post_cpp_tangential",
    "vx_post_cpp_linearCOR", "vy_post_cpp_linearCOR", "vz_post_cpp_linearCOR",
    "wx_post_cpp_linearCOR", "wy_post_cpp_linearCOR", "wz_post_cpp_linearCOR",
    "vx_post_cpp_new_exp", "vy_post_cpp_new_exp", "vz_post_cpp_new_exp",
    "wx_post_cpp_new_exp", "wy_post_cpp_new_exp", "wz_post_cpp_new_exp",
    "vx_post_cpp_original_exp", "vy_post_cpp_original_exp", "vz_post_cpp_original_exp",
    "wx_post_cpp_original_exp", "wy_post_cpp_original_exp", "wz_post_cpp_original_exp",
    "vx_post_RCM_tangential_refined", "vy_post_RCM_tangential_refined", "vz_post_RCM_tangential_refined",
    "wx_post_RCM_tangential_refined", "wy_post_RCM_tangential_refined", "wz_post_RCM_tangential_refined",
    # ── Nakashima (refined) predictions ──
    "vx_post_Nakashima_refined", "vy_post_Nakashima_refined", "vz_post_Nakashima_refined",
    "wx_post_Nakashima_refined", "wy_post_Nakashima_refined", "wz_post_Nakashima_refined",
    # ── C++ no-residual (refined) predictions ──
    "vx_post_Nakashima_cpp_refined", "vy_post_Nakashima_cpp_refined", "vz_post_Nakashima_cpp_refined",
    "wx_post_Nakashima_cpp_refined", "wy_post_Nakashima_cpp_refined", "wz_post_Nakashima_cpp_refined",
    # ── Polyfit-refined quantities ──
    "x_polyfit", "y_polyfit", "z_polyfit",
    "vx_polyfit_pre", "vy_polyfit_pre", "vz_polyfit_pre",
    "vx_polyfit_post", "vy_polyfit_post", "vz_polyfit_post",
    "x_racket_polyfit", "y_racket_polyfit", "z_racket_polyfit",
    "qx_racket_polyfit", "qy_racket_polyfit", "qz_racket_polyfit", "qw_racket_polyfit",
    "vx_racket_polyfit", "vy_racket_polyfit", "vz_racket_polyfit",
    "vx_racket_w_angvel_polyfit", "vy_racket_w_angvel_polyfit", "vz_racket_w_angvel_polyfit",
    "wx_racket_polyfit", "wy_racket_polyfit", "wz_racket_polyfit",
    "polyfit_mae_pre", "polyfit_mae_post", "polyfit_pos_diff",
    "vx_post_RCM_tangential_polyfit", "vy_post_RCM_tangential_polyfit", "vz_post_RCM_tangential_polyfit",
    "wx_post_RCM_tangential_polyfit", "wy_post_RCM_tangential_polyfit", "wz_post_RCM_tangential_polyfit",
    # ── ONNX RCM (OldRcm) predictions ──
    "vx_post_onnx_rcm", "vy_post_onnx_rcm", "vz_post_onnx_rcm",
    "wx_post_onnx_rcm", "wy_post_onnx_rcm", "wz_post_onnx_rcm",
    "vx_post_onnx_rcm_refined", "vy_post_onnx_rcm_refined", "vz_post_onnx_rcm_refined",
    "wx_post_onnx_rcm_refined", "wy_post_onnx_rcm_refined", "wz_post_onnx_rcm_refined",
    "vx_post_onnx_rcm_polyfit", "vy_post_onnx_rcm_polyfit", "vz_post_onnx_rcm_polyfit",
    "wx_post_onnx_rcm_polyfit", "wy_post_onnx_rcm_polyfit", "wz_post_onnx_rcm_polyfit",
    # ── ONNX alex predictions ──
    "vx_post_onnx_alex", "vy_post_onnx_alex", "vz_post_onnx_alex",
    "wx_post_onnx_alex", "wy_post_onnx_alex", "wz_post_onnx_alex",
    "vx_post_onnx_alex_refined", "vy_post_onnx_alex_refined", "vz_post_onnx_alex_refined",
    "wx_post_onnx_alex_refined", "wy_post_onnx_alex_refined", "wz_post_onnx_alex_refined",
    "vx_post_onnx_alex_polyfit", "vy_post_onnx_alex_polyfit", "vz_post_onnx_alex_polyfit",
    "wx_post_onnx_alex_polyfit", "wy_post_onnx_alex_polyfit", "wz_post_onnx_alex_polyfit",
    # ── ONNX RCM 0426 (commit 1757e3ec, 6-input model) ──
    "vx_post_onnx_0426", "vy_post_onnx_0426", "vz_post_onnx_0426",
    "wx_post_onnx_0426", "wy_post_onnx_0426", "wz_post_onnx_0426",
    # ── ONNX RCM Plus (RcmPlus, angular-velocity-aware) predictions ──
    # NOTE: Commented out until the RcmPlus ONNX model is ready.
    # "vx_post_onnx_rcm_plus", "vy_post_onnx_rcm_plus", "vz_post_onnx_rcm_plus",
    # "wx_post_onnx_rcm_plus", "wy_post_onnx_rcm_plus", "wz_post_onnx_rcm_plus",
    # "vx_post_onnx_rcm_plus_refined", "vy_post_onnx_rcm_plus_refined", "vz_post_onnx_rcm_plus_refined",
    # "wx_post_onnx_rcm_plus_refined", "wy_post_onnx_rcm_plus_refined", "wz_post_onnx_rcm_plus_refined",
    # "vx_post_onnx_rcm_plus_polyfit", "vy_post_onnx_rcm_plus_polyfit", "vz_post_onnx_rcm_plus_polyfit",
    # "wx_post_onnx_rcm_plus_polyfit", "wy_post_onnx_rcm_plus_polyfit", "wz_post_onnx_rcm_plus_polyfit",
]

# HDF5 scalar fields (all float64 except string fields handled separately)
_HDF5_FLOAT_FIELDS = [c for c in RACKET_CONTACT_COLUMNS if c not in (
    "sequence_name", "log", "last_modified", "ball_type", "date", "court",
    "usage_type",
    "event_timestamp",  # stored as float, but listed separately for clarity
)]
_HDF5_FLOAT_FIELDS.insert(0, "event_timestamp")  # ensure timestamp is included

_HDF5_STRING_FIELDS = ["ball_type", "usage_type"]


# ── RacketContactOptimizer ────────────────────────────────────────────────

class RacketContactOptimizer:
    """
    Extract racket contact parameters and store in HDF5 / CSV.

    For each labelled shot_p1 event, runs aerodynamics optimisation on the
    pre- and post-contact trajectories, estimates contact parameters using
    bisection search, and computes Nakashima model predictions.
    """

    def __init__(self, data_folder, data: MatchCollection, recompute: bool,
                 export_csv: bool = False, zero_toss_spin: bool = False):
        self.data_folder = data_folder
        self.data = data
        self.recompute = recompute
        self.export_csv = export_csv
        self.zero_toss_spin = zero_toss_spin

        # RCM model setup (computed once)
        self._rcm_params = get_params()
        self._rcm_kp = [self._rcm_params.get("coeff_elasticity_factor_racket", 1.928e-3)]

        # Nakashima aerodynamics: constant Cd=0.54, Cm=0.069, rho_air=1.184
        # Matches the Python ODE in update_hdf5.py
        self._nakashima_aero_params = get_params(
            drag_coeff=0.54,
            magnus_coeff=0.069,
            magnus_coeff_linear=0.0,
            magnus_coeff_max=0.069,
            test_new_model=False,
        )
        self._nakashima_aero_params["air_density"] = 1.184

        # Python physics params for trajectory simulation
        self._rcm_pp = make_physics_params(self._rcm_params)
        self._nakashima_aero_pp = make_physics_params(self._nakashima_aero_params)

        # Resolve models directory — configurable via ACE_MODELS_DIR env var
        import os
        _models_dir_env = os.environ.get("ACE_MODELS_DIR")
        if _models_dir_env:
            _models_dir = pathlib.Path(_models_dir_env)
        else:
            # Default: look in ace_evaluation/models/ (bundled with the package)
            _models_dir = pathlib.Path(__file__).resolve().parents[1] / "models"

        _required_models = {
            "tangential":   "RCM-175_wo1203_tangential_COR.onnx",
        }
        _optional_models = {
            "alex":         "alex_refined_contact_FK_tangential_COR_residual.onnx",
            "0426":         "residual_model_compact.exponential.onnx",
        }
        _missing = [
            f"  {label}: {name}  (expected at {_models_dir / name})"
            for label, name in _required_models.items()
            if not (_models_dir / name).exists()
        ]
        if _missing:
            raise FileNotFoundError(
                f"Missing required ONNX model(s) in {_models_dir}:\n" + "\n".join(_missing)
                + "\n\nSet the ACE_MODELS_DIR environment variable to the directory "
                  "containing the ONNX model files."
            )

        # ONNX RCM (OldRcm — no angular velocity)
        # Also used as drop-in replacement for C++ resolve_contact_9dof
        # with compact_residual_model=True (default and tangential configs).
        _onnx_rcm_path = _models_dir / "RCM-175_wo1203_tangential_COR.onnx"
        (self._onnx_session, self._onnx_inputs, self._onnx_velxyz,
         self._onnx_spinxyz, self._onnx_posyz, self._onnx_vel_out,
         self._onnx_angvel_out, self._onnx_pos_out) = _init_onnx_rcm(str(_onnx_rcm_path))

        # ONNX RCM for "Nakashima_cpp" / "Nakashima_cpp_refined" columns
        # (default compact_residual_model ONNX — previously ran through C++)
        _default_onnx_name = self._rcm_params.get(
            "compact_residual_model_name",
            "residual_model_compact.exponential.dim8.pre1002_wo0428.RCM-118.onnx",
        )
        _default_onnx_path = _models_dir / _default_onnx_name
        if _default_onnx_path.exists():
            (self._onnx_default_session, self._onnx_default_inputs,
             self._onnx_default_velxyz, self._onnx_default_spinxyz,
             self._onnx_default_posyz, self._onnx_default_vel_out,
             self._onnx_default_angvel_out, self._onnx_default_pos_out) = _init_onnx_rcm(str(_default_onnx_path))
        else:
            # Fall back to the tangential model if the default model is not available
            (self._onnx_default_session, self._onnx_default_inputs,
             self._onnx_default_velxyz, self._onnx_default_spinxyz,
             self._onnx_default_posyz, self._onnx_default_vel_out,
             self._onnx_default_angvel_out, self._onnx_default_pos_out) = (
                self._onnx_session, self._onnx_inputs,
                self._onnx_velxyz, self._onnx_spinxyz,
                self._onnx_posyz, self._onnx_vel_out,
                self._onnx_angvel_out, self._onnx_pos_out)

        # ONNX alex (optional)
        _onnx_alex_path = _models_dir / "alex_refined_contact_FK_tangential_COR_residual.onnx"
        if _onnx_alex_path.exists():
            (self._onnx_alex_session, self._onnx_alex_inputs, self._onnx_alex_velxyz,
             self._onnx_alex_spinxyz, self._onnx_alex_posyz, self._onnx_alex_vel_out,
             self._onnx_alex_angvel_out, self._onnx_alex_pos_out) = _init_onnx_rcm(str(_onnx_alex_path))
        else:
            warnings.warn(f"Optional ONNX model not found: {_onnx_alex_path}  — onnx_alex columns will be NaN")
            self._onnx_alex_session = None

        # ONNX RCM 0426 (commit 1757e3ec, old 6-input model — optional)
        _onnx_0426_path = _models_dir / "residual_model_compact.exponential.onnx"
        if _onnx_0426_path.exists():
            self._onnx_0426_session = _init_onnx_rcm_0426(str(_onnx_0426_path))
        else:
            warnings.warn(f"Optional ONNX model not found: {_onnx_0426_path}  — onnx_0426 columns will be NaN")
            self._onnx_0426_session = None

        print("=============================================")
        print("\t\tRacket Contact")
        print("=============================================")
        if self.export_csv:
            print("  CSV export: enabled (extracted.csv)")
        if self.zero_toss_spin:
            print("  Zero toss spin: enabled")

    # ── Clean / recompute ─────────────────────────────────────────────

    def _simulate_post_contact(self, position, velocity, spin, duration,
                               _physics=None, aero_model="new"):
        """Simulate post-contact ball trajectory using Python aerodynamics integrator.

        Args:
            physics: Unused, kept for API compatibility.
            aero_model: "new" for piecewise 0226 model, "old" for constant Cd/Cm.

        Returns a trajectory dict or None if the simulation has too few steps.
        """
        if duration <= DT:
            return None
        state = BallState(
            timestamp=0.0,
            position=np.asarray(position, dtype=np.float64),
            linear_velocity=np.asarray(velocity, dtype=np.float64),
            angular_velocity=np.asarray(spin, dtype=np.float64),
        )

        ts  = [state.timestamp]
        xs  = [state.position[0]];  ys = [state.position[1]];  zs = [state.position[2]]
        vxs = [state.linear_velocity[0]];  vys = [state.linear_velocity[1]];  vzs = [state.linear_velocity[2]]
        wxs = [state.angular_velocity[0]]; wys = [state.angular_velocity[1]]; wzs = [state.angular_velocity[2]]

        prev_x = state.position[0]
        while True:
            if aero_model == "old":
                state = predict_ball_state_old_model(
                    state, DT, self._nakashima_aero_pp,
                    self._nakashima_aero_params,
                )
            else:
                state = predict_ball_state_new_model(
                    state, DT, self._rcm_pp,
                    get_drag_coefficient_new, get_magnus_coefficient_new,
                )
            ts.append(state.timestamp)
            xs.append(state.position[0]);   ys.append(state.position[1]);   zs.append(state.position[2])
            vxs.append(state.linear_velocity[0]); vys.append(state.linear_velocity[1]); vzs.append(state.linear_velocity[2])
            wxs.append(state.angular_velocity[0]); wys.append(state.angular_velocity[1]); wzs.append(state.angular_velocity[2])
            # Stop when ball hits the ground
            if state.position[2] < BALL_RADIUS:
                break
            # Stop when ball crosses the net (x=0) and is below net top
            if prev_x * state.position[0] < 0 and state.position[2] < NET_HEIGHT + BALL_RADIUS:
                break
            prev_x = state.position[0]

        if len(ts) <= 1:
            return None
        return {
            "t": np.array(ts, dtype=np.float64),
            "x": np.array(xs, dtype=np.float64), "y": np.array(ys, dtype=np.float64), "z": np.array(zs, dtype=np.float64),
            "vx": np.array(vxs, dtype=np.float64), "vy": np.array(vys, dtype=np.float64), "vz": np.array(vzs, dtype=np.float64),
            "wx": np.array(wxs, dtype=np.float64), "wy": np.array(wys, dtype=np.float64), "wz": np.array(wzs, dtype=np.float64),
        }

    def _clean_hdf5_data(self):
        """Remove all existing racket_contacts data from HDF5 files."""
        print("Cleaning existing racket contact data...")
        hdf5_files = list(pathlib.Path(self.data_folder).rglob("data.h5"))
        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, "a") as h5f:
                    for game_key in [k for k in h5f.keys() if k.startswith("game_")]:
                        for rally_key in [k for k in h5f[game_key].keys() if k.startswith("rally_")]:
                            rally_grp = h5f[game_key][rally_key]
                            if "ground_truth_200" in rally_grp and "racket_contacts" in rally_grp["ground_truth_200"]:
                                del rally_grp["ground_truth_200"]["racket_contacts"]
                print(f"  Cleaned: {file_path}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"  Warning: Failed to clean {file_path}: {e}")
        print("Racket contact data cleanup complete.\n")

    # ── Main update ───────────────────────────────────────────────────

    def update_hdf(self):
        if self.recompute:
            self._clean_hdf5_data()

        all_csv_rows: list[dict] = []
        parameters = get_params()

        # Counters for summary
        total_shot_p1 = 0
        bad_optimization = 0
        bad_length = 0
        bad_spin_confidence = 0
        bad_data = 0
        bad_racket_pos = 0
        bad_start_end = 0
        bad_no_before_after = 0
        bad_racket_distance = 0
        bad_segmentation = 0
        bad_others = 0
        extracted_count = 0

        for match in self.data.matches:
            match_file = match.file
            match_parent_name = match.file.parent.name

            for game in match.games:
                for rally in game.rallies:
                    rally_data_raw = rally.rally
                    if isinstance(rally_data_raw, list) and len(rally_data_raw) == 0:
                        continue
                    if isinstance(rally_data_raw, pd.DataFrame) and rally_data_raw.empty:
                        continue

                    events = rally.events
                    if len(events) < 3:
                        continue

                    sequence_name = f"{match_parent_name}_game_{game.game_id}_rally_{rally.rally_id}"
                    ball_type = _get_ball_type(sequence_name)

                    # Check if data already exists
                    if not self.recompute:
                        try:
                            with h5py.File(match_file, "r") as h5f:
                                grp_path = f"game_{game.game_id}/rally_{rally.rally_id}/ground_truth_200/racket_contacts"
                                if grp_path in h5f:
                                    continue
                        except Exception:  # pylint: disable=broad-exception-caught
                            pass

                    # Rename columns for extraction compatibility
                    rally_data = rename_to_flat(rally_data_raw)

                    print(f"Processing Match {match.match_id}, Game {game.game_id}, "
                          f"Rally {rally.rally_id} — {sequence_name}")

                    contact_rows = []

                    for event_idx, event in enumerate(events):
                        if not isinstance(event, RacketContactEvent):
                            continue
                        if event.type != "shot_p1":
                            continue

                        total_shot_p1 += 1

                        # Need a next event to define the post-contact window
                        if event_idx >= len(events) - 1:
                            bad_start_end += 1
                            continue

                        # For serves (event_idx == 0), use rally data start
                        # as the pre-contact window origin instead of skipping.
                        if event_idx == 0:
                            start = rally_data["time"].iloc[0]
                        else:
                            prev_event = events[event_idx - 1]
                            start = prev_event.timestamp
                        next_event = events[event_idx + 1]
                        end = next_event.timestamp
                        _contact_time_approx = event.timestamp

                        # Extract shot window
                        shot = rally_data[(rally_data["time"] > start) & (rally_data["time"] < end)].copy()
                        if len(shot) == 0:
                            bad_data += 1
                            continue

                        start = shot.head(1)["time"].values[0]
                        end = shot.tail(1)["time"].values[0]

                        # Drop NaN racket orientation rows, reset index
                        racket_orient_cols = ['racket_1_orientation_x', 'racket_1_orientation_y',
                                              'racket_1_orientation_z', 'racket_1_orientation_w']
                        if not all(c in shot.columns for c in racket_orient_cols):
                            bad_racket_pos += 1
                            continue

                        shot = shot.dropna(subset=racket_orient_cols).reset_index(drop=True)
                        if len(shot) == 0:
                            bad_data += 1
                            continue

                        # Refine contact time
                        contact_time = _refine_racket_contacts(shot)
                        if contact_time is None:
                            bad_segmentation += 1
                            continue

                        shot_contact_index = shot.index[shot["time"] > contact_time].min()
                        if pd.isna(shot_contact_index):
                            bad_segmentation += 1
                            continue

                        # Split pre/post
                        shot_pre = shot[shot["time"] < contact_time].dropna(
                            subset=['time', 'ball_position_x', 'ball_position_y', 'ball_position_z'])
                        shot_post = shot[shot["time"] >= contact_time].dropna(
                            subset=['time', 'ball_position_x', 'ball_position_y', 'ball_position_z'])

                        if len(shot_pre) <= 3 or len(shot_post) <= 8:
                            bad_length += 1
                            continue

                        # Check data completeness (>50% samples present)
                        samples_pct_pre = len(shot_pre) * DT / (shot_pre.iloc[-1].time - shot_pre.iloc[0].time + 1e-9)
                        samples_pct_post = len(shot_post) * DT / (shot_post.iloc[-1].time - shot_post.iloc[0].time + 1e-9)

                        racket_pos_ok = (
                            shot.racket_1_position_x.loc[shot_contact_index - 1:shot_contact_index + 1].notna().all()
                            if shot_contact_index >= 1 and shot_contact_index + 1 < len(shot) else False
                        )
                        racket_orient_ok = (
                            shot.racket_1_orientation_x.loc[shot_contact_index - 1:shot_contact_index + 1].notna().all()
                            if shot_contact_index >= 1 and shot_contact_index + 1 < len(shot) else False
                        )

                        if samples_pct_pre <= 0.5 or samples_pct_post <= 0.5:
                            bad_data += 1
                            continue
                        if not racket_pos_ok or not racket_orient_ok:
                            bad_racket_pos += 1
                            continue

                        # Check spin confidence
                        min_spin_confidence = 0.5
                        conf_cols = ['ball_gt200_wx_confidence', 'ball_gt200_wy_confidence', 'ball_gt200_wz_confidence']
                        has_conf = all(c in shot_pre.columns for c in conf_cols)
                        if has_conf:
                            min_conf_pre = min(shot_pre[c].min() for c in conf_cols)
                            min_conf_post = min(shot_post[c].min() for c in conf_cols)
                            if min_conf_pre < min_spin_confidence or min_conf_post < min_spin_confidence:
                                bad_spin_confidence += 1
                                continue

                        # Estimate spin
                        wx_pre, wy_pre, wz_pre = _estimate_spin(shot_pre)

                        # Ball toss detection
                        if self.zero_toss_spin and _is_ball_toss(events, event_idx, shot_pre):
                            print(f"\tBall toss detected, setting pre-contact spin to 0")
                            wx_pre, wy_pre, wz_pre = 0.0, 0.0, 0.0

                        # ── Pre-contact optimisation ──────────────────────
                        const_params = [
                            wx_pre, wy_pre, wz_pre,
                            parameters["air_density"], parameters["ball_density"],
                            parameters["ball_radius"], parameters["drag_coeff"],
                        ]
                        # Use finite-diff velocity as initial guess (not zero)
                        # to avoid flat-gradient regions where speed-dependent
                        # magnus/drag make the landscape insensitive at v≈0.
                        _dt_pre = shot_pre["time"].diff()
                        _vx0_pre = (shot_pre["ball_position_x"].diff() / _dt_pre).dropna().iloc[0] if len(shot_pre) > 1 else 0.0
                        _vy0_pre = (shot_pre["ball_position_y"].diff() / _dt_pre).dropna().iloc[0] if len(shot_pre) > 1 else 0.0
                        _vz0_pre = (shot_pre["ball_position_z"].diff() / _dt_pre).dropna().iloc[0] if len(shot_pre) > 1 else 0.0
                        opt_pre = minimize(
                            _solve_aerodynamics_vel, (_vx0_pre, _vy0_pre, _vz0_pre),
                            args=(shot_pre, start, contact_time + DT / 2., DT, const_params),
                            method="L-BFGS-B", tol=0.0001,
                        )
                        fitness_pre = opt_pre.fun

                        t_eval_pre = np.arange(start, contact_time, DT)
                        t_eval_pre = t_eval_pre[(t_eval_pre >= start) & (t_eval_pre <= contact_time)]
                        sim_data_pre = _sim_python(
                            start, contact_time,
                            [shot_pre.iloc[0].ball_position_x, shot_pre.iloc[0].ball_position_y, shot_pre.iloc[0].ball_position_z],
                            const_params, opt_pre, t_eval_pre,
                        )
                        contact_idx = len(sim_data_pre)

                        # ── Post-contact optimisation ─────────────────────
                        # Skip first 5 timesteps after contact to avoid transient
                        shot_post_p5dt = shot_post[shot_post["time"] > contact_time + 5 * DT]
                        wx_post, wy_post, wz_post = _estimate_spin(shot_post_p5dt if len(shot_post_p5dt) > 1 else shot_post)
                        const_params_post = [
                            wx_post, wy_post, wz_post,
                            parameters["air_density"], parameters["ball_density"],
                            parameters["ball_radius"], parameters["drag_coeff"],
                        ]
                        # Use finite-diff velocity from the first two post-
                        # contact observations as initial guess (see pre-contact
                        # comment for rationale).
                        _dt_post = shot_post["time"].diff()
                        _vx0_post = (shot_post["ball_position_x"].diff() / _dt_post).dropna().iloc[0] if len(shot_post) > 1 else 0.0
                        _vy0_post = (shot_post["ball_position_y"].diff() / _dt_post).dropna().iloc[0] if len(shot_post) > 1 else 0.0
                        _vz0_post = (shot_post["ball_position_z"].diff() / _dt_post).dropna().iloc[0] if len(shot_post) > 1 else 0.0
                        opt_post = minimize(
                            _solve_aerodynamics_vel, (_vx0_post, _vy0_post, _vz0_post),
                            args=(shot_post, contact_time, end, DT, const_params_post),
                            method="L-BFGS-B", tol=0.0001,
                        )
                        fitness_post = opt_post.fun

                        closest_row = shot.dropna(subset=['time', 'ball_position_x', 'ball_position_y', 'ball_position_z'])
                        closest_row = closest_row[closest_row["time"] > contact_time]
                        if len(closest_row) == 0:
                            bad_data += 1
                            continue
                        closest_row = closest_row.iloc[0]

                        t_eval_post = np.arange(closest_row.time, end, DT)
                        t_eval_post = t_eval_post[(t_eval_post >= closest_row.time) & (t_eval_post <= end)]
                        sim_data_post = _sim_python(
                            closest_row.time, end,
                            [closest_row.ball_position_x, closest_row.ball_position_y, closest_row.ball_position_z],
                            const_params_post, opt_post, t_eval_post,
                        )
                        sim_data = pd.concat([sim_data_pre, sim_data_post], ignore_index=True)

                        # ── Estimate quantities at impact ─────────────────
                        # Need enough context around contact_idx
                        if contact_idx < 2 or contact_idx + 2 >= len(sim_data):
                            bad_others += 1
                            continue
                        if shot_contact_index < 2 or shot_contact_index + 2 >= len(shot):
                            bad_others += 1
                            continue

                        try:
                            racket_angular_vel = math_utilities.compute_angular_velocity(
                                np.array([shot.loc[shot_contact_index - 1][c] for c in racket_orient_cols]),
                                np.array([shot.loc[shot_contact_index + 1][c] for c in racket_orient_cols]),
                                shot.loc[shot_contact_index + 1]['time'] - shot.loc[shot_contact_index - 1]['time'],
                            )

                            (t_contact_pre, x_pre_refined, x_racket_pre_refined, v_racket_pre_refined,
                             q_racket_pre_refined, r_pre, v_racket_w_angvel_pre,
                             t_contact_post, x_post_refined, x_racket_post_refined, v_racket_post_refined,
                             q_racket_post_refined, r_post, v_racket_w_angvel_post) = \
                                _estimate_quantities_at_impact(sim_data, contact_idx, shot, shot_contact_index, racket_angular_vel)
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tFailed impact estimation: {e}")
                            bad_others += 1
                            continue

                        # ── Validate distance ─────────────────────────────
                        _racket_pos = np.array([
                            shot.loc[shot_contact_index]['racket_1_position_x'],
                            shot.loc[shot_contact_index]['racket_1_position_y'],
                            shot.loc[shot_contact_index]['racket_1_position_z'],
                        ])
                        _ball_pos_sim = np.array([
                            sim_data.iloc[contact_idx - 1]['x'],
                            sim_data.iloc[contact_idx - 1]['y'],
                            sim_data.iloc[contact_idx - 1]['z'],
                        ])
                        if _racket_pos[0] > 0:
                            bad_racket_distance += 1
                            continue
                        if np.linalg.norm(_racket_pos - _ball_pos_sim) > 0.2:
                            bad_racket_distance += 1
                            continue

                        # ── Build result row ──────────────────────────────
                        row = {
                            "event_timestamp": float(event.timestamp),
                            "ball_type": ball_type,
                            "usage_type": match.data_usage or "",
                            "x_pre": float(sim_data.iloc[contact_idx - 1]['x']),
                            "y_pre": float(sim_data.iloc[contact_idx - 1]['y']),
                            "z_pre": float(sim_data.iloc[contact_idx - 1]['z']),
                            "x_pre_refined": float(x_pre_refined[0]),
                            "y_pre_refined": float(x_pre_refined[1]),
                            "z_pre_refined": float(x_pre_refined[2]),
                            "x_post_refined": float(x_post_refined[0]),
                            "y_post_refined": float(x_post_refined[1]),
                            "z_post_refined": float(x_post_refined[2]),
                            "vx_pre": float(sim_data.iloc[contact_idx - 1]['vx']),
                            "vy_pre": float(sim_data.iloc[contact_idx - 1]['vy']),
                            "vz_pre": float(sim_data.iloc[contact_idx - 1]['vz']),
                            "vx_post": float(sim_data.iloc[contact_idx]['vx']),
                            "vy_post": float(sim_data.iloc[contact_idx]['vy']),
                            "vz_post": float(sim_data.iloc[contact_idx]['vz']),
                            "wx_pre": float(wx_pre),
                            "wy_pre": float(wy_pre),
                            "wz_pre": float(wz_pre),
                            "wx_pre_opt": float(sim_data.iloc[contact_idx - 1]['wx']),
                            "wy_pre_opt": float(sim_data.iloc[contact_idx - 1]['wy']),
                            "wz_pre_opt": float(sim_data.iloc[contact_idx - 1]['wz']),
                            "wx_post": float(wx_post),
                            "wy_post": float(wy_post),
                            "wz_post": float(wz_post),
                            "wx_post_opt": float(sim_data.iloc[contact_idx]['wx']),
                            "wy_post_opt": float(sim_data.iloc[contact_idx]['wy']),
                            "wz_post_opt": float(sim_data.iloc[contact_idx]['wz']),
                            "qx_racket": float(shot.loc[shot_contact_index]['racket_1_orientation_x']),
                            "qy_racket": float(shot.loc[shot_contact_index]['racket_1_orientation_y']),
                            "qz_racket": float(shot.loc[shot_contact_index]['racket_1_orientation_z']),
                            "qw_racket": float(shot.loc[shot_contact_index]['racket_1_orientation_w']),
                            "qx_racket_pre_refined": float(q_racket_pre_refined[0]),
                            "qy_racket_pre_refined": float(q_racket_pre_refined[1]),
                            "qz_racket_pre_refined": float(q_racket_pre_refined[2]),
                            "qw_racket_pre_refined": float(q_racket_pre_refined[3]),
                            "qx_racket_post_refined": float(q_racket_post_refined[0]),
                            "qy_racket_post_refined": float(q_racket_post_refined[1]),
                            "qz_racket_post_refined": float(q_racket_post_refined[2]),
                            "qw_racket_post_refined": float(q_racket_post_refined[3]),
                            "x_racket": float(shot.loc[shot_contact_index]['racket_1_position_x']),
                            "y_racket": float(shot.loc[shot_contact_index]['racket_1_position_y']),
                            "z_racket": float(shot.loc[shot_contact_index]['racket_1_position_z']),
                            "x_racket_pre_refined": float(x_racket_pre_refined[0]),
                            "y_racket_pre_refined": float(x_racket_pre_refined[1]),
                            "z_racket_pre_refined": float(x_racket_pre_refined[2]),
                            "x_racket_post_refined": float(x_racket_post_refined[0]),
                            "y_racket_post_refined": float(x_racket_post_refined[1]),
                            "z_racket_post_refined": float(x_racket_post_refined[2]),
                            "vx_racket": float(shot.loc[shot_contact_index]['racket_1_velocity_x']),
                            "vy_racket": float(shot.loc[shot_contact_index]['racket_1_velocity_y']),
                            "vz_racket": float(shot.loc[shot_contact_index]['racket_1_velocity_z']),
                            "vx_racket_pre_refined": float(v_racket_pre_refined[0]),
                            "vy_racket_pre_refined": float(v_racket_pre_refined[1]),
                            "vz_racket_pre_refined": float(v_racket_pre_refined[2]),
                            "vx_racket_post_refined": float(v_racket_post_refined[0]),
                            "vy_racket_post_refined": float(v_racket_post_refined[1]),
                            "vz_racket_post_refined": float(v_racket_post_refined[2]),
                            "vx_racket_w_angvel_pre": float(v_racket_w_angvel_pre[0]),
                            "vy_racket_w_angvel_pre": float(v_racket_w_angvel_pre[1]),
                            "vz_racket_w_angvel_pre": float(v_racket_w_angvel_pre[2]),
                            "vx_racket_w_angvel_post": float(v_racket_w_angvel_post[0]),
                            "vy_racket_w_angvel_post": float(v_racket_w_angvel_post[1]),
                            "vz_racket_w_angvel_post": float(v_racket_w_angvel_post[2]),
                            "wx_racket": float(racket_angular_vel[0]),
                            "wy_racket": float(racket_angular_vel[1]),
                            "wz_racket": float(racket_angular_vel[2]),
                            "dt_racket": float(shot.loc[shot_contact_index + 1]['time'] - shot.loc[shot_contact_index - 1]['time']),
                            "fitness_pre": float(fitness_pre),
                            "fitness_post": float(fitness_post),
                            "pre_duration": float(contact_time - start),
                            "post_duration": float(end - contact_time),
                            "t_contact_pre_offset": float(t_contact_pre),
                            "t_contact_post_offset": float(t_contact_post),
                            "dx_pre": float(r_pre[0]),
                            "dy_pre": float(r_pre[1]),
                            "dz_pre": float(r_pre[2]),
                            "dx_post": float(r_post[0]),
                            "dy_post": float(r_post[1]),
                            "dz_post": float(r_post[2]),
                        }

                        # ── Nakashima model predictions ───────────────────
                        _b_vel = np.array([row["vx_pre"], row["vy_pre"], row["vz_pre"]])
                        _b_spin = np.array([row["wx_pre"], row["wy_pre"], row["wz_pre"]])
                        _r_vel = np.array([row["vx_racket"], row["vy_racket"], row["vz_racket"]])
                        _r_pos = np.array([row["x_racket"], row["y_racket"], row["z_racket"]])
                        _b_pos = np.array([row["x_pre"], row["y_pre"], row["z_pre"]])
                        _r_quat = np.array([row["qx_racket"], row["qy_racket"], row["qz_racket"], row["qw_racket"]])
                        _r_quat_fixed = compute_racket_contact_quat_v(_b_vel, _r_vel, _r_quat)

                        _contact = SimpleContact(
                            orientation=_r_quat_fixed,
                            position=R.from_quat(_r_quat_fixed).inv().apply(_b_pos - _r_pos),
                            linear_velocity=_r_vel,
                            angular_velocity=np.asarray(racket_angular_vel),
                        )

                        # Helper to fill NaN for a model prediction
                        # pylint: disable=cell-var-from-loop
                        def _fill_nan(prefix, comps=("vx", "vy", "vz", "wx", "wy", "wz")):
                            for comp in comps:
                                row[f"{comp}_post_{prefix}"] = float("nan")

                        # Helper to store a (vel, spin) prediction tuple
                        def _store_pred(pred, prefix):
                            row[f"vx_post_{prefix}"] = float(pred[0][0])
                            row[f"vy_post_{prefix}"] = float(pred[0][1])
                            row[f"vz_post_{prefix}"] = float(pred[0][2])
                            row[f"wx_post_{prefix}"] = float(pred[1][0])
                            row[f"wy_post_{prefix}"] = float(pred[1][1])
                            row[f"wz_post_{prefix}"] = float(pred[1][2])
                        # pylint: enable=cell-var-from-loop

                        # ── Python Nakashima models (independent try/except each) ──
                        try:
                            _store_pred(racket_contacts.nakashima(_b_vel, _b_spin, _contact, self._rcm_kp), "Nakashima_default")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: Nakashima default failed: {e}")
                            _fill_nan("Nakashima_default")

                        # ── C++ models replaced by ONNX equivalents ──
                        try:
                            _cpp_vel, _cpp_spin = _run_onnx_rcm(
                                self._onnx_default_session, self._onnx_default_inputs,
                                self._onnx_default_velxyz, self._onnx_default_spinxyz,
                                self._onnx_default_posyz, self._onnx_default_vel_out,
                                self._onnx_default_angvel_out,
                                _b_pos.astype(np.float32), _b_vel.astype(np.float32), _b_spin.astype(np.float32),
                                _r_pos.astype(np.float32), _r_quat.astype(np.float32), _r_vel.astype(np.float32),
                            )
                            _store_pred((_cpp_vel, _cpp_spin), "Nakashima_cpp")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: ONNX default (Nakashima_cpp) failed: {e}")
                            _fill_nan("Nakashima_cpp")

                        # NOTE: cpp_p7p disabled — model no longer loaded
                        _fill_nan("cpp_p7p")

                        try:
                            _tang_vel, _tang_spin = _run_onnx_rcm(
                                self._onnx_session, self._onnx_inputs,
                                self._onnx_velxyz, self._onnx_spinxyz,
                                self._onnx_posyz, self._onnx_vel_out,
                                self._onnx_angvel_out,
                                _b_pos.astype(np.float32), _b_vel.astype(np.float32), _b_spin.astype(np.float32),
                                _r_pos.astype(np.float32), _r_quat.astype(np.float32), _r_vel.astype(np.float32),
                            )
                            _store_pred((_tang_vel, _tang_spin), "cpp_tangential")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: ONNX tangential (cpp_tangential) failed: {e}")
                            _fill_nan("cpp_tangential")

                        # NOTE: cpp_linearCOR disabled — model no longer loaded
                        _fill_nan("cpp_linearCOR")

                        # NOTE: cpp_new_exp disabled — model no longer loaded
                        _fill_nan("cpp_new_exp")

                        # NOTE: cpp_original_exp disabled — model no longer loaded
                        _fill_nan("cpp_original_exp")

                        # ── Parametric / tangential Python models ──
                        try:
                            _p7p_new = racket_contacts.PARAMETRIC_7P_NEW
                            _p7p_params = [_p7p_new["e0"], _p7p_new["e1"], _p7p_new["et0"],
                                           _p7p_new["et1"], _p7p_new["c_spin"], _p7p_new["alpha"]]
                            _store_pred(racket_contacts.parametric_7p(_b_vel, _b_spin, _contact, _p7p_params), "Parametric_7p")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: Parametric_7p failed: {e}")
                            _fill_nan("Parametric_7p")

                        try:
                            _rcm_new = racket_contacts.RCM_TANGENTIAL_NEW
                            _rcm_params = [_rcm_new["e0"], _rcm_new["e1"], _rcm_new["et0"],
                                              _rcm_new["et1"], _rcm_new["c_spin"]]
                            _store_pred(racket_contacts.rcm_tangential(_b_vel, _b_spin, _contact, _rcm_params), "RCM_tangential")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: RCM_tangential failed: {e}")
                            _fill_nan("RCM_tangential")

                        # ── ONNX RCM (OldRcm — no angular velocity) ──
                        try:
                            _onnx_vel, _onnx_spin = _run_onnx_rcm(
                                self._onnx_session, self._onnx_inputs,
                                self._onnx_velxyz, self._onnx_spinxyz, self._onnx_posyz,
                                self._onnx_vel_out, self._onnx_angvel_out,
                                _b_pos.astype(np.float32), _b_vel.astype(np.float32), _b_spin.astype(np.float32),
                                _r_pos.astype(np.float32), _r_quat.astype(np.float32), _r_vel.astype(np.float32),
                            )
                            _store_pred((_onnx_vel, _onnx_spin), "onnx_rcm")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: ONNX RCM failed: {e}")
                            _fill_nan("onnx_rcm")

                        # NOTE: ONNX RCM Plus disabled — model not yet ready.
                        # try:
                        #     _onnx_plus_vel, _onnx_plus_spin = _run_onnx_rcm(
                        #         self._onnx_plus_session, self._onnx_plus_inputs,
                        #         self._onnx_plus_velxyz, self._onnx_plus_spinxyz, self._onnx_plus_posyz,
                        #         self._onnx_plus_vel_out, self._onnx_plus_angvel_out,
                        #         _b_pos.astype(np.float32), _b_vel.astype(np.float32), _b_spin.astype(np.float32),
                        #         _r_pos.astype(np.float32), _r_quat.astype(np.float32), _r_vel.astype(np.float32),
                        #         racket_angular_vel=np.array(racket_angular_vel, dtype=np.float32),
                        #     )
                        #     _store_pred((_onnx_plus_vel, _onnx_plus_spin), "onnx_rcm_plus")
                        # except Exception as e:  # pylint: disable=broad-exception-caught
                        #     print(f"\tWarning: ONNX RCM Plus failed: {e}")
                        #     _fill_nan("onnx_rcm_plus")

                        # ── ONNX alex ──
                        if self._onnx_alex_session is not None:
                            try:
                                _alex_vel, _alex_spin = _run_onnx_rcm(
                                    self._onnx_alex_session, self._onnx_alex_inputs,
                                    self._onnx_alex_velxyz, self._onnx_alex_spinxyz, self._onnx_alex_posyz,
                                    self._onnx_alex_vel_out, self._onnx_alex_angvel_out,
                                    _b_pos.astype(np.float32), _b_vel.astype(np.float32), _b_spin.astype(np.float32),
                                    _r_pos.astype(np.float32), _r_quat.astype(np.float32), _r_vel.astype(np.float32),
                                )
                                _store_pred((_alex_vel, _alex_spin), "onnx_alex")
                            except Exception as e:  # pylint: disable=broad-exception-caught
                                print(f"\tWarning: ONNX alex failed: {e}")
                                _fill_nan("onnx_alex")
                        else:
                            _fill_nan("onnx_alex")

                        # ── ONNX RCM 0426 (commit 1757e3ec, 6-input model) ──
                        if self._onnx_0426_session is not None:
                            try:
                                _0426_vel, _0426_spin = _run_onnx_rcm_0426(
                                    self._onnx_0426_session,
                                    _b_vel.astype(np.float64), _b_spin.astype(np.float64),
                                    _r_pos.astype(np.float64), _r_quat.astype(np.float64), _r_vel.astype(np.float64),
                                )
                                _store_pred((_0426_vel, _0426_spin), "onnx_0426")
                            except Exception as e:  # pylint: disable=broad-exception-caught
                                print(f"\tWarning: ONNX 0426 failed: {e}")
                                _fill_nan("onnx_0426")
                        else:
                            _fill_nan("onnx_0426")

                        # ── Models with refined contact quantities ────
                        _b_vel_ref = np.array([row["vx_pre"], row["vy_pre"], row["vz_pre"]])
                        _b_spin_ref = np.array([row["wx_pre"], row["wy_pre"], row["wz_pre"]])
                        _b_pos_ref = np.array([row["x_pre_refined"], row["y_pre_refined"], row["z_pre_refined"]])
                        _r_pos_ref = np.array([row["x_racket_pre_refined"], row["y_racket_pre_refined"], row["z_racket_pre_refined"]])
                        _r_vel_ref = np.array([row["vx_racket_w_angvel_pre"], row["vy_racket_w_angvel_pre"], row["vz_racket_w_angvel_pre"]])
                        _r_quat_ref = np.array([row["qx_racket_pre_refined"], row["qy_racket_pre_refined"], row["qz_racket_pre_refined"], row["qw_racket_pre_refined"]])
                        _r_quat_ref_fixed = compute_racket_contact_quat_v(_b_vel_ref, _r_vel_ref, _r_quat_ref)

                        _contact_ref = SimpleContact(
                            orientation=_r_quat_ref_fixed,
                            position=R.from_quat(_r_quat_ref_fixed).inv().apply(_b_pos_ref - _r_pos_ref),
                            linear_velocity=_r_vel_ref,
                            angular_velocity=np.asarray(racket_angular_vel),
                        )

                        # ── Nakashima with refined contact quantities ──
                        try:
                            _store_pred(racket_contacts.nakashima(_b_vel_ref, _b_spin_ref, _contact_ref, self._rcm_kp), "Nakashima_refined")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: Nakashima refined failed: {e}")
                            _fill_nan("Nakashima_refined")

                        # ── ONNX default with refined contact quantities ──
                        try:
                            _cpp_ref_vel, _cpp_ref_spin = _run_onnx_rcm(
                                self._onnx_default_session, self._onnx_default_inputs,
                                self._onnx_default_velxyz, self._onnx_default_spinxyz,
                                self._onnx_default_posyz, self._onnx_default_vel_out,
                                self._onnx_default_angvel_out,
                                _b_pos_ref.astype(np.float32), _b_vel_ref.astype(np.float32), _b_spin_ref.astype(np.float32),
                                _r_pos_ref.astype(np.float32), _r_quat_ref.astype(np.float32), _r_vel_ref.astype(np.float32),
                            )
                            _store_pred((_cpp_ref_vel, _cpp_ref_spin), "Nakashima_cpp_refined")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: ONNX default refined (Nakashima_cpp_refined) failed: {e}")
                            _fill_nan("Nakashima_cpp_refined")

                        try:
                            _rcm_new = racket_contacts.RCM_TANGENTIAL_NEW
                            _rcm_params = [_rcm_new["e0"], _rcm_new["e1"], _rcm_new["et0"],
                                           _rcm_new["et1"], _rcm_new["c_spin"]]
                            _store_pred(racket_contacts.rcm_tangential(_b_vel_ref, _b_spin_ref, _contact_ref, _rcm_params), "RCM_tangential_refined")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: RCM_tangential_refined failed: {e}")
                            _fill_nan("RCM_tangential_refined")

                        # ── ONNX RCM with refined contact quantities ──
                        try:
                            _onnx_ref_vel, _onnx_ref_spin = _run_onnx_rcm(
                                self._onnx_session, self._onnx_inputs,
                                self._onnx_velxyz, self._onnx_spinxyz, self._onnx_posyz,
                                self._onnx_vel_out, self._onnx_angvel_out,
                                _b_pos_ref.astype(np.float32), _b_vel_ref.astype(np.float32), _b_spin_ref.astype(np.float32),
                                _r_pos_ref.astype(np.float32), _r_quat_ref.astype(np.float32), _r_vel_ref.astype(np.float32),
                            )
                            _store_pred((_onnx_ref_vel, _onnx_ref_spin), "onnx_rcm_refined")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: ONNX RCM refined failed: {e}")
                            _fill_nan("onnx_rcm_refined")

                        # NOTE: ONNX RCM Plus refined disabled — model not yet ready.
                        # try:
                        #     _onnx_plus_ref_vel, _onnx_plus_ref_spin = _run_onnx_rcm(
                        #         self._onnx_plus_session, self._onnx_plus_inputs,
                        #         self._onnx_plus_velxyz, self._onnx_plus_spinxyz, self._onnx_plus_posyz,
                        #         self._onnx_plus_vel_out, self._onnx_plus_angvel_out,
                        #         _b_pos_ref.astype(np.float32), _b_vel_ref.astype(np.float32), _b_spin_ref.astype(np.float32),
                        #         _r_pos_ref.astype(np.float32), _r_quat_ref.astype(np.float32), _r_vel_ref.astype(np.float32),
                        #         racket_angular_vel=np.array(racket_angular_vel, dtype=np.float32),
                        #     )
                        #     _store_pred((_onnx_plus_ref_vel, _onnx_plus_ref_spin), "onnx_rcm_plus_refined")
                        # except Exception as e:  # pylint: disable=broad-exception-caught
                        #     print(f"\tWarning: ONNX RCM Plus refined failed: {e}")
                        #     _fill_nan("onnx_rcm_plus_refined")

                        # ── ONNX alex with refined contact quantities ──
                        if self._onnx_alex_session is not None:
                            try:
                                _alex_ref_vel, _alex_ref_spin = _run_onnx_rcm(
                                    self._onnx_alex_session, self._onnx_alex_inputs,
                                    self._onnx_alex_velxyz, self._onnx_alex_spinxyz, self._onnx_alex_posyz,
                                    self._onnx_alex_vel_out, self._onnx_alex_angvel_out,
                                    _b_pos_ref.astype(np.float32), _b_vel_ref.astype(np.float32), _b_spin_ref.astype(np.float32),
                                    _r_pos_ref.astype(np.float32), _r_quat_ref.astype(np.float32), _r_vel_ref.astype(np.float32),
                                )
                                _store_pred((_alex_ref_vel, _alex_ref_spin), "onnx_alex_refined")
                            except Exception as e:  # pylint: disable=broad-exception-caught
                                print(f"\tWarning: ONNX alex refined failed: {e}")
                                _fill_nan("onnx_alex_refined")
                        else:
                            _fill_nan("onnx_alex_refined")


                        # ── Polyfit-refined contact (Method B) + RCM tangential ──
                        polyfit = _polyfit_refine_contact(shot, event.timestamp)
                        if polyfit is not None:
                            row.update(polyfit)

                            _b_vel_pf = np.array([polyfit["vx_polyfit_pre"], polyfit["vy_polyfit_pre"], polyfit["vz_polyfit_pre"]])
                            _b_spin_pf = np.array([row["wx_pre"], row["wy_pre"], row["wz_pre"]])
                            _b_pos_pf = np.array([polyfit["x_polyfit"], polyfit["y_polyfit"], polyfit["z_polyfit"]])
                            _r_pos_pf = np.array([polyfit["x_racket_polyfit"], polyfit["y_racket_polyfit"], polyfit["z_racket_polyfit"]])
                            _r_vel_pf = np.array([polyfit["vx_racket_w_angvel_polyfit"], polyfit["vy_racket_w_angvel_polyfit"], polyfit["vz_racket_w_angvel_polyfit"]])
                            _r_quat_pf = np.array([polyfit["qx_racket_polyfit"], polyfit["qy_racket_polyfit"], polyfit["qz_racket_polyfit"], polyfit["qw_racket_polyfit"]])
                            _r_angvel_pf = np.array([polyfit["wx_racket_polyfit"], polyfit["wy_racket_polyfit"], polyfit["wz_racket_polyfit"]])
                            _r_quat_pf_fixed = compute_racket_contact_quat_v(_b_vel_pf, _r_vel_pf, _r_quat_pf)

                            _contact_pf = SimpleContact(
                                orientation=_r_quat_pf_fixed,
                                position=R.from_quat(_r_quat_pf_fixed).inv().apply(_b_pos_pf - _r_pos_pf),
                                linear_velocity=_r_vel_pf,
                                angular_velocity=_r_angvel_pf,
                            )

                            try:
                                _rcm_new = racket_contacts.RCM_TANGENTIAL_NEW
                                _rcm_params = [_rcm_new["e0"], _rcm_new["e1"], _rcm_new["et0"],
                                               _rcm_new["et1"], _rcm_new["c_spin"]]
                                _store_pred(racket_contacts.rcm_tangential(_b_vel_pf, _b_spin_pf, _contact_pf, _rcm_params), "RCM_tangential_polyfit")
                            except Exception as e:  # pylint: disable=broad-exception-caught
                                print(f"\tWarning: RCM_tangential_polyfit failed: {e}")
                                _fill_nan("RCM_tangential_polyfit")

                            # ── ONNX RCM with polyfit contact quantities ──
                            try:
                                _onnx_pf_vel, _onnx_pf_spin = _run_onnx_rcm(
                                    self._onnx_session, self._onnx_inputs,
                                    self._onnx_velxyz, self._onnx_spinxyz, self._onnx_posyz,
                                    self._onnx_vel_out, self._onnx_angvel_out,
                                    _b_pos_pf.astype(np.float32), _b_vel_pf.astype(np.float32), _b_spin_pf.astype(np.float32),
                                    _r_pos_pf.astype(np.float32), _r_quat_pf.astype(np.float32), _r_vel_pf.astype(np.float32),
                                )
                                _store_pred((_onnx_pf_vel, _onnx_pf_spin), "onnx_rcm_polyfit")
                            except Exception as e:  # pylint: disable=broad-exception-caught
                                print(f"\tWarning: ONNX RCM polyfit failed: {e}")
                                _fill_nan("onnx_rcm_polyfit")

                            # NOTE: ONNX RCM Plus polyfit disabled — model not yet ready.
                            # try:
                            #     _onnx_plus_pf_vel, _onnx_plus_pf_spin = _run_onnx_rcm(
                            #         self._onnx_plus_session, self._onnx_plus_inputs,
                            #         self._onnx_plus_velxyz, self._onnx_plus_spinxyz, self._onnx_plus_posyz,
                            #         self._onnx_plus_vel_out, self._onnx_plus_angvel_out,
                            #         _b_pos_pf.astype(np.float32), _b_vel_pf.astype(np.float32), _b_spin_pf.astype(np.float32),
                            #         _r_pos_pf.astype(np.float32), _r_quat_pf.astype(np.float32), _r_vel_pf.astype(np.float32),
                            #         racket_angular_vel=_r_angvel_pf.astype(np.float32),
                            #     )
                            #     _store_pred((_onnx_plus_pf_vel, _onnx_plus_pf_spin), "onnx_rcm_plus_polyfit")
                            # except Exception as e:  # pylint: disable=broad-exception-caught
                            #     print(f"\tWarning: ONNX RCM Plus polyfit failed: {e}")
                            #     _fill_nan("onnx_rcm_plus_polyfit")

                            # ── ONNX alex with polyfit contact quantities ──
                            if self._onnx_alex_session is not None:
                                try:
                                    _alex_pf_vel, _alex_pf_spin = _run_onnx_rcm(
                                        self._onnx_alex_session, self._onnx_alex_inputs,
                                        self._onnx_alex_velxyz, self._onnx_alex_spinxyz, self._onnx_alex_posyz,
                                        self._onnx_alex_vel_out, self._onnx_alex_angvel_out,
                                        _b_pos_pf.astype(np.float32), _b_vel_pf.astype(np.float32), _b_spin_pf.astype(np.float32),
                                        _r_pos_pf.astype(np.float32), _r_quat_pf.astype(np.float32), _r_vel_pf.astype(np.float32),
                                    )
                                    _store_pred((_alex_pf_vel, _alex_pf_spin), "onnx_alex_polyfit")
                                except Exception as e:  # pylint: disable=broad-exception-caught
                                    print(f"\tWarning: ONNX alex polyfit failed: {e}")
                                    _fill_nan("onnx_alex_polyfit")
                            else:
                                _fill_nan("onnx_alex_polyfit")


                        else:
                            _fill_nan("RCM_tangential_polyfit")
                            _fill_nan("onnx_rcm_polyfit")
                            _fill_nan("onnx_alex_polyfit")
                            # _fill_nan("onnx_rcm_plus_polyfit")

                        # ── Simulated post-contact trajectory ─────────
                        # Use the cpp model post-contact vel/spin to simulate
                        # the ball trajectory with the C++ physics layer for
                        # the same duration as the observed post-contact flight
                        # segment (matching the aerodynamics update_hdf5 path).
                        _post_dur = float(end - contact_time)

                        sim_traj = None
                        try:
                            _v_post_sim = np.array([
                                row.get("vx_post_Nakashima_cpp", float("nan")),
                                row.get("vy_post_Nakashima_cpp", float("nan")),
                                row.get("vz_post_Nakashima_cpp", float("nan")),
                            ])
                            _w_post_sim = np.array([
                                row.get("wx_post_Nakashima_cpp", float("nan")),
                                row.get("wy_post_Nakashima_cpp", float("nan")),
                                row.get("wz_post_Nakashima_cpp", float("nan")),
                            ])
                            if not (np.any(np.isnan(_v_post_sim)) or np.any(np.isnan(_w_post_sim))):
                                sim_traj = self._simulate_post_contact(
                                    [row["x_pre"], row["y_pre"], row["z_pre"]],
                                    _v_post_sim, _w_post_sim, _post_dur,
                                    aero_model="old",
                                )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: Simulated trajectory failed: {e}")

                        # NOTE: RCM refined, RCM polyfit, ONNX RCM, ONNX alex
                        # trajectory simulations disabled — only Nakashima (cpp),
                        # Nakashima (refined), and latest (ONNX alex refined +
                        # 0226 aero) are active.

                        # ── Simulated trajectory: Nakashima refined + Nakashima aero ──
                        sim_traj_nakashima_refined = None
                        try:
                            _v_nak_ref = np.array([
                                row.get("vx_post_Nakashima_refined", float("nan")),
                                row.get("vy_post_Nakashima_refined", float("nan")),
                                row.get("vz_post_Nakashima_refined", float("nan")),
                            ])
                            _w_nak_ref = np.array([
                                row.get("wx_post_Nakashima_refined", float("nan")),
                                row.get("wy_post_Nakashima_refined", float("nan")),
                                row.get("wz_post_Nakashima_refined", float("nan")),
                            ])
                            if not (np.any(np.isnan(_v_nak_ref)) or np.any(np.isnan(_w_nak_ref))):
                                sim_traj_nakashima_refined = self._simulate_post_contact(
                                    [row["x_pre_refined"], row["y_pre_refined"], row["z_pre_refined"]],
                                    _v_nak_ref, _w_nak_ref, _post_dur,
                                    aero_model="old",
                                )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: Simulated trajectory (Nakashima refined) failed: {e}")

                        # ── Simulated trajectory: C++ no-residual refined + 0226 aero ──
                        sim_traj_cpp_no_residual_refined = None
                        try:
                            _v_cpp_nores_ref = np.array([
                                row.get("vx_post_Nakashima_cpp_refined", float("nan")),
                                row.get("vy_post_Nakashima_cpp_refined", float("nan")),
                                row.get("vz_post_Nakashima_cpp_refined", float("nan")),
                            ])
                            _w_cpp_nores_ref = np.array([
                                row.get("wx_post_Nakashima_cpp_refined", float("nan")),
                                row.get("wy_post_Nakashima_cpp_refined", float("nan")),
                                row.get("wz_post_Nakashima_cpp_refined", float("nan")),
                            ])
                            if not (np.any(np.isnan(_v_cpp_nores_ref)) or np.any(np.isnan(_w_cpp_nores_ref))):
                                sim_traj_cpp_no_residual_refined = self._simulate_post_contact(
                                    [row["x_pre_refined"], row["y_pre_refined"], row["z_pre_refined"]],
                                    _v_cpp_nores_ref, _w_cpp_nores_ref, _post_dur,
                                )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: Simulated trajectory (C++ no-residual refined) failed: {e}")

                        # ── Simulated trajectory: latest (ONNX alex refined + 0226 aero) ──
                        sim_traj_latest = None
                        try:
                            _v_alex_ref = np.array([
                                row.get("vx_post_onnx_alex_refined", float("nan")),
                                row.get("vy_post_onnx_alex_refined", float("nan")),
                                row.get("vz_post_onnx_alex_refined", float("nan")),
                            ])
                            _w_alex_ref = np.array([
                                row.get("wx_post_onnx_alex_refined", float("nan")),
                                row.get("wy_post_onnx_alex_refined", float("nan")),
                                row.get("wz_post_onnx_alex_refined", float("nan")),
                            ])
                            if not (np.any(np.isnan(_v_alex_ref)) or np.any(np.isnan(_w_alex_ref))):
                                sim_traj_latest = self._simulate_post_contact(
                                    [row["x_pre_refined"], row["y_pre_refined"], row["z_pre_refined"]],
                                    _v_alex_ref, _w_alex_ref, _post_dur,
                                )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"\tWarning: Simulated trajectory (latest) failed: {e}")

                        row["_simulated_trajectory"] = sim_traj
                        row["_simulated_trajectory_nakashima_refined"] = sim_traj_nakashima_refined
                        row["_simulated_trajectory_cpp_no_residual_refined"] = sim_traj_cpp_no_residual_refined
                        row["_simulated_trajectory_latest"] = sim_traj_latest
                        contact_rows.append(row)
                        extracted_count += 1

                    # ── Write rally contact rows to HDF5 ──────────────
                    if contact_rows:
                        self._write_to_hdf5(match_file, game.game_id, rally.rally_id, contact_rows)

                    # ── Collect for CSV ────────────────────────────────
                    if self.export_csv:
                        for row in contact_rows:
                            csv_row = {
                                "sequence_name": sequence_name,
                                "log": rally.log_identifier or "",
                                "last_modified": match.last_modified or "",
                                "date": match.date or "",
                                "court": match.court or "",
                            }
                            csv_row.update({k: v for k, v in row.items() if not k.startswith("_")})
                            all_csv_rows.append(csv_row)

        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  RACKET CONTACT EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"  Total shot_p1 events:         {total_shot_p1}")
        print(f"  Extracted contacts:            {extracted_count}")
        print(f"  Bad spin confidence:           {bad_spin_confidence}")
        print(f"  Bad optimization:              {bad_optimization}")
        print(f"  Bad racket distance (>20cm):   {bad_racket_distance}")
        print(f"  Bad data quality (>50% miss):  {bad_data}")
        print(f"  Bad racket pos/quat:           {bad_racket_pos}")
        print(f"  Bad length (too short):        {bad_length}")
        print(f"  Bad no before/after:           {bad_no_before_after}")
        print(f"  Bad start/end:                 {bad_start_end}")
        print(f"  Bad segmentation:              {bad_segmentation}")
        print(f"  Bad others:                    {bad_others}")
        print("=" * 60 + "\n")

        # ── Write CSV ─────────────────────────────────────────────────
        if self.export_csv and all_csv_rows:
            csv_path = pathlib.Path(self.data_folder) / "extracted.csv"
            df = pd.DataFrame(all_csv_rows)
            # Ensure all expected columns are present
            for col in RACKET_CONTACT_COLUMNS:
                if col not in df.columns:
                    df[col] = float("nan")
            df.to_csv(csv_path, index=False)
            print(f"Wrote {len(df)} racket contacts to {csv_path}")

    # ── HDF5 write ────────────────────────────────────────────────────

    @staticmethod
    def _write_to_hdf5(match_file, game_id, rally_id, contact_rows: list[dict]):
        """Write racket contact data to HDF5 under ground_truth_200/racket_contacts."""
        with h5py.File(match_file, "a") as h5f:
            grp_path = f"game_{game_id}/rally_{rally_id}/ground_truth_200"
            if grp_path not in h5f:
                return
            rally_grp = h5f[grp_path]
            if "racket_contacts" in rally_grp:
                del rally_grp["racket_contacts"]
            rc_grp = rally_grp.create_group("racket_contacts")

            n = len(contact_rows)
            if n == 0:
                return

            # Float datasets
            for field in _HDF5_FLOAT_FIELDS:
                data = np.array([row.get(field, float("nan")) for row in contact_rows], dtype=np.float64)
                rc_grp.create_dataset(field, data=data, dtype=np.float64)

            # String datasets
            for field in _HDF5_STRING_FIELDS:
                vals = [str(row.get(field, "")) for row in contact_rows]
                rc_grp.create_dataset(field, data=np.array(vals, dtype=h5py.special_dtype(vlen=str)))

            # Simulated trajectories (variable-length per contact)
            _TRAJ_FIELDS = ("t", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz")
            _TRAJ_KEYS = [
                ("_simulated_trajectory",              "simulated_trajectories"),
                ("_simulated_trajectory_nakashima_refined", "simulated_trajectories_nakashima_refined"),
                ("_simulated_trajectory_cpp_no_residual_refined", "simulated_trajectories_cpp_no_residual_refined"),
                ("_simulated_trajectory_latest",       "simulated_trajectories_latest"),
            ]
            for row_key, grp_name in _TRAJ_KEYS:
                has_any = any(row.get(row_key) is not None for row in contact_rows)
                if not has_any:
                    continue
                traj_grp = rc_grp.create_group(grp_name)
                for idx, row in enumerate(contact_rows):
                    traj = row.get(row_key)
                    if traj is None:
                        continue
                    ev_grp = traj_grp.create_group(str(idx))
                    for field in _TRAJ_FIELDS:
                        ev_grp.create_dataset(field, data=traj[field], dtype=np.float64)

        print(f"  Wrote {n} racket contacts to HDF5")

    # ── Plotting (stub) ──────────────────────────────────────────────

    def plot(self):
        print("  [Racket contacts] Plotting not yet implemented")
