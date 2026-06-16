# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Table contact (bounce) extraction and HDF5 storage.

Extracts pre- and post-contact ball velocity and spin at every labelled table
bounce by running independent aerodynamics optimisations on the flight segments
bracketing each bounce.  Results are stored in the HDF5 file under
``ground_truth_200/table_contacts``.

Optionally writes ``extracted_TCM.csv`` for backward compatibility.
"""

import pathlib

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from ace_evaluation.utilities.data_classes import (
    MatchCollection,
    TableContactEvent,
)

# ── Physical constants ────────────────────────────────────────────────────
BALL_RADIUS = 0.02  # metres
RHO_AIR = 1.204
RHO_BALL = 81.2
DT = 0.005


# ── Aerodynamics helpers (same ODE as in update_hdf5.py) ─────────────────

def _aerodynamics_diff(_, s, params, const_params):
    """Aerodynamics differential equations."""
    _, _, _, vx, vy, vz = s
    c_drag, c_magnus = params
    wx, wy, wz, rho_air, rho_ball, radius = const_params
    vmag = np.sqrt(vx * vx + vy * vy + vz * vz)
    kd = c_drag * 3.0 / (8.0 * radius) * rho_air / rho_ball
    km = c_magnus * rho_air / rho_ball
    dsdt = [
        vx,
        vy,
        vz,
        -kd * vmag * vx - km * (vy * wz - vz * wy),
        -kd * vmag * vy - km * (vz * wx - vx * wz),
        -kd * vmag * vz - km * (vx * wy - vy * wx) - (1.0 - rho_air / rho_ball) * 9.8,
    ]
    return dsdt


def _solve_aerodynamics(params, flight_segment_data, tstart, tend, dt, const_params, pos_cols):
    """Solve the ODE and return fitness vs observations."""
    px, py, pz = pos_cols
    t_eval = np.arange(tstart, tend, dt)
    # Drop the last point — fp rounding in arange can push it to >= tend,
    # which makes solve_ivp reject it.  Same guard used by racket_contacts.
    if len(t_eval) > 1:
        t_eval = t_eval[:-1]
    if len(t_eval) == 0:
        return float("inf")
    aero_params = np.array([params[3], params[4]])
    result = solve_ivp(
        _aerodynamics_diff,
        [tstart, tend],
        (
            flight_segment_data[px].iloc[0],
            flight_segment_data[py].iloc[0],
            flight_segment_data[pz].iloc[0],
            params[0], params[1], params[2],
        ),
        t_eval=t_eval,
        method="RK45",
        args=(aero_params, const_params),
    )
    if not result.success or len(result.y) < 3:
        return float("inf")

    ref_pos = np.stack([
        flight_segment_data["time"].to_numpy(),
        flight_segment_data[px].to_numpy(),
        flight_segment_data[py].to_numpy(),
        flight_segment_data[pz].to_numpy(),
    ]).T
    sim_pos = np.stack([result.t, result.y[0], result.y[1], result.y[2]]).T
    return _compute_fitness(ref_pos, sim_pos, dt)


def _compute_fitness(ref_pos, sim_pos, dt, delta=0.05):
    """Compute fitness using Huber loss over time-aligned 3D positions."""
    ref_times = ref_pos[:, 0]
    sim_times = sim_pos[:, 0]
    ref_coords = ref_pos[:, 1:4]
    sim_coords = sim_pos[:, 1:4]
    losses = []
    for i, t_ref in enumerate(ref_times):
        mask = np.abs(sim_times - t_ref) < dt * 0.5
        if np.any(mask):
            diffs = sim_coords[mask] - ref_coords[i]
            dists = np.linalg.norm(diffs, axis=1)
            squared = 0.5 * dists**2
            linear = delta * (dists - 0.5 * delta)
            losses.extend(np.where(dists <= delta, squared, linear))
    if not losses:
        return float("inf")
    return float(np.sqrt(np.mean(losses)))


def _optimise_segment(flight_segment_data, tstart, tend, tend_sim, dt, pos_cols, default_cd=0.55, default_cm=0.08):
    """Run aerodynamics optimisation on a single flight segment.

    Returns (optimised_result_df, const_params) or None on failure.
    optimised_result_df has columns: t, pos (list), vel (list), spin (list), cd, cm.
    """
    if len(flight_segment_data) < 2:
        return None

    # Estimate initial velocity with finite differences
    time_arr = flight_segment_data["time"].values
    px, py, pz = pos_cols
    x_arr = flight_segment_data[px].values
    y_arr = flight_segment_data[py].values
    z_arr = flight_segment_data[pz].values

    if len(time_arr) < 2 or (time_arr[1] - time_arr[0]) == 0:
        return None

    vx_fd = (x_arr[1] - x_arr[0]) / (time_arr[1] - time_arr[0])
    vy_fd = (y_arr[1] - y_arr[0]) / (time_arr[1] - time_arr[0])
    vz_fd = (z_arr[1] - z_arr[0]) / (time_arr[1] - time_arr[0])

    # Get spin via mode (matching the aerodynamics optimizer logic)
    wx_col = "wx" if "wx" in flight_segment_data.columns else "ball_gt200_wx"
    wy_col = "wy" if "wy" in flight_segment_data.columns else "ball_gt200_wy"
    wz_col = "wz" if "wz" in flight_segment_data.columns else "ball_gt200_wz"

    try:
        if flight_segment_data[wx_col].isna().all():
            return None
        wx = stats.mode(flight_segment_data[wx_col], keepdims=False, nan_policy='omit')[0]
        wy = stats.mode(flight_segment_data[wy_col], keepdims=False, nan_policy='omit')[0]
        wz = stats.mode(flight_segment_data[wz_col], keepdims=False, nan_policy='omit')[0]
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    const_params = (wx, wy, wz, RHO_AIR, RHO_BALL, BALL_RADIUS)
    params = (vx_fd, vy_fd, vz_fd, default_cd, default_cm)

    opt_result = minimize(
        _solve_aerodynamics,
        params,
        args=(flight_segment_data, tstart, tend, dt, const_params, pos_cols),
        method="SLSQP",
        tol=0.0001,
    )

    if not opt_result.success or opt_result.fun == float("inf"):
        return None

    # Reconstruct trajectory up to tend_sim
    t_sim = np.arange(tstart, tend_sim + dt, dt)
    t_sim = t_sim[t_sim <= tend_sim + dt]  # clamp fp rounding
    if len(t_sim) == 0:
        return None
    aero_params = np.array([opt_result.x[3], opt_result.x[4]])
    sol = solve_ivp(
        _aerodynamics_diff,
        [tstart, t_sim[-1] + dt],
        (
            flight_segment_data[px].iloc[0],
            flight_segment_data[py].iloc[0],
            flight_segment_data[pz].iloc[0],
            opt_result.x[0], opt_result.x[1], opt_result.x[2],
        ),
        t_eval=t_sim,
        method="RK45",
        args=(aero_params, const_params),
    )

    df = pd.DataFrame({
        "t": sol.t,
        "pos": [list(sol.y[0:3, i]) for i in range(len(sol.t))],
        "vel": [list(sol.y[3:6, i]) for i in range(len(sol.t))],
        "spin": [[wx, wy, wz]] * len(sol.t),
        "cd": [opt_result.x[3]] * len(sol.t),
        "cm": [opt_result.x[4]] * len(sol.t),
    })
    return df


def _compute_epsilon_opt(vz_pre, vz_post):
    """Observed coefficient of restitution: |vz_post / vz_pre|."""
    if abs(vz_pre) < 1e-9:
        return float("nan")
    return abs(vz_post / vz_pre)


# Fixed CoR for Nakashima model variants
NAKASHIMA_EPSILON_ITTF = 0.876   # ITTF rules (ball dropped from 30 cm, bounces ~23 cm)
NAKASHIMA_EPSILON_PAPER = 0.93   # Value from Nakashima's paper


def _compute_epsilon_velocity_dependent(vz):
    """Velocity-dependent coefficient of restitution (used by ResidualCorrectedNakashima / C++)."""
    return vz * 0.02 + 0.98


def _nakashima_contact(v, w, epsilon):
    """Nakashima table contact model (no residual correction).

    Pure analytical model: given pre-contact velocity *v*, spin *w* and
    coefficient of restitution *epsilon*, returns (v_post, w_post).
    """
    r = BALL_RADIUS

    ub_z = -v[2]
    ub_x = v[0] - r * w[1]
    ub_y = v[1] + r * w[0]
    ub_t = np.sqrt(ub_x * ub_x + ub_y * ub_y)
    if ub_t == 0:
        ub_t = 1e9

    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = min(ap, 0.4)

    a_u = np.array([[1 - a, 0, 0], [0, 1 - a, 0], [0, 0, -epsilon]])
    b_u = np.array([[0, a * r, 0], [-a * r, 0, 0], [0, 0, 0]])
    a_o = np.array([[0, -1.5 * a / r, 0], [1.5 * a / r, 0, 0], [0, 0, 0]])
    b_o = np.array([[1 - 1.5 * a, 0, 0], [0, 1 - 1.5 * a, 0], [0, 0, 1]])

    v_post = a_u @ v + b_u @ w
    w_post = a_o @ v + b_o @ w
    return v_post, w_post


def _0426_table_contact(v, w):
    """Table contact model from commit 1757e3ec (2025-04-26).

    Same Nakashima base as :func:`_nakashima_contact` but with a simple
    column-2 correction to b_u and b_o matrices (only the spin-to-output
    coupling in the z-direction is adjusted).  The correction is rotated
    by the velocity direction in the xy-plane.

    The coefficient of restitution is computed internally from the vertical
    velocity component (matching the C++ code with
    ``compute_table_restitution_coefficient = False``):
        epsilon = vz * 0.02 + 0.98

    This matches the C++ ``ResolveContactTable`` at that commit with
    identity table orientation.
    """
    r = BALL_RADIUS

    # Velocity-dependent restitution coefficient:
    # epsilon = vz * coeff_scale + coeff_base
    # (vz is negative at a downward bounce, giving epsilon < 0.98)
    epsilon = v[2] * 0.02 + 0.98

    ub_z = -v[2]
    ub_x = v[0] - r * w[1]
    ub_y = v[1] + r * w[0]
    ub_t = np.sqrt(ub_x * ub_x + ub_y * ub_y)
    if ub_t == 0:
        ub_t = 1e9

    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = min(ap, 0.4)

    a_u = np.array([[1 - a, 0, 0], [0, 1 - a, 0], [0, 0, -epsilon]])
    b_u = np.array([[0, a * r, 0], [-a * r, 0, 0], [0, 0, 0]])
    a_o = np.array([[0, -1.5 * a / r, 0], [1.5 * a / r, 0, 0], [0, 0, 0]])
    b_o = np.array([[1 - 1.5 * a, 0, 0], [0, 1 - 1.5 * a, 0], [0, 0, 1]])

    v_post = a_u @ v + b_u @ w
    w_post = a_o @ v + b_o @ w
    return v_post, w_post


def _residual_corrected_nakashima_contact(v, w, epsilon):
    """Nakashima table contact model with learned residual correction.

    Same as :func:`_nakashima_contact` but subtracts the residual correction
    matrices (fitted in a rotated frame where v_y=0).  Matches the C++
    ``ResolveContactTable`` implementation (with identity table orientation).
    """
    r = BALL_RADIUS

    ub_z = -v[2]
    ub_x = v[0] - r * w[1]
    ub_y = v[1] + r * w[0]
    ub_t = np.sqrt(ub_x * ub_x + ub_y * ub_y)
    if ub_t == 0:
        ub_t = 1e9

    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = min(ap, 0.4)

    a_u = np.array([[1 - a, 0, 0], [0, 1 - a, 0], [0, 0, -epsilon]])
    b_u = np.array([[0, a * r, 0], [-a * r, 0, 0], [0, 0, 0]])
    a_o = np.array([[0, -1.5 * a / r, 0], [1.5 * a / r, 0, 0], [0, 0, 0]])
    b_o = np.array([[1 - 1.5 * a, 0, 0], [0, 1 - 1.5 * a, 0], [0, 0, 1]])

    # Rotation into the frame where v_y = 0
    v_x, v_y, _ = v
    norm_xy = np.sqrt(v_x**2 + v_y**2)
    if norm_xy == 0:
        rot = np.eye(3)
    else:
        u_x = np.array([v_x / norm_xy, v_y / norm_xy, 0])
        u_y = np.array([-v_y / norm_xy, v_x / norm_xy, 0])
        u_z = np.array([0, 0, 1])
        rot = np.vstack([u_x, u_y, u_z])

    # Residual correction matrices (learned, same as C++ ResolveContactTable)
    a_u_corr = np.array([[-0.0108710967, 0.0, 0.0],
                         [ 0.0,           0.0, 0.0],
                         [-0.0143258878, 0.0, 0.0]])
    b_u_corr = np.array([[-0.0000073021,  0.0005743815,  0.0000834545],
                         [-0.0005860303, -0.0000306265, -0.0001665864],
                         [-0.0000248477, -0.0001321961, -0.0000142774]])
    a_o_corr = np.array([[-0.0735163773, 0.0,          0.0],
                         [ 3.0883899621, 0.0,          3.0314342691],
                         [-0.0382839144, 0.0,          0.2482747068]])
    b_o_corr = np.array([[ 0.0133803508,  0.0022481343,  0.0110659132],
                         [-0.0046491227,  0.0018867774, -0.0055119878],
                         [ 0.0450644266, -0.0034796966,  0.0576889524]])

    a_u -= rot.T @ a_u_corr @ rot
    b_u -= rot.T @ b_u_corr @ rot
    a_o -= rot.T @ a_o_corr @ rot
    b_o -= rot.T @ b_o_corr @ rot

    v_post = a_u @ v + b_u @ w
    w_post = a_o @ v + b_o @ w
    return v_post, w_post


def _residual0805_nakashima_contact(v, w, epsilon):
    """Nakashima table contact model with residual correction (LassoCV 2025-08-05).

    Same structure as :func:`_residual_corrected_nakashima_contact` but uses
    updated correction matrices fitted on the 30K-contact dataset
    (extracted_TCM.csv, May 2026).
    """
    r = BALL_RADIUS

    ub_z = -v[2]
    ub_x = v[0] - r * w[1]
    ub_y = v[1] + r * w[0]
    ub_t = np.sqrt(ub_x * ub_x + ub_y * ub_y)
    if ub_t == 0:
        ub_t = 1e9

    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = min(ap, 0.4)

    a_u = np.array([[1 - a, 0, 0], [0, 1 - a, 0], [0, 0, -epsilon]])
    b_u = np.array([[0, a * r, 0], [-a * r, 0, 0], [0, 0, 0]])
    a_o = np.array([[0, -1.5 * a / r, 0], [1.5 * a / r, 0, 0], [0, 0, 0]])
    b_o = np.array([[1 - 1.5 * a, 0, 0], [0, 1 - 1.5 * a, 0], [0, 0, 1]])

    v_x, v_y, _ = v
    norm_xy = np.sqrt(v_x**2 + v_y**2)
    if norm_xy == 0:
        rot = np.eye(3)
    else:
        u_x = np.array([v_x / norm_xy, v_y / norm_xy, 0])
        u_y = np.array([-v_y / norm_xy, v_x / norm_xy, 0])
        u_z = np.array([0, 0, 1])
        rot = np.vstack([u_x, u_y, u_z])

    # LassoCV correction matrices (30K dataset, May 2026)
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

    a_u -= rot.T @ a_u_corr @ rot
    b_u -= rot.T @ b_u_corr @ rot
    a_o -= rot.T @ a_o_corr @ rot
    b_o -= rot.T @ b_o_corr @ rot

    v_post = a_u @ v + b_u @ w
    w_post = a_o @ v + b_o @ w
    return v_post, w_post


def _pysr_nakashima_contact(v, w, epsilon):
    """Nakashima table contact model with PySR symbolic residual correction.

    Uses PySR-discovered nonlinear expressions (denormalized to real units)
    fitted on the 30K-contact dataset.  The base Nakashima model outputs are
    computed first, then the residuals are subtracted in the rotated frame
    (where v_y = 0) and rotated back.
    """
    r = BALL_RADIUS

    ub_z = -v[2]
    ub_x = v[0] - r * w[1]
    ub_y = v[1] + r * w[0]
    ub_t = np.sqrt(ub_x * ub_x + ub_y * ub_y)
    if ub_t == 0:
        ub_t = 1e9

    ap = 0.25 * (1.0 + epsilon) * ub_z / ub_t
    a = min(ap, 0.4)

    a_u = np.array([[1 - a, 0, 0], [0, 1 - a, 0], [0, 0, -epsilon]])
    b_u = np.array([[0, a * r, 0], [-a * r, 0, 0], [0, 0, 0]])
    a_o = np.array([[0, -1.5 * a / r, 0], [1.5 * a / r, 0, 0], [0, 0, 0]])
    b_o = np.array([[1 - 1.5 * a, 0, 0], [0, 1 - 1.5 * a, 0], [0, 0, 1]])

    # Base Nakashima output
    v_base = a_u @ v + b_u @ w
    w_base = a_o @ v + b_o @ w

    # Rotation into the frame where v_y = 0
    v_x, v_y, _ = v
    norm_xy = np.sqrt(v_x**2 + v_y**2)
    if norm_xy == 0:
        return v_base, w_base

    u_x = np.array([v_x / norm_xy, v_y / norm_xy, 0])
    u_y = np.array([-v_y / norm_xy, v_x / norm_xy, 0])
    u_z = np.array([0, 0, 1])
    rot = np.vstack([u_x, u_y, u_z])

    # Rotated pre-contact state
    v_rot = rot @ v   # vrx_pre = v_rot[0], vz_pre = v_rot[2]
    w_rot = rot @ w   # wrx_pre, wry_pre, wz_pre
    vrx = v_rot[0]
    _vz = v_rot[2]
    wrx = w_rot[0]
    wry = w_rot[1]
    wz = w_rot[2]

    # PySR residual expressions (denormalized, real units)
    res_vrx = 0.000331228454808625 * wry - 0.00234328506995007
    res_vry = -2.59237126684253e-6 * wrx * wry + 0.000334357282237056 * wrx + 7.75028081762634e-6 * wry - 0.019965477425329
    res_vz = 0.0353459618428551 - 0.0185605475825958 * vrx
    res_wrx = (1.80864194820441 - 0.0140229379398023 * wry) * np.cos(0.00305392436946761 * wry - 0.393887197174651) - 1.56985386606186
    res_wry = 40.6577706332294 * np.cos(np.sin(0.00305392436946761 * wry - 0.393887197174651)) - 27.5348325673878
    res_wz = 0.0249048361999532 * wz + 2.75505998061377

    # Subtract residuals (rotated back to world frame)
    v_corr = rot.T @ np.array([res_vrx, res_vry, res_vz])
    w_corr = rot.T @ np.array([res_wrx, res_wry, res_wz])

    v_post = v_base - v_corr
    w_post = w_base - w_corr
    return v_post, w_post


# ── TableContactOptimizer ────────────────────────────────────────────────

class TableContactOptimizer:
    """
    Extract table contact parameters and store in HDF5 / CSV.

    For each table bounce, runs independent aerodynamics optimisation on the
    pre- and post-bounce flight segments and extracts pre/post velocity, spin,
    and coefficient of restitution at the bounce boundary.
    """

    def __init__(self, data_folder, data: MatchCollection, recompute: bool,
                 export_csv: bool = False, use_gt200_positions: bool = False):
        self.data_folder = data_folder
        self.data = data
        self.recompute = recompute
        self.export_csv = export_csv
        self.pos_cols = ("x_gt200", "y_gt200", "z_gt200") if use_gt200_positions else ("x_aps", "y_aps", "z_aps")

        print("=============================================")
        print("\t\tTable Contact")
        print("=============================================")
        print(f"  Position source: {self.pos_cols[0].split('_', 1)[1]}")
        if self.export_csv:
            print("  CSV export: enabled (extracted_TCM.csv)")

    # ── Clean / recompute ─────────────────────────────────────────────

    def _clean_hdf5_data(self):
        """Remove all existing table_contacts data from HDF5 files."""
        print("Cleaning existing table contact data...")
        hdf5_files = list(pathlib.Path(self.data_folder).rglob("data.h5"))
        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, "a") as h5f:
                    for game_key in [k for k in h5f.keys() if k.startswith("game_")]:
                        for rally_key in [k for k in h5f[game_key].keys() if k.startswith("rally_")]:
                            rally_grp = h5f[game_key][rally_key]
                            if "ground_truth_200" in rally_grp and "table_contacts" in rally_grp["ground_truth_200"]:
                                del rally_grp["ground_truth_200"]["table_contacts"]
                print(f"  Cleaned: {file_path}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"  Warning: Failed to clean {file_path}: {e}")
        print("Table contact data cleanup complete.\n")

    # ── Main update ───────────────────────────────────────────────────

    def update_hdf(self):
        if self.recompute:
            self._clean_hdf5_data()

        all_csv_rows: list[dict] = []

        for match in self.data.matches:
            match_file = match.file
            match_parent_name = match.file.parent.name

            for game in match.games:
                for rally in game.rallies:
                    print(f"Processing Match {match.match_id} ({match.file}), "
                          f"Game {game.game_id}, Rally {rally.rally_id}")

                    rally_data = rally.rally
                    if isinstance(rally_data, list) and len(rally_data) == 0:
                        continue
                    if isinstance(rally_data, pd.DataFrame) and rally_data.empty:
                        continue

                    events = rally.events
                    if len(events) < 3:
                        continue

                    # Build sequence name (matches CSV convention)
                    sequence_name = f"{match_parent_name}_game_{game.game_id}_rally_{rally.rally_id}"

                    # Check if data already exists (skip if not recomputing)
                    if not self.recompute:
                        try:
                            with h5py.File(match_file, "r") as h5f:
                                grp_path = f"game_{game.game_id}/rally_{rally.rally_id}/ground_truth_200/table_contacts"
                                if grp_path in h5f:
                                    print(f"  Skipping (already computed)")
                                    continue
                        except Exception:  # pylint: disable=broad-exception-caught
                            pass

                    contact_rows = self._process_rally(
                        rally, events, rally_data, sequence_name,
                        match, game,
                    )

                    if not contact_rows:
                        print(f"  No table contacts found for Rally {rally.rally_id}")
                        continue

                    # ── Write to HDF5 ─────────────────────────────────
                    self._write_to_hdf5(match_file, game.game_id, rally.rally_id, contact_rows)

                    # ── Collect CSV rows ──────────────────────────────
                    if self.export_csv:
                        for row in contact_rows:
                            csv_row = {
                                "sequence_name": sequence_name,
                                "log": rally.log_identifier or "",
                                "last_modified": match.last_modified or "",
                                "date": match.date or "",
                                "court": match.court or "",
                            }
                            csv_row.update(row)
                            all_csv_rows.append(csv_row)

        # ── Write CSV ─────────────────────────────────────────────────
        if self.export_csv and all_csv_rows:
            csv_path = pathlib.Path(self.data_folder) / "extracted_TCM.csv"
            df = pd.DataFrame(all_csv_rows)
            df.to_csv(csv_path, index=False)
            print(f"\nWrote {len(df)} table contacts to {csv_path}")

    # ── Per-rally processing ──────────────────────────────────────────

    def _process_rally(self, _rally, events, rally_data, _sequence_name, _match, _game):
        """Extract table contact data for one rally.

        For each bounce event, optimises the pre- and post-bounce flight
        segments independently and extracts pre/post velocity + spin.
        """
        results = []

        for i, event in enumerate(events):
            if not isinstance(event, TableContactEvent):
                continue
            if i == 0 or i == len(events) - 1:
                continue  # need a previous and next event

            t_bounce = event.timestamp
            t_prev = events[i - 1].timestamp
            t_next = events[i + 1].timestamp

            # Extract pre-contact segment data (from previous event to bounce)
            pre_mask = (rally_data["time"] >= t_prev) & (rally_data["time"] < t_bounce)
            pre_data = rally_data.loc[pre_mask].copy()

            # Extract post-contact segment data (from bounce to next event)
            post_mask = (rally_data["time"] >= t_bounce) & (rally_data["time"] <= t_next)
            post_data = rally_data.loc[post_mask].copy()

            if len(pre_data) < 2 or len(post_data) < 2:
                continue

            # Check we have valid position data
            px, py, pz = self.pos_cols
            if pre_data[px].isna().all() or post_data[px].isna().all():
                continue

            # Drop NaN rows in position columns
            pre_data = pre_data.dropna(subset=[px, py, pz])
            post_data = post_data.dropna(subset=[px, py, pz])
            if len(pre_data) < 2 or len(post_data) < 2:
                continue

            # ── Pre-bounce segment optimisation ───────────────────────
            tstart_pre = pre_data["time"].iloc[0]
            tend_pre = pre_data["time"].iloc[-1] + DT
            # Extend simulation to the bounce boundary
            pre_result = _optimise_segment(
                pre_data, tstart_pre, tend_pre, t_bounce, DT, self.pos_cols,
            )
            if pre_result is None:
                continue

            # ── Post-bounce segment optimisation ──────────────────────
            tstart_post = post_data["time"].iloc[0]
            tend_post = post_data["time"].iloc[-1] + DT
            post_result = _optimise_segment(
                post_data, tstart_post, tend_post, t_next, DT, self.pos_cols,
            )
            if post_result is None:
                continue

            # ── Extract values at bounce boundary ─────────────────────
            # Pre: last point of pre-contact simulation (closest to bounce)
            pre_last_idx = (pre_result["t"] - t_bounce).abs().idxmin()
            pre_vel = pre_result.loc[pre_last_idx, "vel"]
            pre_spin = pre_result.loc[pre_last_idx, "spin"]
            t_pre_contact = pre_result.loc[pre_last_idx, "t"]

            # Post: first point of post-contact simulation (closest to bounce)
            post_first_idx = (post_result["t"] - t_bounce).abs().idxmin()
            post_vel = post_result.loc[post_first_idx, "vel"]
            post_spin = post_result.loc[post_first_idx, "spin"]

            # Sanity: endpoints should be close to the bounce label
            if abs(t_pre_contact - t_bounce) > 0.05:
                continue
            t_post_contact = post_result.loc[post_first_idx, "t"]
            if abs(t_post_contact - t_bounce) > 0.05:
                continue

            # Require total segment duration < 1 s (matches original)
            total_duration = post_data["time"].iloc[-1] - pre_data["time"].iloc[0]
            if total_duration > 1.0:
                continue

            vx_pre, vy_pre, vz_pre = pre_vel
            wx_pre, wy_pre, wz_pre = pre_spin
            vx_post, vy_post, vz_post = post_vel
            wx_post, wy_post, wz_post = post_spin

            # ── Coefficient of restitution ────────────────────────────
            eps_opt = _compute_epsilon_opt(vz_pre, vz_post)
            if np.isnan(eps_opt):
                continue
            # Reject super-elastic contacts
            if eps_opt >= 10.0 and vz_pre <= -10:
                continue

            # ── Nakashima model predictions ──────────────────────
            v_pre = np.array([vx_pre, vy_pre, vz_pre])
            w_pre = np.array([wx_pre, wy_pre, wz_pre])
            eps_ittf = NAKASHIMA_EPSILON_ITTF
            eps_paper = NAKASHIMA_EPSILON_PAPER
            eps_rcn = _compute_epsilon_velocity_dependent(vz_pre)

            v_post_ittf, w_post_ittf = _nakashima_contact(v_pre, w_pre, eps_ittf)
            v_post_paper, w_post_paper = _nakashima_contact(v_pre, w_pre, eps_paper)
            v_post_rcn, w_post_rcn = _residual_corrected_nakashima_contact(v_pre, w_pre, eps_rcn)
            v_post_r0805, w_post_r0805 = _residual0805_nakashima_contact(v_pre, w_pre, eps_rcn)
            v_post_pysr, w_post_pysr = _pysr_nakashima_contact(v_pre, w_pre, eps_rcn)
            v_post_0426, w_post_0426 = _0426_table_contact(v_pre, w_pre)

            results.append({
                "event_timestamp": float(t_bounce),
                "contact_type": event.type,
                "t_contact": float(t_pre_contact),
                "vx_pre": float(vx_pre), "vy_pre": float(vy_pre), "vz_pre": float(vz_pre),
                "wx_pre": float(wx_pre), "wy_pre": float(wy_pre), "wz_pre": float(wz_pre),
                "vx_post": float(vx_post), "vy_post": float(vy_post), "vz_post": float(vz_post),
                "wx_post": float(wx_post), "wy_post": float(wy_post), "wz_post": float(wz_post),
                "epsilon_opt": float(eps_opt),
                "epsilon_ittf": float(eps_ittf),
                "epsilon_paper": float(eps_paper),
                "epsilon_rcn": float(eps_rcn),
                # NakashimaITTF model
                "vx_post_NakashimaITTF": float(v_post_ittf[0]),
                "vy_post_NakashimaITTF": float(v_post_ittf[1]),
                "vz_post_NakashimaITTF": float(v_post_ittf[2]),
                "wx_post_NakashimaITTF": float(w_post_ittf[0]),
                "wy_post_NakashimaITTF": float(w_post_ittf[1]),
                "wz_post_NakashimaITTF": float(w_post_ittf[2]),
                # NakashimaPaper model
                "vx_post_NakashimaPaper": float(v_post_paper[0]),
                "vy_post_NakashimaPaper": float(v_post_paper[1]),
                "vz_post_NakashimaPaper": float(v_post_paper[2]),
                "wx_post_NakashimaPaper": float(w_post_paper[0]),
                "wy_post_NakashimaPaper": float(w_post_paper[1]),
                "wz_post_NakashimaPaper": float(w_post_paper[2]),
                # ResidualCorrectedNakashima model
                "vx_post_ResidualCorrectedNakashima": float(v_post_rcn[0]),
                "vy_post_ResidualCorrectedNakashima": float(v_post_rcn[1]),
                "vz_post_ResidualCorrectedNakashima": float(v_post_rcn[2]),
                "wx_post_ResidualCorrectedNakashima": float(w_post_rcn[0]),
                "wy_post_ResidualCorrectedNakashima": float(w_post_rcn[1]),
                "wz_post_ResidualCorrectedNakashima": float(w_post_rcn[2]),
                # Residual0805 model (LassoCV, 30K dataset)
                "vx_post_Residual0805": float(v_post_r0805[0]),
                "vy_post_Residual0805": float(v_post_r0805[1]),
                "vz_post_Residual0805": float(v_post_r0805[2]),
                "wx_post_Residual0805": float(w_post_r0805[0]),
                "wy_post_Residual0805": float(w_post_r0805[1]),
                "wz_post_Residual0805": float(w_post_r0805[2]),
                # PySR model (symbolic regression)
                "vx_post_PySR": float(v_post_pysr[0]),
                "vy_post_PySR": float(v_post_pysr[1]),
                "vz_post_PySR": float(v_post_pysr[2]),
                "wx_post_PySR": float(w_post_pysr[0]),
                "wy_post_PySR": float(w_post_pysr[1]),
                "wz_post_PySR": float(w_post_pysr[2]),
                # 0426 model (commit 1757e3ec, simple column-2 correction)
                "vx_post_0426": float(v_post_0426[0]),
                "vy_post_0426": float(v_post_0426[1]),
                "vz_post_0426": float(v_post_0426[2]),
                "wx_post_0426": float(w_post_0426[0]),
                "wy_post_0426": float(w_post_0426[1]),
                "wz_post_0426": float(w_post_0426[2]),
            })

        return results

    # ── HDF5 write ────────────────────────────────────────────────────

    @staticmethod
    def _write_to_hdf5(match_file, game_id, rally_id, contact_rows: list[dict]):
        """Write table contact data to HDF5 under ground_truth_200/table_contacts."""
        with h5py.File(match_file, "a") as h5f:
            grp_path = f"game_{game_id}/rally_{rally_id}/ground_truth_200"
            if grp_path not in h5f:
                return

            rally_grp = h5f[grp_path]
            if "table_contacts" in rally_grp:
                del rally_grp["table_contacts"]
            tc_grp = rally_grp.create_group("table_contacts")

            n = len(contact_rows)
            if n == 0:
                return

            # Scalar datasets
            scalar_fields = [
                "event_timestamp", "t_contact", "epsilon_opt", "epsilon_ittf", "epsilon_paper", "epsilon_rcn",
                "vx_pre", "vy_pre", "vz_pre", "wx_pre", "wy_pre", "wz_pre",
                "vx_post", "vy_post", "vz_post", "wx_post", "wy_post", "wz_post",
                "vx_post_NakashimaITTF", "vy_post_NakashimaITTF", "vz_post_NakashimaITTF",
                "wx_post_NakashimaITTF", "wy_post_NakashimaITTF", "wz_post_NakashimaITTF",
                "vx_post_NakashimaPaper", "vy_post_NakashimaPaper", "vz_post_NakashimaPaper",
                "wx_post_NakashimaPaper", "wy_post_NakashimaPaper", "wz_post_NakashimaPaper",
                "vx_post_ResidualCorrectedNakashima", "vy_post_ResidualCorrectedNakashima", "vz_post_ResidualCorrectedNakashima",
                "wx_post_ResidualCorrectedNakashima", "wy_post_ResidualCorrectedNakashima", "wz_post_ResidualCorrectedNakashima",
                "vx_post_Residual0805", "vy_post_Residual0805", "vz_post_Residual0805",
                "wx_post_Residual0805", "wy_post_Residual0805", "wz_post_Residual0805",
                "vx_post_PySR", "vy_post_PySR", "vz_post_PySR",
                "wx_post_PySR", "wy_post_PySR", "wz_post_PySR",
                "vx_post_0426", "vy_post_0426", "vz_post_0426",
                "wx_post_0426", "wy_post_0426", "wz_post_0426",
            ]
            for field in scalar_fields:
                data = np.array([row[field] for row in contact_rows], dtype=np.float64)
                tc_grp.create_dataset(field, data=data, dtype=np.float64)

            # String dataset: contact_type
            types = [row["contact_type"] for row in contact_rows]
            tc_grp.create_dataset(
                "contact_type",
                data=np.array(types, dtype=h5py.special_dtype(vlen=str)),
            )

        print(f"  Wrote {n} table contacts to HDF5")

    # ── Plotting (stub) ──────────────────────────────────────────────

    def plot(self):
        print("  [Table contacts] Plotting not yet implemented")
