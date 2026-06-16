# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""Data classes for ACE evaluation"""

from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import h5py
import pathlib
from scipy.spatial.transform import Rotation as _R


# ── Standalone helpers (previously in utilities.py) ──────────────────────


def list_files(path: str, extension: str) -> list[str]:
    """Return all files under *path* whose name ends with *extension*."""
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list


def _clean_trajectory(s, x, y, z):
    """Keep only the best candidate at each time-step of a multi-hypothesis trajectory."""

    def best_candidate_3d(
        prev, next_candidates_x, next_candidates_y, next_candidates_z
    ):
        best_value = None
        min_distance = float("inf")
        for nx, ny, nz in zip(next_candidates_x, next_candidates_y, next_candidates_z):
            avg_x = (prev[0] + nx) / 2
            avg_y = (prev[1] + ny) / 2
            avg_z = (prev[2] + nz) / 2
            distance = np.sqrt(
                (prev[0] - avg_x) ** 2 + (prev[1] - avg_y) ** 2 + (prev[2] - avg_z) ** 2
            )
            if nz < -0.05:
                distance += 100.0
            if distance < min_distance:
                min_distance = distance
                best_value = (nx, ny, nz)
        return best_value

    clean_s, clean_x, clean_y, clean_z = [], [], [], []
    start = 0
    while len(clean_s) == 0:
        if (
            not np.isnan(x[start][0])
            and not np.isnan(y[start][0])
            and not np.isnan(z[start][0])
        ):
            clean_s.append(s[start])
            clean_x.append(x[start][0])
            clean_y.append(y[start][0])
            clean_z.append(z[start][0])
        start += 1

    for i in range(start, len(x) - 1):
        if np.isnan(x[i][0]) and np.isnan(y[i][0]) and np.isnan(z[i][0]):
            continue
        prev = (clean_x[-1], clean_y[-1], clean_z[-1])
        best_x, best_y, best_z = best_candidate_3d(prev, x[i], y[i], z[i])
        clean_s.append(s[i])
        clean_x.append(best_x)
        clean_y.append(best_y)
        clean_z.append(best_z)

    clean_s.append(s[-1])
    clean_x.append(x[-1][0])
    clean_y.append(y[-1][0])
    clean_z.append(z[-1][0])
    return clean_s, clean_x, clean_y, clean_z


def get_ball_trajectory(aps_ball_triangulations: h5py.Group) -> pd.DataFrame:
    """Extract a clean ball trajectory DataFrame from an HDF5 APS group."""
    s, x, y, z = _clean_trajectory(
        aps_ball_triangulations["sequence_number"],
        aps_ball_triangulations["positions"][:, 0],
        aps_ball_triangulations["positions"][:, 1],
        aps_ball_triangulations["positions"][:, 2],
    )
    df = pd.DataFrame(
        {
            "sequence_number": s,
            "ball_position_x": x,
            "ball_position_y": y,
            "ball_position_z": z,
        }
    )
    df = df.dropna()
    return df


# ── Lightweight value types (NamedTuple for performance) ─────────────────


class Vec3(NamedTuple):
    """3-component vector (position, velocity, angular velocity, …)."""

    x: float = float("nan")
    y: float = float("nan")
    z: float = float("nan")


class Quat(NamedTuple):
    """Quaternion in (x, y, z, w) convention."""

    x: float = float("nan")
    y: float = float("nan")
    z: float = float("nan")
    w: float = float("nan")


class BallState(NamedTuple):
    """Ball position, velocity and spin in the global frame."""

    x: float = float("nan")
    y: float = float("nan")
    z: float = float("nan")
    vx: float = float("nan")
    vy: float = float("nan")
    vz: float = float("nan")
    wx: float = float("nan")
    wy: float = float("nan")
    wz: float = float("nan")


# ── Event hierarchy ──────────────────────────────────────────────────────


@dataclass
class Event:
    """Base event class.

    Every labelled instant in a rally is an Event.  Subclasses carry
    physics data specific to the contact type (racket hit, table bounce,
    net crossing, rally start/end).
    """

    timestamp: float
    type: str


@dataclass
class RacketContactEvent(Event):
    """Racket–ball contact (*shot_p1* / *shot_p2*).

    Ball state sourced from ``extracted.csv`` (physics-optimised).
    Racket state from forward kinematics (direct HDF5 copy in the CSV).
    All optional fields default to ``None`` / ``NaN`` and are populated
    when ``extracted.csv`` is loaded alongside the HDF5 data.
    """

    # ── Ball states (physics-optimised) ──────────────────────────────
    ball_pre: Optional[BallState] = None
    ball_post: Optional[BallState] = None

    # ── Model predictions (post-contact velocity/spin) ───────────────
    ball_post_default: Optional[BallState] = None  # Nakashima constant COR
    ball_post_exp: Optional[BallState] = None  # Nakashima exponential COR
    ball_post_piecewise: Optional[BallState] = None  # Nakashima piecewise COR
    ball_post_cpp: Optional[BallState] = None  # C++ residual model
    ball_post_cpp_refined: Optional[
        BallState
    ] = None  # C++ no-residual (refined inputs)
    ball_post_parametric_7p: Optional[BallState] = None  # Parametric 7-parameter model
    ball_post_rcm_tangential: Optional[BallState] = None  # RCM tangential
    ball_post_cpp_tangential: Optional[BallState] = None  # C++ with tangential COR ONNX
    ball_post_rcm_tangential_refined: Optional[
        BallState
    ] = None  # RCM tangential (refined inputs)
    ball_post_rcm_tangential_polyfit: Optional[
        BallState
    ] = None  # RCM tangential (polyfit inputs)
    ball_post_onnx_rcm: Optional[BallState] = None  # ONNX RCM (synapse simulation)
    ball_post_onnx_alex: Optional[BallState] = None  # ONNX alex
    ball_post_nakashima_refined: Optional[
        BallState
    ] = None  # Nakashima (refined inputs + angular vel)
    ball_post_onnx_alex_refined: Optional[
        BallState
    ] = None  # ONNX alex (refined inputs)
    ball_post_onnx_alex_polyfit: Optional[
        BallState
    ] = None  # ONNX alex (polyfit inputs)
    ball_post_onnx_0426: Optional[BallState] = None  # ONNX 0426 (old 6-input model)

    # ── Model predictions in racket (local) frame ────────────────────
    ball_vel_post_default_racket: Optional[Vec3] = None
    ball_spin_post_default_racket: Optional[Vec3] = None
    ball_vel_post_exp_racket: Optional[Vec3] = None
    ball_spin_post_exp_racket: Optional[Vec3] = None
    ball_vel_post_piecewise_racket: Optional[Vec3] = None
    ball_spin_post_piecewise_racket: Optional[Vec3] = None
    ball_vel_post_cpp_racket: Optional[Vec3] = None
    ball_spin_post_cpp_racket: Optional[Vec3] = None
    ball_vel_post_cpp_refined_racket: Optional[Vec3] = None
    ball_spin_post_cpp_refined_racket: Optional[Vec3] = None
    ball_vel_post_parametric_7p_racket: Optional[Vec3] = None
    ball_spin_post_parametric_7p_racket: Optional[Vec3] = None
    ball_vel_post_rcm_tangential_racket: Optional[Vec3] = None
    ball_spin_post_rcm_tangential_racket: Optional[Vec3] = None
    ball_vel_post_cpp_tangential_racket: Optional[Vec3] = None
    ball_spin_post_cpp_tangential_racket: Optional[Vec3] = None
    ball_vel_post_rcm_tangential_refined_racket: Optional[Vec3] = None
    ball_spin_post_rcm_tangential_refined_racket: Optional[Vec3] = None
    ball_vel_post_rcm_tangential_polyfit_racket: Optional[Vec3] = None
    ball_spin_post_rcm_tangential_polyfit_racket: Optional[Vec3] = None
    ball_vel_post_onnx_rcm_racket: Optional[Vec3] = None
    ball_spin_post_onnx_rcm_racket: Optional[Vec3] = None
    ball_vel_post_onnx_alex_racket: Optional[Vec3] = None
    ball_spin_post_onnx_alex_racket: Optional[Vec3] = None
    ball_vel_post_nakashima_refined_racket: Optional[Vec3] = None
    ball_spin_post_nakashima_refined_racket: Optional[Vec3] = None
    ball_vel_post_onnx_alex_refined_racket: Optional[Vec3] = None
    ball_spin_post_onnx_alex_refined_racket: Optional[Vec3] = None
    ball_vel_post_onnx_alex_polyfit_racket: Optional[Vec3] = None
    ball_spin_post_onnx_alex_polyfit_racket: Optional[Vec3] = None
    ball_vel_post_onnx_0426_racket: Optional[Vec3] = None
    ball_spin_post_onnx_0426_racket: Optional[Vec3] = None

    # ── Racket state at contact (FK snapshot) ────────────────────────
    racket_pos: Optional[Vec3] = None  # x/y/z_racket
    racket_quat: Optional[Quat] = None  # qx/qy/qz/qw_racket
    racket_vel: Optional[Vec3] = None  # vx/vy/vz_racket
    racket_ang_vel: Optional[Vec3] = None  # wx/wy/wz_racket

    # ── Refined racket state (angular-velocity-corrected) ────────────
    racket_pos_pre_refined: Optional[Vec3] = None  # x/y/z_racket_pre_refined
    racket_pos_post_refined: Optional[Vec3] = None  # x/y/z_racket_post_refined
    racket_quat_pre_refined: Optional[Quat] = None  # qx/qy/qz/qw_racket_pre_refined
    racket_quat_post_refined: Optional[Quat] = None  # qx/qy/qz/qw_racket_post_refined
    racket_vel_pre_refined: Optional[Vec3] = None  # vx/vy/vz_racket_w_angvel_pre
    racket_vel_post_refined: Optional[Vec3] = None  # vx/vy/vz_racket_w_angvel_post

    # ── Refined ball position at contact ─────────────────────────────
    ball_pos_pre_refined: Optional[Vec3] = None  # x/y/z_pre_refined
    ball_pos_post_refined: Optional[Vec3] = None  # x/y/z_post_refined

    # ── Quality metrics from physics optimisation ────────────────────
    fitness_pre: float = float("nan")
    fitness_post: float = float("nan")
    dt_racket: float = float("nan")
    pre_duration: float = float("nan")
    post_duration: float = float("nan")

    # ── Contact time offsets ─────────────────────────────────────────
    t_contact_pre_offset: float = float("nan")
    t_contact_post_offset: float = float("nan")

    # ── Position deltas (ball displacement during opt window) ────────
    delta_pre: Optional[Vec3] = None  # dx/dy/dz_pre
    delta_post: Optional[Vec3] = None  # dx/dy/dz_post

    # ── Derived quantities (computed after CSV load + quat fix) ──────

    # Racket-frame ball velocity/spin (standard racket quat)
    ball_vel_pre_racket: Optional[Vec3] = None  # vrx/vry/vrz_pre
    ball_vel_post_racket: Optional[Vec3] = None  # vrx/vry/vrz_post
    ball_spin_pre_racket: Optional[Vec3] = None  # wrx/wry/wrz_pre
    ball_spin_post_racket: Optional[Vec3] = None  # wrx/wry/wrz_post

    # Racket-frame ball velocity/spin (refined, using pre_refined quat)
    ball_vel_pre_refined_racket: Optional[Vec3] = None  # vrx/vry/vrz_pre_refined
    ball_vel_post_refined_racket: Optional[Vec3] = None  # vrx/vry/vrz_post_refined
    ball_spin_pre_refined_racket: Optional[Vec3] = None  # wrx/wry/wrz_pre_refined
    ball_spin_post_refined_racket: Optional[Vec3] = None  # wrx/wry/wrz_post_refined

    # Racket velocity/spin projected to racket frame
    racket_vel_racket: Optional[Vec3] = None  # vrx/vry/vrz_racket
    racket_spin_racket: Optional[Vec3] = None  # wrx/wry/wrz_racket
    racket_spin_body: Optional[
        Vec3
    ] = None  # wbx/wby/wbz_racket (pure R^T, no v_ref alignment)

    # Relative velocity in racket frame (ball_pre_racket - racket_vel_racket)
    relative_vel_racket: Optional[Vec3] = None  # vrx/vry/vrz_relative

    # Ball position relative to racket, in racket frame
    ball_pos_pre_racket: Optional[Vec3] = None  # drx/dry/drz_pre

    # Kinetic energy
    ke_lin_pre: float = float("nan")
    ke_lin_post: float = float("nan")
    ke_ang_pre: float = float("nan")
    ke_ang_post: float = float("nan")

    # Angles
    theta_angle: float = float("nan")  # angle between relative vel and racket plane
    racket_open_angle: float = float("nan")  # racket tilt from horizontal

    # Simulated post-contact trajectory (from RCM ball simulation)
    simulated_trajectory: Optional[
        Dict
    ] = None  # keys: t, x, y, z, vx, vy, vz, wx, wy, wz
    simulated_trajectory_rcm_refined: Optional[Dict] = None
    simulated_trajectory_rcm_polyfit: Optional[Dict] = None
    simulated_trajectory_onnx_rcm: Optional[Dict] = None
    simulated_trajectory_onnx_alex: Optional[Dict] = None
    simulated_trajectory_nakashima_refined: Optional[Dict] = None
    simulated_trajectory_cpp_no_residual_refined: Optional[Dict] = None
    simulated_trajectory_latest: Optional[Dict] = None


@dataclass
class TableContactEvent(Event):
    """Table bounce (*bounce_p1* / *bounce_p2*).

    Ball state from HDF5 boundary samples (approximate), or from
    physics-optimised ODE trajectories when ``extracted_TCM.csv`` is loaded.
    """

    ball_pre: Optional[BallState] = None
    ball_post: Optional[BallState] = None
    epsilon_opt: float = float(
        "nan"
    )  # observed coefficient of restitution |vz_post/vz_pre|

    # Model predictions (populated from HDF5 table_contacts group)
    ball_post_nakashima_ittf: Optional[BallState] = None
    ball_post_nakashima_paper: Optional[BallState] = None
    ball_post_residual_corrected: Optional[BallState] = None
    ball_post_residual0805: Optional[BallState] = None
    ball_post_pysr: Optional[BallState] = None
    ball_post_0426: Optional[BallState] = None


@dataclass
class NetEvent(Event):
    """Ball crosses the net."""

    ball_pre: Optional[BallState] = None
    ball_post: Optional[BallState] = None


@dataclass
class RallyStartEvent(Event):
    """Rally begins.  Only post-contact state available (first observed sample)."""

    ball_post: Optional[BallState] = None


@dataclass
class RallyEndEvent(Event):
    """Rally ends (tracking lost or ball out).  Only pre-contact state available."""

    ball_pre: Optional[BallState] = None


# ── Derived-quantity helpers (pure math, no physics_layer dep) ───────


def _fix_quat_orientation(
    b_vel: np.ndarray, r_vel: np.ndarray, r_quat_xyzw: np.ndarray
) -> np.ndarray:
    """Fix racket quaternion so the normal faces the incoming ball.

    If ``dot(ball_vel - racket_vel, racket_normal) > 0`` the racket normal
    is pointing *away* from the ball.  In that case we flip the quaternion
    180° around the local z-axis (equivalent to ``quat_quat_rot`` with
    ``[0, 0, 1, 0]`` in xyzw convention).
    """
    v_relative = b_vel - r_vel
    rot_mat = _R.from_quat(r_quat_xyzw).as_matrix()
    if np.dot(v_relative, rot_mat[:, 0]) > 0:
        flip = np.array([0.0, 0.0, 1.0, 0.0])  # 180° around z, xyzw
        return (_R.from_quat(r_quat_xyzw) * _R.from_quat(flip)).as_quat()
    return r_quat_xyzw


def _project_to_racket(
    v_ref: np.ndarray, v: np.ndarray, rot_quat_xyzw: np.ndarray
) -> np.ndarray:
    """Project *v* into the racket frame aligned with reference velocity.

    Mirrors ``math_utilities.project_to_racket`` but takes a raw xyzw
    quaternion instead of a ``scipy.spatial.transform.Rotation``.
    """
    rotation = _R.from_quat(rot_quat_xyzw).inv().as_matrix()
    tmp_v_ref = rotation @ v_ref
    tmp_v = rotation @ v

    _, vry, vrz = tmp_v_ref
    norm_yz = np.sqrt(vry**2 + vrz**2)
    if norm_yz < 1e-12:
        return tmp_v  # degenerate case – cannot align plane

    u_x = np.array([1.0, 0.0, 0.0])
    u_y = np.array([0.0, vry / norm_yz, vrz / norm_yz])
    u_z = np.array([0.0, -vrz / norm_yz, vry / norm_yz])
    return np.vstack([u_x, u_y, u_z]) @ tmp_v


def _compute_theta_angle(
    b_vel: np.ndarray, r_vel: np.ndarray, r_quat_xyzw: np.ndarray
) -> float:
    """Angle (degrees) between relative velocity and the racket plane."""
    v = r_vel - b_vel
    n = _R.from_quat(r_quat_xyzw).apply([1, 0, 0])
    norms = np.linalg.norm(v) * np.linalg.norm(n)
    if norms < 1e-12:
        return float("nan")
    angle_to_normal = np.arccos(np.clip(np.dot(v, n) / norms, -1.0, 1.0))
    return float(np.degrees(np.pi / 2 - angle_to_normal))


def _compute_racket_angle(r_quat_xyzw: np.ndarray) -> float:
    """Racket tilt from horizontal (degrees).  Positive = open face."""
    rot_mat = _R.from_quat(r_quat_xyzw).as_matrix()
    return float(np.degrees(np.arcsin(rot_mat[:, 0][2])))


_BALL_MASS = 81.2 * 0.02**3 * 4.0 * np.pi / 3.0
_BALL_INERTIA = 2.0 / 3.0 * _BALL_MASS * 0.02**2


def _compute_kinetic_energy(
    v_pre: np.ndarray, v_post: np.ndarray, w_pre: np.ndarray, w_post: np.ndarray
) -> np.ndarray:
    """Return ``[ke_lin_pre, ke_lin_post, ke_ang_pre, ke_ang_post]``."""
    I = _BALL_INERTIA * np.eye(3)
    return np.array(
        [
            0.5 * _BALL_MASS * np.dot(v_pre, v_pre),
            0.5 * _BALL_MASS * np.dot(v_post, v_post),
            0.5 * np.dot(w_pre, I @ w_pre),
            0.5 * np.dot(w_post, I @ w_post),
        ]
    )


@dataclass
class FlightSegment:
    """flight segment class"""

    data: pd.DataFrame
    trigger_event: Optional[Event]

    def estimate_velocity(self) -> pd.DataFrame:
        """Estimate velocity using central finite differences with one-sided at edges"""
        if len(self.data) < 2:
            return self.data

        time = self.data["time"].values
        x = self.data["x_aps"].values
        y = self.data["y_aps"].values
        z = self.data["z_aps"].values

        self.data["vx"] = np.zeros(len(time))
        self.data["vy"] = np.zeros(len(time))
        self.data["vz"] = np.zeros(len(time))

        # Forward difference for first point
        if len(time) > 1:
            dt_forward = time[1] - time[0]
            self.data.loc[self.data.index[0], "vx"] = (x[1] - x[0]) / dt_forward
            self.data.loc[self.data.index[0], "vy"] = (y[1] - y[0]) / dt_forward
            self.data.loc[self.data.index[0], "vz"] = (z[1] - z[0]) / dt_forward

        # Central differences for interior points
        for i in range(1, len(time) - 1):
            dt_central = time[i + 1] - time[i - 1]
            self.data.loc[self.data.index[i], "vx"] = (x[i + 1] - x[i - 1]) / dt_central
            self.data.loc[self.data.index[i], "vy"] = (y[i + 1] - y[i - 1]) / dt_central
            self.data.loc[self.data.index[i], "vz"] = (z[i + 1] - z[i - 1]) / dt_central

        # Backward difference for last point
        if len(time) > 1:
            dt_backward = time[-1] - time[-2]
            self.data.loc[self.data.index[-1], "vx"] = (x[-1] - x[-2]) / dt_backward
            self.data.loc[self.data.index[-1], "vy"] = (y[-1] - y[-2]) / dt_backward
            self.data.loc[self.data.index[-1], "vz"] = (z[-1] - z[-2]) / dt_backward

        return self.data

    def estimate_velocity_polyfit(
        self, n_samples: int = 10, deg: int = 2
    ) -> pd.DataFrame:
        """Estimate velocity by fitting a polynomial to positions.

        Fits a degree-`deg` polynomial to the first `n_samples` points of the
        segment and evaluates its derivative at each timestep. This is more
        robust than finite differences near contacts, where positions across
        a contact event can produce wildly incorrect velocity estimates.

        For the initial velocity (iloc[0]), the polynomial derivative is
        particularly valuable since it uses information from multiple
        subsequent points rather than just the immediate neighbor.

        Args:
            n_samples: Number of samples from the start of the segment to use
                for the polynomial fit. Clamped to segment length.
            deg: Degree of the polynomial fit (default 2, i.e. quadratic,
                which matches ballistic motion).
        """
        if len(self.data) < 2:
            return self.data

        time = self.data["time"].values
        x = self.data["x_aps"].values
        y = self.data["y_aps"].values
        z = self.data["z_aps"].values

        n = min(len(time), n_samples)
        t_fit = time[:n] - time[0]  # shift to start at 0 for numerical stability

        if n < deg + 1:
            # Not enough points for the requested degree, fall back to FD
            return self.estimate_velocity()

        # Fit polynomials per coordinate
        cx = np.polyfit(t_fit, x[:n], deg)
        cy = np.polyfit(t_fit, y[:n], deg)
        cz = np.polyfit(t_fit, z[:n], deg)

        # Derivative coefficients: d/dt of polynomial
        dcx = np.polyder(cx)
        dcy = np.polyder(cy)
        dcz = np.polyder(cz)

        # Evaluate derivative at all timesteps within the fit range
        t_all = time - time[0]
        self.data["vx"] = np.polyval(dcx, t_all)
        self.data["vy"] = np.polyval(dcy, t_all)
        self.data["vz"] = np.polyval(dcz, t_all)

        return self.data


@dataclass
class Shot:
    """shot class"""

    flight_segments: List[FlightSegment]  # 3 if serve, 2 otherwise
    is_serve: bool
    start_time: float
    end_time: float
    player: int


@dataclass
class Rally:
    """rally class"""

    rally: pd.DataFrame
    shots: List[Shot]
    events: List[Event]
    rally_id: int
    point_winner: Optional[str] = None  # "player1" (robot) or "player2" (human)
    log_identifier: Optional[str] = None  # log identifier from HDF5 sensor attrs
    gcs_source: Optional[str] = None  # which gcs sensor was used, or reason for absence


@dataclass
class Game:
    """game class"""

    rallies: List[Rally]
    game_id: int
    game_name: str
    game_score_player: Optional[int] = None
    game_score_robot: Optional[int] = None


@dataclass
class Match:
    """match class"""

    games: List[Game]
    date: datetime.date
    match_id: int
    player: str
    policy: str
    court: Optional[str] = None
    last_modified: Optional[str] = None  # file modification time string
    data_usage: Optional[str] = None  # file/data_usage HDF5 attribute
    version_data_processing: Optional[
        str
    ] = None  # file/version_data_processing HDF5 attribute
    match_score_player: Optional[int] = None
    match_score_robot: Optional[int] = None
    file: pathlib.Path = pathlib.Path("")


@dataclass
class MatchCollection:
    """match collection class"""

    matches: List[Match]
    base_folder: Optional[pathlib.Path] = None

    def __init__(
        self,
        data_folder: pathlib.Path,
        parallel: bool = True,
        max_workers: int = None,
        gcs_fallback: bool = False,
        allowed_files: Optional[List[str]] = None,
    ):
        """load match collection

        Args:
            data_folder: Path to folder containing HDF5 files
            parallel: If True, load files in parallel (default: True)
            max_workers: Maximum number of parallel workers. If None, uses min(cpu_count(), num_files)
            gcs_fallback: If True, fall back to gcs_filtered / gcs when gcs_offline is not
                available.  If False (default), rallies without gcs_offline are excluded
                (a warning is printed for each).
            allowed_files: If provided, only load these specific HDF5 file paths
                (overrides the default directory scan).
        """
        self.base_folder = pathlib.Path(data_folder)
        self.gcs_fallback = gcs_fallback
        if allowed_files is not None:
            files_h5 = allowed_files
        else:
            files_h5 = list_files(data_folder, ".h5")
        # exclude files with optitrack in the name
        files_h5 = [
            f
            for f in files_h5
            if "optitrack" not in pathlib.Path(f).name.lower()
            and "with_opt" not in pathlib.Path(f).name.lower()
        ]

        if parallel and len(files_h5) > 1:
            # Parallel loading
            if max_workers is None:
                max_workers = min(cpu_count(), len(files_h5))

            print(
                f"Loading {len(files_h5)} files in parallel using {max_workers} workers..."
            )
            self.matches = []

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks with their original index for ordering
                future_to_idx = {
                    executor.submit(
                        self._load_match_static, file_path, idx, gcs_fallback
                    ): idx
                    for idx, file_path in enumerate(files_h5)
                }

                # Collect results as they complete
                results = [None] * len(files_h5)
                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        match = future.result()
                        results[idx] = match
                        completed += 1
                        if completed % 10 == 0 or completed == len(files_h5):
                            print(f"Loaded {completed}/{len(files_h5)} files")
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"Error loading file {files_h5[idx]}: {e}")
                        results[idx] = None

                # Filter out failed loads and store matches in order
                self.matches = [match for match in results if match is not None]  # type: ignore[misc]
        else:
            # Sequential loading (original behavior)
            self.matches = []
            match_id = 0
            for file_path in files_h5:
                match = self._load_match_static(file_path, match_id, gcs_fallback)
                self.matches.append(match)
                match_id += 1

        # ── Summary ──────────────────────────────────────────────────
        total_games = sum(len(m.games) for m in self.matches)
        total_rallies = sum(len(g.rallies) for m in self.matches for g in m.games)
        print(
            f"Loaded {len(self.matches)} matches, {total_games} games, {total_rallies} rallies"
        )

        # ── Enrich contact events from HDF5 ─────────────────────────
        self._load_racket_contacts_from_hdf5(files_h5)
        self._load_table_contacts_from_hdf5(files_h5)
        self._report_unenriched_events()

        # ── Compute derived quantities (quat fix, projections, KE, …) ─
        self._compute_derived_quantities()

    # ------------------------------------------------------------------
    # Alternative constructor: load specific rallies from a JSON manifest
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, json_path: str, gcs_fallback: bool = False) -> "MatchCollection":
        """Create a :class:`MatchCollection` by loading only the rallies listed in a JSON file.

        The JSON file must contain a list of objects, each with:

        - ``lab``             – e.g. ``"tyo01"``
        - ``date``            – e.g. ``"20260325"``
        - ``experiment``      – e.g. ``"ace_t_vs_kihara_match_0"``
        - ``game_id``         – e.g. ``"game_0"``
        - ``rally_id``        – e.g. ``"rally_1"``
        - ``sequence_number`` – *(optional)* reference sequence number

        The HDF5 path is constructed as
        ``<json_parent>/<lab>/<date>/<experiment>/data.h5``.

        Alternatively, entries may provide ``hdf5_filepath`` directly
        (relative paths are resolved against the JSON file's parent directory).

        Entries are grouped by file so each HDF5 is opened only once.
        The resulting :class:`MatchCollection` goes through the same
        enrichment and derived-quantity pipeline as a folder-based load.

        Args:
            json_path: Path to the JSON manifest file.
            gcs_fallback: Fall back to ``gcs_filtered``/``gcs`` when
                ``gcs_offline`` is not available.
        """
        import json as _json

        json_p = pathlib.Path(json_path).resolve()
        if not json_p.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        json_parent = json_p.parent

        with open(json_p, "r") as fh:
            entries = _json.load(fh)

        if not isinstance(entries, list):
            raise ValueError("JSON root must be a list of rally entries")

        # Resolve HDF5 filepath for each entry and group by file
        from collections import defaultdict as _defaultdict

        by_file: dict = _defaultdict(list)
        for i, entry in enumerate(entries):
            for key in ("game_id", "rally_id"):
                if key not in entry:
                    raise ValueError(f"Entry {i} is missing required key '{key}'")

            # Construct path from lab/date/experiment (preferred)
            lab = entry.get("lab") or entry.get("court")
            date = entry.get("date")
            experiment = entry.get("experiment")

            if lab and date and experiment:
                h5path = str(json_parent / lab / date / experiment / "data.h5")
            elif "hdf5_filepath" in entry:
                # Fallback: use explicit hdf5_filepath
                h5p = pathlib.Path(entry["hdf5_filepath"])
                if not h5p.is_absolute():
                    h5p = json_parent / h5p
                h5path = str(h5p)
            else:
                raise ValueError(
                    f"Entry {i}: needs either (lab, date, experiment) or hdf5_filepath"
                )

            by_file[h5path].append(entry)

        # Build Match objects, loading only requested game/rally pairs
        instance = object.__new__(cls)
        instance.matches = []
        instance.base_folder = None
        instance.gcs_fallback = gcs_fallback

        files_h5: List[str] = []
        match_id = 0
        _n_good = 0
        _n_warn = 0
        _n_bad = 0

        for file_path, file_entries in by_file.items():
            files_h5.append(file_path)

            # Determine which (game_id, rally_id) pairs we need
            requested: dict = _defaultdict(set)  # game_key -> set of rally_keys
            for e in file_entries:
                requested[e["game_id"]].add(e["rally_id"])

            print(
                f"Opening {file_path} ({sum(len(v) for v in requested.values())} rallies) ..."
            )
            try:
                with h5py.File(file_path, "r") as f:
                    games: List[Game] = []
                    for game_key in sorted(requested.keys()):
                        if game_key not in f:
                            print(f"  ✗ {game_key} not found in {file_path}")
                            continue
                        game_group = f[game_key]
                        game_idx = int(game_key.split("_")[1])
                        rallies: List[Rally] = []
                        for rally_key in sorted(requested[game_key]):
                            if rally_key not in game_group:
                                print(f"  ✗ {game_key}/{rally_key} not found")
                                continue
                            rally_idx = int(rally_key.split("_")[1])
                            try:
                                rally = cls._load_rally_static(
                                    game_group[rally_key],
                                    rally_idx,
                                    gcs_fallback,
                                    file_path=file_path,
                                )
                                # Report quality issues for this rally
                                _YELLOW = "\033[93m"
                                _RED = "\033[91m"
                                _RESET = "\033[0m"
                                src = rally.gcs_source or "none"
                                is_empty = not rally.shots
                                if "no_ground_truth_200" in src:
                                    print(
                                        f"  {_RED}✗ {game_key}/{rally_key}: missing ground_truth_200{_RESET}"
                                    )
                                    _n_bad += 1
                                elif "no_racket1" in src:
                                    print(
                                        f"  {_RED}✗ {game_key}/{rally_key}: missing racket1 in ground_truth_200{_RESET}"
                                    )
                                    _n_bad += 1
                                elif "excluded" in src:
                                    print(
                                        f"  {_YELLOW}⚠ {game_key}/{rally_key}: {src} — rally excluded{_RESET}"
                                    )
                                    _n_warn += 1
                                elif src == "none":
                                    print(
                                        f"  {_YELLOW}⚠ {game_key}/{rally_key}: no gcs sensor — spin will be NaN{_RESET}"
                                    )
                                    _n_warn += 1
                                elif is_empty:
                                    print(
                                        f"  {_YELLOW}⚠ {game_key}/{rally_key}: empty rally (gcs_source={src}){_RESET}"
                                    )
                                    _n_warn += 1
                                elif src != "gcs_offline":
                                    print(
                                        f"  {_YELLOW}⚠ {game_key}/{rally_key}: using {src} (not gcs_offline){_RESET}"
                                    )
                                    _n_warn += 1
                                else:
                                    n_shots = len(rally.shots)
                                    n_events = len(rally.events)
                                    print(
                                        f"  ✓ {game_key}/{rally_key} ({n_shots} shots, {n_events} events, gcs={src})"
                                    )
                                    _n_good += 1
                                rallies.append(rally)
                            except Exception as exc:  # pylint: disable=broad-exception-caught
                                print(f"  ✗ {game_key}/{rally_key}: {exc}")

                        if rallies:
                            path = pathlib.Path(file_path)
                            experiment = path.parent.name
                            games.append(
                                Game(
                                    rallies=rallies,
                                    game_id=game_idx,
                                    game_name=experiment,
                                )
                            )

                    if games:
                        path = pathlib.Path(file_path)
                        parent_dirs = path.parent.parts
                        date_str = parent_dirs[-2] if len(parent_dirs) >= 2 else ""
                        player_dir = parent_dirs[-1] if len(parent_dirs) >= 1 else ""

                        experiment_name = f.attrs.get("experiment/name", "")
                        if experiment_name and "vs_" in experiment_name:
                            try:
                                player_name = experiment_name.split("vs_")[1].split(
                                    "_match_"
                                )[0]
                            except IndexError:
                                player_name = player_dir
                        else:
                            player_name = player_dir

                        court = None
                        for part in parent_dirs:
                            lp = part.lower()
                            if "court" in lp or lp.startswith(("tyo", "osa", "cmax")):
                                court = part
                                break

                        import time as _time_mod

                        try:
                            last_modified = _time_mod.ctime(path.stat().st_mtime)
                        except OSError:
                            last_modified = None

                        instance.matches.append(
                            Match(
                                games=games,
                                date=f.attrs.get("experiment/date", date_str),
                                player=player_name,
                                policy=f.attrs.get("experiment/name", player_dir),
                                court=court,
                                last_modified=last_modified,
                                data_usage=f.attrs.get("file/data_usage", ""),
                                match_id=match_id,
                                file=path,
                            )
                        )
                        match_id += 1

            except OSError as exc:
                print(f"  ✗ Cannot open file: {exc}")

        # Use the JSON file's parent directory as base_folder (the manifest
        # sits at the root of the dataset, so plots/ should live next to it).
        instance.base_folder = json_p.parent

        total_games = sum(len(m.games) for m in instance.matches)
        total_rallies = sum(len(g.rallies) for m in instance.matches for g in m.games)
        _YELLOW = "\033[93m"
        _RED = "\033[91m"
        _RESET = "\033[0m"
        summary_parts = [
            f"Loaded {total_rallies} rallies ({len(instance.matches)} matches, {total_games} games) from JSON"
        ]
        if _n_good:
            summary_parts.append(f"✓ {_n_good} good")
        if _n_warn:
            summary_parts.append(f"{_YELLOW}⚠ {_n_warn} warnings{_RESET}")
        if _n_bad:
            summary_parts.append(f"{_RED}✗ {_n_bad} errors{_RESET}")
        print(" | ".join(summary_parts))

        # Run the same enrichment pipeline as the standard constructor
        instance._load_racket_contacts_from_hdf5(files_h5)
        instance._load_table_contacts_from_hdf5(files_h5)
        instance._report_unenriched_events()
        instance._compute_derived_quantities()

        return instance

    def to_json(self, json_path: str, *, relative_to: str = None) -> None:
        """Export every rally in this collection as a JSON manifest.

        The file produced is compatible with :meth:`from_json` so it can be
        loaded back.  Each entry contains ``lab``, ``date``, ``experiment``,
        ``game_id``, ``rally_id``, and a running ``sequence_number``.

        Args:
            json_path: Destination path for the JSON file.
            relative_to: If given, ``hdf5_filepath`` is also emitted as a
                path relative to this directory.  Otherwise omitted.
        """
        import json as _json

        entries: list = []
        seq = 0
        for match in self.matches:
            h5_path = pathlib.Path(match.file).resolve()
            # Derive lab / date / experiment from path:
            #   <...>/<lab>/<date>/<experiment>/data.h5
            experiment = h5_path.parent.name
            date = h5_path.parent.parent.name
            lab = h5_path.parent.parent.parent.name

            for game in match.games:
                game_key = f"game_{game.game_id}"
                for rally in game.rallies:
                    rally_key = f"rally_{rally.rally_id}"
                    entry: dict = {
                        "lab": lab,
                        "date": date,
                        "experiment": experiment,
                        "game_id": game_key,
                        "rally_id": rally_key,
                        "sequence_number": seq,
                    }
                    if relative_to is not None:
                        try:
                            entry["hdf5_filepath"] = str(
                                h5_path.relative_to(pathlib.Path(relative_to).resolve())
                            )
                        except ValueError:
                            entry["hdf5_filepath"] = str(h5_path)
                    entries.append(entry)
                    seq += 1

        out = pathlib.Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            _json.dump(entries, fh, indent=2)
        print(f"Saved {len(entries)} rally entries to {out}")

    def enrich_from_csv(self) -> None:
        """Re-run HDF5-based enrichment and derived-quantity computation.

        This is useful after deserialising a cached :class:`MatchCollection`
        whose pickle was created before the enrichment pipeline existed,
        or when the HDF5 contact data has been regenerated.

        .. note:: Despite the legacy name, this method no longer reads CSV
           files.  Contact data is sourced exclusively from HDF5.
        """
        files_h5 = []
        for root, _, files in os.walk(str(self.base_folder)):
            for f in files:
                if f.lower().endswith(".h5"):
                    files_h5.append(os.path.join(root, f))
        files_h5 = [
            f
            for f in files_h5
            if "optitrack" not in pathlib.Path(f).name.lower()
            and "with_opt" not in pathlib.Path(f).name.lower()
        ]
        self._load_racket_contacts_from_hdf5(files_h5)
        self._load_table_contacts_from_hdf5(files_h5)
        self._report_unenriched_events()
        self._compute_derived_quantities()

    # ── Unenriched-event diagnostics ─────────────────────────────────

    _RED = "\033[91m"
    _YELLOW = "\033[93m"
    _RESET = "\033[0m"

    def _report_unenriched_events(self) -> None:
        """Report contact enrichment statistics.

        Distinguishes between:
        - Rallies where the HDF5 group is entirely missing (update_hdf5 was
          never run) → printed in **red**.
        - Events that exist but weren't extracted because they failed quality
          checks (bad spin confidence, short trajectory, etc.) → printed as a
          normal info line, since this is expected behaviour.
        """
        rc_robot_total = 0
        rc_robot_enriched = 0
        rc_player_total = 0
        rc_player_enriched = 0
        rc_rallies_missing_group = 0
        rc_rallies_total = 0
        tc_total = 0
        tc_enriched = 0
        tc_rallies_missing_group = 0
        tc_rallies_total = 0
        rc_files_missing_group: set = set()
        tc_files_missing_group: set = set()

        for match in self.matches:
            try:
                f = h5py.File(str(match.file), "r")
            except Exception:  # pylint: disable=broad-exception-caught
                f = None
            for game in match.games:
                for rally in game.rallies:
                    has_rc_events = False
                    has_tc_events = False
                    for event in rally.events:
                        if isinstance(event, RacketContactEvent):
                            is_robot = event.type == "shot_p1"
                            has_rc_events = True
                            enriched = event.ball_pre is not None and not np.isnan(
                                event.ball_pre.vx
                            )
                            if is_robot:
                                rc_robot_total += 1
                                if enriched:
                                    rc_robot_enriched += 1
                            else:
                                rc_player_total += 1
                                if enriched:
                                    rc_player_enriched += 1
                        elif isinstance(event, TableContactEvent):
                            tc_total += 1
                            has_tc_events = True
                            if event.ball_pre is not None and not np.isnan(
                                event.ball_pre.vx
                            ):
                                tc_enriched += 1

                    # Check whether the HDF5 group exists at all
                    if f is not None:
                        rc_path = f"game_{game.game_id}/rally_{rally.rally_id}/ground_truth_200/racket_contacts"
                        tc_path = f"game_{game.game_id}/rally_{rally.rally_id}/ground_truth_200/table_contacts"
                        if has_rc_events:
                            rc_rallies_total += 1
                            if rc_path not in f:
                                rc_rallies_missing_group += 1
                                rc_files_missing_group.add(str(match.file))
                        if has_tc_events:
                            tc_rallies_total += 1
                            if tc_path not in f:
                                tc_rallies_missing_group += 1
                                tc_files_missing_group.add(str(match.file))
            if f is not None:
                f.close()

        # ── Racket contacts ──
        rc_total = rc_robot_total + rc_player_total
        rc_enriched = rc_robot_enriched + rc_player_enriched
        if rc_rallies_missing_group > 0:
            rc_files_str = "\n    ".join(sorted(rc_files_missing_group))
            print(
                f"{self._RED}ERROR: {rc_rallies_missing_group}/{rc_rallies_total} rallies "
                f"have no racket_contacts HDF5 group — run update_hdf5.py with "
                f"--recompute_racket_contacts True\n"
                f"  Affected files:\n    {rc_files_str}{self._RESET}"
            )
        if rc_total > 0:
            rc_robot_not = rc_robot_total - rc_robot_enriched
            print(
                f"[HDF5] Racket contacts: {rc_enriched}/{rc_total} enriched "
                f"(robot: {rc_robot_enriched}/{rc_robot_total}, "
                f"player: {rc_player_enriched}/{rc_player_total})"
            )
            if rc_robot_not > 0:
                print(
                    f"  Robot contacts not enriched: {rc_robot_not} "
                    f"(no racket_contacts entry in HDF5 — run update_hdf5.py "
                    f"--recompute_racket_contacts True)"
                )

        # ── Table contacts ──
        tc_not_extracted = tc_total - tc_enriched
        if tc_rallies_missing_group > 0:
            tc_files_str = "\n    ".join(sorted(tc_files_missing_group))
            print(
                f"{self._RED}ERROR: {tc_rallies_missing_group}/{tc_rallies_total} rallies "
                f"have no table_contacts HDF5 group — run update_hdf5.py with "
                f"--recompute_table_contacts True\n"
                f"  Affected files:\n    {tc_files_str}{self._RESET}"
            )
        if tc_total > 0:
            print(
                f"[HDF5] Table contacts: {tc_enriched}/{tc_total} enriched "
                f"({tc_not_extracted} not enriched)"
            )

    # ── HDF5-based enrichment (primary source) ───────────────────────

    def _load_racket_contacts_from_hdf5(self, _files_h5: List[str]) -> None:
        """Enrich :class:`RacketContactEvent` objects from ``ground_truth_200/racket_contacts`` in HDF5.

        This is the primary enrichment path.  Events already enriched here
        will be skipped by the CSV fallback.
        """
        enriched = 0
        for match in self.matches:
            h5_path = str(match.file)
            try:
                f = h5py.File(h5_path, "r")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"  Warning: Cannot open HDF5 {h5_path}: {e}")
                continue
            try:
                for game in match.games:
                    for rally in game.rallies:
                        grp_path = f"game_{game.game_id}/rally_{rally.rally_id}/ground_truth_200/racket_contacts"
                        if grp_path not in f:
                            continue
                        rc_grp = f[grp_path]

                        # Read all datasets once
                        try:
                            timestamps = np.array(
                                rc_grp["event_timestamp"], dtype=np.float64
                            )
                        except KeyError:
                            continue
                        n = len(timestamps)
                        if n == 0:
                            continue

                        # Helper to safely read a float dataset
                        def _get_float_array(name, _grp=rc_grp, _n=n):
                            if name in _grp:
                                return np.array(_grp[name], dtype=np.float64)
                            return np.full(_n, np.nan)

                        def _get_str_array(name, _grp=rc_grp, _n=n):
                            if name in _grp:
                                return [
                                    s.decode("utf-8")
                                    if isinstance(s, bytes)
                                    else str(s)
                                    for s in _grp[name][()]
                                ]
                            return [""] * _n

                        # Build per-contact dicts (skip HDF5 sub-groups)
                        contact_data = {}
                        for key in rc_grp.keys():
                            if isinstance(rc_grp[key], h5py.Group):
                                continue
                            if key in ("ball_type",):
                                contact_data[key] = _get_str_array(key)
                            else:
                                try:
                                    contact_data[key] = _get_float_array(key)
                                except Exception:  # pylint: disable=broad-exception-caught
                                    pass

                        # Match to events by timestamp
                        for event in rally.events:
                            if not isinstance(event, RacketContactEvent):
                                continue
                            if event.ball_pre is not None and not np.isnan(
                                event.ball_pre.vx
                            ):
                                continue  # already enriched
                            # Find matching timestamp
                            diffs = np.abs(timestamps - event.timestamp)
                            best_idx = np.argmin(diffs)
                            if diffs[best_idx] > 0.001:
                                continue
                            # Build a Series matching the CSV column layout
                            row_dict = {}
                            for key, arr in contact_data.items():
                                row_dict[key] = arr[best_idx]
                            row_series = pd.Series(row_dict)
                            MatchCollection._enrich_racket_event(event, row_series)

                            # Load simulated trajectory if available
                            sim_traj_path = (
                                f"{grp_path}/simulated_trajectories/{best_idx}"
                            )
                            if sim_traj_path in f:
                                sim_grp = f[sim_traj_path]
                                _TRAJ_FIELDS = (
                                    "t",
                                    "x",
                                    "y",
                                    "z",
                                    "vx",
                                    "vy",
                                    "vz",
                                    "wx",
                                    "wy",
                                    "wz",
                                )
                                sim_traj = {}
                                for tf in _TRAJ_FIELDS:
                                    if tf in sim_grp:
                                        sim_traj[tf] = np.array(
                                            sim_grp[tf], dtype=np.float64
                                        )
                                if "t" in sim_traj and len(sim_traj["t"]) >= 2:
                                    # Store time relative to contact (offset from event timestamp)
                                    sim_traj["t"] = sim_traj["t"] - sim_traj["t"][0]
                                    event.simulated_trajectory = sim_traj

                            # Load additional simulated trajectories
                            for attr, grp_suffix in [
                                (
                                    "simulated_trajectory_nakashima_refined",
                                    "simulated_trajectories_nakashima_refined",
                                ),
                                (
                                    "simulated_trajectory_cpp_no_residual_refined",
                                    "simulated_trajectories_cpp_no_residual_refined",
                                ),
                                (
                                    "simulated_trajectory_latest",
                                    "simulated_trajectories_latest",
                                ),
                            ]:
                                traj_path = f"{grp_path}/{grp_suffix}/{best_idx}"
                                if traj_path in f:
                                    sim_grp = f[traj_path]
                                    _TRAJ_FIELDS = (
                                        "t",
                                        "x",
                                        "y",
                                        "z",
                                        "vx",
                                        "vy",
                                        "vz",
                                        "wx",
                                        "wy",
                                        "wz",
                                    )
                                    traj = {}
                                    for tf in _TRAJ_FIELDS:
                                        if tf in sim_grp:
                                            traj[tf] = np.array(
                                                sim_grp[tf], dtype=np.float64
                                            )
                                    if "t" in traj and len(traj["t"]) >= 2:
                                        traj["t"] = traj["t"] - traj["t"][0]
                                        setattr(event, attr, traj)

                            enriched += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(
                    f"  Warning: Failed to load racket contacts from HDF5 {h5_path}: {e}"
                )
            finally:
                f.close()

        if enriched:
            print(f"[HDF5] Enriched {enriched} racket contact events from HDF5")

    def _load_table_contacts_from_hdf5(self, _files_h5: List[str]) -> None:
        """Enrich :class:`TableContactEvent` objects from ``ground_truth_200/table_contacts`` in HDF5.

        This is the primary enrichment path.  Events already enriched here
        will be skipped by the CSV fallback.
        """
        enriched = 0
        for match in self.matches:
            h5_path = str(match.file)
            try:
                with h5py.File(h5_path, "r") as f:
                    for game in match.games:
                        for rally in game.rallies:
                            grp_path = f"game_{game.game_id}/rally_{rally.rally_id}/ground_truth_200/table_contacts"
                            if grp_path not in f:
                                continue
                            tc_grp = f[grp_path]

                            try:
                                timestamps = np.array(
                                    tc_grp["event_timestamp"], dtype=np.float64
                                )
                            except KeyError:
                                continue
                            n = len(timestamps)
                            if n == 0:
                                continue

                            # pylint: disable=cell-var-from-loop
                            def _get_float_array(name):
                                if name in tc_grp:
                                    return np.array(tc_grp[name], dtype=np.float64)
                                return np.full(n, np.nan)
                            # pylint: enable=cell-var-from-loop

                            # Pre-read all float fields
                            field_data = {}
                            for key in tc_grp.keys():
                                if key == "contact_type":
                                    continue
                                try:
                                    field_data[key] = _get_float_array(key)
                                except Exception:  # pylint: disable=broad-exception-caught
                                    pass

                            for event in rally.events:
                                if not isinstance(event, TableContactEvent):
                                    continue
                                if event.ball_pre is not None and not np.isnan(
                                    event.ball_pre.vx
                                ):
                                    continue  # already enriched
                                diffs = np.abs(timestamps - event.timestamp)
                                best_idx = np.argmin(diffs)
                                if diffs[best_idx] > 0.001:
                                    continue
                                row_dict = {
                                    key: arr[best_idx]
                                    for key, arr in field_data.items()
                                }
                                row_series = pd.Series(row_dict)
                                MatchCollection._enrich_table_event(event, row_series)
                                enriched += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(
                    f"  Warning: Failed to load table contacts from HDF5 {h5_path}: {e}"
                )

        if enriched:
            print(f"[HDF5] Enriched {enriched} table contact events from HDF5")

    def _load_extracted_csv(self, files_h5: List[str]) -> None:
        """Load per-match ``extracted.csv`` files and enrich :class:`RacketContactEvent` objects.

        Each HDF5 file may have a sibling ``extracted.csv`` in the same
        directory.  A CSV is used only when its modification time is **newer**
        than the companion HDF5 file.

        Each CSV row is matched to a :class:`RacketContactEvent` via the
        composite key ``(sequence_name, event_timestamp)``.

        Events already enriched from HDF5 are skipped.
        """
        # Collect all per-match CSVs.
        # The CSV may live either next to the H5 file *or* at the top-level
        # base_folder (where extract_racket_contacts.py writes it).
        csv_lookup: dict = {}
        csv_files_loaded = 0
        csv_rows_total = 0

        _already_loaded_csvs: set = set()  # avoid loading the same CSV twice

        for h5_path_str in files_h5:
            h5_path = pathlib.Path(h5_path_str)
            # Try the sibling directory first, then fall back to base_folder
            csv_path = h5_path.parent / "extracted.csv"
            if not csv_path.exists() and self.base_folder is not None:
                csv_path = pathlib.Path(self.base_folder) / "extracted.csv"
            if not csv_path.exists():
                print(
                    f"[RCM-diag] No extracted.csv found for {h5_path.name} "
                    f"(checked {h5_path.parent} and {self.base_folder})"
                )
                continue

            # De-duplicate: if multiple H5s share the same CSV, load it once
            csv_path_resolved = csv_path.resolve()
            if csv_path_resolved in _already_loaded_csvs:
                continue

            csv_mtime = os.path.getmtime(csv_path)
            h5_mtime = os.path.getmtime(h5_path)
            if csv_mtime < h5_mtime:
                # Warn but still load — the CSV may have been generated before
                # the H5 was last touched (e.g. by rsync/cp).
                print(
                    f"[RCM-diag] WARNING: {csv_path.name} is older than {h5_path.name} "
                    f"(csv mtime={csv_mtime:.0f}, h5 mtime={h5_mtime:.0f}) — loading anyway"
                )

            try:
                csv_data = pd.read_csv(csv_path)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Warning: Could not read {csv_path}: {e}")
                continue

            if csv_data.empty:
                continue

            _already_loaded_csvs.add(csv_path_resolved)
            csv_files_loaded += 1
            csv_rows_total += len(csv_data)

            for _, row in csv_data.iterrows():  # pylint: disable=no-member
                key = (
                    str(row.get("sequence_name", "")),
                    float(row.get("event_timestamp", 0.0)),
                )
                csv_lookup[key] = row

        print(
            f"[RCM-diag] _load_extracted_csv: loaded {csv_files_loaded} CSV files, "
            f"{csv_rows_total} rows, {len(csv_lookup)} unique keys"
        )

        if not csv_lookup:
            return

        enriched = 0
        for match in self.matches:
            match_parent_name = match.file.parent.name
            for game in match.games:
                for rally in game.rallies:
                    seq_name = f"{match_parent_name}_game_{game.game_id}_rally_{rally.rally_id}"
                    for event in rally.events:
                        if not isinstance(event, RacketContactEvent):
                            continue
                        # Skip events already enriched from HDF5
                        if event.ball_pre is not None and not np.isnan(
                            event.ball_pre.vx
                        ):
                            continue
                        key = (seq_name, event.timestamp)
                        csv_row = csv_lookup.get(key)
                        if csv_row is None:
                            # Try with small tolerance for float matching
                            for csv_key, csv_row_candidate in csv_lookup.items():
                                if (
                                    csv_key[0] == seq_name
                                    and abs(csv_key[1] - event.timestamp) < 0.0005
                                ):
                                    csv_row = csv_row_candidate
                                    break
                        if csv_row is not None:
                            MatchCollection._enrich_racket_event(event, csv_row)
                            enriched += 1

        print(
            f"[RCM-diag] Enriched {enriched} racket contact events "
            f"from {csv_files_loaded} extracted.csv files "
            f"({csv_rows_total} CSV rows, {len(csv_lookup)} unique keys)"
        )

    @staticmethod
    def _enrich_racket_event(event: "RacketContactEvent", row: pd.Series) -> None:
        """Populate a :class:`RacketContactEvent` from a single CSV row."""

        def _float(col: str) -> float:
            try:
                v = float(row[col])
                return v
            except (KeyError, TypeError, ValueError):
                return float("nan")

        def _vec3(cx: str, cy: str, cz: str) -> Optional[Vec3]:
            x, y, z = _float(cx), _float(cy), _float(cz)
            if np.isnan(x) and np.isnan(y) and np.isnan(z):
                return None
            return Vec3(x, y, z)

        def _quat(cx: str, cy: str, cz: str, cw: str) -> Optional[Quat]:
            x, y, z, w = _float(cx), _float(cy), _float(cz), _float(cw)
            if np.isnan(x) and np.isnan(y) and np.isnan(z) and np.isnan(w):
                return None
            return Quat(x, y, z, w)

        # ── Ball states ──────────────────────────────────────────────
        event.ball_pre = BallState(
            x=_float("x_pre"),
            y=_float("y_pre"),
            z=_float("z_pre"),
            vx=_float("vx_pre"),
            vy=_float("vy_pre"),
            vz=_float("vz_pre"),
            wx=_float("wx_pre"),
            wy=_float("wy_pre"),
            wz=_float("wz_pre"),
        )
        event.ball_post = BallState(
            x=_float("x_pre"),
            y=_float("y_pre"),
            z=_float("z_pre"),  # post position ≈ pre position (contact point)
            vx=_float("vx_post"),
            vy=_float("vy_post"),
            vz=_float("vz_post"),
            wx=_float("wx_post"),
            wy=_float("wy_post"),
            wz=_float("wz_post"),
        )

        # ── Racket state (FK snapshot) ───────────────────────────────
        event.racket_pos = _vec3("x_racket", "y_racket", "z_racket")
        event.racket_quat = _quat("qx_racket", "qy_racket", "qz_racket", "qw_racket")
        event.racket_vel = _vec3("vx_racket", "vy_racket", "vz_racket")
        event.racket_ang_vel = _vec3("wx_racket", "wy_racket", "wz_racket")

        # ── Refined racket state ─────────────────────────────────────
        event.racket_pos_pre_refined = _vec3(
            "x_racket_pre_refined", "y_racket_pre_refined", "z_racket_pre_refined"
        )
        event.racket_pos_post_refined = _vec3(
            "x_racket_post_refined", "y_racket_post_refined", "z_racket_post_refined"
        )
        event.racket_quat_pre_refined = _quat(
            "qx_racket_pre_refined",
            "qy_racket_pre_refined",
            "qz_racket_pre_refined",
            "qw_racket_pre_refined",
        )
        event.racket_quat_post_refined = _quat(
            "qx_racket_post_refined",
            "qy_racket_post_refined",
            "qz_racket_post_refined",
            "qw_racket_post_refined",
        )
        event.racket_vel_pre_refined = _vec3(
            "vx_racket_w_angvel_pre", "vy_racket_w_angvel_pre", "vz_racket_w_angvel_pre"
        )
        event.racket_vel_post_refined = _vec3(
            "vx_racket_w_angvel_post",
            "vy_racket_w_angvel_post",
            "vz_racket_w_angvel_post",
        )

        # ── Refined ball position ────────────────────────────────────
        event.ball_pos_pre_refined = _vec3(
            "x_pre_refined", "y_pre_refined", "z_pre_refined"
        )
        event.ball_pos_post_refined = _vec3(
            "x_post_refined", "y_post_refined", "z_post_refined"
        )

        # ── Model predictions ────────────────────────────────────────
        # Maps: attr_name -> (dataclass_field, column_template with {comp} placeholder)
        _model_columns = [
            ("ball_post_default", "{comp}_post_Nakashima_default"),
            ("ball_post_exp", "{comp}_post_Nakashima_exp"),
            ("ball_post_piecewise", "{comp}_post_Nakashima_piecewise"),
            ("ball_post_cpp", "{comp}_post_Nakashima_cpp"),
            ("ball_post_cpp_refined", "{comp}_post_Nakashima_cpp_refined"),
            ("ball_post_parametric_7p", "{comp}_post_Parametric_7p"),
            ("ball_post_rcm_tangential", "{comp}_post_RCM_tangential"),
            ("ball_post_cpp_tangential", "{comp}_post_cpp_tangential"),
            ("ball_post_nakashima_refined", "{comp}_post_Nakashima_refined"),
            ("ball_post_rcm_tangential_refined", "{comp}_post_RCM_tangential_refined"),
            ("ball_post_rcm_tangential_polyfit", "{comp}_post_RCM_tangential_polyfit"),
            ("ball_post_onnx_rcm", "{comp}_post_onnx_rcm"),
            ("ball_post_onnx_alex", "{comp}_post_onnx_alex"),
            ("ball_post_onnx_alex_refined", "{comp}_post_onnx_alex_refined"),
            ("ball_post_onnx_alex_polyfit", "{comp}_post_onnx_alex_polyfit"),
            ("ball_post_onnx_0426", "{comp}_post_onnx_0426"),
        ]
        for attr, col_tpl in _model_columns:
            vx_col = col_tpl.format(comp="vx")
            if vx_col in row.index:
                setattr(
                    event,
                    attr,
                    BallState(
                        x=_float("x_pre"),
                        y=_float("y_pre"),
                        z=_float("z_pre"),
                        vx=_float(col_tpl.format(comp="vx")),
                        vy=_float(col_tpl.format(comp="vy")),
                        vz=_float(col_tpl.format(comp="vz")),
                        wx=_float(col_tpl.format(comp="wx")),
                        wy=_float(col_tpl.format(comp="wy")),
                        wz=_float(col_tpl.format(comp="wz")),
                    ),
                )

        # ── Quality metrics ──────────────────────────────────────────
        event.fitness_pre = _float("fitness_pre")
        event.fitness_post = _float("fitness_post")
        event.dt_racket = _float("dt_racket")
        event.pre_duration = _float("pre_duration")
        event.post_duration = _float("post_duration")

        # ── Contact time offsets ─────────────────────────────────────
        event.t_contact_pre_offset = _float("t_contact_pre_offset")
        event.t_contact_post_offset = _float("t_contact_post_offset")

        # ── Position deltas ──────────────────────────────────────────
        event.delta_pre = _vec3("dx_pre", "dy_pre", "dz_pre")
        event.delta_post = _vec3("dx_post", "dy_post", "dz_post")

    # ── TCM CSV enrichment ───────────────────────────────────────────

    def _load_extracted_tcm_csv(self, files_h5: List[str]) -> None:
        """Load per-match ``extracted_TCM.csv`` and enrich :class:`TableContactEvent` objects.

        Each HDF5 file may have a sibling ``extracted_TCM.csv`` in the same
        directory (or at the top-level ``base_folder``).  Each CSV row is
        matched to a :class:`TableContactEvent` via the composite key
        ``(sequence_name, event_timestamp)``.
        """
        csv_lookup: dict = {}
        csv_files_loaded = 0
        csv_rows_total = 0
        _already_loaded_csvs: set = set()

        for h5_path_str in files_h5:
            h5_path = pathlib.Path(h5_path_str)
            csv_path = h5_path.parent / "extracted_TCM.csv"
            if not csv_path.exists() and self.base_folder is not None:
                csv_path = pathlib.Path(self.base_folder) / "extracted_TCM.csv"
            if not csv_path.exists():
                continue

            csv_path_resolved = csv_path.resolve()
            if csv_path_resolved in _already_loaded_csvs:
                continue

            try:
                csv_data = pd.read_csv(csv_path)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Warning: Could not read {csv_path}: {e}")
                continue

            if csv_data.empty:
                continue

            _already_loaded_csvs.add(csv_path_resolved)
            csv_files_loaded += 1
            csv_rows_total += len(csv_data)

            for _, row in csv_data.iterrows():  # pylint: disable=no-member
                key = (
                    str(row.get("sequence_name", "")),
                    float(row.get("event_timestamp", 0.0)),
                )
                csv_lookup[key] = row

        print(
            f"[TCM-diag] _load_extracted_tcm_csv: loaded {csv_files_loaded} CSV files, "
            f"{csv_rows_total} rows, {len(csv_lookup)} unique keys"
        )

        if not csv_lookup:
            return

        enriched = 0
        for match in self.matches:
            match_parent_name = match.file.parent.name
            for game in match.games:
                for rally in game.rallies:
                    seq_name = f"{match_parent_name}_game_{game.game_id}_rally_{rally.rally_id}"
                    for event in rally.events:
                        if not isinstance(event, TableContactEvent):
                            continue
                        # Skip events already enriched from HDF5
                        if event.ball_pre is not None and not np.isnan(
                            event.ball_pre.vx
                        ):
                            continue
                        key = (seq_name, event.timestamp)
                        csv_row = csv_lookup.get(key)
                        if csv_row is None:
                            # Try with small tolerance for float matching
                            for csv_key, csv_row_candidate in csv_lookup.items():
                                if (
                                    csv_key[0] == seq_name
                                    and abs(csv_key[1] - event.timestamp) < 0.0005
                                ):
                                    csv_row = csv_row_candidate
                                    break
                        if csv_row is not None:
                            MatchCollection._enrich_table_event(event, csv_row)
                            enriched += 1

        print(
            f"[TCM-diag] Enriched {enriched} table contact events "
            f"from {csv_files_loaded} extracted_TCM.csv files "
            f"({csv_rows_total} CSV rows, {len(csv_lookup)} unique keys)"
        )

    @staticmethod
    def _enrich_table_event(event: "TableContactEvent", row: pd.Series) -> None:
        """Populate a :class:`TableContactEvent` from a single ``extracted_TCM.csv`` row."""

        def _float(col: str) -> float:
            try:
                v = float(row[col])
                return v
            except (KeyError, TypeError, ValueError):
                return float("nan")

        event.ball_pre = BallState(
            x=float("nan"),
            y=float("nan"),
            z=float("nan"),  # position not in CSV
            vx=_float("vx_pre"),
            vy=_float("vy_pre"),
            vz=_float("vz_pre"),
            wx=_float("wx_pre"),
            wy=_float("wy_pre"),
            wz=_float("wz_pre"),
        )
        event.ball_post = BallState(
            x=float("nan"),
            y=float("nan"),
            z=float("nan"),
            vx=_float("vx_post"),
            vy=_float("vy_post"),
            vz=_float("vz_post"),
            wx=_float("wx_post"),
            wy=_float("wy_post"),
            wz=_float("wz_post"),
        )
        event.epsilon_opt = _float("epsilon_opt")

        # Model predictions (from HDF5 table_contacts group)
        if not np.isnan(_float("vx_post_NakashimaITTF")):
            event.ball_post_nakashima_ittf = BallState(
                x=float("nan"),
                y=float("nan"),
                z=float("nan"),
                vx=_float("vx_post_NakashimaITTF"),
                vy=_float("vy_post_NakashimaITTF"),
                vz=_float("vz_post_NakashimaITTF"),
                wx=_float("wx_post_NakashimaITTF"),
                wy=_float("wy_post_NakashimaITTF"),
                wz=_float("wz_post_NakashimaITTF"),
            )
        if not np.isnan(_float("vx_post_NakashimaPaper")):
            event.ball_post_nakashima_paper = BallState(
                x=float("nan"),
                y=float("nan"),
                z=float("nan"),
                vx=_float("vx_post_NakashimaPaper"),
                vy=_float("vy_post_NakashimaPaper"),
                vz=_float("vz_post_NakashimaPaper"),
                wx=_float("wx_post_NakashimaPaper"),
                wy=_float("wy_post_NakashimaPaper"),
                wz=_float("wz_post_NakashimaPaper"),
            )
        if not np.isnan(_float("vx_post_ResidualCorrectedNakashima")):
            event.ball_post_residual_corrected = BallState(
                x=float("nan"),
                y=float("nan"),
                z=float("nan"),
                vx=_float("vx_post_ResidualCorrectedNakashima"),
                vy=_float("vy_post_ResidualCorrectedNakashima"),
                vz=_float("vz_post_ResidualCorrectedNakashima"),
                wx=_float("wx_post_ResidualCorrectedNakashima"),
                wy=_float("wy_post_ResidualCorrectedNakashima"),
                wz=_float("wz_post_ResidualCorrectedNakashima"),
            )
        if not np.isnan(_float("vx_post_Residual0805")):
            event.ball_post_residual0805 = BallState(
                x=float("nan"),
                y=float("nan"),
                z=float("nan"),
                vx=_float("vx_post_Residual0805"),
                vy=_float("vy_post_Residual0805"),
                vz=_float("vz_post_Residual0805"),
                wx=_float("wx_post_Residual0805"),
                wy=_float("wy_post_Residual0805"),
                wz=_float("wz_post_Residual0805"),
            )
        if not np.isnan(_float("vx_post_PySR")):
            event.ball_post_pysr = BallState(
                x=float("nan"),
                y=float("nan"),
                z=float("nan"),
                vx=_float("vx_post_PySR"),
                vy=_float("vy_post_PySR"),
                vz=_float("vz_post_PySR"),
                wx=_float("wx_post_PySR"),
                wy=_float("wy_post_PySR"),
                wz=_float("wz_post_PySR"),
            )
        if not np.isnan(_float("vx_post_0426")):
            event.ball_post_0426 = BallState(
                x=float("nan"),
                y=float("nan"),
                z=float("nan"),
                vx=_float("vx_post_0426"),
                vy=_float("vy_post_0426"),
                vz=_float("vz_post_0426"),
                wx=_float("wx_post_0426"),
                wy=_float("wy_post_0426"),
                wz=_float("wz_post_0426"),
            )

    # ── Derived quantities ───────────────────────────────────────────

    def _compute_derived_quantities(self) -> None:
        """Fix racket orientations and compute derived quantities in-place.

        Iterates every :class:`RacketContactEvent` that has been enriched
        from ``extracted.csv`` and:
        1. Corrects the racket quaternion so the normal faces the incoming
           ball (overwrites the original quaternion fields).
        2. Projects ball velocity / spin / position and racket velocity /
           spin into the racket coordinate frame.
        3. Computes kinetic energy, theta angle, and racket-open angle.

        Called automatically at the end of :meth:`__init__`.
        """
        computed = 0
        for match in self.matches:
            for game in match.games:
                for rally in game.rallies:
                    for event in rally.events:
                        if not isinstance(event, RacketContactEvent):
                            continue
                        if event.ball_pre is None or event.racket_quat is None:
                            continue  # not enriched
                        self._compute_derived_for_event(event)
                        computed += 1
        if computed:
            print(f"Computed derived quantities for {computed} racket contact events")

    @staticmethod
    def _compute_derived_for_event(ev: "RacketContactEvent") -> None:
        """Compute derived quantities *in-place* for one enriched event."""

        # ── helpers to extract numpy arrays ──────────────────────────
        def _v3(v: Optional[Vec3]) -> Optional[np.ndarray]:
            return np.array([v.x, v.y, v.z]) if v is not None else None

        def _q4(q: Optional[Quat]) -> Optional[np.ndarray]:
            return np.array([q.x, q.y, q.z, q.w]) if q is not None else None

        # ── unpack essential arrays ──────────────────────────────────
        b_vel_pre = np.array([ev.ball_pre.vx, ev.ball_pre.vy, ev.ball_pre.vz])
        b_spin_pre = np.array([ev.ball_pre.wx, ev.ball_pre.wy, ev.ball_pre.wz])
        b_pos_pre = np.array([ev.ball_pre.x, ev.ball_pre.y, ev.ball_pre.z])

        b_vel_post = (
            np.array([ev.ball_post.vx, ev.ball_post.vy, ev.ball_post.vz])
            if ev.ball_post
            else None
        )
        b_spin_post = (
            np.array([ev.ball_post.wx, ev.ball_post.wy, ev.ball_post.wz])
            if ev.ball_post
            else None
        )

        r_vel = _v3(ev.racket_vel)
        r_quat = _q4(ev.racket_quat)
        r_pos = _v3(ev.racket_pos)
        r_ang_vel = _v3(ev.racket_ang_vel)

        if r_vel is None or r_quat is None:
            return  # cannot compute anything without racket state

        # ── 1. Fix racket orientations (overwrite) ───────────────────
        r_quat = _fix_quat_orientation(b_vel_pre, r_vel, r_quat)
        ev.racket_quat = Quat(*r_quat)

        r_vel_pre_ref = _v3(ev.racket_vel_pre_refined)
        r_quat_pre_ref = _q4(ev.racket_quat_pre_refined)
        if r_vel_pre_ref is not None and r_quat_pre_ref is not None:
            r_quat_pre_ref = _fix_quat_orientation(
                b_vel_pre, r_vel_pre_ref, r_quat_pre_ref
            )
            ev.racket_quat_pre_refined = Quat(*r_quat_pre_ref)

        r_vel_post_ref = _v3(ev.racket_vel_post_refined)
        r_quat_post_ref = _q4(ev.racket_quat_post_refined)
        if r_vel_post_ref is not None and r_quat_post_ref is not None:
            r_quat_post_ref = _fix_quat_orientation(
                b_vel_pre, r_vel_post_ref, r_quat_post_ref
            )
            ev.racket_quat_post_refined = Quat(*r_quat_post_ref)

        # ── 2. Standard racket-frame projections ─────────────────────
        v_ref = b_vel_pre - r_vel  # reference direction for alignment

        ev.ball_vel_pre_racket = Vec3(
            *_project_to_racket(v_ref, b_vel_pre - r_vel, r_quat)
        )
        ev.ball_spin_pre_racket = Vec3(*_project_to_racket(v_ref, b_spin_pre, r_quat))
        if b_vel_post is not None:
            ev.ball_vel_post_racket = Vec3(
                *_project_to_racket(v_ref, b_vel_post - r_vel, r_quat)
            )
        if b_spin_post is not None:
            ev.ball_spin_post_racket = Vec3(
                *_project_to_racket(v_ref, b_spin_post, r_quat)
            )

        # ── 3. Refined racket-frame projections ──────────────────────
        if r_quat_pre_ref is not None and r_vel_pre_ref is not None:
            v_ref_ref = b_vel_pre - r_vel_pre_ref
            ev.ball_vel_pre_refined_racket = Vec3(
                *_project_to_racket(
                    v_ref_ref, b_vel_pre - r_vel_pre_ref, r_quat_pre_ref
                )
            )
            ev.ball_spin_pre_refined_racket = Vec3(
                *_project_to_racket(v_ref_ref, b_spin_pre, r_quat_pre_ref)
            )
            if b_vel_post is not None:
                ev.ball_vel_post_refined_racket = Vec3(
                    *_project_to_racket(
                        v_ref_ref, b_vel_post - r_vel_pre_ref, r_quat_pre_ref
                    )
                )
            if b_spin_post is not None:
                ev.ball_spin_post_refined_racket = Vec3(
                    *_project_to_racket(v_ref_ref, b_spin_post, r_quat_pre_ref)
                )

        # ── 3b. Model predictions projected into racket frame ────────
        for _m_attr, _m_vel_attr, _m_spin_attr in [
            (
                "ball_post_default",
                "ball_vel_post_default_racket",
                "ball_spin_post_default_racket",
            ),
            ("ball_post_exp", "ball_vel_post_exp_racket", "ball_spin_post_exp_racket"),
            (
                "ball_post_piecewise",
                "ball_vel_post_piecewise_racket",
                "ball_spin_post_piecewise_racket",
            ),
            ("ball_post_cpp", "ball_vel_post_cpp_racket", "ball_spin_post_cpp_racket"),
            (
                "ball_post_cpp_refined",
                "ball_vel_post_cpp_refined_racket",
                "ball_spin_post_cpp_refined_racket",
            ),
            (
                "ball_post_parametric_7p",
                "ball_vel_post_parametric_7p_racket",
                "ball_spin_post_parametric_7p_racket",
            ),
            (
                "ball_post_rcm_tangential",
                "ball_vel_post_rcm_tangential_racket",
                "ball_spin_post_rcm_tangential_racket",
            ),
            (
                "ball_post_cpp_tangential",
                "ball_vel_post_cpp_tangential_racket",
                "ball_spin_post_cpp_tangential_racket",
            ),
            (
                "ball_post_nakashima_refined",
                "ball_vel_post_nakashima_refined_racket",
                "ball_spin_post_nakashima_refined_racket",
            ),
            (
                "ball_post_rcm_tangential_refined",
                "ball_vel_post_rcm_tangential_refined_racket",
                "ball_spin_post_rcm_tangential_refined_racket",
            ),
            (
                "ball_post_rcm_tangential_polyfit",
                "ball_vel_post_rcm_tangential_polyfit_racket",
                "ball_spin_post_rcm_tangential_polyfit_racket",
            ),
            (
                "ball_post_onnx_rcm",
                "ball_vel_post_onnx_rcm_racket",
                "ball_spin_post_onnx_rcm_racket",
            ),
            (
                "ball_post_onnx_alex",
                "ball_vel_post_onnx_alex_racket",
                "ball_spin_post_onnx_alex_racket",
            ),
            (
                "ball_post_onnx_alex_refined",
                "ball_vel_post_onnx_alex_refined_racket",
                "ball_spin_post_onnx_alex_refined_racket",
            ),
            (
                "ball_post_onnx_alex_polyfit",
                "ball_vel_post_onnx_alex_polyfit_racket",
                "ball_spin_post_onnx_alex_polyfit_racket",
            ),
            (
                "ball_post_onnx_0426",
                "ball_vel_post_onnx_0426_racket",
                "ball_spin_post_onnx_0426_racket",
            ),
        ]:
            _m_bs = getattr(ev, _m_attr, None)
            if _m_bs is not None and not (
                np.isnan(_m_bs.vx) and np.isnan(_m_bs.vy) and np.isnan(_m_bs.vz)
            ):
                _m_vel = np.array([_m_bs.vx, _m_bs.vy, _m_bs.vz])
                _m_spin = np.array([_m_bs.wx, _m_bs.wy, _m_bs.wz])
                setattr(
                    ev,
                    _m_vel_attr,
                    Vec3(*_project_to_racket(v_ref, _m_vel - r_vel, r_quat)),
                )
                setattr(
                    ev, _m_spin_attr, Vec3(*_project_to_racket(v_ref, _m_spin, r_quat))
                )

        # ── 4. Racket velocity/spin projected into racket frame ──────
        ev.racket_vel_racket = Vec3(*_project_to_racket(v_ref, r_vel, r_quat))
        if r_ang_vel is not None:
            ev.racket_spin_racket = Vec3(*_project_to_racket(v_ref, r_ang_vel, r_quat))
            # Body-frame angular velocity: pure R^T, no v_ref y-z alignment
            ev.racket_spin_body = Vec3(
                *(_R.from_quat(r_quat).inv().as_matrix() @ r_ang_vel)
            )

        # ── 5. Relative velocity in racket frame ─────────────────────
        if ev.ball_vel_pre_racket is not None and ev.racket_vel_racket is not None:
            ev.relative_vel_racket = Vec3(
                ev.ball_vel_pre_racket.x - ev.racket_vel_racket.x,
                ev.ball_vel_pre_racket.y - ev.racket_vel_racket.y,
                ev.ball_vel_pre_racket.z - ev.racket_vel_racket.z,
            )

        # ── 6. Ball position relative to racket in racket frame ──────
        if r_pos is not None:
            ev.ball_pos_pre_racket = Vec3(
                *_project_to_racket(v_ref, b_pos_pre - r_pos, r_quat)
            )

        # ── 7. Kinetic energy ────────────────────────────────────────
        if b_vel_post is not None and b_spin_post is not None:
            ke = _compute_kinetic_energy(b_vel_pre, b_vel_post, b_spin_pre, b_spin_post)
            ev.ke_lin_pre = float(ke[0])
            ev.ke_lin_post = float(ke[1])
            ev.ke_ang_pre = float(ke[2])
            ev.ke_ang_post = float(ke[3])

        # ── 8. Angles ────────────────────────────────────────────────
        ev.theta_angle = _compute_theta_angle(b_vel_pre, r_vel, r_quat)
        ev.racket_open_angle = _compute_racket_angle(r_quat)

    @staticmethod
    def _load_match_static(file_path, match_id: int, gcs_fallback: bool = False):
        """Static method for loading a single match file (used for parallel processing)"""
        print(file_path)
        try:
            with h5py.File(file_path, "r") as f:
                games = [
                    MatchCollection._load_game_static(
                        f[f"game_{i}"], i, file_path, gcs_fallback
                    )
                    for i in range(len(f.items()))
                    if f"game_{i}" in f
                ]
                games = [game for game in games if len(game.rallies) > 0]
                path = pathlib.Path(file_path)
                parent_dirs = path.parent.parts

                date, player = parent_dirs[-2], parent_dirs[-1]

                # Extract court from directory path
                court = None
                for part in parent_dirs:
                    lp = part.lower()
                    if "court" in lp or lp.startswith(("tyo", "osa", "cmax")):
                        court = part
                        break

                # Get file modification time
                import time as _time

                try:
                    last_modified = _time.ctime(pathlib.Path(file_path).stat().st_mtime)
                except OSError:
                    last_modified = None

                # Safely extract player name from experiment/name attribute
                experiment_name = f.attrs.get("experiment/name", "")
                if experiment_name and "vs_" in experiment_name:
                    try:
                        player_from_attr = experiment_name.split("vs_")[1].split(
                            "_match_"
                        )[0]
                    except IndexError:
                        print(
                            f"  Warning: Could not parse player name from experiment/name='{experiment_name}', using folder name '{player}'"
                        )
                        player_from_attr = player
                else:
                    if experiment_name:
                        print(
                            f"  Warning: experiment/name='{experiment_name}' does not contain 'vs_', using folder name '{player}'"
                        )
                    else:
                        print(
                            f"  Warning: experiment/name attribute is empty or missing, using folder name '{player}'"
                        )
                    player_from_attr = player

                # Read version_data_processing; decode bytes to str if needed
                raw_ver = f.attrs.get("file/version_data_processing", None)
                if isinstance(raw_ver, bytes):
                    raw_ver = raw_ver.decode("utf-8", errors="replace")  # pylint: disable=no-member
                version_dp = str(raw_ver) if raw_ver is not None else None

                return Match(
                    games=games,
                    date=f.attrs.get("experiment/date", date),
                    player=player_from_attr,
                    policy=f.attrs.get("experiment/name", player),
                    court=court,
                    last_modified=last_modified,
                    data_usage=f.attrs.get("file/data_usage", ""),
                    version_data_processing=version_dp,
                    match_id=match_id,
                    file=pathlib.Path(file_path),
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  ERROR loading match from {file_path}: {type(e).__name__}: {e}")
            raise

    def load_match(self, file_path, match_id: int):
        """load match"""
        return self._load_match_static(
            file_path, match_id, getattr(self, "gcs_fallback", False)
        )

    def load_game(self, group: h5py.Group, game_id: int, file_path: str):
        """load game"""
        return self._load_game_static(
            group, game_id, file_path, getattr(self, "gcs_fallback", False)
        )

    @staticmethod
    def _load_game_static(
        group: h5py.Group, game_id: int, file_path: str, gcs_fallback: bool = False
    ):
        """Static version of load_game for parallel processing"""
        rallies = [
            MatchCollection._load_rally_static(
                group[f"rally_{i}"], i, gcs_fallback, file_path=file_path
            )
            for i in range(len(group))
            if f"rally_{i}" in group
        ]
        path = pathlib.Path(file_path)
        parent_dirs = path.parent.parts
        experiment = parent_dirs[-1]
        return Game(
            rallies=rallies,
            game_id=game_id,
            game_name=experiment,
        )

    def load_rally(self, group: h5py.Group, rally_id: int, file_path: str = ""):
        """load rally"""
        return self._load_rally_static(
            group, rally_id, getattr(self, "gcs_fallback", False), file_path=file_path
        )

    @staticmethod
    def _load_rally_static(
        group: h5py.Group,
        rally_id: int,
        gcs_fallback: bool = False,
        file_path: str = "",
    ):
        """Static version of load_rally for parallel processing"""
        if "ground_truth_200" in list(group) and "racket1" in list(
            group["ground_truth_200"].keys()
        ):
            # Build base trajectory data
            trajectory_data = {
                "time": group["ground_truth_200/ball/time"][()],
                "x_gt200": group["ground_truth_200/ball/position"][()][:, 0],
                "y_gt200": group["ground_truth_200/ball/position"][()][:, 1],
                "z_gt200": group["ground_truth_200/ball/position"][()][:, 2],
                "vx_gt200": group["ground_truth_200/ball/linear_velocity"][()][:, 0],
                "vy_gt200": group["ground_truth_200/ball/linear_velocity"][()][:, 1],
                "vz_gt200": group["ground_truth_200/ball/linear_velocity"][()][:, 2],
                "wx": group["ground_truth_200/ball/angular_velocity"][()][:, 0],
                "wy": group["ground_truth_200/ball/angular_velocity"][()][:, 1],
                "wz": group["ground_truth_200/ball/angular_velocity"][()][:, 2],
            }

            # Add confidence data if it exists in the file
            if "confidences" in group["ground_truth_200/ball"].keys():
                trajectory_data.update(
                    {
                        "x_confidence": group["ground_truth_200/ball/confidences"][()][
                            :, 0
                        ],
                        "y_confidence": group["ground_truth_200/ball/confidences"][()][
                            :, 1
                        ],
                        "z_confidence": group["ground_truth_200/ball/confidences"][()][
                            :, 2
                        ],
                        "wx_confidence": group["ground_truth_200/ball/confidences"][()][
                            :, 10
                        ],
                        "wy_confidence": group["ground_truth_200/ball/confidences"][()][
                            :, 11
                        ],
                        "wz_confidence": group["ground_truth_200/ball/confidences"][()][
                            :, 12
                        ],
                    }
                )

            trajectory = pd.DataFrame(trajectory_data)

            # ==================== TEMPORARY CODE - REMOVE IN FUTURE ====================
            # Load ground_truth_curr data if available (at same level as ground_truth_200)
            if "ground_truth_curr" in list(group) and "ball" in list(
                group["ground_truth_curr"].keys()
            ):
                try:
                    gt_curr_group = group["ground_truth_curr/ball"]
                    gt_curr_data = {
                        "time_curr": gt_curr_group["time"][()],
                        "x_curr": gt_curr_group["position"][()][:, 0],
                        "y_curr": gt_curr_group["position"][()][:, 1],
                        "z_curr": gt_curr_group["position"][()][:, 2],
                        "vx_curr": gt_curr_group["linear_velocity"][()][:, 0],
                        "vy_curr": gt_curr_group["linear_velocity"][()][:, 1],
                        "vz_curr": gt_curr_group["linear_velocity"][()][:, 2],
                        "wx_curr": gt_curr_group["angular_velocity"][()][:, 0],
                        "wy_curr": gt_curr_group["angular_velocity"][()][:, 1],
                        "wz_curr": gt_curr_group["angular_velocity"][()][:, 2],
                    }
                    # Add CURR confidences if available
                    if "confidences" in gt_curr_group.keys():
                        gt_curr_data.update(
                            {
                                "x_confidence_curr": gt_curr_group["confidences"][()][
                                    :, 0
                                ],
                                "y_confidence_curr": gt_curr_group["confidences"][()][
                                    :, 1
                                ],
                                "z_confidence_curr": gt_curr_group["confidences"][()][
                                    :, 2
                                ],
                                "wx_confidence_curr": gt_curr_group["confidences"][()][
                                    :, 10
                                ],
                                "wy_confidence_curr": gt_curr_group["confidences"][()][
                                    :, 11
                                ],
                                "wz_confidence_curr": gt_curr_group["confidences"][()][
                                    :, 12
                                ],
                            }
                        )
                    gt_curr_df = pd.DataFrame(gt_curr_data)
                    gt_curr_df = gt_curr_df.rename(columns={"time_curr": "time"})
                    trajectory = pd.merge(
                        trajectory, gt_curr_df, on="time", how="outer"
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"    Warning: Failed to load ground_truth_curr data: {e}")
            # ==================== END TEMPORARY CODE ====================

            if "aerodynamics" in list(group["ground_truth_200"].keys()):
                aero_group = group["ground_truth_200/aerodynamics"]

                # Helper function to safely get dataset
                def safe_get(path, default=None):
                    try:
                        return aero_group[path][()]
                    except KeyError:
                        return default

                # Helper function to safely extract vector components
                def safe_extract_vector(data, component_idx):
                    """Extract a component from vector data, handling both 1D and 2D arrays"""
                    if data is None:
                        return None
                    try:
                        # Try 2D indexing first (normal case)
                        if data.ndim == 2:
                            return data[:, component_idx]
                        elif data.ndim == 1:
                            # If 1D, this is corrupted data - skip it
                            print(
                                f"    Warning: Corrupted aerodynamics data (1D instead of 2D) - skipping"
                            )
                            return None
                        else:
                            return None
                    except (IndexError, AttributeError):
                        return None

                # Build trajectory_aero dict, only including fields that exist
                aero_data = {"time": safe_get("t")}

                # Optimized data
                pos_opt = safe_get("pos_optimized")
                if pos_opt is not None and pos_opt.ndim == 2:
                    aero_data["x_opt"] = safe_extract_vector(pos_opt, 0)
                    aero_data["y_opt"] = safe_extract_vector(pos_opt, 1)
                    aero_data["z_opt"] = safe_extract_vector(pos_opt, 2)

                vel_opt = safe_get("vel_optimized")
                if vel_opt is not None and vel_opt.ndim == 2:
                    aero_data["vx_opt"] = safe_extract_vector(vel_opt, 0)
                    aero_data["vy_opt"] = safe_extract_vector(vel_opt, 1)
                    aero_data["vz_opt"] = safe_extract_vector(vel_opt, 2)

                aero_data["cd_opt"] = safe_get("cd_optimized")
                aero_data["cm_opt"] = safe_get("cm_optimized")
                aero_data["v_eff_drag_opt"] = safe_get("v_eff_drag_optimized")
                aero_data["v_eff_magnus_opt"] = safe_get("v_eff_magnus_optimized")
                aero_data["w_eff_perp_magnus_opt"] = safe_get(
                    "w_eff_perp_magnus_optimized"
                )

                # 0226 data
                pos_0226 = safe_get("pos_0226")
                if pos_0226 is not None and pos_0226.ndim == 2:
                    aero_data["x_0226"] = safe_extract_vector(pos_0226, 0)
                    aero_data["y_0226"] = safe_extract_vector(pos_0226, 1)
                    aero_data["z_0226"] = safe_extract_vector(pos_0226, 2)

                vel_0226 = safe_get("vel_0226")
                if vel_0226 is not None and vel_0226.ndim == 2:
                    aero_data["vx_0226"] = safe_extract_vector(vel_0226, 0)
                    aero_data["vy_0226"] = safe_extract_vector(vel_0226, 1)
                    aero_data["vz_0226"] = safe_extract_vector(vel_0226, 2)

                aero_data["cd_0226"] = safe_get("cd_0226")
                aero_data["cm_0226"] = safe_get("cm_0226")
                aero_data["v_eff_drag_0226"] = safe_get("v_eff_drag_0226")
                aero_data["v_eff_magnus_0226"] = safe_get("v_eff_magnus_0226")
                aero_data["w_eff_perp_magnus_0226"] = safe_get("w_eff_perp_magnus_0226")

                # Nakashima data
                pos_nakashima = safe_get("pos_nakashima")
                if pos_nakashima is not None and pos_nakashima.ndim == 2:
                    aero_data["x_nakashima"] = safe_extract_vector(pos_nakashima, 0)
                    aero_data["y_nakashima"] = safe_extract_vector(pos_nakashima, 1)
                    aero_data["z_nakashima"] = safe_extract_vector(pos_nakashima, 2)

                vel_nakashima = safe_get("vel_nakashima")
                if vel_nakashima is not None and vel_nakashima.ndim == 2:
                    aero_data["vx_nakashima"] = safe_extract_vector(vel_nakashima, 0)
                    aero_data["vy_nakashima"] = safe_extract_vector(vel_nakashima, 1)
                    aero_data["vz_nakashima"] = safe_extract_vector(vel_nakashima, 2)

                aero_data["cd_nakashima"] = safe_get("cd_nakashima")
                aero_data["cm_nakashima"] = safe_get("cm_nakashima")

                # 0426 data
                pos_0426 = safe_get("pos_0426")
                if pos_0426 is not None and pos_0426.ndim == 2:
                    aero_data["x_0426"] = safe_extract_vector(pos_0426, 0)
                    aero_data["y_0426"] = safe_extract_vector(pos_0426, 1)
                    aero_data["z_0426"] = safe_extract_vector(pos_0426, 2)

                vel_0426 = safe_get("vel_0426")
                if vel_0426 is not None and vel_0426.ndim == 2:
                    aero_data["vx_0426"] = safe_extract_vector(vel_0426, 0)
                    aero_data["vy_0426"] = safe_extract_vector(vel_0426, 1)
                    aero_data["vz_0426"] = safe_extract_vector(vel_0426, 2)

                # Spin data
                spin = safe_get("spin")
                if spin is not None and spin.ndim == 2:
                    aero_data["wx_opt"] = safe_extract_vector(spin, 0)
                    aero_data["wy_opt"] = safe_extract_vector(spin, 1)
                    aero_data["wz_opt"] = safe_extract_vector(spin, 2)

                # Remove None values and create DataFrame
                aero_data = {k: v for k, v in aero_data.items() if v is not None}
                trajectory_aero = pd.DataFrame(aero_data)
                trajectory = pd.merge(
                    trajectory, trajectory_aero, on="time", how="outer"
                )

            robot_racket = pd.DataFrame(
                {
                    "time": group["ground_truth_200/racket1/time"][()],
                    "robot_racket_x": group["ground_truth_200/racket1/position"][()][
                        :, 0
                    ],
                    "robot_racket_y": group["ground_truth_200/racket1/position"][()][
                        :, 1
                    ],
                    "robot_racket_z": group["ground_truth_200/racket1/position"][()][
                        :, 2
                    ],
                    "robot_racket_vx": group[
                        "ground_truth_200/racket1/linear_velocity"
                    ][()][:, 0],
                    "robot_racket_vy": group[
                        "ground_truth_200/racket1/linear_velocity"
                    ][()][:, 1],
                    "robot_racket_vz": group[
                        "ground_truth_200/racket1/linear_velocity"
                    ][()][:, 2],
                    "robot_racket_qx": group["ground_truth_200/racket1/orientation"][
                        ()
                    ][:, 0],
                    "robot_racket_qy": group["ground_truth_200/racket1/orientation"][
                        ()
                    ][:, 1],
                    "robot_racket_qz": group["ground_truth_200/racket1/orientation"][
                        ()
                    ][:, 2],
                    "robot_racket_qw": group["ground_truth_200/racket1/orientation"][
                        ()
                    ][:, 3],
                }
            )

            robot_racket["sequence_number"] = (
                np.round(robot_racket["time"] * 200).astype(int) * 5
            )
            trajectory = pd.merge(robot_racket, trajectory, on="time", how="outer")

            # Load spin data: prefer gcs_offline; optionally fall back to gcs_filtered / gcs
            spin = None
            gcs_source = None
            sensors = group["sensors"]
            if gcs_fallback:
                gcs_keys_to_try = ("gcs_offline", "gcs_filtered", "gcs")
            else:
                gcs_keys_to_try = ("gcs_offline",)
            for gcs_key in gcs_keys_to_try:
                if gcs_key in sensors:
                    try:
                        gcs_group = sensors[gcs_key]
                        spin = pd.DataFrame(
                            {
                                "sequence_number": gcs_group["sequence_number"][()],
                                "wx_gcs": gcs_group["spins"][()][:, 0],
                                "wy_gcs": gcs_group["spins"][()][:, 1],
                                "wz_gcs": gcs_group["spins"][()][:, 2],
                                "w_confidence": gcs_group["covariance"][()][:, 0],
                            }
                        )
                        spin["w_confidence"] = spin["w_confidence"].fillna(0.0)
                        gcs_source = gcs_key
                        break
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"    Warning: Failed to load spin from {gcs_key}: {e}")
                        spin = None
            if gcs_source is None:
                if not gcs_fallback and any(
                    k in sensors for k in ("gcs_filtered", "gcs")
                ):
                    available = [k for k in ("gcs_filtered", "gcs") if k in sensors]
                    YELLOW = "\033[93m"
                    RESET = "\033[0m"
                    file_hint = f" [{file_path}]" if file_path else ""
                    print(
                        f"    {YELLOW}⚠  Rally {rally_id}: gcs_offline not available "
                        f"(has {', '.join(available)}) — excluding rally. "
                        f"Use --gcs-fallback to include with fallback sources."
                        f"{file_hint}{RESET}"
                    )
                    return Rally(
                        shots=[],
                        events=[],
                        rally_id=rally_id,
                        rally=[],
                        gcs_source=f"excluded (has {', '.join(available)}, no gcs_offline)",
                    )
                YELLOW = "\033[93m"
                RESET = "\033[0m"
                file_hint = f" [{file_path}]" if file_path else ""
                print(
                    f"    {YELLOW}⚠  Rally {rally_id}: no gcs sensor found — spin columns will be NaN{file_hint}{RESET}"
                )

            trajectory_raw = get_ball_trajectory(
                group["sensors/aps_ball_triangulation"]
            )
            trajectory_raw = trajectory_raw.rename(
                columns={
                    "ball_position_x": "x_aps",
                    "ball_position_y": "y_aps",
                    "ball_position_z": "z_aps",
                }
            )

            if spin is not None:
                trajectory_raw = pd.merge(
                    trajectory_raw, spin, on="sequence_number", how="outer"
                )
            trajectory = pd.merge(
                trajectory, trajectory_raw, on="sequence_number", how="outer"
            )
            # Tag the gcs source so downstream code can display it
            trajectory.attrs["gcs_source"] = gcs_source if gcs_source else "none"
            trajectory["time"] = trajectory["sequence_number"] / 1000.0

            trajectory = trajectory.dropna(subset=["x_aps", "y_aps", "z_aps"])

            # add finite difference estimates for velocity
            trajectory["vx_aps"] = (
                trajectory["x_aps"].diff() / trajectory["time"].diff()
            )
            trajectory["vy_aps"] = (
                trajectory["y_aps"].diff() / trajectory["time"].diff()
            )
            trajectory["vz_aps"] = (
                trajectory["z_aps"].diff() / trajectory["time"].diff()
            )

            trajectory.drop(trajectory[trajectory["z_aps"] < -0.2].index, inplace=True)

            times = group["labels/timestamps"][()]
            types = [t.decode("utf-8") for t in group["labels/type"][()]]

            _EVENT_TYPE_MAP = {
                "shot_p1": RacketContactEvent,
                "shot_p2": RacketContactEvent,
                "bounce_p1": TableContactEvent,
                "bounce_p2": TableContactEvent,
                "net": NetEvent,
                "start": RallyStartEvent,
                "end": RallyEndEvent,
            }
            events = [
                _EVENT_TYPE_MAP.get(typ, Event)(timestamp=t, type=typ)
                for t, typ in zip(times, types)
            ]

            shots = MatchCollection.split_rally(events, trajectory)

            point_winner = MatchCollection._determine_point_winner(events, trajectory)

            # Extract log_identifier from sensor attributes
            log_identifier = None
            try:
                log_identifier = str(
                    group["sensors"]["aps_ball_triangulation"].attrs.get(
                        "log_identifier", ""
                    )
                )
            except (KeyError, AttributeError):
                pass

            return Rally(
                shots=shots,
                events=events,
                rally_id=rally_id,
                rally=trajectory,
                point_winner=point_winner,
                log_identifier=log_identifier,
                gcs_source=gcs_source if gcs_source else "none",
            )
        # ground_truth_200 or racket1 missing
        has_gt200 = "ground_truth_200" in list(group)
        has_racket1 = has_gt200 and "racket1" in list(group["ground_truth_200"].keys())
        if not has_gt200:
            reason = "no_ground_truth_200"
        elif not has_racket1:
            reason = "no_racket1_in_ground_truth_200"
        else:
            reason = "unknown"
        return Rally(
            shots=[], events=[], rally_id=rally_id, rally=[], gcs_source=reason
        )

    # Net height above the table surface (metres, ITTF standard).
    _NET_HEIGHT: float = 0.1525

    @staticmethod
    def _determine_point_winner(
        events: List[Event],
        trajectory: Optional["pd.DataFrame"] = None,
    ) -> Optional[str]:
        """Determine the point winner from the event sequence.

        Returns ``"player1"`` (robot) or ``"player2"`` (human), or
        ``None`` when the outcome is ambiguous or a serve let.

        Parameters
        ----------
        events : list[Event]
            Chronological event labels for the rally.
        trajectory : pd.DataFrame, optional
            Ball-tracking DataFrame (must contain ``time`` and ``z_aps``
            columns).  Used to resolve net contacts that occur right
            before the rally ends: if the ball's height at the net
            is **≥ net height** (0.1525 m) the ball is considered to
            cross → hitter wins; otherwise the ball is blocked → hitter
            loses.

        Rules (standard table tennis):
        - After a player hits the ball, it must bounce on the
          **opponent's** side.  If the rally ends before that bounce,
          the hitter loses (ball went off table or missed).
        - If the ball bounces on a player's side and that player
          fails to return it (rally ends), that player loses.
        - If the ball hits the net and the rally ends shortly after
          (within 20 ms), the ball height at the net determines the
          winner (above net → hitter wins, below → hitter loses).
          Falls back to ``None`` if trajectory data is unavailable.
        - On a **serve**, if the ball hits the net but still bounces
          on the opponent's side, it is a **let** (no point) → ``None``.
          A serve net fault (no bounce on opponent's side) → server loses.
        - During regular play a net cord that lands on the opponent's
          side is a valid shot (not a let).

        Implementation: walk backward from the ``end`` event and look
        at the last meaningful contact to decide.
        """
        contact_types = {"shot_p1", "shot_p2", "bounce_p1", "bounce_p2", "net"}
        contacts = [e for e in events if e.type in contact_types]
        end_events = [e for e in events if e.type == "end"]

        if not contacts or not end_events:
            return None

        # ── Serve-let detection ──────────────────────────────────────
        # A serve is the first shot_pX in the rally.  If the sequence
        # contains  shot_pX → bounce_pX → net → bounce_pY  (where Y≠X)
        # that is a let serve → no point awarded.
        racket_hits = [e for e in contacts if e.type in ("shot_p1", "shot_p2")]
        if racket_hits:
            first_shot = racket_hits[0]
            server = first_shot.type  # "shot_p1" or "shot_p2"
            server_bounce = "bounce_p1" if server == "shot_p1" else "bounce_p2"
            opponent_bounce = "bounce_p2" if server == "shot_p1" else "bounce_p1"

            # Check for the serve pattern:
            #   shot_pX → bounce_pX → net → bounce_pY
            # All four must be among the first few contacts.
            serve_contacts = contacts[:5]  # serve involves at most ~4 events
            serve_types = [e.type for e in serve_contacts]
            if (
                len(serve_types) >= 4
                and serve_types[0] == server
                and serve_types[1] == server_bounce
                and serve_types[2] == "net"
                and serve_types[3] == opponent_bounce
            ):
                return None  # let serve — no point

        # ── Normal point-winner logic ────────────────────────────────
        last_contact = contacts[-1]
        rally_end = end_events[-1]

        # Net hit right before end → resolve via ball height at contact
        if (
            last_contact.type == "net"
            and (rally_end.timestamp - last_contact.timestamp) < 0.020
        ):
            shots_before = [
                e
                for e in contacts
                if e.type in ("shot_p1", "shot_p2")
                and e.timestamp < last_contact.timestamp
            ]
            if not shots_before:
                return None
            last_hitter = shots_before[-1].type  # "shot_p1" or "shot_p2"
            # Try to look up the ball z from the trajectory
            if trajectory is not None and "z_aps" in trajectory.columns:
                idx = (trajectory["time"] - last_contact.timestamp).abs().idxmin()
                z_at_net = trajectory.loc[idx, "z_aps"]
                if np.isfinite(z_at_net):
                    if z_at_net >= MatchCollection._NET_HEIGHT:
                        # Ball above net → crosses → hitter wins
                        return "player1" if last_hitter == "shot_p1" else "player2"
                    else:
                        # Ball below net → blocked → hitter loses
                        return "player2" if last_hitter == "shot_p1" else "player1"
            return None  # no trajectory available — ambiguous

        # Net hit → ball didn't cross.  Find who hit it last.
        if last_contact.type == "net":
            # Walk back to find the last racket hit before the net
            shots_before_net = [
                e
                for e in contacts
                if e.type in ("shot_p1", "shot_p2")
                and e.timestamp < last_contact.timestamp
            ]
            if not shots_before_net:
                return None
            last_hitter = shots_before_net[-1].type  # "shot_p1" or "shot_p2"
            # The hitter loses (hit into the net)
            return "player2" if last_hitter == "shot_p1" else "player1"

        # Last contact is a bounce → the ball bounced on someone's side
        # and the rally ended (they failed to return it → they lose).
        if last_contact.type == "bounce_p1":
            return "player2"  # bounced on robot's side → human wins
        if last_contact.type == "bounce_p2":
            return "player1"  # bounced on human's side → robot wins

        # Last contact is a racket hit → the hitter sent the ball but
        # it never bounced on the opponent's side (off the table) → hitter loses.
        if last_contact.type == "shot_p1":
            return "player2"  # robot hit it off → human wins
        if last_contact.type == "shot_p2":
            return "player1"  # human hit it off → robot wins

        return None

    @staticmethod
    def split_rally(events: List[Event], trajectory: pd.DataFrame) -> List[Shot]:
        """split rally into shots"""
        shots = []
        racket_events = [
            e for e in events if e.type in ("start", "shot_p1", "shot_p2", "end")
        ]

        for i in range(len(racket_events) - 1):
            start_event = racket_events[i]
            end_event = racket_events[i + 1]

            player = 1 if start_event.type == "shot_p1" else 2
            mask = (trajectory["time"] >= start_event.timestamp) & (
                trajectory["time"] < end_event.timestamp
            )
            traj_slice = trajectory.loc[mask].copy()

            # Internal net or table events
            internal_events = [
                e
                for e in events
                if start_event.timestamp < e.timestamp < end_event.timestamp
                and e.type in ("bounce_p1", "bounce_p2", "net")
            ]

            # Sort by time and add dummy start/end markers
            boundaries = [start_event] + internal_events + [end_event]

            flight_segments = []
            for j in range(len(boundaries) - 1):
                e0, e1 = boundaries[j], boundaries[j + 1]
                seg_mask = (traj_slice["time"] >= e0.timestamp) & (
                    traj_slice["time"] < e1.timestamp
                )
                segment_df = traj_slice.loc[seg_mask].copy()
                if not segment_df.empty:
                    flight_segments.append(
                        FlightSegment(data=segment_df, trigger_event=e0)
                    )

            if flight_segments:
                is_serve = False
                if len(flight_segments) == 3:
                    is_serve = True
                shots.append(
                    Shot(
                        flight_segments=flight_segments,
                        is_serve=is_serve,
                        player=player,
                        start_time=start_event.timestamp,
                        end_time=end_event.timestamp,
                    )
                )

        return shots
