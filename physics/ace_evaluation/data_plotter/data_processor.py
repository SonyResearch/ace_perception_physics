# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Data Processing Module

Handles loading and processing of HDF5 match data files.
"""

import pathlib
import hashlib
import pickle
import time as _time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ace_evaluation.utilities.data_classes import MatchCollection, FlightSegment, Shot, Rally, RacketContactEvent, TableContactEvent, NetEvent

# Try to import numba for JIT compilation, fall back gracefully if not available
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create no-op decorators as fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ============================================================================
# Numba-accelerated computation functions
# ============================================================================

@njit(cache=True)
def compute_magnitudes_3d(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """Compute 3D vector magnitudes using Numba JIT."""
    n = len(vx)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
    return result


@njit(cache=True)
def filter_valid_velocities(vel_mag: np.ndarray, threshold: float = 1e-9) -> np.ndarray:
    """Return boolean mask for valid (non-zero) velocities."""
    n = len(vel_mag)
    result = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        result[i] = vel_mag[i] > threshold
    return result


@njit(cache=True)
def apply_nan_mask(values: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    """Replace NaN values with corresponding values from replacement array."""
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        if np.isnan(values[i]):
            result[i] = replacement[i]
        else:
            result[i] = values[i]
    return result


@njit(cache=True)
def build_enabled_mask(match_indices: np.ndarray, enabled_set_arr: np.ndarray) -> np.ndarray:
    """Build mask for enabled match indices (Numba-compatible)."""
    n = len(match_indices)
    m = len(enabled_set_arr)
    result = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        match_idx = match_indices[i]
        for j in range(m):
            if match_idx == enabled_set_arr[j]:
                result[i] = True
                break
    return result


# ============================================================================
# Fast mode helper – replaces expensive pd.Series.mode() double-calls
# ============================================================================

def _fast_mode(series) -> float:
    """Return the most frequent value in a pandas Series (or first value if empty).

    ~3-5× faster than ``s.mode().iloc[0]`` because:
    • Operates on the underlying numpy array directly.
    • Uses ``np.unique(return_counts=True)`` — a single pass.
    • Avoids the overhead of constructing a new Series for the result.
    """
    arr = series.values
    if len(arr) == 0:
        return np.nan
    # Drop NaNs for float arrays (np.unique sorts NaN to the end but
    # counts them as distinct values on some numpy versions).
    if arr.dtype.kind == 'f':
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan
    vals, counts = np.unique(arr, return_counts=True)
    return vals[counts.argmax()]


# ============================================================================
# Confidence filtering helper (TEMPORARY FIX for data processing bug)
# ============================================================================

# Global flag to track if warning has been printed
_CONFIDENCE_WARNING_PRINTED = False

CONFIDENCE_COLS = ['x_confidence', 'y_confidence', 'z_confidence',
                   'wx_confidence', 'wy_confidence', 'wz_confidence']

# Spin-only confidence columns – used for the UI confidence filter so that
# position/velocity confidence does not mask otherwise good spin data.
SPIN_CONFIDENCE_COLS = ['wx_confidence', 'wy_confidence', 'wz_confidence']

# Trajectory comparison columns used by the cache to pre-compute RMSE values.
# Each entry: (name, position_cols, velocity_cols)
_TRAJ_COLS = [
    ('gt200',      ['x_gt200',      'y_gt200',      'z_gt200'],      ['vx_gt200',      'vy_gt200',      'vz_gt200']),
    ('opt',        ['x_opt',        'y_opt',        'z_opt'],        ['vx_opt',        'vy_opt',        'vz_opt']),
    ('0226',       ['x_0226',       'y_0226',       'z_0226'],       ['vx_0226',       'vy_0226',       'vz_0226']),
    ('nakashima',  ['x_nakashima',  'y_nakashima',  'z_nakashima'],  ['vx_nakashima',  'vy_nakashima',  'vz_nakashima']),
    ('0426',       ['x_0426',       'y_0426',       'z_0426'],       ['vx_0426',       'vy_0426',       'vz_0426']),
]
_N_TRAJ = len(_TRAJ_COLS)  # 5


def filter_to_majority_confidence(data: pd.DataFrame, print_warning: bool = True) -> pd.DataFrame:
    """
    Filter DataFrame rows to keep only those with the majority confidence value.

    TEMPORARY FIX: This handles a bug in data_processing where EKS corrects event
    timestamps but calculate_confidences() uses the original timestamps, causing
    confidence values to change within a flight segment.

    Only applies to columns with a small number of distinct values (discrete
    jumps from the timestamp bug).  Columns with many unique values (continuous
    per-point confidence) are left unfiltered.

    Args:
        data: DataFrame with confidence columns
        print_warning: Whether to print warning (will only print once globally)

    Returns:
        Filtered DataFrame with only rows matching majority confidence
    """
    global _CONFIDENCE_WARNING_PRINTED

    # Check if all confidence columns exist
    if not all(col in data.columns for col in CONFIDENCE_COLS):
        return data

    # Identify columns with a small number of distinct values (discrete jump).
    # The timestamp bug produces exactly 2 (or rarely 3) distinct values.
    # Continuous per-point confidence (many unique values) is NOT a bug.
    _MAX_DISCRETE_UNIQUE = 5
    cols_to_filter = []
    for col in CONFIDENCE_COLS:
        col_data = data[col].values
        unique_vals = np.unique(col_data[~np.isnan(col_data)])
        if 1 < len(unique_vals) <= _MAX_DISCRETE_UNIQUE:
            cols_to_filter.append(col)

    if not cols_to_filter:
        return data

    # Print warning once
    if print_warning and not _CONFIDENCE_WARNING_PRINTED:
        _CONFIDENCE_WARNING_PRINTED = True
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        print(f"\n{YELLOW}" + "!"*80)
        print("WARNING: TEMPORARY FIX IN PLACE FOR DATA PROCESSING BUG")
        print("!"*80)
        print("Detected segments with inconsistent confidence values (confidence changes within segment).")
        print("This is caused by a bug in data_processing where EKS corrects event timestamps")
        print("but calculate_confidences() uses the original timestamps.")
        print("Samples with anomalous confidence values will be EXCLUDED from analysis.")
        print("TODO: Fix data_processing/ball.py to use events_corrected in calculate_confidences()")
        print("!"*80 + f"{RESET}\n")

    # Determine majority confidence: use the mode (most common value) for each
    # column that has a discrete jump.
    mask = np.ones(len(data), dtype=bool)

    for col in cols_to_filter:
        col_data = data[col].values
        valid_data = col_data[~np.isnan(col_data)]
        if len(valid_data) == 0:
            continue

        # Find majority value (mode)
        unique_vals, counts = np.unique(valid_data, return_counts=True)
        majority_val = unique_vals[np.argmax(counts)]

        # Mark rows that don't match majority as invalid
        mask &= (np.isnan(col_data) | (col_data == majority_val))

    return data[mask]


@dataclass
class FlightSegmentCache:
    """
    Pre-extracted data from all flight segments for fast filtering and computation.

    This cache extracts all relevant numeric data once during loading, allowing
    extraction methods to simply filter and slice arrays instead of traversing
    the entire data hierarchy repeatedly.
    """
    # Indices and references
    n_segments: int
    match_indices: np.ndarray      # Which match each segment belongs to
    player_types: np.ndarray       # 1=robot, 2=human for each segment
    flight_segments: List[FlightSegment]  # References to original objects
    shots: List[Shot]              # References to shot objects
    rallies: List[Rally]           # References to rally objects

    # Metadata
    match_dates: List[str]
    match_players: List[str]
    game_names: List[str]
    game_ids: np.ndarray
    rally_ids: np.ndarray
    segment_start_times: np.ndarray
    match_files: List[pathlib.Path]
    log_identifiers: List[str]

    # GT200 velocity data (last point) - for histogram
    vx_gt200: np.ndarray
    vy_gt200: np.ndarray
    vz_gt200: np.ndarray
    has_gt200: np.ndarray  # Boolean mask

    # Optimized velocity data (last point) - for scatter plots
    vx_opt: np.ndarray
    vy_opt: np.ndarray
    vz_opt: np.ndarray
    has_opt: np.ndarray  # Boolean mask

    # Spin data (last point)
    wx: np.ndarray
    wy: np.ndarray
    wz: np.ndarray

    # Aerodynamics data (first point) - for drag/magnus plots
    cd_opt: np.ndarray
    cm_opt: np.ndarray
    vx_opt_first: np.ndarray
    vy_opt_first: np.ndarray
    vz_opt_first: np.ndarray
    wx_first: np.ndarray
    wy_first: np.ndarray
    wz_first: np.ndarray
    v_eff_drag: np.ndarray
    v_eff_magnus: np.ndarray
    w_eff_perp_magnus: np.ndarray
    has_aero: np.ndarray  # Boolean mask

    # Confidence data
    min_confidence: np.ndarray  # Minimum confidence across all points in segment
    has_confidence: np.ndarray  # Boolean mask
    has_inconsistent_confidence: np.ndarray  # Boolean mask - True if confidence varies within segment (data processing bug)

    # Confidence at last point (for confidence distribution)
    conf_last_point: np.ndarray  # Min of spin confidence values (wx,wy,wz) at last point

    # Duration
    durations: np.ndarray  # Time from first to last point

    # RMSE error vs APS (for error filtering)
    rmse_opt: np.ndarray  # RMSE between x_opt and x_aps

    # Magnus force sensitivity (time-weighted integral of ||v x w||)
    magnus_impulse_proxy: np.ndarray

    # Drag force sensitivity (time-weighted integral of ||v||²)
    drag_impulse_proxy: np.ndarray

    # ── Trajectory error metrics (pre-computed during cache build) ──
    # Column order: gt200=0, opt=1, 0226=2, nakashima=3
    pos_rmse_all: np.ndarray       # shape (n, 4) — position RMSE (APS vs traj)
    pos_max_err_all: np.ndarray    # shape (n, 4) — position max error
    pos_mean_err_all: np.ndarray   # shape (n, 4) — position mean error
    vel_rmse_all: np.ndarray       # shape (n, 4) — velocity RMSE
    pos_rmse_mad_all: np.ndarray   # shape (n, 4) — MAD-filtered position RMSE

    # GCS spin confidence medians (for extract_gcs_spin_confidence_data)
    median_w_confidence: np.ndarray   # median of w_confidence per segment
    median_wx_confidence: np.ndarray  # median of wx_confidence per segment

    # Spin observation metrics
    spin_density: np.ndarray          # % samples with spin conf > 0.5
    spin_variance: np.ndarray         # RMSE of GCS spin vs GT200 spin
    gcs_sources: List[str]            # per-segment GCS source tag

    # Per-segment spin observation details (for extract_spin_observation_data)
    mean_delta_spin: np.ndarray       # mean |‖w_GT200‖ − ‖w_gcs‖| per segment
    spin_obs_var: np.ndarray          # variance of ‖w_gcs‖ per segment
    n_gcs_samples: np.ndarray         # count of valid GCS spin samples (w_confidence >= 0.5)
    shot_position: np.ndarray         # 0=other, 1=second-to-last, 2=last in rally
    mean_x_aps: np.ndarray            # mean x_aps position per segment
    mean_y_aps: np.ndarray            # mean y_aps position per segment
    mean_z_aps: np.ndarray            # mean z_aps position per segment
    contact_types: List[str]          # trigger_event.type per segment

    # ── Derived first-point magnitudes (avoid recomputing in extract methods) ──
    vel_mag_first: np.ndarray         # ||v_opt|| at first point (m/s)
    spin_mag_first: np.ndarray        # ||w|| at first point (rad/s)
    spin_ratio_first: np.ndarray      # r*||w|| / ||v|| at first point (dimensionless)

    # ── Derived last-point magnitudes ──
    vel_mag_last: np.ndarray          # ||v|| at last point (uses opt, fallback gt200)
    spin_mag_last: np.ndarray         # ||w|| at last point

    # ── Match-level metadata ──
    match_policies: List[str]         # policy name per segment

    # ── Rally-level pre-computed data (for extract_rally_analysis_data) ──
    # Each entry is a dict with match_idx + the rally row data.
    rally_rows: List[Dict[str, Any]]

    # ── Contact-pair pre-computed data ──
    # Racket contacts: pre/post velocity+spin at each racket hit
    racket_contact_data: Optional[Dict[str, Any]]
    # Table contacts: pre/post velocity+spin at each table bounce
    table_contact_data: Optional[Dict[str, Any]]

    # ── RCM (Racket Contact Model) pre-computed data ──
    # Enriched from extracted.csv RacketContactEvent objects
    rcm_data: Optional[Dict[str, Any]]

    # Base folder reference
    base_folder: pathlib.Path


def build_flight_segment_cache(match_collection: MatchCollection) -> FlightSegmentCache:
    """
    Build a cache of pre-extracted flight segment data.

    This traverses the data hierarchy once and extracts all relevant data
    into efficient numpy arrays for fast subsequent filtering.
    """
    # First pass: count segments to pre-allocate
    n_segments = 0
    for match in match_collection.matches:
        for game in match.games:
            for rally in game.rallies:
                for shot in rally.shots:
                    for fs in shot.flight_segments:
                        if len(fs.data) > 2:
                            n_segments += 1

    if n_segments == 0:
        # Return empty cache
        return FlightSegmentCache(
            n_segments=0,
            match_indices=np.array([], dtype=np.int32),
            player_types=np.array([], dtype=np.int32),
            flight_segments=[],
            shots=[],
            rallies=[],
            match_dates=[],
            match_players=[],
            game_names=[],
            game_ids=np.array([], dtype=np.int32),
            rally_ids=np.array([], dtype=np.int32),
            segment_start_times=np.array([], dtype=np.float64),
            match_files=[],
            log_identifiers=[],
            vx_gt200=np.array([]), vy_gt200=np.array([]), vz_gt200=np.array([]),
            has_gt200=np.array([], dtype=bool),
            vx_opt=np.array([]), vy_opt=np.array([]), vz_opt=np.array([]),
            has_opt=np.array([], dtype=bool),
            wx=np.array([]), wy=np.array([]), wz=np.array([]),
            cd_opt=np.array([]), cm_opt=np.array([]),
            vx_opt_first=np.array([]), vy_opt_first=np.array([]), vz_opt_first=np.array([]),
            wx_first=np.array([]), wy_first=np.array([]), wz_first=np.array([]),
            v_eff_drag=np.array([]), v_eff_magnus=np.array([]), w_eff_perp_magnus=np.array([]),
            has_aero=np.array([], dtype=bool),
            min_confidence=np.array([]),
            has_confidence=np.array([], dtype=bool),
            has_inconsistent_confidence=np.array([], dtype=bool),
            conf_last_point=np.array([]),
            durations=np.array([]),
            rmse_opt=np.array([]),
            magnus_impulse_proxy=np.array([]),
            drag_impulse_proxy=np.array([]),
            pos_rmse_all=np.empty((0, _N_TRAJ)),
            pos_max_err_all=np.empty((0, _N_TRAJ)),
            pos_mean_err_all=np.empty((0, _N_TRAJ)),
            vel_rmse_all=np.empty((0, _N_TRAJ)),
            pos_rmse_mad_all=np.empty((0, _N_TRAJ)),
            median_w_confidence=np.array([]),
            median_wx_confidence=np.array([]),
            spin_density=np.array([]),
            spin_variance=np.array([]),
            gcs_sources=[],
            mean_delta_spin=np.array([]),
            spin_obs_var=np.array([]),
            n_gcs_samples=np.array([], dtype=np.int32),
            shot_position=np.array([], dtype=np.int32),
            mean_x_aps=np.array([]),
            mean_y_aps=np.array([]),
            mean_z_aps=np.array([]),
            contact_types=[],
            vel_mag_first=np.array([]),
            spin_mag_first=np.array([]),
            spin_ratio_first=np.array([]),
            vel_mag_last=np.array([]),
            spin_mag_last=np.array([]),
            match_policies=[],
            rally_rows=[],
            racket_contact_data=None,
            table_contact_data=None,
            rcm_data=None,
            base_folder=match_collection.base_folder,
        )

    # Pre-allocate arrays
    match_indices = np.zeros(n_segments, dtype=np.int32)
    player_types = np.zeros(n_segments, dtype=np.int32)
    game_ids = np.zeros(n_segments, dtype=np.int32)
    rally_ids = np.zeros(n_segments, dtype=np.int32)
    segment_start_times = np.zeros(n_segments, dtype=np.float64)

    vx_gt200 = np.full(n_segments, np.nan)
    vy_gt200 = np.full(n_segments, np.nan)
    vz_gt200 = np.full(n_segments, np.nan)
    has_gt200 = np.zeros(n_segments, dtype=bool)

    vx_opt = np.full(n_segments, np.nan)
    vy_opt = np.full(n_segments, np.nan)
    vz_opt = np.full(n_segments, np.nan)
    has_opt = np.zeros(n_segments, dtype=bool)

    wx = np.full(n_segments, np.nan)
    wy = np.full(n_segments, np.nan)
    wz = np.full(n_segments, np.nan)

    cd_opt_arr = np.full(n_segments, np.nan)
    cm_opt_arr = np.full(n_segments, np.nan)
    vx_opt_first = np.full(n_segments, np.nan)
    vy_opt_first = np.full(n_segments, np.nan)
    vz_opt_first = np.full(n_segments, np.nan)
    wx_first = np.full(n_segments, np.nan)
    wy_first = np.full(n_segments, np.nan)
    wz_first = np.full(n_segments, np.nan)
    v_eff_drag = np.full(n_segments, np.nan)
    v_eff_magnus = np.full(n_segments, np.nan)
    w_eff_perp_magnus = np.full(n_segments, np.nan)
    has_aero = np.zeros(n_segments, dtype=bool)

    min_confidence = np.ones(n_segments)
    has_confidence = np.zeros(n_segments, dtype=bool)
    has_inconsistent_confidence = np.zeros(n_segments, dtype=bool)
    conf_last_point = np.ones(n_segments)
    durations = np.zeros(n_segments)
    rmse_opt = np.full(n_segments, np.nan)  # RMSE between x_opt and x_aps
    magnus_impulse_proxy = np.full(n_segments, np.nan)  # Time-weighted integral of ||v x w||
    drag_impulse_proxy = np.full(n_segments, np.nan)  # Time-weighted integral of ||v||²

    # Trajectory error metrics: shape (n_segments, 5)
    pos_rmse_all = np.full((n_segments, _N_TRAJ), np.nan)
    pos_max_err_all = np.full((n_segments, _N_TRAJ), np.nan)
    pos_mean_err_all = np.full((n_segments, _N_TRAJ), np.nan)
    vel_rmse_all = np.full((n_segments, _N_TRAJ), np.nan)
    pos_rmse_mad_all = np.full((n_segments, _N_TRAJ), np.nan)

    # GCS spin confidence medians
    median_w_confidence_arr = np.full(n_segments, np.nan)
    median_wx_confidence_arr = np.full(n_segments, np.nan)

    # Spin metrics
    spin_density_arr = np.full(n_segments, np.nan)
    spin_variance_arr = np.full(n_segments, np.nan)
    gcs_sources_list: List[str] = []

    # Per-segment spin observation details
    mean_delta_spin_arr = np.full(n_segments, np.nan)
    spin_obs_var_arr = np.full(n_segments, np.nan)
    n_gcs_samples_arr = np.zeros(n_segments, dtype=np.int32)
    shot_position_arr = np.zeros(n_segments, dtype=np.int32)
    mean_x_aps_arr = np.full(n_segments, np.nan)
    mean_y_aps_arr = np.full(n_segments, np.nan)
    mean_z_aps_arr = np.full(n_segments, np.nan)
    contact_types_list: List[str] = []

    # ── Rally-level pre-computed data ──
    rally_rows_list: List[Dict[str, Any]] = []

    # ── Contact-pair accumulators ──
    # Racket contacts
    rc_vx_pre, rc_vy_pre, rc_vz_pre = [], [], []
    rc_wx_pre, rc_wy_pre, rc_wz_pre = [], [], []
    rc_vx_post, rc_vy_post, rc_vz_post = [], [], []
    rc_wx_post, rc_wy_post, rc_wz_post = [], [], []
    rc_player, rc_policy, rc_confidence, rc_max_rmse = [], [], [], []
    rc_metadata, rc_shot, rc_rally, rc_fs_pre, rc_fs_post = [], [], [], [], []

    # Table contacts
    tc_vx_pre, tc_vy_pre, tc_vz_pre = [], [], []
    tc_wx_pre, tc_wy_pre, tc_wz_pre = [], [], []
    tc_vx_post, tc_vy_post, tc_vz_post = [], [], []
    tc_wx_post, tc_wy_post, tc_wz_post = [], [], []
    tc_metadata, tc_shot, tc_rally, tc_fs_pre, tc_fs_post = [], [], [], [], []
    tc_seg_idx, tc_confidence, tc_max_rmse = [], [], []
    # Model predictions (from HDF5 table_contacts)
    tc_model_keys = [
        'vx_post_ittf', 'vy_post_ittf', 'vz_post_ittf', 'wx_post_ittf', 'wy_post_ittf', 'wz_post_ittf',
        'vx_post_paper', 'vy_post_paper', 'vz_post_paper', 'wx_post_paper', 'wy_post_paper', 'wz_post_paper',
        'vx_post_rcn', 'vy_post_rcn', 'vz_post_rcn', 'wx_post_rcn', 'wy_post_rcn', 'wz_post_rcn',
        'vx_post_res0805', 'vy_post_res0805', 'vz_post_res0805', 'wx_post_res0805', 'wy_post_res0805', 'wz_post_res0805',
        'vx_post_pysr', 'vy_post_pysr', 'vz_post_pysr', 'wx_post_pysr', 'wy_post_pysr', 'wz_post_pysr',
        'vx_post_0426', 'vy_post_0426', 'vz_post_0426', 'wx_post_0426', 'wy_post_0426', 'wz_post_0426',
    ]
    tc_model_lists = {k: [] for k in tc_model_keys}

    # RCM (Racket Contact Model) accumulators — from RacketContactEvent objects
    rcm_arrays: Dict[str, list] = {
        'vx_pre': [], 'vy_pre': [], 'vz_pre': [],
        'wx_pre': [], 'wy_pre': [], 'wz_pre': [],
        'vx_post': [], 'vy_post': [], 'vz_post': [],
        'wx_post': [], 'wy_post': [], 'wz_post': [],
        'vrx_pre': [], 'vry_pre': [],
        'wrx_pre': [], 'wry_pre': [], 'wrz_pre': [],
        'vrx_post': [], 'vry_post': [], 'vrz_post': [],
        'wrx_post': [], 'wry_post': [], 'wrz_post': [],
        'theta_angle': [], 'racket_open_angle': [],
        'fitness_pre': [], 'fitness_post': [],
        # Position deltas (ball displacement during opt window)
        'dx_pre': [], 'dy_pre': [], 'dz_pre': [],
        'dx_post': [], 'dy_post': [], 'dz_post': [],
        # Contact location in racket frame (ball pos relative to racket)
        'drx_pre': [], 'dry_pre': [], 'drz_pre': [],
        # Racket position in global frame at contact
        'rx_global': [], 'ry_global': [], 'rz_global': [],
        # Racket velocity / angular velocity in racket frame
        'vrx_racket': [], 'vry_racket': [], 'vrz_racket': [],
        'wrx_racket': [], 'wry_racket': [], 'wrz_racket': [],
        # Racket angular velocity in global frame
        'wgx_racket': [], 'wgy_racket': [], 'wgz_racket': [],
        # Racket angular velocity in body frame (pure R^T, no v_ref alignment)
        'wbx_racket': [], 'wby_racket': [], 'wbz_racket': [],
        # Model predictions (global frame)
        'vx_post_default': [], 'vy_post_default': [], 'vz_post_default': [],
        'wx_post_default': [], 'wy_post_default': [], 'wz_post_default': [],
        'vx_post_cpp': [], 'vy_post_cpp': [], 'vz_post_cpp': [],
        'wx_post_cpp': [], 'wy_post_cpp': [], 'wz_post_cpp': [],
        'vx_post_parametric_7p': [], 'vy_post_parametric_7p': [], 'vz_post_parametric_7p': [],
        'wx_post_parametric_7p': [], 'wy_post_parametric_7p': [], 'wz_post_parametric_7p': [],
        'vx_post_rcm_tangential': [], 'vy_post_rcm_tangential': [], 'vz_post_rcm_tangential': [],
        'wx_post_rcm_tangential': [], 'wy_post_rcm_tangential': [], 'wz_post_rcm_tangential': [],
        'vx_post_cpp_tangential': [], 'vy_post_cpp_tangential': [], 'vz_post_cpp_tangential': [],
        'wx_post_cpp_tangential': [], 'wy_post_cpp_tangential': [], 'wz_post_cpp_tangential': [],
        'vx_post_nakashima_refined': [], 'vy_post_nakashima_refined': [], 'vz_post_nakashima_refined': [],
        'wx_post_nakashima_refined': [], 'wy_post_nakashima_refined': [], 'wz_post_nakashima_refined': [],
        'vx_post_cpp_refined': [], 'vy_post_cpp_refined': [], 'vz_post_cpp_refined': [],
        'wx_post_cpp_refined': [], 'wy_post_cpp_refined': [], 'wz_post_cpp_refined': [],
        'vx_post_rcm_tangential_refined': [], 'vy_post_rcm_tangential_refined': [], 'vz_post_rcm_tangential_refined': [],
        'wx_post_rcm_tangential_refined': [], 'wy_post_rcm_tangential_refined': [], 'wz_post_rcm_tangential_refined': [],
        'vx_post_rcm_tangential_polyfit': [], 'vy_post_rcm_tangential_polyfit': [], 'vz_post_rcm_tangential_polyfit': [],
        'wx_post_rcm_tangential_polyfit': [], 'wy_post_rcm_tangential_polyfit': [], 'wz_post_rcm_tangential_polyfit': [],
        'vx_post_onnx_rcm': [], 'vy_post_onnx_rcm': [], 'vz_post_onnx_rcm': [],
        'wx_post_onnx_rcm': [], 'wy_post_onnx_rcm': [], 'wz_post_onnx_rcm': [],
        'vx_post_onnx_alex': [], 'vy_post_onnx_alex': [], 'vz_post_onnx_alex': [],
        'wx_post_onnx_alex': [], 'wy_post_onnx_alex': [], 'wz_post_onnx_alex': [],
        'vx_post_onnx_alex_refined': [], 'vy_post_onnx_alex_refined': [], 'vz_post_onnx_alex_refined': [],
        'wx_post_onnx_alex_refined': [], 'wy_post_onnx_alex_refined': [], 'wz_post_onnx_alex_refined': [],
        'vx_post_onnx_alex_polyfit': [], 'vy_post_onnx_alex_polyfit': [], 'vz_post_onnx_alex_polyfit': [],
        'wx_post_onnx_alex_polyfit': [], 'wy_post_onnx_alex_polyfit': [], 'wz_post_onnx_alex_polyfit': [],
        'vx_post_onnx_0426': [], 'vy_post_onnx_0426': [], 'vz_post_onnx_0426': [],
        'wx_post_onnx_0426': [], 'wy_post_onnx_0426': [], 'wz_post_onnx_0426': [],
        # Model predictions (racket / local frame)
        'vrx_post_default': [], 'vry_post_default': [], 'vrz_post_default': [],
        'wrx_post_default': [], 'wry_post_default': [], 'wrz_post_default': [],
        'vrx_post_cpp': [], 'vry_post_cpp': [], 'vrz_post_cpp': [],
        'wrx_post_cpp': [], 'wry_post_cpp': [], 'wrz_post_cpp': [],
        'vrx_post_parametric_7p': [], 'vry_post_parametric_7p': [], 'vrz_post_parametric_7p': [],
        'wrx_post_parametric_7p': [], 'wry_post_parametric_7p': [], 'wrz_post_parametric_7p': [],
        'vrx_post_rcm_tangential': [], 'vry_post_rcm_tangential': [], 'vrz_post_rcm_tangential': [],
        'wrx_post_rcm_tangential': [], 'wry_post_rcm_tangential': [], 'wrz_post_rcm_tangential': [],
        'vrx_post_cpp_tangential': [], 'vry_post_cpp_tangential': [], 'vrz_post_cpp_tangential': [],
        'wrx_post_cpp_tangential': [], 'wry_post_cpp_tangential': [], 'wrz_post_cpp_tangential': [],
        'vrx_post_nakashima_refined': [], 'vry_post_nakashima_refined': [], 'vrz_post_nakashima_refined': [],
        'wrx_post_nakashima_refined': [], 'wry_post_nakashima_refined': [], 'wrz_post_nakashima_refined': [],
        'vrx_post_cpp_refined': [], 'vry_post_cpp_refined': [], 'vrz_post_cpp_refined': [],
        'wrx_post_cpp_refined': [], 'wry_post_cpp_refined': [], 'wrz_post_cpp_refined': [],
        'vrx_post_rcm_tangential_refined': [], 'vry_post_rcm_tangential_refined': [], 'vrz_post_rcm_tangential_refined': [],
        'wrx_post_rcm_tangential_refined': [], 'wry_post_rcm_tangential_refined': [], 'wrz_post_rcm_tangential_refined': [],
        'vrx_post_rcm_tangential_polyfit': [], 'vry_post_rcm_tangential_polyfit': [], 'vrz_post_rcm_tangential_polyfit': [],
        'wrx_post_rcm_tangential_polyfit': [], 'wry_post_rcm_tangential_polyfit': [], 'wrz_post_rcm_tangential_polyfit': [],
        'vrx_post_onnx_rcm': [], 'vry_post_onnx_rcm': [], 'vrz_post_onnx_rcm': [],
        'wrx_post_onnx_rcm': [], 'wry_post_onnx_rcm': [], 'wrz_post_onnx_rcm': [],
        'vrx_post_onnx_alex': [], 'vry_post_onnx_alex': [], 'vrz_post_onnx_alex': [],
        'wrx_post_onnx_alex': [], 'wry_post_onnx_alex': [], 'wrz_post_onnx_alex': [],
        'vrx_post_onnx_alex_refined': [], 'vry_post_onnx_alex_refined': [], 'vrz_post_onnx_alex_refined': [],
        'wrx_post_onnx_alex_refined': [], 'wry_post_onnx_alex_refined': [], 'wrz_post_onnx_alex_refined': [],
        'vrx_post_onnx_alex_polyfit': [], 'vry_post_onnx_alex_polyfit': [], 'vrz_post_onnx_alex_polyfit': [],
        'wrx_post_onnx_alex_polyfit': [], 'wry_post_onnx_alex_polyfit': [], 'wrz_post_onnx_alex_polyfit': [],
        'vrx_post_onnx_0426': [], 'vry_post_onnx_0426': [], 'vrz_post_onnx_0426': [],
        'wrx_post_onnx_0426': [], 'wry_post_onnx_0426': [], 'wrz_post_onnx_0426': [],
    }
    rcm_player: List[int] = []
    rcm_shot_type: List[int] = []  # 0=serve, 1=rally, 2=last shot
    rcm_metadata: List[dict] = []
    rcm_shot: List[Any] = []
    rcm_rally: List[Any] = []
    rcm_fs_pre: List[Any] = []
    rcm_fs_post: List[Any] = []
    rcm_confidence: List[float] = []
    rcm_max_rmse: List[float] = []
    rcm_worst_spin_density: List[float] = []
    rcm_worst_spin_variance: List[float] = []
    rcm_point_winner: List[int] = []  # 0=unknown, 1=robot won, 2=player won

    # Lists for variable-size data
    flight_segments = []
    shots_list = []
    rallies_list = []
    match_dates = []
    match_players = []
    match_policies_list: List[str] = []
    game_names = []
    match_files = []
    log_identifiers = []

    confidence_cols = ['x_confidence', 'y_confidence', 'z_confidence',
                       'wx_confidence', 'wy_confidence', 'wz_confidence']
    spin_confidence_cols = SPIN_CONFIDENCE_COLS

    # Helper for extracting racket-frame vector components (used by RCM extraction)
    def _v3_cache(v, i):
        return v[i] if v is not None else float('nan')

    # Second pass: extract data and pre-build shot DataFrames
    idx = 0
    for match_idx, match in enumerate(match_collection.matches):
        match_date_str = str(match.date)
        match_player = match.player
        match_policy = getattr(match, 'policy', 'unknown')
        match_file = match.file

        for game in match.games:
            game_name = game.game_name
            game_id = game.game_id

            for rally in game.rallies:
                rally_id = rally.rally_id
                # Extract GCS source for this rally (stored as trajectory attr)
                _rally_gcs_source = (
                    getattr(rally.rally, 'attrs', {}).get('gcs_source', 'unknown')
                    if hasattr(rally, 'rally') and hasattr(rally.rally, 'attrs')
                    else 'unknown'
                )

                # ── Rally-level aggregate (for extract_rally_analysis_data) ──
                if len(rally.shots) > 0:
                    _r_time_from_last = []
                    _r_pos = []
                    _r_vel = []
                    _r_spin = []
                    _r_prev_time = 0.0
                    for _r_si, _r_shot in enumerate(rally.shots):
                        _r_valid_segs = [_fs for _fs in _r_shot.flight_segments
                                         if hasattr(_fs, 'trigger_event') and
                                            hasattr(_fs.trigger_event, 'type') and
                                            _fs.trigger_event.type != 'START']
                        if not _r_valid_segs:
                            continue
                        _r_fs = _r_valid_segs[-1]
                        if len(_r_fs.data) == 0:
                            continue
                        _r_last_idx = _r_fs.data.index[-1]
                        _r_cols = _r_fs.data.columns
                        _r_time = _r_fs.data.loc[_r_last_idx, 'time'] if 'time' in _r_cols else np.nan
                        _r_tfl = (_r_time - _r_prev_time) if _r_si > 0 else 0.0
                        _r_prev_time = _r_time
                        # Position
                        if 'x_gt200' in _r_cols:
                            _rx = _r_fs.data.loc[_r_last_idx, 'x_gt200']
                            _ry = _r_fs.data.loc[_r_last_idx, 'y_gt200']
                            _rz = _r_fs.data.loc[_r_last_idx, 'z_gt200']
                        elif 'x_aps' in _r_cols:
                            _rx = _r_fs.data.loc[_r_last_idx, 'x_aps']
                            _ry = _r_fs.data.loc[_r_last_idx, 'y_aps']
                            _rz = _r_fs.data.loc[_r_last_idx, 'z_aps']
                        else:
                            continue
                        # Velocity
                        if 'vx_gt200' in _r_cols:
                            _rvx = _r_fs.data.loc[_r_last_idx, 'vx_gt200']
                            _rvy = _r_fs.data.loc[_r_last_idx, 'vy_gt200']
                            _rvz = _r_fs.data.loc[_r_last_idx, 'vz_gt200']
                        elif 'vx_aps' in _r_cols:
                            _rvx = _r_fs.data.loc[_r_last_idx, 'vx_aps']
                            _rvy = _r_fs.data.loc[_r_last_idx, 'vy_aps']
                            _rvz = _r_fs.data.loc[_r_last_idx, 'vz_aps']
                        else:
                            continue
                        # Spin
                        if 'wx' in _r_cols:
                            _rwx = _r_fs.data.loc[_r_last_idx, 'wx']
                            _rwy = _r_fs.data.loc[_r_last_idx, 'wy']
                            _rwz = _r_fs.data.loc[_r_last_idx, 'wz']
                        elif 'wx_gcs' in _r_cols:
                            _rwx = _r_fs.data.loc[_r_last_idx, 'wx_gcs']
                            _rwy = _r_fs.data.loc[_r_last_idx, 'wy_gcs']
                            _rwz = _r_fs.data.loc[_r_last_idx, 'wz_gcs']
                        else:
                            continue
                        if not any(pd.isna(v) for v in [_rx, _ry, _rz, _rvx, _rvy, _rvz, _rwx, _rwy, _rwz]):
                            _r_time_from_last.append(_r_tfl)
                            _r_pos.append([_rx, _ry, _rz])
                            _r_vel.append([_rvx, _rvy, _rvz])
                            _r_spin.append([_rwx, _rwy, _rwz])
                    if _r_time_from_last:
                        _r_last_shot = rally.shots[-1]
                        _r_last_valid = [_fs for _fs in _r_last_shot.flight_segments
                                         if hasattr(_fs, 'trigger_event') and
                                            hasattr(_fs.trigger_event, 'type') and
                                            _fs.trigger_event.type != 'START']
                        _r_winner = _r_last_shot.player if _r_last_valid else (3 - _r_last_shot.player)
                        rally_rows_list.append({
                            'match_idx': match_idx,
                            'date': match.date,
                            'match_player': match_player,
                            'game_id': game_id,
                            'rally_id': rally_id,
                            'rally_length': len(_r_time_from_last),
                            'rally_last_shot': 'robot' if _r_last_shot.player == 1 else 'human',
                            'rally_winner': 'robot' if _r_winner == 1 else 'human',
                            'time_from_last': np.array(_r_time_from_last),
                            'pos': np.array(_r_pos).flatten(),
                            'vel': np.array(_r_vel).flatten(),
                            'spin': np.array(_r_spin).flatten(),
                        })

                # ── Racket contact pairs (for extract_racket_contact_data) ──
                for _rc_si in range(1, len(rally.shots)):
                    _rc_shot_post = rally.shots[_rc_si]
                    if not _rc_shot_post.flight_segments:
                        continue
                    _rc_fs_post = _rc_shot_post.flight_segments[0]
                    if _rc_fs_post.trigger_event is None:
                        continue
                    if _rc_fs_post.trigger_event.type not in ('shot_p1', 'shot_p2'):
                        continue
                    _rc_shot_pre = rally.shots[_rc_si - 1]
                    if not _rc_shot_pre.flight_segments:
                        continue
                    _rc_fs_pre = _rc_shot_pre.flight_segments[-1]
                    _cols_pre = _rc_fs_pre.data.columns
                    _cols_post = _rc_fs_post.data.columns
                    _has_aero_pre = 'vx_opt' in _cols_pre
                    _has_aero_post = 'vx_opt' in _cols_post
                    if not (_has_aero_pre and _has_aero_post):
                        continue
                    try:
                        _vx_col = 'vx_opt'
                        _vy_col = 'vy_opt'
                        _vz_col = 'vz_opt'
                        if 'wx_opt' in _cols_pre:
                            _wx_col, _wy_col, _wz_col = 'wx_opt', 'wy_opt', 'wz_opt'
                        elif 'wx' in _cols_pre:
                            _wx_col, _wy_col, _wz_col = 'wx', 'wy', 'wz'
                        else:
                            continue
                        _pre_row = _rc_fs_pre.data.iloc[-1]
                        _post_row = _rc_fs_post.data.iloc[0]
                        _vals = [_pre_row[_vx_col], _pre_row[_vy_col], _pre_row[_vz_col],
                                 _pre_row[_wx_col], _pre_row[_wy_col], _pre_row[_wz_col],
                                 _post_row.get(_vx_col, np.nan), _post_row.get(_vy_col, np.nan),
                                 _post_row.get(_vz_col, np.nan),
                                 _post_row.get(_wx_col, np.nan), _post_row.get(_wy_col, np.nan),
                                 _post_row.get(_wz_col, np.nan)]
                        if any(np.isnan(v) for v in _vals):
                            continue
                        rc_vx_pre.append(_vals[0]); rc_vy_pre.append(_vals[1]); rc_vz_pre.append(_vals[2])
                        rc_wx_pre.append(_vals[3]); rc_wy_pre.append(_vals[4]); rc_wz_pre.append(_vals[5])
                        rc_vx_post.append(_vals[6]); rc_vy_post.append(_vals[7]); rc_vz_post.append(_vals[8])
                        rc_wx_post.append(_vals[9]); rc_wy_post.append(_vals[10]); rc_wz_post.append(_vals[11])
                        rc_player.append(1 if _rc_fs_post.trigger_event.type == 'shot_p1' else 2)
                        rc_policy.append(match_policy)
                        # Confidence: min across both segments
                        _rc_min_conf = 1.0
                        for _rc_fs in [_rc_fs_pre, _rc_fs_post]:
                            if all(c in _rc_fs.data.columns for c in spin_confidence_cols):
                                _rc_cv = _rc_fs.data[spin_confidence_cols].values
                                _rc_sm = np.nanmin(_rc_cv)
                                if not np.isnan(_rc_sm):
                                    _rc_min_conf = min(_rc_min_conf, _rc_sm)
                        rc_confidence.append(_rc_min_conf)
                        # Max RMSE across both segments
                        _rc_mr = np.nan
                        for _rc_fs in [_rc_fs_pre, _rc_fs_post]:
                            _rc_ho = all(c in _rc_fs.data.columns for c in ['x_opt', 'y_opt', 'z_opt'])
                            _rc_ha = all(c in _rc_fs.data.columns for c in ['x_aps', 'y_aps', 'z_aps'])
                            if _rc_ho and _rc_ha:
                                _xo = _rc_fs.data['x_opt'].values; _yo = _rc_fs.data['y_opt'].values; _zo = _rc_fs.data['z_opt'].values
                                _xa = _rc_fs.data['x_aps'].values; _ya = _rc_fs.data['y_aps'].values; _za = _rc_fs.data['z_aps'].values
                                _v = ~(np.isnan(_xa) | np.isnan(_xo))
                                if np.sum(_v) > 0:
                                    _sr = np.sqrt(np.mean((_xa[_v]-_xo[_v])**2 + (_ya[_v]-_yo[_v])**2 + (_za[_v]-_zo[_v])**2))
                                    if np.isnan(_rc_mr) or _sr > _rc_mr:
                                        _rc_mr = _sr
                        rc_max_rmse.append(_rc_mr)
                        rc_metadata.append({
                            'match_idx': match_idx,
                            'match_date': match_date_str, 'match_player': match_player,
                            'game_name': game_name, 'game_id': game_id, 'rally_id': rally_id,
                            'segment_start_time': float(_rc_fs_post.data.index[0]) if len(_rc_fs_post.data) > 0 else 0.0,
                            'contact_type': _rc_fs_post.trigger_event.type,
                            'contact_time': _rc_fs_post.trigger_event.timestamp,
                            'match_file': match_file, 'base_folder': match_collection.base_folder,
                            'log_identifier': getattr(rally, 'log_identifier', None) or '',
                        })
                        rc_shot.append(_rc_shot_post); rc_rally.append(rally)
                        rc_fs_pre.append(_rc_fs_pre); rc_fs_post.append(_rc_fs_post)
                    except Exception:
                        continue

                # ── Table contact pairs (for extract_table_contact_data) ──
                # Prefer enriched TableContactEvent (from extracted_TCM.csv,
                # using ODE trajectory endpoints) over iloc-based extraction.
                for _tc_shot in rally.shots:
                    _tc_segs = _tc_shot.flight_segments
                    for _tc_i in range(len(_tc_segs) - 1):
                        _tc_fspre = _tc_segs[_tc_i]
                        _tc_fspost = _tc_segs[_tc_i + 1]
                        if _tc_fspost.trigger_event is None:
                            continue
                        if _tc_fspost.trigger_event.type not in ('bounce_p1', 'bounce_p2'):
                            continue
                        try:
                            # ── Try enriched TableContactEvent first ──
                            _tc_evt = _tc_fspost.trigger_event
                            _tc_enriched = (
                                isinstance(_tc_evt, TableContactEvent)
                                and _tc_evt.ball_pre is not None
                                and _tc_evt.ball_post is not None
                                and not np.isnan(_tc_evt.ball_pre.vx)
                            )
                            if _tc_enriched:
                                _bp = _tc_evt.ball_pre
                                _ba = _tc_evt.ball_post
                                _tc_vals = [_bp.vx, _bp.vy, _bp.vz,
                                            _bp.wx, _bp.wy, _bp.wz,
                                            _ba.vx, _ba.vy, _ba.vz,
                                            _ba.wx, _ba.wy, _ba.wz]
                            else:
                                # ── Fallback: iloc-based extraction ──
                                _tc_cols_pre = _tc_fspre.data.columns
                                _tc_cols_post = _tc_fspost.data.columns
                                if 'vx_opt' not in _tc_cols_pre or 'vx_opt' not in _tc_cols_post:
                                    continue
                                if 'wx_opt' in _tc_cols_pre:
                                    _twx, _twy, _twz = 'wx_opt', 'wy_opt', 'wz_opt'
                                elif 'wx' in _tc_cols_pre:
                                    _twx, _twy, _twz = 'wx', 'wy', 'wz'
                                else:
                                    continue
                                _tc_pre_row = _tc_fspre.data.iloc[-1]
                                if len(_tc_fspost.data) < 2:
                                    continue
                                _tc_post_row = _tc_fspost.data.iloc[1]
                                _tc_vals = [_tc_pre_row['vx_opt'], _tc_pre_row['vy_opt'], _tc_pre_row['vz_opt'],
                                            _tc_pre_row[_twx], _tc_pre_row[_twy], _tc_pre_row[_twz],
                                            _tc_post_row.get('vx_opt', np.nan), _tc_post_row.get('vy_opt', np.nan),
                                            _tc_post_row.get('vz_opt', np.nan),
                                            _tc_post_row.get(_twx, np.nan), _tc_post_row.get(_twy, np.nan),
                                            _tc_post_row.get(_twz, np.nan)]
                            if any(np.isnan(v) for v in _tc_vals):
                                continue
                            tc_vx_pre.append(_tc_vals[0]); tc_vy_pre.append(_tc_vals[1]); tc_vz_pre.append(_tc_vals[2])
                            tc_wx_pre.append(_tc_vals[3]); tc_wy_pre.append(_tc_vals[4]); tc_wz_pre.append(_tc_vals[5])
                            tc_vx_post.append(_tc_vals[6]); tc_vy_post.append(_tc_vals[7]); tc_vz_post.append(_tc_vals[8])
                            tc_wx_post.append(_tc_vals[9]); tc_wy_post.append(_tc_vals[10]); tc_wz_post.append(_tc_vals[11])
                            # Model predictions from enriched TableContactEvent
                            _bp_ittf = getattr(_tc_evt, 'ball_post_nakashima_ittf', None) if _tc_enriched else None
                            _bp_paper = getattr(_tc_evt, 'ball_post_nakashima_paper', None) if _tc_enriched else None
                            _bp_rcn = getattr(_tc_evt, 'ball_post_residual_corrected', None) if _tc_enriched else None
                            _bp_res0805 = getattr(_tc_evt, 'ball_post_residual0805', None) if _tc_enriched else None
                            _bp_pysr = getattr(_tc_evt, 'ball_post_pysr', None) if _tc_enriched else None
                            _bp_0426 = getattr(_tc_evt, 'ball_post_0426', None) if _tc_enriched else None
                            for _comp, _src in [('ittf', _bp_ittf), ('paper', _bp_paper), ('rcn', _bp_rcn),
                                                ('res0805', _bp_res0805), ('pysr', _bp_pysr), ('0426', _bp_0426)]:
                                if _src is not None:
                                    tc_model_lists[f'vx_post_{_comp}'].append(_src.vx)
                                    tc_model_lists[f'vy_post_{_comp}'].append(_src.vy)
                                    tc_model_lists[f'vz_post_{_comp}'].append(_src.vz)
                                    tc_model_lists[f'wx_post_{_comp}'].append(_src.wx)
                                    tc_model_lists[f'wy_post_{_comp}'].append(_src.wy)
                                    tc_model_lists[f'wz_post_{_comp}'].append(_src.wz)
                                else:
                                    for _ax in ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']:
                                        tc_model_lists[f'{_ax}_post_{_comp}'].append(np.nan)
                            # Confidence + RMSE
                            _tc_min_conf = 1.0
                            _tc_mr = np.nan
                            for _tc_fs in [_tc_fspre, _tc_fspost]:
                                if all(c in _tc_fs.data.columns for c in spin_confidence_cols):
                                    _cv = _tc_fs.data[spin_confidence_cols].values
                                    _sm = np.nanmin(_cv)
                                    if not np.isnan(_sm):
                                        _tc_min_conf = min(_tc_min_conf, _sm)
                                _ho = all(c in _tc_fs.data.columns for c in ['x_opt', 'y_opt', 'z_opt'])
                                _ha = all(c in _tc_fs.data.columns for c in ['x_aps', 'y_aps', 'z_aps'])
                                if _ho and _ha:
                                    _xo = _tc_fs.data['x_opt'].values; _yo = _tc_fs.data['y_opt'].values; _zo = _tc_fs.data['z_opt'].values
                                    _xa = _tc_fs.data['x_aps'].values; _ya = _tc_fs.data['y_aps'].values; _za = _tc_fs.data['z_aps'].values
                                    _v = ~(np.isnan(_xa) | np.isnan(_xo))
                                    if np.sum(_v) > 0:
                                        _sr = np.sqrt(np.mean((_xa[_v]-_xo[_v])**2 + (_ya[_v]-_yo[_v])**2 + (_za[_v]-_zo[_v])**2))
                                        if np.isnan(_tc_mr) or _sr > _tc_mr:
                                            _tc_mr = _sr
                            tc_confidence.append(_tc_min_conf)
                            tc_max_rmse.append(_tc_mr)
                            tc_metadata.append({
                                'match_idx': match_idx,
                                'match_date': match_date_str, 'match_player': match_player,
                                'game_name': game_name, 'game_id': game_id, 'rally_id': rally_id,
                                'contact_time': _tc_fspost.trigger_event.timestamp,
                                'contact_type': _tc_fspost.trigger_event.type,
                                'match_file': match_file, 'base_folder': match_collection.base_folder,
                                'segment_index': _tc_i,
                                'log_identifier': getattr(rally, 'log_identifier', None) or '',
                            })
                            tc_shot.append(_tc_shot); tc_rally.append(rally)
                            tc_fs_pre.append(_tc_fspre); tc_fs_post.append(_tc_fspost)
                            tc_seg_idx.append(_tc_i)
                        except Exception:
                            continue

                # ── RCM data from RacketContactEvent objects ──
                # Pre-scan: collect ALL racket-contact timestamps (not just
                # accepted ones) so serve/last classification uses the rally's
                # actual first and last shots.
                _all_rce = [e for e in rally.events if isinstance(e, RacketContactEvent)]
                _rally_first_ts = _all_rce[0].timestamp if _all_rce else None
                _rally_last_ts = _all_rce[-1].timestamp if _all_rce else None
                # Also collect accepted timestamps for last-shot logic:
                # the last accepted contact is only "last" if no shot
                # (accepted or not) follows it.
                _rcm_accepted_ts: List[float] = []
                for _rcm_ev_pre in rally.events:
                    if not isinstance(_rcm_ev_pre, RacketContactEvent):
                        continue
                    if _rcm_ev_pre.ball_pre is None or _rcm_ev_pre.ball_post is None:
                        continue
                    _bp0 = _rcm_ev_pre.ball_pre; _ba0 = _rcm_ev_pre.ball_post
                    if any(np.isnan(v) for v in [_bp0.vx, _bp0.vy, _bp0.vz, _bp0.wx, _bp0.wy, _bp0.wz,
                                                 _ba0.vx, _ba0.vy, _ba0.vz, _ba0.wx, _ba0.wy, _ba0.wz]):
                        continue
                    _rcm_accepted_ts.append(_rcm_ev_pre.timestamp)
                _rcm_last_ts = _rcm_accepted_ts[-1] if _rcm_accepted_ts else None
                _last_accepted_is_truly_last = (
                    _rcm_last_ts is not None
                    and _rally_last_ts is not None
                    and abs(_rcm_last_ts - _rally_last_ts) < 1e-6
                )

                for _rcm_event in rally.events:
                    if not isinstance(_rcm_event, RacketContactEvent):
                        continue
                    if _rcm_event.ball_pre is None or _rcm_event.ball_post is None:
                        continue
                    _bp = _rcm_event.ball_pre
                    _ba = _rcm_event.ball_post
                    if any(np.isnan(v) for v in [_bp.vx, _bp.vy, _bp.vz, _bp.wx, _bp.wy, _bp.wz,
                                                 _ba.vx, _ba.vy, _ba.vz, _ba.wx, _ba.wy, _ba.wz]):
                        continue
                    rcm_arrays['vx_pre'].append(_bp.vx); rcm_arrays['vy_pre'].append(_bp.vy); rcm_arrays['vz_pre'].append(_bp.vz)
                    rcm_arrays['wx_pre'].append(_bp.wx); rcm_arrays['wy_pre'].append(_bp.wy); rcm_arrays['wz_pre'].append(_bp.wz)
                    rcm_arrays['vx_post'].append(_ba.vx); rcm_arrays['vy_post'].append(_ba.vy); rcm_arrays['vz_post'].append(_ba.vz)
                    rcm_arrays['wx_post'].append(_ba.wx); rcm_arrays['wy_post'].append(_ba.wy); rcm_arrays['wz_post'].append(_ba.wz)
                    # Racket-frame quantities
                    _vpre_r = _rcm_event.ball_vel_pre_racket
                    _vpost_r = _rcm_event.ball_vel_post_racket
                    _wpre_r = _rcm_event.ball_spin_pre_racket
                    _wpost_r = _rcm_event.ball_spin_post_racket
                    rcm_arrays['vrx_pre'].append(_v3_cache(_vpre_r, 0)); rcm_arrays['vry_pre'].append(_v3_cache(_vpre_r, 1))
                    rcm_arrays['wrx_pre'].append(_v3_cache(_wpre_r, 0)); rcm_arrays['wry_pre'].append(_v3_cache(_wpre_r, 1)); rcm_arrays['wrz_pre'].append(_v3_cache(_wpre_r, 2))
                    rcm_arrays['vrx_post'].append(_v3_cache(_vpost_r, 0)); rcm_arrays['vry_post'].append(_v3_cache(_vpost_r, 1)); rcm_arrays['vrz_post'].append(_v3_cache(_vpost_r, 2))
                    rcm_arrays['wrx_post'].append(_v3_cache(_wpost_r, 0)); rcm_arrays['wry_post'].append(_v3_cache(_wpost_r, 1)); rcm_arrays['wrz_post'].append(_v3_cache(_wpost_r, 2))
                    rcm_arrays['theta_angle'].append(_rcm_event.theta_angle)
                    rcm_arrays['racket_open_angle'].append(_rcm_event.racket_open_angle)
                    rcm_arrays['fitness_pre'].append(_rcm_event.fitness_pre)
                    rcm_arrays['fitness_post'].append(_rcm_event.fitness_post)
                    # Position deltas
                    _dpre = _rcm_event.delta_pre
                    _dpost = _rcm_event.delta_post
                    rcm_arrays['dx_pre'].append(_v3_cache(_dpre, 0)); rcm_arrays['dy_pre'].append(_v3_cache(_dpre, 1)); rcm_arrays['dz_pre'].append(_v3_cache(_dpre, 2))
                    rcm_arrays['dx_post'].append(_v3_cache(_dpost, 0)); rcm_arrays['dy_post'].append(_v3_cache(_dpost, 1)); rcm_arrays['dz_post'].append(_v3_cache(_dpost, 2))
                    # Contact location in racket frame
                    _bpr = _rcm_event.ball_pos_pre_racket
                    rcm_arrays['drx_pre'].append(_v3_cache(_bpr, 0)); rcm_arrays['dry_pre'].append(_v3_cache(_bpr, 1)); rcm_arrays['drz_pre'].append(_v3_cache(_bpr, 2))
                    # Racket position in global frame at contact
                    _rpos = _rcm_event.racket_pos
                    rcm_arrays['rx_global'].append(_rpos.x if _rpos is not None else float('nan'))
                    rcm_arrays['ry_global'].append(_rpos.y if _rpos is not None else float('nan'))
                    rcm_arrays['rz_global'].append(_rpos.z if _rpos is not None else float('nan'))
                    # Racket velocity / angular velocity in racket frame
                    _rvr = _rcm_event.racket_vel_racket
                    _rsr = _rcm_event.racket_spin_racket
                    rcm_arrays['vrx_racket'].append(_v3_cache(_rvr, 0)); rcm_arrays['vry_racket'].append(_v3_cache(_rvr, 1)); rcm_arrays['vrz_racket'].append(_v3_cache(_rvr, 2))
                    rcm_arrays['wrx_racket'].append(_v3_cache(_rsr, 0)); rcm_arrays['wry_racket'].append(_v3_cache(_rsr, 1)); rcm_arrays['wrz_racket'].append(_v3_cache(_rsr, 2))
                    # Racket angular velocity in global frame
                    _rav = _rcm_event.racket_ang_vel
                    rcm_arrays['wgx_racket'].append(_v3_cache(_rav, 0)); rcm_arrays['wgy_racket'].append(_v3_cache(_rav, 1)); rcm_arrays['wgz_racket'].append(_v3_cache(_rav, 2))
                    # Racket angular velocity in body frame (pure R^T)
                    _rsb = _rcm_event.racket_spin_body
                    rcm_arrays['wbx_racket'].append(_v3_cache(_rsb, 0)); rcm_arrays['wby_racket'].append(_v3_cache(_rsb, 1)); rcm_arrays['wbz_racket'].append(_v3_cache(_rsb, 2))
                    # Model predictions (global frame)
                    for _sfx, _attr in [('default', 'ball_post_default'),
                                        ('cpp', 'ball_post_cpp'),
                                        ('parametric_7p', 'ball_post_parametric_7p'),
                                        ('rcm_tangential', 'ball_post_rcm_tangential'),
                                        ('cpp_tangential', 'ball_post_cpp_tangential'),
                                        ('nakashima_refined', 'ball_post_nakashima_refined'),
                                        ('cpp_refined', 'ball_post_cpp_refined'),
                                        ('rcm_tangential_refined', 'ball_post_rcm_tangential_refined'),
                                        ('rcm_tangential_polyfit', 'ball_post_rcm_tangential_polyfit'),
                                        ('onnx_rcm', 'ball_post_onnx_rcm'),
                                        ('onnx_alex', 'ball_post_onnx_alex'),
                                        ('onnx_alex_refined', 'ball_post_onnx_alex_refined'),
                                        ('onnx_alex_polyfit', 'ball_post_onnx_alex_polyfit'),
                                        ('onnx_0426', 'ball_post_onnx_0426')]:
                        _mbp = getattr(_rcm_event, _attr, None)
                        rcm_arrays[f'vx_post_{_sfx}'].append(_mbp.vx if _mbp is not None else float('nan'))
                        rcm_arrays[f'vy_post_{_sfx}'].append(_mbp.vy if _mbp is not None else float('nan'))
                        rcm_arrays[f'vz_post_{_sfx}'].append(_mbp.vz if _mbp is not None else float('nan'))
                        rcm_arrays[f'wx_post_{_sfx}'].append(_mbp.wx if _mbp is not None else float('nan'))
                        rcm_arrays[f'wy_post_{_sfx}'].append(_mbp.wy if _mbp is not None else float('nan'))
                        rcm_arrays[f'wz_post_{_sfx}'].append(_mbp.wz if _mbp is not None else float('nan'))
                    # Model predictions (racket / local frame)
                    for _sfx, _vel_attr, _spin_attr in [
                        ('default', 'ball_vel_post_default_racket', 'ball_spin_post_default_racket'),
                        ('cpp', 'ball_vel_post_cpp_racket', 'ball_spin_post_cpp_racket'),
                        ('parametric_7p', 'ball_vel_post_parametric_7p_racket', 'ball_spin_post_parametric_7p_racket'),
                        ('rcm_tangential', 'ball_vel_post_rcm_tangential_racket', 'ball_spin_post_rcm_tangential_racket'),
                        ('cpp_tangential', 'ball_vel_post_cpp_tangential_racket', 'ball_spin_post_cpp_tangential_racket'),
                        ('nakashima_refined', 'ball_vel_post_nakashima_refined_racket', 'ball_spin_post_nakashima_refined_racket'),
                        ('cpp_refined', 'ball_vel_post_cpp_refined_racket', 'ball_spin_post_cpp_refined_racket'),
                        ('rcm_tangential_refined', 'ball_vel_post_rcm_tangential_refined_racket', 'ball_spin_post_rcm_tangential_refined_racket'),
                        ('rcm_tangential_polyfit', 'ball_vel_post_rcm_tangential_polyfit_racket', 'ball_spin_post_rcm_tangential_polyfit_racket'),
                        ('onnx_rcm', 'ball_vel_post_onnx_rcm_racket', 'ball_spin_post_onnx_rcm_racket'),
                        ('onnx_alex', 'ball_vel_post_onnx_alex_racket', 'ball_spin_post_onnx_alex_racket'),
                        ('onnx_alex_refined', 'ball_vel_post_onnx_alex_refined_racket', 'ball_spin_post_onnx_alex_refined_racket'),
                        ('onnx_alex_polyfit', 'ball_vel_post_onnx_alex_polyfit_racket', 'ball_spin_post_onnx_alex_polyfit_racket'),
                        ('onnx_0426', 'ball_vel_post_onnx_0426_racket', 'ball_spin_post_onnx_0426_racket'),
                    ]:
                        _mv = getattr(_rcm_event, _vel_attr, None)
                        _ms = getattr(_rcm_event, _spin_attr, None)
                        rcm_arrays[f'vrx_post_{_sfx}'].append(_v3_cache(_mv, 0))
                        rcm_arrays[f'vry_post_{_sfx}'].append(_v3_cache(_mv, 1))
                        rcm_arrays[f'vrz_post_{_sfx}'].append(_v3_cache(_mv, 2))
                        rcm_arrays[f'wrx_post_{_sfx}'].append(_v3_cache(_ms, 0))
                        rcm_arrays[f'wry_post_{_sfx}'].append(_v3_cache(_ms, 1))
                        rcm_arrays[f'wrz_post_{_sfx}'].append(_v3_cache(_ms, 2))
                    rcm_player.append(1 if _rcm_event.type == 'shot_p1' else 2)
                    # Classify shot type: 0=serve, 1=rally, 2=last shot
                    # Serve = this event IS the rally's first shot (not just
                    # the first *accepted* one).  Last = truly the last
                    # contact of the rally.
                    _is_serve = (
                        _rally_first_ts is not None
                        and abs(_rcm_event.timestamp - _rally_first_ts) < 1e-6
                    )
                    _is_last = (
                        _rcm_event.timestamp == _rcm_last_ts
                        and _last_accepted_is_truly_last
                    )
                    if _is_serve:
                        rcm_shot_type.append(0)
                    elif _is_last:
                        rcm_shot_type.append(2)
                    else:
                        rcm_shot_type.append(1)
                    rcm_metadata.append({
                        'match_idx': match_idx,
                        'match_date': match_date_str, 'match_player': match_player,
                        'match_policy': match_policy,
                        'game_id': game_id, 'game_name': game_name,
                        'rally_id': rally_id,
                        'contact_type': _rcm_event.type,
                        'contact_time': _rcm_event.timestamp,
                        'match_file': match_file, 'base_folder': match_collection.base_folder,
                        'log_identifier': getattr(rally, 'log_identifier', None) or '',
                    })
                    # Find matching shot and flight segments around this event
                    _rcm_fs_pre_ref, _rcm_fs_post_ref, _rcm_shot_ref = None, None, None
                    for _rcm_si in range(1, len(rally.shots)):
                        _rcm_sp = rally.shots[_rcm_si]
                        if len(_rcm_sp.flight_segments) == 0:
                            continue
                        _rcm_fsp = _rcm_sp.flight_segments[0]
                        if (_rcm_fsp.trigger_event is not None and
                                abs(_rcm_fsp.trigger_event.timestamp - _rcm_event.timestamp) < 1e-6):
                            _rcm_shot_ref = _rcm_sp
                            _rcm_fs_post_ref = _rcm_fsp
                            _rcm_sp_prev = rally.shots[_rcm_si - 1]
                            if len(_rcm_sp_prev.flight_segments) > 0:
                                _rcm_fs_pre_ref = _rcm_sp_prev.flight_segments[-1]
                            break
                    rcm_shot.append(_rcm_shot_ref)
                    rcm_rally.append(rally)
                    rcm_fs_pre.append(_rcm_fs_pre_ref)
                    rcm_fs_post.append(_rcm_fs_post_ref)
                    _pw = getattr(rally, 'point_winner', None)
                    rcm_point_winner.append(1 if _pw == 'player1' else (2 if _pw == 'player2' else 0))
                    # Confidence: min across both adjacent segments
                    _rcm_min_conf = 1.0
                    for _rcm_cfs in [_rcm_fs_pre_ref, _rcm_fs_post_ref]:
                        if _rcm_cfs is not None and all(c in _rcm_cfs.data.columns for c in spin_confidence_cols):
                            _rcm_cv = _rcm_cfs.data[spin_confidence_cols].values
                            _rcm_sm = np.nanmin(_rcm_cv)
                            if not np.isnan(_rcm_sm):
                                _rcm_min_conf = min(_rcm_min_conf, _rcm_sm)
                    rcm_confidence.append(_rcm_min_conf)
                    # Max RMSE across both segments
                    _rcm_mr = np.nan
                    for _rcm_cfs in [_rcm_fs_pre_ref, _rcm_fs_post_ref]:
                        if _rcm_cfs is None:
                            continue
                        _rcm_ho = all(c in _rcm_cfs.data.columns for c in ['x_opt', 'y_opt', 'z_opt'])
                        _rcm_ha = all(c in _rcm_cfs.data.columns for c in ['x_aps', 'y_aps', 'z_aps'])
                        if _rcm_ho and _rcm_ha:
                            _xo = _rcm_cfs.data['x_opt'].values; _yo = _rcm_cfs.data['y_opt'].values; _zo = _rcm_cfs.data['z_opt'].values
                            _xa = _rcm_cfs.data['x_aps'].values; _ya = _rcm_cfs.data['y_aps'].values; _za = _rcm_cfs.data['z_aps'].values
                            _v = ~(np.isnan(_xa) | np.isnan(_xo))
                            if np.sum(_v) > 0:
                                _sr = np.sqrt(np.mean((_xa[_v]-_xo[_v])**2 + (_ya[_v]-_yo[_v])**2 + (_za[_v]-_zo[_v])**2))
                                if np.isnan(_rcm_mr) or _sr > _rcm_mr:
                                    _rcm_mr = _sr
                    rcm_max_rmse.append(_rcm_mr)
                    # Worst spin density (min across pre/post) and worst spin variance (max across pre/post)
                    _rcm_best_sd = np.nan  # will take min → worst = lowest density
                    _rcm_worst_sv = np.nan  # will take max → worst = highest RMSE
                    for _rcm_cfs in [_rcm_fs_pre_ref, _rcm_fs_post_ref]:
                        if _rcm_cfs is None:
                            continue
                        _rcm_cols = _rcm_cfs.data.columns
                        _rcm_dv = _rcm_cfs.data.values
                        # Spin density
                        _sd_cols = ['wx_confidence', 'wy_confidence', 'wz_confidence']
                        if all(c in _rcm_cols for c in _sd_cols):
                            try:
                                _sc = np.column_stack([_rcm_dv[:, _rcm_cols.get_loc(c)] for c in _sd_cols])
                                _sc_min = np.min(_sc, axis=1)
                                _wc_sd = _rcm_dv[:, _rcm_cols.get_loc('w_confidence')] if 'w_confidence' in _rcm_cols else np.ones(len(_rcm_dv))
                                _high = np.sum((_sc_min > 0.5) & (_wc_sd >= 0.5))
                                # Compute duration from time column (FlightSegment has no .duration attr)
                                _dur = 0.0
                                if 'time' in _rcm_cols and len(_rcm_dv) > 1:
                                    _t = _rcm_dv[:, _rcm_cols.get_loc('time')]
                                    _dur = _t[-1] - _t[0]
                                _exp = max(1.0, _dur * 200.0 + 1)
                                _sd_val = 100.0 * _high / _exp
                                if np.isnan(_rcm_best_sd) or _sd_val < _rcm_best_sd:
                                    _rcm_best_sd = _sd_val
                            except (KeyError, IndexError):
                                pass
                        # Spin variance (std of observed spin magnitude)
                        _gcs_sp = ['wx_curr', 'wy_curr', 'wz_curr']
                        if not all(c in _rcm_cols for c in _gcs_sp):
                            _gcs_sp = ['wx_gcs', 'wy_gcs', 'wz_gcs']  # fallback column names
                        if all(c in _rcm_cols for c in _gcs_sp):
                            try:
                                _wxg = _rcm_dv[:, _rcm_cols.get_loc(_gcs_sp[0])]
                                _wyg = _rcm_dv[:, _rcm_cols.get_loc(_gcs_sp[1])]
                                _wzg = _rcm_dv[:, _rcm_cols.get_loc(_gcs_sp[2])]
                                _wc_sv = _rcm_dv[:, _rcm_cols.get_loc('w_confidence')] if 'w_confidence' in _rcm_cols else np.ones(len(_rcm_dv))
                                _sv_mask = (~(np.isnan(_wxg) | np.isnan(_wyg) | np.isnan(_wzg)) &
                                            (_wc_sv >= 0.5))
                                if np.sum(_sv_mask) > 1:
                                    _wmag = np.sqrt(_wxg[_sv_mask]**2 + _wyg[_sv_mask]**2 + _wzg[_sv_mask]**2)
                                    _sv_val = np.std(_wmag)
                                    if np.isnan(_rcm_worst_sv) or _sv_val > _rcm_worst_sv:
                                        _rcm_worst_sv = _sv_val
                            except (KeyError, IndexError):
                                pass
                    rcm_worst_spin_density.append(_rcm_best_sd)
                    rcm_worst_spin_variance.append(_rcm_worst_sv)

                for shot_idx, shot in enumerate(rally.shots):
                    # Compute shot position in rally
                    _n_shots = len(rally.shots)
                    if shot_idx == _n_shots - 1:
                        _shot_position = 2  # last
                    elif shot_idx == _n_shots - 2:
                        _shot_position = 1  # second-to-last
                    else:
                        _shot_position = 0  # other

                    for fs_idx, fs in enumerate(shot.flight_segments):
                        if len(fs.data) <= 2:
                            continue

                        # TEMPORARY FIX: Filter to majority confidence before extracting values
                        # This handles a bug in data_processing where confidence varies within segment
                        filtered_data = filter_to_majority_confidence(fs.data, print_warning=True)

                        if len(filtered_data) <= 2:
                            # After filtering, not enough data points remain
                            continue

                        cols = filtered_data.columns
                        data_values = filtered_data.values

                        # Store indices and references
                        match_indices[idx] = match_idx
                        player_types[idx] = shot.player
                        game_ids[idx] = game_id
                        rally_ids[idx] = rally_id

                        flight_segments.append(fs)
                        shots_list.append(shot)
                        rallies_list.append(rally)
                        match_dates.append(match_date_str)
                        match_players.append(match_player)
                        match_policies_list.append(match_policy)
                        game_names.append(game_name)
                        match_files.append(match_file)
                        log_identifiers.append(getattr(rally, 'log_identifier', None) or '')

                        # Track if this segment had inconsistent confidence (for reporting)
                        if len(filtered_data) < len(fs.data):
                            has_inconsistent_confidence[idx] = True

                        # Get time column index
                        time_idx = cols.get_loc('time') if 'time' in cols else None
                        if time_idx is not None:
                            segment_start_times[idx] = data_values[0, time_idx]
                            durations[idx] = data_values[-1, time_idx] - data_values[0, time_idx]

                        # Extract GT200 data (last point)
                        if 'vx_gt200' in cols and 'wx' in cols:
                            try:
                                vx_idx = cols.get_loc('vx_gt200')
                                vy_idx = cols.get_loc('vy_gt200')
                                vz_idx = cols.get_loc('vz_gt200')
                                wx_idx = cols.get_loc('wx')
                                wy_idx = cols.get_loc('wy')
                                wz_idx = cols.get_loc('wz')

                                last_row = data_values[-1]
                                vx_gt200[idx] = last_row[vx_idx]
                                vy_gt200[idx] = last_row[vy_idx]
                                vz_gt200[idx] = last_row[vz_idx]
                                wx[idx] = last_row[wx_idx]
                                wy[idx] = last_row[wy_idx]
                                wz[idx] = last_row[wz_idx]

                                # Check if valid (no NaN)
                                if not np.any(np.isnan([vx_gt200[idx], vy_gt200[idx], vz_gt200[idx],
                                                        wx[idx], wy[idx], wz[idx]])):
                                    has_gt200[idx] = True
                            except (KeyError, IndexError):
                                pass

                        # Extract OPT data (last point)
                        if 'vx_opt' in cols:
                            try:
                                vx_idx = cols.get_loc('vx_opt')
                                vy_idx = cols.get_loc('vy_opt')
                                vz_idx = cols.get_loc('vz_opt')

                                last_row = data_values[-1]
                                vx_opt[idx] = last_row[vx_idx]
                                vy_opt[idx] = last_row[vy_idx]
                                vz_opt[idx] = last_row[vz_idx]

                                # Also need spin for opt
                                if 'wx' in cols and not np.isnan(wx[idx]):
                                    if not np.any(np.isnan([vx_opt[idx], vy_opt[idx], vz_opt[idx]])):
                                        has_opt[idx] = True
                            except (KeyError, IndexError):
                                pass

                        # Extract aerodynamics data (first point)
                        if 'cd_opt' in cols and 'vx_opt' in cols:
                            try:
                                first_row = data_values[0]
                                cd_opt_arr[idx] = first_row[cols.get_loc('cd_opt')]
                                vx_opt_first[idx] = first_row[cols.get_loc('vx_opt')]
                                vy_opt_first[idx] = first_row[cols.get_loc('vy_opt')]
                                vz_opt_first[idx] = first_row[cols.get_loc('vz_opt')]

                                # Get spin - try optimized first
                                if 'wx_opt' in cols:
                                    wx_first[idx] = first_row[cols.get_loc('wx_opt')]
                                    wy_first[idx] = first_row[cols.get_loc('wy_opt')]
                                    wz_first[idx] = first_row[cols.get_loc('wz_opt')]
                                elif 'wx' in cols:
                                    wx_first[idx] = first_row[cols.get_loc('wx')]
                                    wy_first[idx] = first_row[cols.get_loc('wy')]
                                    wz_first[idx] = first_row[cols.get_loc('wz')]

                                if 'v_eff_drag_opt' in cols:
                                    v_eff_drag[idx] = first_row[cols.get_loc('v_eff_drag_opt')]

                                # Check if valid
                                if not np.any(np.isnan([cd_opt_arr[idx], vx_opt_first[idx], vy_opt_first[idx],
                                                        vz_opt_first[idx], wx_first[idx], wy_first[idx], wz_first[idx]])):
                                    has_aero[idx] = True
                            except (KeyError, IndexError):
                                pass

                        # Magnus coefficient
                        if 'cm_opt' in cols:
                            try:
                                cm_opt_arr[idx] = data_values[0, cols.get_loc('cm_opt')]
                                if 'v_eff_magnus_opt' in cols:
                                    v_eff_magnus[idx] = data_values[0, cols.get_loc('v_eff_magnus_opt')]
                                if 'w_eff_perp_magnus_opt' in cols:
                                    w_eff_perp_magnus[idx] = data_values[0, cols.get_loc('w_eff_perp_magnus_opt')]
                            except (KeyError, IndexError):
                                pass

                        # Extract confidence data (spin-only for UI filter)
                        # Note: After filter_to_majority_confidence, confidence should be constant
                        if all(col in cols for col in spin_confidence_cols):
                            try:
                                conf_indices = [cols.get_loc(c) for c in spin_confidence_cols]
                                conf_data = data_values[:, conf_indices]

                                # Min across all points
                                min_conf: float = np.nanmin(conf_data)
                                if not np.isnan(min_conf):
                                    min_confidence[idx] = min_conf
                                    has_confidence[idx] = True

                                # Min at last point
                                last_conf = conf_data[-1]
                                if not np.any(np.isnan(last_conf)):
                                    conf_last_point[idx] = np.min(last_conf)
                            except (KeyError, IndexError):
                                pass

                        # Compute RMSE between x_opt and x_aps
                        has_opt_pos = 'x_opt' in cols and 'y_opt' in cols and 'z_opt' in cols
                        has_aps_pos = 'x_aps' in cols and 'y_aps' in cols and 'z_aps' in cols
                        if has_opt_pos and has_aps_pos:
                            try:
                                x_aps = data_values[:, cols.get_loc('x_aps')]
                                y_aps = data_values[:, cols.get_loc('y_aps')]
                                z_aps = data_values[:, cols.get_loc('z_aps')]
                                x_opt_pos = data_values[:, cols.get_loc('x_opt')]
                                y_opt_pos = data_values[:, cols.get_loc('y_opt')]
                                z_opt_pos = data_values[:, cols.get_loc('z_opt')]

                                # Only compute where both have valid data
                                valid = ~(np.isnan(x_aps) | np.isnan(x_opt_pos))
                                if np.sum(valid) > 0:
                                    dx = x_aps[valid] - x_opt_pos[valid]
                                    dy = y_aps[valid] - y_opt_pos[valid]
                                    dz = z_aps[valid] - z_opt_pos[valid]
                                    rmse_opt[idx] = np.sqrt(np.mean(dx**2 + dy**2 + dz**2))
                            except (KeyError, IndexError):
                                pass

                        # Compute Magnus force sensitivity as time-weighted integral:
                        # Sum_i ||v_i x w|| * (t_end - t_i) * dt
                        # where v_i is the OPT velocity at each timestep
                        # The (t_end - t_i) term weights earlier forces more heavily
                        has_opt_vel = 'vx_opt' in cols and 'vy_opt' in cols and 'vz_opt' in cols
                        has_spin = 'wx' in cols and 'wy' in cols and 'wz' in cols
                        has_time = 'time' in cols

                        # Compute Drag force sensitivity (only needs velocity and time)
                        if has_opt_vel and has_time and len(data_values) > 1:
                            try:
                                vx_vals = data_values[:, cols.get_loc('vx_opt')]
                                vy_vals = data_values[:, cols.get_loc('vy_opt')]
                                vz_vals = data_values[:, cols.get_loc('vz_opt')]
                                time_vals = data_values[:, cols.get_loc('time')]

                                # Compute time values
                                t_end = time_vals[-1]
                                dt = np.diff(time_vals)

                                # Compute (t_end - t_i) for each point (excluding last)
                                time_to_end = t_end - time_vals[:-1]

                                # Compute drag force sensitivity: Sum ||v||² * (t_end - t_i) * dt
                                vel_mag_sq = vx_vals**2 + vy_vals**2 + vz_vals**2
                                valid_vel_mask = ~np.isnan(vel_mag_sq[:-1])
                                if np.any(valid_vel_mask):
                                    drag_impulse_proxy[idx] = np.sum(
                                        vel_mag_sq[:-1][valid_vel_mask] *
                                        time_to_end[valid_vel_mask] *
                                        dt[valid_vel_mask]
                                    )
                            except (KeyError, IndexError, Exception):
                                pass

                        # Compute Magnus force sensitivity (needs velocity, spin, and time)
                        if has_opt_vel and has_spin and has_time and len(data_values) > 1:
                            try:
                                vx_vals = data_values[:, cols.get_loc('vx_opt')]
                                vy_vals = data_values[:, cols.get_loc('vy_opt')]
                                vz_vals = data_values[:, cols.get_loc('vz_opt')]
                                time_vals = data_values[:, cols.get_loc('time')]

                                # Use mode of spin (constant spin assumption)
                                wx_vals = data_values[:, cols.get_loc('wx')]
                                wy_vals = data_values[:, cols.get_loc('wy')]
                                wz_vals = data_values[:, cols.get_loc('wz')]

                                # Fast first-valid-value lookup
                                # (spin is constant within a segment from the optimizer)
                                def _first_valid(arr):
                                    mask = ~np.isnan(arr)
                                    return arr[mask][0] if np.any(mask) else arr[0]

                                wx_mode = _first_valid(wx_vals)
                                wy_mode = _first_valid(wy_vals)
                                wz_mode = _first_valid(wz_vals)

                                # Compute ||v_i x w|| for each timestep
                                cross_x = vy_vals * wz_mode - vz_vals * wy_mode
                                cross_y = vz_vals * wx_mode - vx_vals * wz_mode
                                cross_z = vx_vals * wy_mode - vy_vals * wx_mode
                                cross_mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

                                # Compute time values
                                t_end = time_vals[-1]
                                dt = np.diff(time_vals)

                                # Compute (t_end - t_i) for each point (excluding last)
                                time_to_end = t_end - time_vals[:-1]

                                # Integrate: sum of ||v_i x w|| * (t_end - t_i) * dt
                                valid_mask = ~np.isnan(cross_mag[:-1])
                                if np.any(valid_mask):
                                    magnus_impulse_proxy[idx] = np.sum(
                                        cross_mag[:-1][valid_mask] *
                                        time_to_end[valid_mask] *
                                        dt[valid_mask]
                                    )
                            except (KeyError, IndexError, Exception):
                                pass

                        # ── Trajectory error metrics (APS vs each trajectory) ──
                        if has_aps_pos:
                            # Ensure APS arrays are available (may already be set by rmse_opt block above)
                            try:
                                x_aps = data_values[:, cols.get_loc('x_aps')]
                                y_aps = data_values[:, cols.get_loc('y_aps')]
                                z_aps = data_values[:, cols.get_loc('z_aps')]
                            except (KeyError, IndexError):
                                x_aps = y_aps = z_aps = None
                            if x_aps is not None:
                                for _ti, (_tname, _pcols, _vcols) in enumerate(_TRAJ_COLS):
                                    _has_pos = all(c in cols for c in _pcols)
                                    if _has_pos:
                                        try:
                                            _xp = data_values[:, cols.get_loc(_pcols[0])]
                                            _yp = data_values[:, cols.get_loc(_pcols[1])]
                                            _zp = data_values[:, cols.get_loc(_pcols[2])]
                                            _v = ~(np.isnan(x_aps) | np.isnan(_xp))
                                            if np.sum(_v) > 0:
                                                _dx = x_aps[_v] - _xp[_v]
                                                _dy = y_aps[_v] - _yp[_v]
                                                _dz = z_aps[_v] - _zp[_v]
                                                _sq = _dx**2 + _dy**2 + _dz**2
                                                _dist = np.sqrt(_sq)
                                                pos_rmse_all[idx, _ti] = np.sqrt(np.mean(_sq))
                                                pos_max_err_all[idx, _ti] = np.max(_dist)
                                                pos_mean_err_all[idx, _ti] = np.mean(_dist)
                                                # MAD-filtered RMSE
                                                _med = np.median(_sq)
                                                _mad = np.median(np.abs(_sq - _med))
                                                if _mad > 0:
                                                    _thr = _med + 3.0 * 1.4826 * _mad
                                                    _inl = _sq <= _thr
                                                    if np.sum(_inl) > 0:
                                                        pos_rmse_mad_all[idx, _ti] = np.sqrt(np.mean(_sq[_inl]))
                                                    else:
                                                        pos_rmse_mad_all[idx, _ti] = pos_rmse_all[idx, _ti]
                                                else:
                                                    pos_rmse_mad_all[idx, _ti] = pos_rmse_all[idx, _ti]
                                        except (KeyError, IndexError):
                                            pass

                                    # Velocity RMSE
                                    _has_vel_aps = 'vx_aps' in cols and 'vy_aps' in cols and 'vz_aps' in cols
                                    _has_vel_t = all(c in cols for c in _vcols)
                                    if _has_vel_aps and _has_vel_t:
                                        try:
                                            _vxo = data_values[:, cols.get_loc('vx_aps')]
                                            _vyo = data_values[:, cols.get_loc('vy_aps')]
                                            _vzo = data_values[:, cols.get_loc('vz_aps')]
                                            _vxt = data_values[:, cols.get_loc(_vcols[0])]
                                            _vyt = data_values[:, cols.get_loc(_vcols[1])]
                                            _vzt = data_values[:, cols.get_loc(_vcols[2])]
                                            _vv = ~(np.isnan(_vxo) | np.isnan(_vxt))
                                            if np.sum(_vv) > 0:
                                                _dvx = _vxo[_vv] - _vxt[_vv]
                                                _dvy = _vyo[_vv] - _vyt[_vv]
                                                _dvz = _vzo[_vv] - _vzt[_vv]
                                                vel_rmse_all[idx, _ti] = np.sqrt(np.mean(_dvx**2 + _dvy**2 + _dvz**2))
                                        except (KeyError, IndexError):
                                            pass

                        # ── GCS spin confidence medians ──
                        if 'w_confidence' in cols:
                            try:
                                _wc = data_values[:, cols.get_loc('w_confidence')]
                                _wc_v = _wc[~np.isnan(_wc)]
                                if len(_wc_v) > 0:
                                    median_w_confidence_arr[idx] = np.median(_wc_v)
                            except (KeyError, IndexError):
                                pass
                        if 'wx_confidence' in cols:
                            try:
                                _wxc = data_values[:, cols.get_loc('wx_confidence')]
                                _wxc_v = _wxc[~np.isnan(_wxc)]
                                if len(_wxc_v) > 0:
                                    median_wx_confidence_arr[idx] = np.median(_wxc_v)
                            except (KeyError, IndexError):
                                pass

                        # ── Spin density: % samples with spin conf > 0.5 AND w_confidence >= 0.5 ──
                        _sc_cols = ['wx_confidence', 'wy_confidence', 'wz_confidence']
                        if all(c in cols for c in _sc_cols):
                            try:
                                _sc = np.column_stack([data_values[:, cols.get_loc(c)] for c in _sc_cols])
                                _sc_min = np.min(_sc, axis=1)
                                _wc_sd = data_values[:, cols.get_loc('w_confidence')] if 'w_confidence' in cols else np.ones(len(data_values))
                                _high = np.sum((_sc_min > 0.5) & (_wc_sd >= 0.5))
                                _exp = max(1.0, durations[idx] * 200.0 + 1)
                                spin_density_arr[idx] = 100.0 * _high / _exp
                            except (KeyError, IndexError):
                                pass

                        # ── Spin variance: std of observed spin magnitude ──
                        _gcs_sp = ['wx_curr', 'wy_curr', 'wz_curr']
                        if not all(c in cols for c in _gcs_sp):
                            _gcs_sp = ['wx_gcs', 'wy_gcs', 'wz_gcs']  # fallback column names
                        if all(c in cols for c in _gcs_sp):
                            try:
                                _wxg = data_values[:, cols.get_loc(_gcs_sp[0])]
                                _wyg = data_values[:, cols.get_loc(_gcs_sp[1])]
                                _wzg = data_values[:, cols.get_loc(_gcs_sp[2])]
                                _wc_sv = data_values[:, cols.get_loc('w_confidence')] if 'w_confidence' in cols else np.ones(len(data_values))
                                _sv = (~(np.isnan(_wxg) | np.isnan(_wyg) | np.isnan(_wzg)) &
                                       (_wc_sv >= 0.5))
                                if np.sum(_sv) > 1:
                                    _wmag = np.sqrt(_wxg[_sv]**2 + _wyg[_sv]**2 + _wzg[_sv]**2)
                                    spin_variance_arr[idx] = np.std(_wmag)
                            except (KeyError, IndexError):
                                pass

                        # ── GCS source (from rally level) ──
                        gcs_sources_list.append(_rally_gcs_source)

                        # ── Shot position within rally ──
                        shot_position_arr[idx] = _shot_position

                        # ── Contact type (from trigger event) ──
                        _ct = fs.trigger_event.type if (fs.trigger_event is not None) else 'unknown'
                        contact_types_list.append(_ct)

                        # ── Mean APS position ──
                        if has_aps_pos:
                            try:
                                _xa = data_values[:, cols.get_loc('x_aps')]
                                _ya = data_values[:, cols.get_loc('y_aps')]
                                _za = data_values[:, cols.get_loc('z_aps')]
                                _xa_v = _xa[~np.isnan(_xa)]
                                _ya_v = _ya[~np.isnan(_ya)]
                                _za_v = _za[~np.isnan(_za)]
                                if len(_xa_v) > 0:
                                    mean_x_aps_arr[idx] = np.mean(_xa_v)
                                if len(_ya_v) > 0:
                                    mean_y_aps_arr[idx] = np.mean(_ya_v)
                                if len(_za_v) > 0:
                                    mean_z_aps_arr[idx] = np.mean(_za_v)
                            except (KeyError, IndexError):
                                pass

                        # ── GCS spin observation metrics ──
                        _gcs_spin_cols = ['wx_gcs', 'wy_gcs', 'wz_gcs']
                        _gt_spin_cols_obs = ['wx', 'wy', 'wz']
                        if all(c in cols for c in _gcs_spin_cols) and all(c in cols for c in _gt_spin_cols_obs):
                            try:
                                _wxg_obs = data_values[:, cols.get_loc('wx')]
                                _wyg_obs = data_values[:, cols.get_loc('wy')]
                                _wzg_obs = data_values[:, cols.get_loc('wz')]
                                _wxc_obs = data_values[:, cols.get_loc('wx_gcs')]
                                _wyc_obs = data_values[:, cols.get_loc('wy_gcs')]
                                _wzc_obs = data_values[:, cols.get_loc('wz_gcs')]
                                _wc_obs = data_values[:, cols.get_loc('w_confidence')] if 'w_confidence' in cols else np.ones(len(data_values))
                                _val_obs = (~(np.isnan(_wxg_obs) | np.isnan(_wyg_obs) | np.isnan(_wzg_obs) |
                                              np.isnan(_wxc_obs) | np.isnan(_wyc_obs) | np.isnan(_wzc_obs)) &
                                            (_wc_obs >= 0.5))
                                _nv = int(np.sum(_val_obs))
                                n_gcs_samples_arr[idx] = _nv
                                if _nv > 0:
                                    _smgt = np.sqrt(_wxg_obs[_val_obs]**2 + _wyg_obs[_val_obs]**2 + _wzg_obs[_val_obs]**2)
                                    _smgc = np.sqrt(_wxc_obs[_val_obs]**2 + _wyc_obs[_val_obs]**2 + _wzc_obs[_val_obs]**2)
                                    mean_delta_spin_arr[idx] = float(np.mean(np.abs(_smgt - _smgc)))
                                    spin_obs_var_arr[idx] = float(np.var(_smgc))
                            except (KeyError, IndexError):
                                pass

                        idx += 1

    # Truncate arrays to actual size (some segments may have been skipped due to filtering)
    actual_n_segments = idx
    if actual_n_segments < n_segments:
        match_indices = match_indices[:actual_n_segments]
        player_types = player_types[:actual_n_segments]
        game_ids = game_ids[:actual_n_segments]
        rally_ids = rally_ids[:actual_n_segments]
        segment_start_times = segment_start_times[:actual_n_segments]
        vx_gt200 = vx_gt200[:actual_n_segments]
        vy_gt200 = vy_gt200[:actual_n_segments]
        vz_gt200 = vz_gt200[:actual_n_segments]
        has_gt200 = has_gt200[:actual_n_segments]
        vx_opt = vx_opt[:actual_n_segments]
        vy_opt = vy_opt[:actual_n_segments]
        vz_opt = vz_opt[:actual_n_segments]
        has_opt = has_opt[:actual_n_segments]
        wx = wx[:actual_n_segments]
        wy = wy[:actual_n_segments]
        wz = wz[:actual_n_segments]
        cd_opt_arr = cd_opt_arr[:actual_n_segments]
        cm_opt_arr = cm_opt_arr[:actual_n_segments]
        vx_opt_first = vx_opt_first[:actual_n_segments]
        vy_opt_first = vy_opt_first[:actual_n_segments]
        vz_opt_first = vz_opt_first[:actual_n_segments]
        wx_first = wx_first[:actual_n_segments]
        wy_first = wy_first[:actual_n_segments]
        wz_first = wz_first[:actual_n_segments]
        v_eff_drag = v_eff_drag[:actual_n_segments]
        v_eff_magnus = v_eff_magnus[:actual_n_segments]
        w_eff_perp_magnus = w_eff_perp_magnus[:actual_n_segments]
        has_aero = has_aero[:actual_n_segments]
        min_confidence = min_confidence[:actual_n_segments]
        has_confidence = has_confidence[:actual_n_segments]
        has_inconsistent_confidence = has_inconsistent_confidence[:actual_n_segments]
        conf_last_point = conf_last_point[:actual_n_segments]
        durations = durations[:actual_n_segments]
        rmse_opt = rmse_opt[:actual_n_segments]
        magnus_impulse_proxy = magnus_impulse_proxy[:actual_n_segments]
        drag_impulse_proxy = drag_impulse_proxy[:actual_n_segments]
        pos_rmse_all = pos_rmse_all[:actual_n_segments]
        pos_max_err_all = pos_max_err_all[:actual_n_segments]
        pos_mean_err_all = pos_mean_err_all[:actual_n_segments]
        vel_rmse_all = vel_rmse_all[:actual_n_segments]
        pos_rmse_mad_all = pos_rmse_mad_all[:actual_n_segments]
        median_w_confidence_arr = median_w_confidence_arr[:actual_n_segments]
        median_wx_confidence_arr = median_wx_confidence_arr[:actual_n_segments]
        spin_density_arr = spin_density_arr[:actual_n_segments]
        spin_variance_arr = spin_variance_arr[:actual_n_segments]
        mean_delta_spin_arr = mean_delta_spin_arr[:actual_n_segments]
        spin_obs_var_arr = spin_obs_var_arr[:actual_n_segments]
        n_gcs_samples_arr = n_gcs_samples_arr[:actual_n_segments]
        shot_position_arr = shot_position_arr[:actual_n_segments]
        mean_x_aps_arr = mean_x_aps_arr[:actual_n_segments]
        mean_y_aps_arr = mean_y_aps_arr[:actual_n_segments]
        mean_z_aps_arr = mean_z_aps_arr[:actual_n_segments]
        print(f"Note: {n_segments - actual_n_segments} segments skipped due to confidence filtering")

    return FlightSegmentCache(
        n_segments=actual_n_segments,
        match_indices=match_indices,
        player_types=player_types,
        flight_segments=flight_segments,
        shots=shots_list,
        rallies=rallies_list,
        match_dates=match_dates,
        match_players=match_players,
        game_names=game_names,
        game_ids=game_ids,
        rally_ids=rally_ids,
        segment_start_times=segment_start_times,
        match_files=match_files,
        log_identifiers=log_identifiers,
        vx_gt200=vx_gt200,
        vy_gt200=vy_gt200,
        vz_gt200=vz_gt200,
        has_gt200=has_gt200,
        vx_opt=vx_opt,
        vy_opt=vy_opt,
        vz_opt=vz_opt,
        has_opt=has_opt,
        wx=wx,
        wy=wy,
        wz=wz,
        cd_opt=cd_opt_arr,
        cm_opt=cm_opt_arr,
        vx_opt_first=vx_opt_first,
        vy_opt_first=vy_opt_first,
        vz_opt_first=vz_opt_first,
        wx_first=wx_first,
        wy_first=wy_first,
        wz_first=wz_first,
        v_eff_drag=v_eff_drag,
        v_eff_magnus=v_eff_magnus,
        w_eff_perp_magnus=w_eff_perp_magnus,
        has_aero=has_aero,
        min_confidence=min_confidence,
        has_confidence=has_confidence,
        has_inconsistent_confidence=has_inconsistent_confidence,
        conf_last_point=conf_last_point,
        durations=durations,
        rmse_opt=rmse_opt,
        magnus_impulse_proxy=magnus_impulse_proxy,
        drag_impulse_proxy=drag_impulse_proxy,
        pos_rmse_all=pos_rmse_all,
        pos_max_err_all=pos_max_err_all,
        pos_mean_err_all=pos_mean_err_all,
        vel_rmse_all=vel_rmse_all,
        pos_rmse_mad_all=pos_rmse_mad_all,
        median_w_confidence=median_w_confidence_arr,
        median_wx_confidence=median_wx_confidence_arr,
        spin_density=spin_density_arr,
        spin_variance=spin_variance_arr,
        gcs_sources=gcs_sources_list,
        mean_delta_spin=mean_delta_spin_arr,
        spin_obs_var=spin_obs_var_arr,
        n_gcs_samples=n_gcs_samples_arr,
        shot_position=shot_position_arr,
        mean_x_aps=mean_x_aps_arr,
        mean_y_aps=mean_y_aps_arr,
        mean_z_aps=mean_z_aps_arr,
        contact_types=contact_types_list,
        # ── Derived magnitudes (computed once from cached first/last-point arrays) ──
        vel_mag_first=np.sqrt(vx_opt_first**2 + vy_opt_first**2 + vz_opt_first**2),
        spin_mag_first=np.sqrt(wx_first**2 + wy_first**2 + wz_first**2),
        spin_ratio_first=np.where(
            np.sqrt(vx_opt_first**2 + vy_opt_first**2 + vz_opt_first**2) > 1e-9,
            0.02 * np.sqrt(wx_first**2 + wy_first**2 + wz_first**2)
            / np.sqrt(vx_opt_first**2 + vy_opt_first**2 + vz_opt_first**2),
            0.0,
        ),
        vel_mag_last=np.sqrt(
            np.where(has_opt, vx_opt, vx_gt200)**2
            + np.where(has_opt, vy_opt, vy_gt200)**2
            + np.where(has_opt, vz_opt, vz_gt200)**2,
        ),
        spin_mag_last=np.sqrt(wx**2 + wy**2 + wz**2),
        match_policies=match_policies_list,
        rally_rows=rally_rows_list,
        racket_contact_data={
            'vx_pre': np.array(rc_vx_pre), 'vy_pre': np.array(rc_vy_pre), 'vz_pre': np.array(rc_vz_pre),
            'wx_pre': np.array(rc_wx_pre), 'wy_pre': np.array(rc_wy_pre), 'wz_pre': np.array(rc_wz_pre),
            'vx_post': np.array(rc_vx_post), 'vy_post': np.array(rc_vy_post), 'vz_post': np.array(rc_vz_post),
            'wx_post': np.array(rc_wx_post), 'wy_post': np.array(rc_wy_post), 'wz_post': np.array(rc_wz_post),
            'player': np.array(rc_player, dtype=int), 'policy': rc_policy,
            'confidence': np.array(rc_confidence), 'max_rmse_opt': np.array(rc_max_rmse),
            'metadata_list': rc_metadata, 'shot_list': rc_shot, 'rally_list': rc_rally,
            'fs_pre_list': rc_fs_pre, 'fs_post_list': rc_fs_post,
        } if rc_vx_pre else None,
        table_contact_data={
            'vx_pre': np.array(tc_vx_pre), 'vy_pre': np.array(tc_vy_pre), 'vz_pre': np.array(tc_vz_pre),
            'wx_pre': np.array(tc_wx_pre), 'wy_pre': np.array(tc_wy_pre), 'wz_pre': np.array(tc_wz_pre),
            'vx_post': np.array(tc_vx_post), 'vy_post': np.array(tc_vy_post), 'vz_post': np.array(tc_vz_post),
            'wx_post': np.array(tc_wx_post), 'wy_post': np.array(tc_wy_post), 'wz_post': np.array(tc_wz_post),
            **{k: np.array(v) for k, v in tc_model_lists.items()},
            'metadata_list': tc_metadata, 'shot_list': tc_shot, 'rally_list': tc_rally,
            'fs_pre_list': tc_fs_pre, 'fs_post_list': tc_fs_post,
            'segment_index_list': tc_seg_idx,
            'confidence': np.array(tc_confidence), 'max_rmse_opt': np.array(tc_max_rmse),
        } if tc_vx_pre else None,
        rcm_data={
            k: np.array(v) for k, v in rcm_arrays.items()
        } | {
            'player': np.array(rcm_player, dtype=int),
            'shot_type': np.array(rcm_shot_type, dtype=int),
            'point_winner': np.array(rcm_point_winner, dtype=int),
            'metadata_list': rcm_metadata, 'shot_list': rcm_shot, 'rally_list': rcm_rally,
            'fs_pre_list': rcm_fs_pre, 'fs_post_list': rcm_fs_post,
            'confidence': np.array(rcm_confidence), 'max_rmse_opt': np.array(rcm_max_rmse),
            'worst_spin_density': np.array(rcm_worst_spin_density),
            'worst_spin_variance': np.array(rcm_worst_spin_variance),
        } if rcm_arrays['vx_pre'] else None,
        base_folder=match_collection.base_folder,
    )


class DataProcessor:
    """Processes match data and extracts features for visualization"""

    CACHE_VERSION = 28  # Bump when data format changes (28: fix OOB in racket contact HDF5 enrichment)

    def __init__(self):
        self.match_collection: MatchCollection = None
        self._cache: Optional[FlightSegmentCache] = None

    # ------------------------------------------------------------------
    # Pickle-based cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_cache_key(data_folder: pathlib.Path, *, gcs_fallback: bool = False) -> str:
        """Compute a hash from HDF5 file paths, sizes, modification times,
        the extracted.csv mtime, and the gcs_fallback flag."""
        h5_files = sorted(data_folder.rglob("*.h5"))
        h5_files = [
            f for f in h5_files
            if "optitrack" not in f.name.lower() and "with_opt" not in f.name.lower()
        ]
        hasher = hashlib.md5()
        hasher.update(str(DataProcessor.CACHE_VERSION).encode())
        hasher.update(f"gcs_fallback={gcs_fallback}".encode())
        for f in h5_files:
            hasher.update(str(f).encode())
            st = f.stat()
            hasher.update(str(st.st_mtime_ns).encode())
            hasher.update(str(st.st_size).encode())
        # Also include extracted.csv so regenerating it invalidates the cache
        csv_file = data_folder / "extracted.csv"
        if csv_file.exists():
            csv_st = csv_file.stat()
            hasher.update(b"extracted.csv")
            hasher.update(str(csv_st.st_mtime_ns).encode())
            hasher.update(str(csv_st.st_size).encode())
        # Also include extracted_TCM.csv so regenerating it invalidates the cache
        tcm_csv_file = data_folder / "extracted_TCM.csv"
        if tcm_csv_file.exists():
            tcm_csv_st = tcm_csv_file.stat()
            hasher.update(b"extracted_TCM.csv")
            hasher.update(str(tcm_csv_st.st_mtime_ns).encode())
            hasher.update(str(tcm_csv_st.st_size).encode())
        return hasher.hexdigest()

    @staticmethod
    def _cache_path(data_folder: pathlib.Path) -> pathlib.Path:
        return data_folder / ".data_plotter_cache.pkl"

    def load_data(self, data_folder: pathlib.Path, gcs_fallback: bool = False) -> MatchCollection:
        """
        Load match collection from HDF5 files.

        Uses a pickle cache to avoid re-reading HDF5 on repeated launches.
        The cache is invalidated when any HDF5 file changes (size/mtime).

        Args:
            data_folder: Path to folder containing HDF5 files.
            gcs_fallback: If True, fall back to gcs_filtered/gcs when gcs_offline is
                not available.  If False (default), rallies without gcs_offline
                are excluded.
        """
        data_folder = pathlib.Path(data_folder)
        cache_file = self._cache_path(data_folder)
        cache_key = self._compute_cache_key(data_folder, gcs_fallback=gcs_fallback)

        # --- Try loading from pickle cache ---
        if cache_file.exists():
            try:
                t0 = _time.time()
                with open(cache_file, "rb") as fh:
                    cached = pickle.load(fh)
                if (
                    isinstance(cached, dict)
                    and cached.get("key") == cache_key
                ):
                    self.match_collection = cached["match_collection"]
                    self.match_collection.gcs_fallback = gcs_fallback
                    self._cache = cached["flight_segment_cache"]
                    elapsed = _time.time() - t0
                    print(
                        f"Loaded from cache in {elapsed:.2f}s "
                        f"({self._cache.n_segments} segments)"
                    )
                    # Always re-run CSV enrichment (CSV may have changed
                    # independently of the HDF5 files).
                    self.match_collection.enrich_from_csv()
                    return self.match_collection
                else:
                    print("Cache key mismatch — reloading from HDF5.")
            except Exception as e:
                print(f"Cache load failed ({e}) — reloading from HDF5.")

        # --- Full HDF5 load ---
        self.match_collection = MatchCollection(data_folder, gcs_fallback=gcs_fallback)

        # Build the flight segment cache for fast extraction
        print("Building flight segment cache...")
        t0 = _time.time()
        self._cache = build_flight_segment_cache(self.match_collection)
        print(f"Cache built in {_time.time()-t0:.2f}s ({self._cache.n_segments} segments)")

        # --- Save pickle cache ---
        try:
            t0 = _time.time()
            with open(cache_file, "wb") as fh:
                pickle.dump(
                    {
                        "key": cache_key,
                        "match_collection": self.match_collection,
                        "flight_segment_cache": self._cache,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"Saved cache to {cache_file} in {_time.time()-t0:.2f}s")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

        # Report segments that had inconsistent confidence (anomalous samples were filtered)
        n_inconsistent = np.sum(self._cache.has_inconsistent_confidence)
        if n_inconsistent > 0 and self._cache.n_segments > 0:
            print(f"Note: {n_inconsistent} segments ({100*n_inconsistent/self._cache.n_segments:.1f}%) had samples "
                  f"with anomalous confidence filtered out.")

        # Report Magnus force sensitivity range
        valid_magnus = self._cache.magnus_impulse_proxy[~np.isnan(self._cache.magnus_impulse_proxy)]
        if len(valid_magnus) > 0:
            print(f"Magnus force sensitivity range: [{valid_magnus.min():.1f}, {valid_magnus.max():.1f}] "
                  f"(mean: {valid_magnus.mean():.1f}, median: {np.median(valid_magnus):.1f})")

        # Report Drag force sensitivity range
        valid_drag = self._cache.drag_impulse_proxy[~np.isnan(self._cache.drag_impulse_proxy)]
        if len(valid_drag) > 0:
            print(f"Drag force sensitivity range: [{valid_drag.min():.1f}, {valid_drag.max():.1f}] "
                  f"(mean: {valid_drag.mean():.1f}, median: {np.median(valid_drag):.1f})")

        # Print data validation report
        self.print_data_validation_report(self.match_collection)

        return self.match_collection

    def load_data_from_json(self, json_path: str, gcs_fallback: bool = False) -> MatchCollection:
        """Load specific rallies from a JSON manifest into a MatchCollection.

        The JSON file should contain a list of objects with keys:
        ``hdf5_filepath``, ``game_id``, ``rally_id``, and optionally
        ``sequence_number``.

        Args:
            json_path: Path to the JSON manifest file.
            gcs_fallback: Fall back to gcs_filtered/gcs when gcs_offline
                is not available.
        """
        self.match_collection = MatchCollection.from_json(json_path, gcs_fallback=gcs_fallback)

        # Build the flight segment cache for fast extraction
        print("Building flight segment cache...")
        t0 = _time.time()
        self._cache = build_flight_segment_cache(self.match_collection)
        print(f"Cache built in {_time.time()-t0:.2f}s ({self._cache.n_segments} segments)")

        # Print data validation report
        self.print_data_validation_report(self.match_collection)

        return self.match_collection

    @property
    def cache(self) -> Optional[FlightSegmentCache]:
        """Get or build the flight segment cache"""
        if self._cache is None and self.match_collection is not None:
            self._cache = build_flight_segment_cache(self.match_collection)
        return self._cache

    def print_data_validation_report(self, match_collection: MatchCollection):
        """
        Print comprehensive data validation report showing what data is excluded

        Args:
            match_collection: MatchCollection object
        """
        print("\n" + "="*80)
        print("DATA VALIDATION REPORT")
        print("="*80)

        # Initialize counters
        total_segments = 0
        segments_too_short = 0
        segments_missing_gt200 = 0
        segments_missing_opt = 0
        segments_with_nan = 0
        segments_valid_gt200 = 0
        segments_valid_opt = 0

        total_rallies = 0
        rallies_missing_gt200 = 0
        rallies_missing_opt = 0

        # Iterate through all data
        for match in match_collection.matches:
            for game in match.games:
                for rally in game.rallies:
                    total_rallies += 1
                    rally_has_gt200 = False
                    rally_has_opt = False

                    for shot in rally.shots:
                        for fs in shot.flight_segments:
                            total_segments += 1

                            # Check 1: Length > 2
                            if len(fs.data) <= 2:
                                segments_too_short += 1
                                continue

                            # Check 2: Has GT200 data (vx_gt200 and wx)
                            has_gt200 = "vx_gt200" in fs.data.columns and "wx" in fs.data.columns
                            if not has_gt200:
                                segments_missing_gt200 += 1
                            else:
                                rally_has_gt200 = True

                                # Check for NaN in GT200 data
                                last_idx = fs.data.index[-1]
                                vx = fs.data.loc[last_idx, "vx_gt200"]
                                vy = fs.data.loc[last_idx, "vy_gt200"]
                                vz = fs.data.loc[last_idx, "vz_gt200"]
                                wx = fs.data.loc[last_idx, "wx"]
                                wy = fs.data.loc[last_idx, "wy"]
                                wz = fs.data.loc[last_idx, "wz"]

                                if pd.isna(vx) or pd.isna(vy) or pd.isna(vz) or pd.isna(wx) or pd.isna(wy) or pd.isna(wz):
                                    segments_with_nan += 1
                                else:
                                    segments_valid_gt200 += 1

                            # Check 3: Has OPT data (aerodynamics)
                            has_opt = "cd_opt" in fs.data.columns and "vx_opt" in fs.data.columns
                            if not has_opt:
                                segments_missing_opt += 1
                            else:
                                rally_has_opt = True

                                # Check for valid OPT data
                                try:
                                    idx = fs.data.index[0]
                                    cd_opt = fs.data.loc[idx, "cd_opt"]
                                    vx_opt = fs.data.loc[idx, "vx_opt"]
                                    vy_opt = fs.data.loc[idx, "vy_opt"]
                                    vz_opt = fs.data.loc[idx, "vz_opt"]

                                    # Get spin data
                                    if "wx_opt" in fs.data.columns:
                                        wx = fs.data.loc[idx, "wx_opt"]
                                        wy = fs.data.loc[idx, "wy_opt"]
                                        wz = fs.data.loc[idx, "wz_opt"]
                                    elif "wx" in fs.data.columns:
                                        wx = fs.data.loc[idx, "wx"]
                                        wy = fs.data.loc[idx, "wy"]
                                        wz = fs.data.loc[idx, "wz"]
                                    else:
                                        continue

                                    if not (pd.isna(cd_opt) or pd.isna(vx_opt) or pd.isna(vy_opt) or pd.isna(vz_opt) or
                                           pd.isna(wx) or pd.isna(wy) or pd.isna(wz)):
                                        vel_magnitude = np.sqrt(vx_opt**2 + vy_opt**2 + vz_opt**2)
                                        if vel_magnitude > 1e-9:
                                            segments_valid_opt += 1
                                except:
                                    pass

                    # Rally-level tracking
                    if not rally_has_gt200:
                        rallies_missing_gt200 += 1
                    if not rally_has_opt:
                        rallies_missing_opt += 1

        # Print Flight Segment Statistics
        print("\nFLIGHT SEGMENT STATISTICS:")
        print(f"  Total flight segments: {total_segments}")

        if total_segments == 0:
            print("  (no flight segments — skipping detailed breakdown)")
        else:
            print(f"\n  Segments too short (≤2 points):")
            print(f"    Count: {segments_too_short}")
            print(f"    Percentage: {segments_too_short/total_segments*100:.1f}%")

            print(f"\n  Segments missing GT200 data (vx_gt200, wx):")
            print(f"    Count: {segments_missing_gt200}")
            print(f"    Percentage: {segments_missing_gt200/total_segments*100:.1f}%")

            print(f"\n  Segments with NaN values in GT200 data:")
            print(f"    Count: {segments_with_nan}")
            print(f"    Percentage: {segments_with_nan/total_segments*100:.1f}%")

            print(f"\n  Segments with valid GT200 data (usable for Scatter/Histogram plots):")
            print(f"    Count: {segments_valid_gt200}")
            print(f"    Percentage: {segments_valid_gt200/total_segments*100:.1f}%")

            print(f"\n  Segments missing OPT data (cd_opt, vx_opt, etc.):")
            print(f"    Count: {segments_missing_opt}")
            print(f"    Percentage: {segments_missing_opt/total_segments*100:.1f}%")

            print(f"\n  Segments with valid OPT data (usable for Drag/Magnus plots):")
            print(f"    Count: {segments_valid_opt}")
            print(f"    Percentage: {segments_valid_opt/total_segments*100:.1f}%")

        # Print Rally Statistics
        print("\n" + "-"*80)
        print("RALLY STATISTICS:")
        print(f"  Total rallies: {total_rallies}")

        if total_rallies == 0:
            print("  (no rallies — skipping detailed breakdown)")
        else:
            print(f"\n  Rallies with NO segments having GT200 data:")
            print(f"    Count: {rallies_missing_gt200}")
            print(f"    Percentage: {rallies_missing_gt200/total_rallies*100:.1f}%")

            print(f"\n  Rallies with NO segments having OPT data:")
            print(f"    Count: {rallies_missing_opt}")
            print(f"    Percentage: {rallies_missing_opt/total_rallies*100:.1f}%")

        # Print Plot-specific Segment Counts (using cache)
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache
            print("\n" + "-"*80)
            print("PLOT-SPECIFIC SEGMENT AVAILABILITY:")
            print(f"  (Segments available when all filters set to maximum extent)")

            # Scatter Plot: requires has_opt (vx_opt, vy_opt, vz_opt + wx, wy, wz at last point)
            scatter_robot = int(np.sum((cache.player_types == 1) & cache.has_opt))
            scatter_player = int(np.sum((cache.player_types == 2) & cache.has_opt))
            scatter_total = scatter_robot + scatter_player
            print(f"\n  Scatter Plot (has_opt: vx_opt + spin at last point):")
            print(f"    Robot: {scatter_robot}, Player: {scatter_player}, Total: {scatter_total}")

            # Drag Coefficient: requires has_aero + valid cd_opt + non-zero velocity
            # Compute velocity magnitude at first point for filtering
            vel_mag_first = np.sqrt(cache.vx_opt_first**2 + cache.vy_opt_first**2 + cache.vz_opt_first**2)
            valid_vel = vel_mag_first > 1e-9

            drag_mask_robot = (cache.player_types == 1) & cache.has_aero & ~np.isnan(cache.cd_opt) & valid_vel
            drag_mask_player = (cache.player_types == 2) & cache.has_aero & ~np.isnan(cache.cd_opt) & valid_vel
            drag_robot = int(np.sum(drag_mask_robot))
            drag_player = int(np.sum(drag_mask_player))
            drag_total = drag_robot + drag_player
            print(f"\n  Drag Coefficient Plot (has_aero + valid cd_opt + non-zero velocity):")
            print(f"    Robot: {drag_robot}, Player: {drag_player}, Total: {drag_total}")

            # Magnus Coefficient: requires has_aero + valid cm_opt + non-zero velocity
            magnus_mask_robot = (cache.player_types == 1) & cache.has_aero & ~np.isnan(cache.cm_opt) & valid_vel
            magnus_mask_player = (cache.player_types == 2) & cache.has_aero & ~np.isnan(cache.cm_opt) & valid_vel
            magnus_robot = int(np.sum(magnus_mask_robot))
            magnus_player = int(np.sum(magnus_mask_player))
            magnus_total = magnus_robot + magnus_player
            print(f"\n  Magnus Coefficient Plot (has_aero + valid cm_opt + non-zero velocity):")
            print(f"    Robot: {magnus_robot}, Player: {magnus_player}, Total: {magnus_total}")

            # Show why drag and magnus might differ
            if drag_total != magnus_total:
                cd_valid = ~np.isnan(cache.cd_opt) & cache.has_aero & valid_vel
                cm_valid = ~np.isnan(cache.cm_opt) & cache.has_aero & valid_vel
                cd_only = int(np.sum(cd_valid & ~cm_valid))
                cm_only = int(np.sum(cm_valid & ~cd_valid))
                print(f"\n  Drag/Magnus count difference:")
                print(f"    Segments with valid cd_opt but NaN cm_opt: {cd_only}")
                print(f"    Segments with valid cm_opt but NaN cd_opt: {cm_only}")

            # Confidence statistics
            print(f"\n  CONFIDENCE STATISTICS:")
            conf = cache.min_confidence
            conf_gt_0 = int(np.sum(conf > 0))
            conf_gt_50 = int(np.sum(conf > 0.5))
            conf_gt_90 = int(np.sum(conf > 0.9))
            conf_eq_1 = int(np.sum(conf == 1.0))
            print(f"    Segments with confidence > 0:   {conf_gt_0} ({conf_gt_0/cache.n_segments*100:.1f}%)")
            print(f"    Segments with confidence > 0.5: {conf_gt_50} ({conf_gt_50/cache.n_segments*100:.1f}%)")
            print(f"    Segments with confidence > 0.9: {conf_gt_90} ({conf_gt_90/cache.n_segments*100:.1f}%)")
            print(f"    Segments with confidence = 1.0: {conf_eq_1} ({conf_eq_1/cache.n_segments*100:.1f}%)")

            # For drag/magnus specifically (filtered segments)
            drag_conf = cache.min_confidence[drag_mask_robot | drag_mask_player]
            if len(drag_conf) > 0:
                drag_conf_gt_0 = int(np.sum(drag_conf > 0))
                drag_conf_gt_50 = int(np.sum(drag_conf > 0.5))
                print(f"\n    Among Drag Plot segments ({drag_total} total):")
                print(f"      With confidence > 0:   {drag_conf_gt_0} ({drag_conf_gt_0/drag_total*100:.1f}%)")
                print(f"      With confidence > 0.5: {drag_conf_gt_50} ({drag_conf_gt_50/drag_total*100:.1f}%)")

            magnus_conf = cache.min_confidence[magnus_mask_robot | magnus_mask_player]
            if len(magnus_conf) > 0:
                magnus_conf_gt_0 = int(np.sum(magnus_conf > 0))
                magnus_conf_gt_50 = int(np.sum(magnus_conf > 0.5))
                print(f"\n    Among Magnus Plot segments ({magnus_total} total):")
                print(f"      With confidence > 0:   {magnus_conf_gt_0} ({magnus_conf_gt_0/magnus_total*100:.1f}%)")
                print(f"      With confidence > 0.5: {magnus_conf_gt_50} ({magnus_conf_gt_50/magnus_total*100:.1f}%)")

            # Explain difference
            print(f"\n  Note: Drag/Magnus require aerodynamics data at the FIRST point,")
            print(f"        Scatter requires optimizer velocity + spin at the LAST point.")

        # Table Contact count (computed at runtime, not cached)
        try:
            table_contact_data = self.extract_table_contact_data(match_collection, player_type=None)
            table_contact_count = len(table_contact_data.get('vx_pre', []))
            print(f"\n  Table Contact Plot (consecutive segments with table bounce event):")
            print(f"    Total: {table_contact_count}")
        except Exception as e:
            print(f"\n  Table Contact Plot: Error computing count - {e}")

        print("\n" + "="*80 + "\n")

    def extract_spin_speed_data(
        self, match_collection: MatchCollection, player_type: str = "robot", enabled_indices: List[int] = None
    ) -> Tuple[
        List[float], List[float], List[FlightSegment], List[Dict[str, Any]], List[Any], List[Any], List[Any], List[Any], List[float], List[float]
    ]:
        """
        Extract spin and speed data from all flight segments

        Args:
            match_collection: MatchCollection object
            player_type: "robot" or "player" to filter by player type
            enabled_indices: List of match indices to include (None = all matches)

        Returns:
            Tuple of (speeds, spins, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list, rmse_opt_list)
            metadata_list contains dicts with: match_date, match_player, game_name, game_id, rally_id
            shot_data_list contains combined data for entire shot
            rally_data_list contains rally DataFrame
            shot_object_list contains Shot objects
            rally_object_list contains Rally objects
            confidence_list contains minimum confidence across 6 values (x,y,z,wx,wy,wz) for each point
            rmse_opt_list contains RMSE between x_opt and x_aps for each flight segment (meters, NaN if not available)
        """
        # Use cache if available
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache

            # Build filter mask — prefer OPT velocity, fall back to GT200
            player_filter = 1 if player_type == "robot" else 2
            mask = (cache.player_types == player_filter) & (cache.has_opt | cache.has_gt200)

            if enabled_indices is not None:
                # Use Numba-accelerated mask building
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                match_mask = build_enabled_mask(cache.match_indices, enabled_arr)
                mask = mask & match_mask

            # Get indices where mask is True
            indices = np.where(mask)[0]

            # For each segment, use OPT velocity if available, otherwise GT200
            has_opt_masked = cache.has_opt[mask]
            vx = np.where(has_opt_masked, cache.vx_opt[mask], cache.vx_gt200[mask])
            vy = np.where(has_opt_masked, cache.vy_opt[mask], cache.vy_gt200[mask])
            vz = np.where(has_opt_masked, cache.vz_opt[mask], cache.vz_gt200[mask])
            wx_arr = cache.wx[mask]
            wy_arr = cache.wy[mask]
            wz_arr = cache.wz[mask]

            # Use Numba-accelerated magnitude calculations
            speeds = compute_magnitudes_3d(vx, vy, vz).tolist()
            spins = compute_magnitudes_3d(wx_arr, wy_arr, wz_arr).tolist()
            confidence_list = cache.min_confidence[mask].tolist()

            # Extract object references (list comprehension over filtered indices)
            flight_segments = [cache.flight_segments[i] for i in indices]
            shot_object_list = [cache.shots[i] for i in indices]
            rally_object_list = [cache.rallies[i] for i in indices]

            # Build metadata list
            metadata_list = []
            for i in indices:
                metadata = {
                    "match_date": cache.match_dates[i],
                    "match_player": cache.match_players[i],
                    "game_name": cache.game_names[i],
                    "game_id": int(cache.game_ids[i]),
                    "rally_id": int(cache.rally_ids[i]),
                    "segment_start_time": cache.segment_start_times[i],
                    "match_file": cache.match_files[i],
                    "base_folder": cache.base_folder,
                    "log_identifier": cache.log_identifiers[i],
                }
                metadata_list.append(metadata)

            # Pass Shot/Rally objects — shot_data/rally_data DataFrames built on demand at click time
            shot_data_list = [None] * len(indices)
            rally_data_list = [None] * len(indices)
            # Extract RMSE OPT values from cache
            rmse_opt_list = cache.rmse_opt[mask].tolist()

            return (
                speeds,
                spins,
                flight_segments,
                metadata_list,
                shot_data_list,
                rally_data_list,
                shot_object_list,
                rally_object_list,
                confidence_list,
                rmse_opt_list,
            )

        # Fallback to original implementation if no cache
        speeds = []
        spins = []
        flight_segments = []
        metadata_list = []
        shot_data_list = []
        rally_data_list = []
        shot_object_list = []
        rally_object_list = []
        confidence_list = []
        rmse_opt_list = []

        player_filter = 1 if player_type == "robot" else 2
        confidence_cols = SPIN_CONFIDENCE_COLS

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            match_date_str = str(match.date)
            match_player = match.player
            match_file = match.file

            for game in match.games:
                game_name = game.game_name
                game_id = game.game_id

                for rally in game.rallies:
                    rally_id = rally.rally_id
                    rally_data = rally.rally

                    for shot in rally.shots:
                        if shot.player != player_filter:
                            continue

                        shot_df = None

                        for fs in shot.flight_segments:
                            if len(fs.data) <= 2:
                                continue

                            cols = fs.data.columns
                            # Prefer vx_opt, fall back to vx_gt200
                            if "vx_opt" in cols:
                                vel_col_names = ['vx_opt', 'vy_opt', 'vz_opt']
                            elif "vx_gt200" in cols:
                                vel_col_names = ['vx_gt200', 'vy_gt200', 'vz_gt200']
                            else:
                                continue
                            if "wx" not in cols:
                                continue

                            try:
                                vel_cols = [cols.get_loc(c) for c in vel_col_names]
                                spin_cols = [cols.get_loc(c) for c in ['wx', 'wy', 'wz']]
                            except KeyError:
                                continue

                            last_row = fs.data.values[-1]
                            vx, vy, vz = last_row[vel_cols]
                            wx, wy, wz = last_row[spin_cols]

                            values = np.array([vx, vy, vz, wx, wy, wz])
                            if np.any(np.isnan(values)):
                                continue

                            speed = np.sqrt(vx**2 + vy**2 + vz**2)
                            spin = np.sqrt(wx**2 + wy**2 + wz**2)

                            speeds.append(speed)
                            spins.append(spin)
                            flight_segments.append(fs)

                            min_confidence = 1.0
                            if all(col in cols for col in confidence_cols):
                                try:
                                    conf_col_indices = [cols.get_loc(c) for c in confidence_cols]
                                    conf_data = fs.data.values[:, conf_col_indices]
                                    min_confidence = np.nanmin(conf_data)
                                    if np.isnan(min_confidence):
                                        min_confidence = 1.0
                                except (KeyError, IndexError):
                                    pass
                            confidence_list.append(min_confidence)

                            metadata = {
                                "match_date": match_date_str,
                                "match_player": match_player,
                                "game_name": game_name,
                                "game_id": game_id,
                                "rally_id": rally_id,
                                "segment_start_time": fs.data.values[0, cols.get_loc("time")],
                                "match_file": match_file,
                                "base_folder": match_collection.base_folder,
                                "log_identifier": getattr(rally, 'log_identifier', None) or '',
                            }
                            metadata_list.append(metadata)

                            if shot_df is None:
                                shot_df = pd.concat(
                                    [seg.data for seg in shot.flight_segments], ignore_index=False
                                )
                            shot_data_list.append(shot_df)
                            rally_data_list.append(rally_data)
                            shot_object_list.append(shot)
                            rally_object_list.append(rally)

                            # Compute RMSE OPT (position RMSE between x_aps and x_opt)
                            rmse_opt_val = np.nan
                            if all(c in cols for c in ['x_aps', 'y_aps', 'z_aps', 'x_opt', 'y_opt', 'z_opt']):
                                x_aps = fs.data['x_aps'].values
                                y_aps = fs.data['y_aps'].values
                                z_aps = fs.data['z_aps'].values
                                x_opt = fs.data['x_opt'].values
                                y_opt = fs.data['y_opt'].values
                                z_opt = fs.data['z_opt'].values
                                valid = ~(np.isnan(x_aps) | np.isnan(x_opt))
                                if np.sum(valid) > 0:
                                    dx = x_aps[valid] - x_opt[valid]
                                    dy = y_aps[valid] - y_opt[valid]
                                    dz = z_aps[valid] - z_opt[valid]
                                    rmse_opt_val = np.sqrt(np.mean(dx**2 + dy**2 + dz**2))
                            rmse_opt_list.append(rmse_opt_val)

        return (
            speeds,
            spins,
            flight_segments,
            metadata_list,
            shot_data_list,
            rally_data_list,
            shot_object_list,
            rally_object_list,
            confidence_list,
            rmse_opt_list,
        )

    def get_statistics(self, match_collection: MatchCollection) -> dict:
        """
        Get statistics about the loaded data

        Args:
            match_collection: MatchCollection object

        Returns:
            Dictionary with statistics
        """
        nmatches = len(match_collection.matches)
        ngames = sum(len(match.games) for match in match_collection.matches)
        nrallies = sum(len(game.rallies) for match in match_collection.matches for game in match.games)
        nshots = sum(
            len(rally.shots) for match in match_collection.matches for game in match.games for rally in game.rallies
        )

        # Summarize gcs sources across all rallies
        gcs_sources: Dict[str, int] = {}
        for match in match_collection.matches:
            for game in match.games:
                for rally in game.rallies:
                    src = getattr(rally.rally, 'attrs', {}).get('gcs_source', 'unknown') if hasattr(rally.rally, 'attrs') else 'unknown'
                    gcs_sources[src] = gcs_sources.get(src, 0) + 1

        return {"matches": nmatches, "games": ngames, "rallies": nrallies, "shots": nshots, "gcs_sources": gcs_sources}

    def extract_histogram_data(self, match_collection: MatchCollection, player_type: str = "robot", enabled_indices: List[int] = None):
        """
        Extract velocity and spin data for histogram plotting

        Args:
            match_collection: MatchCollection object
            player_type: "robot" or "player" to filter by player type
            enabled_indices: List of match indices to include (None = all matches)

        Returns:
            Tuple of (speeds, spins) where each is a tuple of (x, y, z, magnitude) arrays
        """
        # Use cache if available
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache

            # Build filter mask
            player_filter = 1 if player_type == "robot" else 2
            mask = (cache.player_types == player_filter) & cache.has_gt200

            if enabled_indices is not None:
                # Use Numba-accelerated mask building
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                match_mask = build_enabled_mask(cache.match_indices, enabled_arr)
                mask = mask & match_mask

            # Extract filtered data (pure array slicing - extremely fast)
            vx_arr = cache.vx_gt200[mask]
            vy_arr = cache.vy_gt200[mask]
            vz_arr = cache.vz_gt200[mask]
            wx_arr = cache.wx[mask]
            wy_arr = cache.wy[mask]
            wz_arr = cache.wz[mask]

            # Use Numba-accelerated magnitude calculations
            vmag_arr = compute_magnitudes_3d(vx_arr, vy_arr, vz_arr)
            wmag_arr = compute_magnitudes_3d(wx_arr, wy_arr, wz_arr)

            speeds = (vx_arr, vy_arr, vz_arr, vmag_arr)
            spins = (wx_arr, wy_arr, wz_arr, wmag_arr)

            return speeds, spins

        # Fallback to original implementation if no cache
        player_filter = 1 if player_type == "robot" else 2
        data_arrays = []

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            for game in match.games:
                for rally in game.rallies:
                    for shot in rally.shots:
                        if shot.player != player_filter:
                            continue

                        for fs in shot.flight_segments:
                            if len(fs.data) > 2:
                                cols = fs.data.columns
                                if 'vx_gt200' not in cols or 'wx' not in cols:
                                    continue

                                try:
                                    col_indices = [fs.data.columns.get_loc(c) for c in
                                                   ['vx_gt200', 'vy_gt200', 'vz_gt200', 'wx', 'wy', 'wz']]
                                except KeyError:
                                    continue

                                last_row = fs.data.values[-1, col_indices]

                                if not np.any(np.isnan(last_row)):
                                    data_arrays.append(last_row)

        if data_arrays:
            data = np.array(data_arrays)
            vx_arr = data[:, 0]
            vy_arr = data[:, 1]
            vz_arr = data[:, 2]
            wx_arr = data[:, 3]
            wy_arr = data[:, 4]
            wz_arr = data[:, 5]
        else:
            vx_arr = vy_arr = vz_arr = wx_arr = wy_arr = wz_arr = np.array([])

        vmag_arr = np.sqrt(vx_arr**2 + vy_arr**2 + vz_arr**2)
        wmag_arr = np.sqrt(wx_arr**2 + wy_arr**2 + wz_arr**2)

        speeds = (vx_arr, vy_arr, vz_arr, vmag_arr)
        spins = (wx_arr, wy_arr, wz_arr, wmag_arr)

        return speeds, spins

    def extract_rally_analysis_data(self, match_collection: MatchCollection, enabled_indices: List[int] = None) -> pd.DataFrame:
        """
        Extract rally-level data for rally analysis scatter plot matrix

        Args:
            match_collection: MatchCollection object
            enabled_indices: Optional list of match indices to include. If None, all matches are included.

        Returns:
            DataFrame with columns: date, match_player, game_id, rally_id, rally_length,
            rally_last_shot, rally_winner, time_from_last, pos, vel, spin
        """
        # ── Fast cache path ──
        if self._cache is not None and self._cache.rally_rows:
            if enabled_indices is not None:
                enabled_set = set(enabled_indices)
                rows = [r for r in self._cache.rally_rows if r['match_idx'] in enabled_set]
            else:
                rows = self._cache.rally_rows
            if rows:
                # Drop the internal match_idx key before building the DataFrame
                data_rally = pd.DataFrame([{k: v for k, v in r.items() if k != 'match_idx'} for r in rows])
            else:
                data_rally = pd.DataFrame(columns=[
                    'date', 'match_player', 'game_id', 'rally_id', 'rally_length',
                    'rally_last_shot', 'rally_winner', 'time_from_last', 'pos', 'vel', 'spin'
                ])
            return data_rally

        # ── Fallback: full traversal ──
        _rally_rows: list[dict] = []  # collect rows; build DataFrame once at end

        for match_idx, match in enumerate(match_collection.matches):
            # Skip if this match is not enabled
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue
            for game in match.games:
                for rally in game.rallies:
                    if len(rally.shots) == 0:
                        continue

                    # Collect trajectory data for the rally
                    time_from_last_t = []
                    pos_t = []
                    vel_t = []
                    spin_t = []
                    prev_time = 0.0

                    for shot_idx, shot in enumerate(rally.shots):
                        # Filter out segments with trigger_event.type == 'START'
                        valid_segments = [fs for fs in shot.flight_segments
                                        if hasattr(fs, 'trigger_event') and
                                           hasattr(fs.trigger_event, 'type') and
                                           fs.trigger_event.type != 'START']

                        if len(valid_segments) == 0:
                            continue

                        # Use the last valid segment in the shot
                        fs = valid_segments[-1]

                        if len(fs.data) > 0:
                            # Get last data point from segment
                            last_idx = fs.data.index[-1]

                            # Extract time (relative to shot start)
                            time = fs.data.loc[last_idx, 'time']
                            if shot_idx > 0:
                                # Time since previous shot's last point
                                time_from_last = time - prev_time
                            else:
                                time_from_last = 0.0
                            prev_time = time

                            # Extract position (use GT200 if available, otherwise APS)
                            if 'x_gt200' in fs.data.columns:
                                x = fs.data.loc[last_idx, 'x_gt200']
                                y = fs.data.loc[last_idx, 'y_gt200']
                                z = fs.data.loc[last_idx, 'z_gt200']
                            elif 'x_aps' in fs.data.columns:
                                x = fs.data.loc[last_idx, 'x_aps']
                                y = fs.data.loc[last_idx, 'y_aps']
                                z = fs.data.loc[last_idx, 'z_aps']
                            else:
                                continue

                            # Extract velocity
                            if 'vx_gt200' in fs.data.columns:
                                vx = fs.data.loc[last_idx, 'vx_gt200']
                                vy = fs.data.loc[last_idx, 'vy_gt200']
                                vz = fs.data.loc[last_idx, 'vz_gt200']
                            elif 'vx_aps' in fs.data.columns:
                                vx = fs.data.loc[last_idx, 'vx_aps']
                                vy = fs.data.loc[last_idx, 'vy_aps']
                                vz = fs.data.loc[last_idx, 'vz_aps']
                            else:
                                continue

                            # Extract spin
                            if 'wx' in fs.data.columns:
                                wx = fs.data.loc[last_idx, 'wx']
                                wy = fs.data.loc[last_idx, 'wy']
                                wz = fs.data.loc[last_idx, 'wz']
                            elif 'wx_gcs' in fs.data.columns:
                                wx = fs.data.loc[last_idx, 'wx_gcs']
                                wy = fs.data.loc[last_idx, 'wy_gcs']
                                wz = fs.data.loc[last_idx, 'wz_gcs']
                            else:
                                continue

                            # Check for valid values
                            if not (pd.isna(x) or pd.isna(y) or pd.isna(z) or
                                   pd.isna(vx) or pd.isna(vy) or pd.isna(vz) or
                                   pd.isna(wx) or pd.isna(wy) or pd.isna(wz)):
                                time_from_last_t.append(time_from_last)
                                pos_t.append([x, y, z])
                                vel_t.append([vx, vy, vz])
                                spin_t.append([wx, wy, wz])

                    if len(time_from_last_t) == 0:
                        continue

                    # Determine rally winner based on last shot outcome
                    last_shot = rally.shots[-1]
                    # Check if last shot has valid segments
                    last_valid_segments = [fs for fs in last_shot.flight_segments
                                          if hasattr(fs, 'trigger_event') and
                                             hasattr(fs.trigger_event, 'type') and
                                             fs.trigger_event.type != 'START']

                    if len(last_valid_segments) > 0:
                        # If there are valid segments, the shot was successful (ball was hit)
                        # Winner is the player who hit the last shot
                        rally_winner = last_shot.player
                    else:
                        # No valid segments means the shot failed
                        # Winner is the opponent
                        rally_winner = 3 - last_shot.player

                    rally_last_shot_player = last_shot.player

                    # Append rally row dict (cheap O(1) list append)
                    _rally_rows.append({
                        "date": match.date,
                        "match_player": match.player,
                        "game_id": game.game_id,
                        "rally_id": rally.rally_id,
                        'rally_length': len(time_from_last_t),
                        'rally_last_shot': "robot" if rally_last_shot_player == 1 else "human",
                        'rally_winner': "robot" if rally_winner == 1 else "human",
                        'time_from_last': np.array(time_from_last_t),
                        'pos': np.array(pos_t).flatten(),
                        'vel': np.array(vel_t).flatten(),
                        'spin': np.array(spin_t).flatten(),
                    })

        # Build DataFrame in one shot — O(n) instead of O(n²)
        if _rally_rows:
            data_rally = pd.DataFrame(_rally_rows)
        else:
            data_rally = pd.DataFrame(columns=[
                'date', 'match_player', 'game_id', 'rally_id', 'rally_length',
                'rally_last_shot', 'rally_winner', 'time_from_last', 'pos', 'vel', 'spin'
            ])
        return data_rally

    def extract_drag_coefficient_data(
        self, match_collection: MatchCollection, player_type: Optional[str] = "robot", enabled_indices: List[int] = None
    ) -> Tuple[
        List[float], List[float], List[float], List[float], List[float], List[Any], List[Dict[str, Any]], List[Any], List[Any], List[Any], List[Any], List[float], List[float], List[bool], List[float]
    ]:
        """
        Extract drag coefficient data from all flight segments with aerodynamics data

        Args:
            match_collection: MatchCollection object
            player_type: "robot" or "player" to filter by player type
            enabled_indices: List of match indices to include (None = all matches)

        Returns:
            Tuple of (vel_mag, v_eff_drag, spin_mag, spin_ratio, cd_opt, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list, rmse_opt_list, inconsistent_confidence_list, drag_impulse_proxy_list)

            confidence_list contains minimum confidence across 6 values (x,y,z,wx,wy,wz) for each point
        """
        radius_ball = 0.02  # 2cm radius

        # Use cache if available
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache

            # Build filter mask: player type + has aero data + valid cd_opt
            # player_type=None means include all players
            if player_type is None:
                mask = cache.has_aero & ~np.isnan(cache.cd_opt)
            else:
                player_filter: Optional[int] = 1 if player_type == "robot" else 2
                mask = (cache.player_types == player_filter) & cache.has_aero & ~np.isnan(cache.cd_opt)

            if enabled_indices is not None:
                # Use Numba-accelerated mask building
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                match_mask = build_enabled_mask(cache.match_indices, enabled_arr)
                mask = mask & match_mask

            # Get indices
            indices = np.where(mask)[0]

            # Use pre-computed magnitudes from cache
            vel_mag_array = cache.vel_mag_first[mask]
            spin_mag_array = cache.spin_mag_first[mask]
            spin_ratio_array = cache.spin_ratio_first[mask]

            # Filter out zero velocity
            valid = vel_mag_array > 1e-9
            if not np.all(valid):
                valid_indices = indices[valid]
                vel_mag_array = vel_mag_array[valid]
                spin_mag_array = spin_mag_array[valid]
                spin_ratio_array = spin_ratio_array[valid]
                indices = valid_indices
                mask_valid = np.zeros(cache.n_segments, dtype=bool)
                mask_valid[indices] = True
            else:
                mask_valid = mask

            cd_opt_array = cache.cd_opt[mask_valid]
            v_eff_drag_arr = cache.v_eff_drag[mask_valid]
            # Use Numba-accelerated NaN replacement
            v_eff_drag_array = apply_nan_mask(v_eff_drag_arr, vel_mag_array)
            confidence_list = cache.min_confidence[mask_valid].tolist()
            rmse_opt_list = cache.rmse_opt[mask_valid].tolist()
            inconsistent_confidence_list = cache.has_inconsistent_confidence[mask_valid].tolist()
            drag_impulse_proxy_list = cache.drag_impulse_proxy[mask_valid].tolist()

            # Extract object references
            flight_segments = [cache.flight_segments[i] for i in indices]
            shot_object_list = [cache.shots[i] for i in indices]
            rally_object_list = [cache.rallies[i] for i in indices]

            # Build metadata
            metadata_list = []
            for i in indices:
                metadata = {
                    "match_date": cache.match_dates[i],
                    "match_player": cache.match_players[i],
                    "game_name": cache.game_names[i],
                    "game_id": int(cache.game_ids[i]),
                    "rally_id": int(cache.rally_ids[i]),
                    "segment_start_time": cache.segment_start_times[i],
                    "match_file": cache.match_files[i],
                    "base_folder": cache.base_folder,
                    "log_identifier": cache.log_identifiers[i],
                }
                metadata_list.append(metadata)

            # Pass Shot/Rally objects — shot_data/rally_data DataFrames built on demand at click time
            shot_data_list = [None] * len(indices)
            rally_data_list = [None] * len(indices)

            return (
                vel_mag_array,
                v_eff_drag_array,
                spin_mag_array,
                spin_ratio_array,
                cd_opt_array,
                flight_segments,
                metadata_list,
                shot_data_list,
                rally_data_list,
                shot_object_list,
                rally_object_list,
                confidence_list,
                rmse_opt_list,
                inconsistent_confidence_list,
                drag_impulse_proxy_list,
            )

        # Fallback to original implementation
        vel_mag_list: List[float] = []
        v_eff_drag_list: List[float] = []
        spin_mag_list: List[float] = []
        spin_ratio_list: List[float] = []
        cd_opt_list: List[float] = []
        flight_segments = []
        metadata_list = []
        shot_data_list = []
        rally_data_list = []
        shot_object_list = []
        rally_object_list = []
        confidence_list = []

        player_filter = None if player_type is None else (1 if player_type == "robot" else 2)
        confidence_cols = SPIN_CONFIDENCE_COLS

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            match_date_str = str(match.date)
            match_player = match.player
            match_file = match.file

            for game in match.games:
                game_name = game.game_name
                game_id = game.game_id

                for rally in game.rallies:
                    rally_id = rally.rally_id
                    rally_data = rally.rally

                    for shot in rally.shots:
                        if player_filter is not None and shot.player != player_filter:
                            continue

                        shot_df = None

                        for fs in shot.flight_segments:
                            if len(fs.data) <= 2:
                                continue

                            cols = fs.data.columns
                            if "cd_opt" not in cols or "vx_opt" not in cols:
                                continue

                            try:
                                cd_idx = cols.get_loc("cd_opt")
                                vx_idx = cols.get_loc("vx_opt")
                                vy_idx = cols.get_loc("vy_opt")
                                vz_idx = cols.get_loc("vz_opt")
                                v_eff_drag_idx = cols.get_loc("v_eff_drag_opt") if "v_eff_drag_opt" in cols else None

                                if "wx_opt" in cols:
                                    wx_idx = cols.get_loc("wx_opt")
                                    wy_idx = cols.get_loc("wy_opt")
                                    wz_idx = cols.get_loc("wz_opt")
                                elif "wx" in cols:
                                    wx_idx = cols.get_loc("wx")
                                    wy_idx = cols.get_loc("wy")
                                    wz_idx = cols.get_loc("wz")
                                else:
                                    continue

                                first_row = fs.data.values[0]
                                cd_opt = first_row[cd_idx]
                                vx_opt = first_row[vx_idx]
                                vy_opt = first_row[vy_idx]
                                vz_opt = first_row[vz_idx]
                                wx = first_row[wx_idx]
                                wy = first_row[wy_idx]
                                wz = first_row[wz_idx]
                                v_eff_drag = first_row[v_eff_drag_idx] if v_eff_drag_idx is not None else None

                                values = np.array([cd_opt, vx_opt, vy_opt, vz_opt, wx, wy, wz])
                                if np.any(np.isnan(values)):
                                    continue

                                vel_magnitude = np.sqrt(vx_opt**2 + vy_opt**2 + vz_opt**2)
                                if vel_magnitude <= 1e-9:
                                    continue

                                spin_magnitude = np.sqrt(wx**2 + wy**2 + wz**2)
                                spin_ratio = (radius_ball * spin_magnitude) / vel_magnitude

                                vel_mag_list.append(vel_magnitude)
                                v_eff_drag_list.append(v_eff_drag if v_eff_drag is not None and not np.isnan(v_eff_drag) else vel_magnitude)
                                spin_mag_list.append(spin_magnitude)
                                spin_ratio_list.append(spin_ratio)
                                cd_opt_list.append(cd_opt)
                                flight_segments.append(fs)

                                min_confidence = 1.0
                                if all(col in cols for col in confidence_cols):
                                    try:
                                        conf_col_indices = [cols.get_loc(c) for c in confidence_cols]
                                        conf_data = fs.data.values[:, conf_col_indices]
                                        min_confidence = np.nanmin(conf_data)
                                        if np.isnan(min_confidence):
                                            min_confidence = 1.0
                                    except (KeyError, IndexError):
                                        pass
                                confidence_list.append(min_confidence)

                                metadata = {
                                    "match_date": match_date_str,
                                    "match_player": match_player,
                                    "game_name": game_name,
                                    "game_id": game_id,
                                    "rally_id": rally_id,
                                    "segment_start_time": fs.data.values[0, cols.get_loc("time")],
                                    "match_file": match_file,
                                    "base_folder": match_collection.base_folder,
                                    "log_identifier": getattr(rally, 'log_identifier', None) or '',
                                }
                                metadata_list.append(metadata)

                                if shot_df is None:
                                    shot_df = pd.concat(
                                        [seg.data for seg in shot.flight_segments], ignore_index=False
                                    )
                                shot_data_list.append(shot_df)
                                rally_data_list.append(rally_data)
                                shot_object_list.append(shot)
                                rally_object_list.append(rally)
                            except Exception:
                                continue

        vel_mag_array = np.array(vel_mag_list)
        v_eff_drag_array = np.array(v_eff_drag_list)
        spin_mag_array = np.array(spin_mag_list)
        spin_ratio_array = np.array(spin_ratio_list)
        cd_opt_array = np.array(cd_opt_list)
        # Fallback doesn't compute RMSE, return NaN array
        rmse_opt_list = [np.nan] * len(vel_mag_list)
        # Fallback doesn't have inconsistent confidence info, assume all consistent
        inconsistent_confidence_list = [False] * len(vel_mag_list)
        # Fallback doesn't compute drag force sensitivity, return NaN array
        drag_impulse_proxy_list = [np.nan] * len(vel_mag_list)

        return (
            vel_mag_array,
            v_eff_drag_array,
            spin_mag_array,
            spin_ratio_array,
            cd_opt_array,
            flight_segments,
            metadata_list,
            shot_data_list,
            rally_data_list,
            shot_object_list,
            rally_object_list,
            confidence_list,
            rmse_opt_list,
            inconsistent_confidence_list,
            drag_impulse_proxy_list,
        )

    def extract_magnus_coefficient_data(
        self, match_collection: MatchCollection, player_type: Optional[str] = "robot", enabled_indices: List[int] = None
    ) -> Tuple[
        List[float], List[float], List[float], List[float], List[float], List[float], List[Any], List[Dict[str, Any]], List[Any], List[Any], List[Any], List[Any], List[float], List[float], List[bool], List[float]
    ]:
        """
        Extract magnus coefficient data from all flight segments with aerodynamics data

        Args:
            match_collection: MatchCollection object
            player_type: "robot" or "player" to filter by player type
            enabled_indices: List of match indices to include (None = all matches)

        Returns:
            Tuple of (vel_mag, v_eff_magnus, spin_mag, spin_ratio, w_eff_perp_magnus, cm_opt, flight_segments, metadata_list, shot_data_list, rally_data_list, shot_object_list, rally_object_list, confidence_list, rmse_opt_list, inconsistent_confidence_list, magnus_impulse_proxy_list)

            confidence_list contains minimum confidence across 6 values (x,y,z,wx,wy,wz) for each point
        """
        radius_ball = 0.02  # 2cm radius

        # Use cache if available
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache

            # Build filter mask: player type + has aero data + valid cm_opt
            # player_type=None means include all players
            if player_type is None:
                mask = cache.has_aero & ~np.isnan(cache.cm_opt)
            else:
                player_filter: Optional[int] = 1 if player_type == "robot" else 2
                mask = (cache.player_types == player_filter) & cache.has_aero & ~np.isnan(cache.cm_opt)

            if enabled_indices is not None:
                # Use Numba-accelerated mask building
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                match_mask = build_enabled_mask(cache.match_indices, enabled_arr)
                mask = mask & match_mask

            # Get indices
            indices = np.where(mask)[0]

            # Use pre-computed magnitudes from cache
            vel_mag_array = cache.vel_mag_first[mask]
            spin_mag_array = cache.spin_mag_first[mask]
            spin_ratio_array = cache.spin_ratio_first[mask]

            # Filter out zero velocity
            valid = vel_mag_array > 1e-9
            if not np.all(valid):
                valid_indices = indices[valid]
                vel_mag_array = vel_mag_array[valid]
                spin_mag_array = spin_mag_array[valid]
                spin_ratio_array = spin_ratio_array[valid]
                indices = valid_indices
                mask_valid = np.zeros(cache.n_segments, dtype=bool)
                mask_valid[indices] = True
            else:
                mask_valid = mask

            cm_opt_array = cache.cm_opt[mask_valid]
            v_eff_magnus_arr = cache.v_eff_magnus[mask_valid]
            # Use Numba-accelerated NaN replacement
            v_eff_magnus_array = apply_nan_mask(v_eff_magnus_arr, vel_mag_array)
            # Extract w_eff_perp_magnus (fall back to spin_mag if not available)
            w_eff_perp_magnus_arr = cache.w_eff_perp_magnus[mask_valid]
            w_eff_perp_magnus_array = apply_nan_mask(w_eff_perp_magnus_arr, spin_mag_array)
            confidence_list = cache.min_confidence[mask_valid].tolist()
            rmse_opt_list = cache.rmse_opt[mask_valid].tolist()
            inconsistent_confidence_list = cache.has_inconsistent_confidence[mask_valid].tolist()
            magnus_impulse_proxy_list = cache.magnus_impulse_proxy[mask_valid].tolist()

            # Extract object references
            flight_segments = [cache.flight_segments[i] for i in indices]
            shot_object_list = [cache.shots[i] for i in indices]
            rally_object_list = [cache.rallies[i] for i in indices]

            # Build metadata
            metadata_list = []
            for i in indices:
                metadata = {
                    "match_date": cache.match_dates[i],
                    "match_player": cache.match_players[i],
                    "game_name": cache.game_names[i],
                    "game_id": int(cache.game_ids[i]),
                    "rally_id": int(cache.rally_ids[i]),
                    "segment_start_time": cache.segment_start_times[i],
                    "match_file": cache.match_files[i],
                    "base_folder": cache.base_folder,
                    "log_identifier": cache.log_identifiers[i],
                }
                metadata_list.append(metadata)

            # Pass Shot/Rally objects — shot_data/rally_data DataFrames built on demand at click time
            shot_data_list = [None] * len(indices)
            rally_data_list = [None] * len(indices)

            return (
                vel_mag_array,
                v_eff_magnus_array,
                spin_mag_array,
                spin_ratio_array,
                w_eff_perp_magnus_array,
                cm_opt_array,
                flight_segments,
                metadata_list,
                shot_data_list,
                rally_data_list,
                shot_object_list,
                rally_object_list,
                confidence_list,
                rmse_opt_list,
                inconsistent_confidence_list,
                magnus_impulse_proxy_list,
            )

        # Fallback to original implementation
        vel_mag_list: List[float] = []
        v_eff_magnus_list: List[float] = []
        spin_mag_list: List[float] = []
        spin_ratio_list: List[float] = []
        w_eff_perp_magnus_list: List[float] = []
        cm_opt_list: List[float] = []
        flight_segments = []
        metadata_list = []
        shot_data_list = []
        rally_data_list = []
        shot_object_list = []
        rally_object_list = []
        confidence_list = []

        player_filter = None if player_type is None else (1 if player_type == "robot" else 2)
        confidence_cols = SPIN_CONFIDENCE_COLS

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            match_date_str = str(match.date)
            match_player = match.player
            match_file = match.file

            for game in match.games:
                game_name = game.game_name
                game_id = game.game_id

                for rally in game.rallies:
                    rally_id = rally.rally_id
                    rally_data = rally.rally

                    for shot in rally.shots:
                        if player_filter is not None and shot.player != player_filter:
                            continue

                        shot_df = None

                        for fs in shot.flight_segments:
                            if len(fs.data) <= 2:
                                continue

                            cols = fs.data.columns
                            if "cm_opt" not in cols or "vx_opt" not in cols:
                                continue

                            try:
                                cm_idx = cols.get_loc("cm_opt")
                                vx_idx = cols.get_loc("vx_opt")
                                vy_idx = cols.get_loc("vy_opt")
                                vz_idx = cols.get_loc("vz_opt")
                                v_eff_magnus_idx = cols.get_loc("v_eff_magnus_opt") if "v_eff_magnus_opt" in cols else None

                                if "wx_opt" in cols:
                                    wx_idx = cols.get_loc("wx_opt")
                                    wy_idx = cols.get_loc("wy_opt")
                                    wz_idx = cols.get_loc("wz_opt")
                                elif "wx" in cols:
                                    wx_idx = cols.get_loc("wx")
                                    wy_idx = cols.get_loc("wy")
                                    wz_idx = cols.get_loc("wz")
                                else:
                                    continue

                                first_row = fs.data.values[0]
                                cm_opt = first_row[cm_idx]
                                vx_opt = first_row[vx_idx]
                                vy_opt = first_row[vy_idx]
                                vz_opt = first_row[vz_idx]
                                wx = first_row[wx_idx]
                                wy = first_row[wy_idx]
                                wz = first_row[wz_idx]
                                v_eff_magnus = first_row[v_eff_magnus_idx] if v_eff_magnus_idx is not None else None
                                w_eff_perp_magnus_idx = cols.get_loc("w_eff_perp_magnus_opt") if "w_eff_perp_magnus_opt" in cols else None
                                w_eff_perp_magnus = first_row[w_eff_perp_magnus_idx] if w_eff_perp_magnus_idx is not None else None

                                values = np.array([cm_opt, vx_opt, vy_opt, vz_opt, wx, wy, wz])
                                if np.any(np.isnan(values)):
                                    continue

                                vel_magnitude = np.sqrt(vx_opt**2 + vy_opt**2 + vz_opt**2)
                                if vel_magnitude <= 1e-9:
                                    continue

                                spin_magnitude = np.sqrt(wx**2 + wy**2 + wz**2)
                                spin_ratio = (radius_ball * spin_magnitude) / vel_magnitude

                                vel_mag_list.append(vel_magnitude)
                                v_eff_magnus_list.append(v_eff_magnus if v_eff_magnus is not None and not np.isnan(v_eff_magnus) else vel_magnitude)
                                spin_mag_list.append(spin_magnitude)
                                spin_ratio_list.append(spin_ratio)
                                w_eff_perp_magnus_list.append(w_eff_perp_magnus if w_eff_perp_magnus is not None and not np.isnan(w_eff_perp_magnus) else spin_magnitude)
                                cm_opt_list.append(cm_opt)
                                flight_segments.append(fs)

                                min_confidence = 1.0
                                if all(col in cols for col in confidence_cols):
                                    try:
                                        conf_col_indices = [cols.get_loc(c) for c in confidence_cols]
                                        conf_data = fs.data.values[:, conf_col_indices]
                                        min_confidence = np.nanmin(conf_data)
                                        if np.isnan(min_confidence):
                                            min_confidence = 1.0
                                    except (KeyError, IndexError):
                                        pass
                                confidence_list.append(min_confidence)

                                metadata = {
                                    "match_date": match_date_str,
                                    "match_player": match_player,
                                    "game_name": game_name,
                                    "game_id": game_id,
                                    "rally_id": rally_id,
                                    "segment_start_time": fs.data.values[0, cols.get_loc("time")],
                                    "match_file": match_file,
                                    "base_folder": match_collection.base_folder,
                                    "log_identifier": getattr(rally, 'log_identifier', None) or '',
                                }
                                metadata_list.append(metadata)

                                if shot_df is None:
                                    shot_df = pd.concat(
                                        [seg.data for seg in shot.flight_segments], ignore_index=False
                                    )
                                shot_data_list.append(shot_df)
                                rally_data_list.append(rally_data)
                                shot_object_list.append(shot)
                                rally_object_list.append(rally)
                            except Exception:
                                continue

        vel_mag_array = np.array(vel_mag_list)
        v_eff_magnus_array = np.array(v_eff_magnus_list)
        spin_mag_array = np.array(spin_mag_list)
        spin_ratio_array = np.array(spin_ratio_list)
        w_eff_perp_magnus_array = np.array(w_eff_perp_magnus_list)
        cm_opt_array = np.array(cm_opt_list)
        # Fallback doesn't compute RMSE, return NaN array
        rmse_opt_list = [np.nan] * len(vel_mag_list)
        # Fallback doesn't have inconsistent confidence info, assume all consistent
        inconsistent_confidence_list = [False] * len(vel_mag_list)
        # Fallback doesn't compute magnus force sensitivity, return NaN array
        magnus_impulse_proxy_list = [np.nan] * len(vel_mag_list)

        return (
            vel_mag_array,
            v_eff_magnus_array,
            spin_mag_array,
            spin_ratio_array,
            w_eff_perp_magnus_array,
            cm_opt_array,
            flight_segments,
            metadata_list,
            shot_data_list,
            rally_data_list,
            shot_object_list,
            rally_object_list,
            confidence_list,
            rmse_opt_list,
            inconsistent_confidence_list,
            magnus_impulse_proxy_list,
        )

    def extract_racket_contact_data(
        self, match_collection: MatchCollection, enabled_indices: List[int] = None
    ) -> Dict[str, Any]:
        """Extract racket contact (shot) data from consecutive shots within a rally.

        A racket contact occurs when a player hits the ball. This function
        extracts pre-contact (end of last segment of previous shot) and
        post-contact (start of first segment of current shot) velocity/spin.

        Returns:
            Dictionary with arrays for vx/vy/vz/wx/wy/wz pre/post,
            player (1=robot, 2=human), confidence, metadata, etc.
        """
        # ── Fast cache path ──
        if self._cache is not None and self._cache.racket_contact_data is not None:
            rcd = self._cache.racket_contact_data
            if enabled_indices is not None:
                enabled_set = set(enabled_indices)
                keep = np.array([
                    i for i, m in enumerate(rcd['metadata_list'])
                    if m.get('match_idx') in enabled_set
                ], dtype=int)
                if len(keep) == 0:
                    return {k: (np.array([]) if isinstance(v, np.ndarray) else [])
                            for k, v in rcd.items()}
                result = {}
                for k, v in rcd.items():
                    if isinstance(v, np.ndarray):
                        result[k] = v[keep]
                    elif isinstance(v, list):
                        result[k] = [v[i] for i in keep]
                    else:
                        result[k] = v
                return result
            return rcd

        # ── Fallback: full traversal ──
        vx_pre_list, vy_pre_list, vz_pre_list = [], [], []
        wx_pre_list, wy_pre_list, wz_pre_list = [], [], []
        vx_post_list, vy_post_list, vz_post_list = [], [], []
        wx_post_list, wy_post_list, wz_post_list = [], [], []
        player_list = []  # 1=robot (shot_p1), 2=human (shot_p2)
        policy_list = []  # policy name from the Match
        confidence_list = []
        max_rmse_opt_list = []  # Track max RMSE error (x_aps vs x_opt) between both segments
        metadata_list = []
        shot_list = []
        rally_list = []
        fs_pre_list = []
        fs_post_list = []

        confidence_cols = SPIN_CONFIDENCE_COLS

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            match_date_str = str(match.date)
            match_player = match.player
            match_policy = getattr(match, 'policy', 'unknown')
            match_file = match.file

            for game in match.games:
                game_name = game.game_name
                game_id = game.game_id

                for rally in game.rallies:
                    rally_id = rally.rally_id
                    shots = rally.shots

                    for s_idx in range(1, len(shots)):
                        shot_post = shots[s_idx]

                        # The first segment of the new shot should start with a racket event
                        if len(shot_post.flight_segments) == 0:
                            continue
                        fs_post = shot_post.flight_segments[0]
                        if fs_post.trigger_event is None:
                            continue
                        if fs_post.trigger_event.type not in ('shot_p1', 'shot_p2'):
                            continue

                        # Pre-contact: last segment of previous shot
                        shot_pre = shots[s_idx - 1]
                        if len(shot_pre.flight_segments) == 0:
                            continue
                        fs_pre = shot_pre.flight_segments[-1]

                        # Need aerodynamics data on both segments
                        cols_pre = fs_pre.data.columns
                        cols_post = fs_post.data.columns

                        has_aero_pre = 'vx_opt' in cols_pre or 'vel_opt_x' in cols_pre
                        has_aero_post = 'vx_opt' in cols_post or 'vel_opt_x' in cols_post
                        if not has_aero_pre or not has_aero_post:
                            continue

                        try:
                            # Velocity columns
                            vx_col = 'vx_opt' if 'vx_opt' in cols_pre else 'vel_opt_x'
                            vy_col = 'vy_opt' if 'vy_opt' in cols_pre else 'vel_opt_y'
                            vz_col = 'vz_opt' if 'vz_opt' in cols_pre else 'vel_opt_z'

                            # Spin columns
                            if 'wx_opt' in cols_pre:
                                wx_col, wy_col, wz_col = 'wx_opt', 'wy_opt', 'wz_opt'
                            elif 'wx' in cols_pre:
                                wx_col, wy_col, wz_col = 'wx', 'wy', 'wz'
                            else:
                                continue

                            # Pre-contact: end of last segment of previous shot
                            pre_row = fs_pre.data.iloc[-1]
                            vx_pre = pre_row[vx_col]
                            vy_pre = pre_row[vy_col]
                            vz_pre = pre_row[vz_col]
                            wx_pre = pre_row[wx_col]
                            wy_pre = pre_row[wy_col]
                            wz_pre = pre_row[wz_col]

                            # Post-contact: start of first segment of current shot
                            post_row = fs_post.data.iloc[0]
                            vx_post = post_row[vx_col if vx_col in cols_post else 'vx_opt']
                            vy_post = post_row[vy_col if vy_col in cols_post else 'vy_opt']
                            vz_post = post_row[vz_col if vz_col in cols_post else 'vz_opt']
                            wx_post = post_row[wx_col if wx_col in cols_post else 'wx_opt']
                            wy_post = post_row[wy_col if wy_col in cols_post else 'wy_opt']
                            wz_post = post_row[wz_col if wz_col in cols_post else 'wz_opt']

                            vals = [vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre,
                                    vx_post, vy_post, vz_post, wx_post, wy_post, wz_post]
                            if any(np.isnan(v) for v in vals):
                                continue

                            vx_pre_list.append(vx_pre)
                            vy_pre_list.append(vy_pre)
                            vz_pre_list.append(vz_pre)
                            wx_pre_list.append(wx_pre)
                            wy_pre_list.append(wy_pre)
                            wz_pre_list.append(wz_pre)
                            vx_post_list.append(vx_post)
                            vy_post_list.append(vy_post)
                            vz_post_list.append(vz_post)
                            wx_post_list.append(wx_post)
                            wy_post_list.append(wy_post)
                            wz_post_list.append(wz_post)

                            player = 1 if fs_post.trigger_event.type == 'shot_p1' else 2
                            player_list.append(player)
                            policy_list.append(match_policy)

                            # Confidence: min across both segments
                            min_conf = 1.0
                            for fs in [fs_pre, fs_post]:
                                if all(col in fs.data.columns for col in confidence_cols):
                                    conf_vals = fs.data[confidence_cols].values
                                    seg_min = np.nanmin(conf_vals)
                                    if not np.isnan(seg_min):
                                        min_conf = min(min_conf, seg_min)
                            confidence_list.append(min_conf)

                            # Compute max RMSE error (x_aps vs x_opt) between both segments
                            max_rmse = np.nan
                            for fs in [fs_pre, fs_post]:
                                has_opt = all(c in fs.data.columns for c in ['x_opt', 'y_opt', 'z_opt'])
                                has_aps = all(c in fs.data.columns for c in ['x_aps', 'y_aps', 'z_aps'])
                                if has_opt and has_aps:
                                    x_opt_p = fs.data['x_opt'].values
                                    y_opt_p = fs.data['y_opt'].values
                                    z_opt_p = fs.data['z_opt'].values
                                    x_aps_p = fs.data['x_aps'].values
                                    y_aps_p = fs.data['y_aps'].values
                                    z_aps_p = fs.data['z_aps'].values
                                    valid = ~(np.isnan(x_aps_p) | np.isnan(x_opt_p))
                                    if np.sum(valid) > 0:
                                        dx = x_aps_p[valid] - x_opt_p[valid]
                                        dy = y_aps_p[valid] - y_opt_p[valid]
                                        dz = z_aps_p[valid] - z_opt_p[valid]
                                        seg_rmse = np.sqrt(np.mean(dx**2 + dy**2 + dz**2))
                                        if np.isnan(max_rmse) or seg_rmse > max_rmse:
                                            max_rmse = seg_rmse
                            max_rmse_opt_list.append(max_rmse)

                            metadata = {
                                'match_date': match_date_str,
                                'match_player': match_player,
                                'game_name': game_name,
                                'game_id': game_id,
                                'rally_id': rally_id,
                                'segment_start_time': float(fs_post.data.index[0]) if len(fs_post.data) > 0 else 0.0,
                                'contact_type': fs_post.trigger_event.type,
                                'contact_time': fs_post.trigger_event.timestamp,
                                'match_file': match_file,
                                'base_folder': match_collection.base_folder,
                                'log_identifier': getattr(rally, 'log_identifier', None) or '',
                            }
                            metadata_list.append(metadata)
                            shot_list.append(shot_post)
                            rally_list.append(rally)
                            fs_pre_list.append(fs_pre)
                            fs_post_list.append(fs_post)

                        except Exception:
                            continue

        return {
            'vx_pre': np.array(vx_pre_list),
            'vy_pre': np.array(vy_pre_list),
            'vz_pre': np.array(vz_pre_list),
            'wx_pre': np.array(wx_pre_list),
            'wy_pre': np.array(wy_pre_list),
            'wz_pre': np.array(wz_pre_list),
            'vx_post': np.array(vx_post_list),
            'vy_post': np.array(vy_post_list),
            'vz_post': np.array(vz_post_list),
            'wx_post': np.array(wx_post_list),
            'wy_post': np.array(wy_post_list),
            'wz_post': np.array(wz_post_list),
            'player': np.array(player_list, dtype=int),
            'policy': policy_list,
            'confidence': np.array(confidence_list),
            'max_rmse_opt': np.array(max_rmse_opt_list),
            'metadata_list': metadata_list,
            'shot_list': shot_list,
            'rally_list': rally_list,
            'fs_pre_list': fs_pre_list,
            'fs_post_list': fs_post_list,
        }

    def extract_rcm_data(
        self, match_collection: MatchCollection, enabled_indices: List[int] = None
    ) -> Dict[str, Any]:
        """Extract RCM (Racket Contact Model) arrays from :class:`RacketContactEvent` objects.

        Unlike :meth:`extract_racket_contact_data`, which reconstructs contact
        quantities from HDF5 flight-segment boundaries, this method reads the
        physics-optimised values that were loaded from ``extracted.csv`` and
        enriched onto the event instances.

        Returns
        -------
        Dict with NumPy arrays keyed by quantity name.  Contains at least:

        * **Global frame** – ``vx_pre … wz_post``
        * **Racket (local) frame** – ``vrx_pre … wrz_post``
        * **Angles** – ``theta_angle``, ``racket_open_angle``
        * **Quality** – ``fitness_pre``, ``fitness_post``
        * **Metadata** – ``player``, ``metadata_list``, ``shot_list``,
          ``rally_list``
        """
        # ── Fast cache path ──
        if self._cache is not None and self._cache.rcm_data is not None:
            rcm = self._cache.rcm_data
            if enabled_indices is not None:
                enabled_set = set(enabled_indices)
                keep = np.array([m['match_idx'] in enabled_set for m in rcm['metadata_list']], dtype=bool)
                if not np.any(keep):
                    return {k: np.array([]) for k in rcm if isinstance(rcm[k], np.ndarray)} | {
                        'metadata_list': [], 'shot_list': [], 'rally_list': [],
                        'fs_pre_list': [], 'fs_post_list': [],
                    }
                result = {}
                for k, v in rcm.items():
                    if isinstance(v, np.ndarray):
                        result[k] = v[keep]
                    elif isinstance(v, list):
                        result[k] = [v[i] for i in range(len(v)) if keep[i]]
                    else:
                        result[k] = v
                return result
            return rcm

        # Accumulators
        arrays: Dict[str, list] = {
            'vx_pre': [], 'vy_pre': [], 'vz_pre': [],
            'wx_pre': [], 'wy_pre': [], 'wz_pre': [],
            'vx_post': [], 'vy_post': [], 'vz_post': [],
            'wx_post': [], 'wy_post': [], 'wz_post': [],
            # racket frame
            'vrx_pre': [], 'vry_pre': [],
            'wrx_pre': [], 'wry_pre': [], 'wrz_pre': [],
            'vrx_post': [], 'vry_post': [], 'vrz_post': [],
            'wrx_post': [], 'wry_post': [], 'wrz_post': [],
            # angles
            'theta_angle': [], 'racket_open_angle': [],
            # quality
            'fitness_pre': [], 'fitness_post': [],
            # position deltas
            'dx_pre': [], 'dy_pre': [], 'dz_pre': [],
            'dx_post': [], 'dy_post': [], 'dz_post': [],
            # contact location in racket frame
            'drx_pre': [], 'dry_pre': [], 'drz_pre': [],
            # racket position in global frame at contact
            'rx_global': [], 'ry_global': [], 'rz_global': [],
            # racket velocity / angular velocity in racket frame
            'vrx_racket': [], 'vry_racket': [], 'vrz_racket': [],
            'wrx_racket': [], 'wry_racket': [], 'wrz_racket': [],
            # racket angular velocity in global frame
            'wgx_racket': [], 'wgy_racket': [], 'wgz_racket': [],
            # racket angular velocity in body frame (pure R^T, no v_ref alignment)
            'wbx_racket': [], 'wby_racket': [], 'wbz_racket': [],
            # model predictions (global frame)
            'vx_post_default': [], 'vy_post_default': [], 'vz_post_default': [],
            'wx_post_default': [], 'wy_post_default': [], 'wz_post_default': [],
            'vx_post_cpp': [], 'vy_post_cpp': [], 'vz_post_cpp': [],
            'wx_post_cpp': [], 'wy_post_cpp': [], 'wz_post_cpp': [],
            'vx_post_parametric_7p': [], 'vy_post_parametric_7p': [], 'vz_post_parametric_7p': [],
            'wx_post_parametric_7p': [], 'wy_post_parametric_7p': [], 'wz_post_parametric_7p': [],
            'vx_post_rcm_tangential': [], 'vy_post_rcm_tangential': [], 'vz_post_rcm_tangential': [],
            'wx_post_rcm_tangential': [], 'wy_post_rcm_tangential': [], 'wz_post_rcm_tangential': [],
            'vx_post_cpp_tangential': [], 'vy_post_cpp_tangential': [], 'vz_post_cpp_tangential': [],
            'wx_post_cpp_tangential': [], 'wy_post_cpp_tangential': [], 'wz_post_cpp_tangential': [],
            'vx_post_cpp_refined': [], 'vy_post_cpp_refined': [], 'vz_post_cpp_refined': [],
            'wx_post_cpp_refined': [], 'wy_post_cpp_refined': [], 'wz_post_cpp_refined': [],
            'vx_post_rcm_tangential_refined': [], 'vy_post_rcm_tangential_refined': [], 'vz_post_rcm_tangential_refined': [],
            'wx_post_rcm_tangential_refined': [], 'wy_post_rcm_tangential_refined': [], 'wz_post_rcm_tangential_refined': [],
            'vx_post_rcm_tangential_polyfit': [], 'vy_post_rcm_tangential_polyfit': [], 'vz_post_rcm_tangential_polyfit': [],
            'wx_post_rcm_tangential_polyfit': [], 'wy_post_rcm_tangential_polyfit': [], 'wz_post_rcm_tangential_polyfit': [],
            'vx_post_nakashima_refined': [], 'vy_post_nakashima_refined': [], 'vz_post_nakashima_refined': [],
            'wx_post_nakashima_refined': [], 'wy_post_nakashima_refined': [], 'wz_post_nakashima_refined': [],
            'vx_post_onnx_rcm': [], 'vy_post_onnx_rcm': [], 'vz_post_onnx_rcm': [],
            'wx_post_onnx_rcm': [], 'wy_post_onnx_rcm': [], 'wz_post_onnx_rcm': [],
            'vx_post_onnx_alex': [], 'vy_post_onnx_alex': [], 'vz_post_onnx_alex': [],
            'wx_post_onnx_alex': [], 'wy_post_onnx_alex': [], 'wz_post_onnx_alex': [],
            'vx_post_onnx_alex_refined': [], 'vy_post_onnx_alex_refined': [], 'vz_post_onnx_alex_refined': [],
            'wx_post_onnx_alex_refined': [], 'wy_post_onnx_alex_refined': [], 'wz_post_onnx_alex_refined': [],
            'vx_post_onnx_alex_polyfit': [], 'vy_post_onnx_alex_polyfit': [], 'vz_post_onnx_alex_polyfit': [],
            'wx_post_onnx_alex_polyfit': [], 'wy_post_onnx_alex_polyfit': [], 'wz_post_onnx_alex_polyfit': [],
            'vx_post_onnx_0426': [], 'vy_post_onnx_0426': [], 'vz_post_onnx_0426': [],
            'wx_post_onnx_0426': [], 'wy_post_onnx_0426': [], 'wz_post_onnx_0426': [],
        }
        player_list: List[int] = []
        shot_type_list: List[int] = []  # 0=serve, 1=rally, 2=last shot
        metadata_list: List[dict] = []
        shot_list: List[Any] = []
        rally_list: List[Any] = []
        fs_pre_list: List[Any] = []
        fs_post_list: List[Any] = []
        confidence_list: List[float] = []
        max_rmse_opt_list: List[float] = []
        # Model prediction accumulators
        _tc_model_keys = [
            'vx_post_ittf', 'vy_post_ittf', 'vz_post_ittf', 'wx_post_ittf', 'wy_post_ittf', 'wz_post_ittf',
            'vx_post_paper', 'vy_post_paper', 'vz_post_paper', 'wx_post_paper', 'wy_post_paper', 'wz_post_paper',
            'vx_post_rcn', 'vy_post_rcn', 'vz_post_rcn', 'wx_post_rcn', 'wy_post_rcn', 'wz_post_rcn',
            'vx_post_res0805', 'vy_post_res0805', 'vz_post_res0805', 'wx_post_res0805', 'wy_post_res0805', 'wz_post_res0805',
            'vx_post_pysr', 'vy_post_pysr', 'vz_post_pysr', 'wx_post_pysr', 'wy_post_pysr', 'wz_post_pysr',
            'vx_post_0426', 'vy_post_0426', 'vz_post_0426', 'wx_post_0426', 'wy_post_0426', 'wz_post_0426',
        ]
        _tc_model_lists = {k: [] for k in _tc_model_keys}

        confidence_cols = SPIN_CONFIDENCE_COLS

        # ── Pre-flight diagnostics ─────────────────────────────────
        import pathlib as _pl
        _bf = getattr(match_collection, 'base_folder', None)
        _csv_path = _pl.Path(_bf) / "extracted.csv" if _bf else None
        _csv_exists = _csv_path.exists() if _csv_path else False
        _csv_size = _csv_path.stat().st_size if _csv_exists else 0
        print(f"[RCM-diag] extract_rcm_data: base_folder={_bf}")
        print(f"[RCM-diag] extract_rcm_data: extracted.csv exists={_csv_exists}, size={_csv_size}")
        print(f"[RCM-diag] extract_rcm_data: num_matches={len(match_collection.matches)}, "
              f"enabled_indices={enabled_indices}")
        # Check if any H5 in folder are newer than CSV
        if _csv_exists:
            import os as _os
            _csv_mt = _os.path.getmtime(_csv_path)
            _h5_files = list(_pl.Path(_bf).rglob("*.h5"))
            _h5_newer = [f.name for f in _h5_files
                         if f.stat().st_mtime > _csv_mt
                         and "optitrack" not in f.name.lower()
                         and "with_opt" not in f.name.lower()]
            if _h5_newer:
                print(f"[RCM-diag] WARNING: {len(_h5_newer)} H5 files are NEWER than extracted.csv — "
                      f"enrichment was likely skipped!  Examples: {_h5_newer[:3]}")
            else:
                print(f"[RCM-diag] extracted.csv is newer than all {len(_h5_files)} H5 files — good")

        _diag_total_events = 0
        _diag_racket_events = 0
        _diag_no_pre_post = 0
        _diag_nan_values = 0
        _diag_accepted = 0

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue
            match_date_str = str(match.date)
            match_player = match.player
            match_policy = getattr(match, 'policy', 'unknown')
            match_file = match.file

            for game in match.games:
                for rally in game.rallies:
                    # Pre-compute ALL racket-contact timestamps (not just
                    # accepted) so serve = rally's actual first shot.
                    _all_rce_ev = [e for e in rally.events if isinstance(e, RacketContactEvent)]
                    _rally_first_rc_ts = _all_rce_ev[0].timestamp if _all_rce_ev else None
                    _rally_last_rc_ts = _all_rce_ev[-1].timestamp if _all_rce_ev else None
                    # Also collect accepted timestamps for last-shot logic.
                    _rc_timestamps_in_rally: List[float] = []
                    for _ev in rally.events:
                        if not isinstance(_ev, RacketContactEvent):
                            continue
                        if _ev.ball_pre is None or _ev.ball_post is None:
                            continue
                        _bp2 = _ev.ball_pre; _ba2 = _ev.ball_post
                        if any(np.isnan(v) for v in [_bp2.vx, _bp2.vy, _bp2.vz, _bp2.wx, _bp2.wy, _bp2.wz,
                                                     _ba2.vx, _ba2.vy, _ba2.vz, _ba2.wx, _ba2.wy, _ba2.wz]):
                            continue
                        _rc_timestamps_in_rally.append(_ev.timestamp)
                    _last_rc_ts = _rc_timestamps_in_rally[-1] if _rc_timestamps_in_rally else None
                    _last_accepted_is_truly_last = (
                        _last_rc_ts is not None
                        and _rally_last_rc_ts is not None
                        and abs(_last_rc_ts - _rally_last_rc_ts) < 1e-6
                    )

                    for event in rally.events:
                        _diag_total_events += 1
                        if not isinstance(event, RacketContactEvent):
                            continue
                        _diag_racket_events += 1
                        # Skip events that were never enriched from CSV
                        if event.ball_pre is None or event.ball_post is None:
                            _diag_no_pre_post += 1
                            continue
                        bp = event.ball_pre
                        ba = event.ball_post
                        # Require finite global-frame values
                        if any(np.isnan(v) for v in [bp.vx, bp.vy, bp.vz, bp.wx, bp.wy, bp.wz,
                                                     ba.vx, ba.vy, ba.vz, ba.wx, ba.wy, ba.wz]):
                            _diag_nan_values += 1
                            continue

                        arrays['vx_pre'].append(bp.vx)
                        arrays['vy_pre'].append(bp.vy)
                        arrays['vz_pre'].append(bp.vz)
                        arrays['wx_pre'].append(bp.wx)
                        arrays['wy_pre'].append(bp.wy)
                        arrays['wz_pre'].append(bp.wz)
                        arrays['vx_post'].append(ba.vx)
                        arrays['vy_post'].append(ba.vy)
                        arrays['vz_post'].append(ba.vz)
                        arrays['wx_post'].append(ba.wx)
                        arrays['wy_post'].append(ba.wy)
                        arrays['wz_post'].append(ba.wz)

                        # Racket-frame quantities
                        def _v3(v, idx):
                            return v[idx] if v is not None else float('nan')

                        vpre_r = event.ball_vel_pre_racket
                        vpost_r = event.ball_vel_post_racket
                        wpre_r = event.ball_spin_pre_racket
                        wpost_r = event.ball_spin_post_racket

                        arrays['vrx_pre'].append(_v3(vpre_r, 0))
                        arrays['vry_pre'].append(_v3(vpre_r, 1))
                        arrays['wrx_pre'].append(_v3(wpre_r, 0))
                        arrays['wry_pre'].append(_v3(wpre_r, 1))
                        arrays['wrz_pre'].append(_v3(wpre_r, 2))

                        arrays['vrx_post'].append(_v3(vpost_r, 0))
                        arrays['vry_post'].append(_v3(vpost_r, 1))
                        arrays['vrz_post'].append(_v3(vpost_r, 2))
                        arrays['wrx_post'].append(_v3(wpost_r, 0))
                        arrays['wry_post'].append(_v3(wpost_r, 1))
                        arrays['wrz_post'].append(_v3(wpost_r, 2))

                        arrays['theta_angle'].append(event.theta_angle)
                        arrays['racket_open_angle'].append(event.racket_open_angle)
                        arrays['fitness_pre'].append(event.fitness_pre)
                        arrays['fitness_post'].append(event.fitness_post)

                        # Position deltas
                        _dpre = event.delta_pre
                        _dpost = event.delta_post
                        arrays['dx_pre'].append(_v3(_dpre, 0) if _dpre is not None else float('nan'))
                        arrays['dy_pre'].append(_v3(_dpre, 1) if _dpre is not None else float('nan'))
                        arrays['dz_pre'].append(_v3(_dpre, 2) if _dpre is not None else float('nan'))
                        arrays['dx_post'].append(_v3(_dpost, 0) if _dpost is not None else float('nan'))
                        arrays['dy_post'].append(_v3(_dpost, 1) if _dpost is not None else float('nan'))
                        arrays['dz_post'].append(_v3(_dpost, 2) if _dpost is not None else float('nan'))

                        # Contact location in racket frame
                        _bpr = event.ball_pos_pre_racket
                        arrays['drx_pre'].append(_v3(_bpr, 0) if _bpr is not None else float('nan'))
                        arrays['dry_pre'].append(_v3(_bpr, 1) if _bpr is not None else float('nan'))
                        arrays['drz_pre'].append(_v3(_bpr, 2) if _bpr is not None else float('nan'))

                        # Racket velocity / angular velocity in racket frame
                        _rvr = event.racket_vel_racket
                        _rsr = event.racket_spin_racket
                        arrays['vrx_racket'].append(_v3(_rvr, 0) if _rvr is not None else float('nan'))
                        arrays['vry_racket'].append(_v3(_rvr, 1) if _rvr is not None else float('nan'))
                        arrays['vrz_racket'].append(_v3(_rvr, 2) if _rvr is not None else float('nan'))
                        arrays['wrx_racket'].append(_v3(_rsr, 0) if _rsr is not None else float('nan'))
                        arrays['wry_racket'].append(_v3(_rsr, 1) if _rsr is not None else float('nan'))
                        arrays['wrz_racket'].append(_v3(_rsr, 2) if _rsr is not None else float('nan'))

                        # Racket position in global frame at contact
                        _rpos = event.racket_pos
                        arrays['rx_global'].append(_rpos.x if _rpos is not None else float('nan'))
                        arrays['ry_global'].append(_rpos.y if _rpos is not None else float('nan'))
                        arrays['rz_global'].append(_rpos.z if _rpos is not None else float('nan'))

                        # Racket angular velocity in global frame
                        _rav = event.racket_ang_vel
                        arrays['wgx_racket'].append(_v3(_rav, 0) if _rav is not None else float('nan'))
                        arrays['wgy_racket'].append(_v3(_rav, 1) if _rav is not None else float('nan'))
                        arrays['wgz_racket'].append(_v3(_rav, 2) if _rav is not None else float('nan'))

                        # Racket angular velocity in body frame (pure R^T)
                        _rsb = event.racket_spin_body
                        arrays['wbx_racket'].append(_v3(_rsb, 0) if _rsb is not None else float('nan'))
                        arrays['wby_racket'].append(_v3(_rsb, 1) if _rsb is not None else float('nan'))
                        arrays['wbz_racket'].append(_v3(_rsb, 2) if _rsb is not None else float('nan'))

                        # Model predictions
                        for _suffix, _attr in [('default', 'ball_post_default'),
                                               ('cpp', 'ball_post_cpp'),
                                               ('parametric_7p', 'ball_post_parametric_7p'),
                                               ('rcm_tangential', 'ball_post_rcm_tangential'),
                                               ('cpp_tangential', 'ball_post_cpp_tangential'),
                                               ('cpp_refined', 'ball_post_cpp_refined'),
                                               ('rcm_tangential_refined', 'ball_post_rcm_tangential_refined'),
                                               ('rcm_tangential_polyfit', 'ball_post_rcm_tangential_polyfit'),
                                               ('nakashima_refined', 'ball_post_nakashima_refined'),
                                               ('onnx_rcm', 'ball_post_onnx_rcm'),
                                               ('onnx_alex', 'ball_post_onnx_alex'),
                                               ('onnx_alex_refined', 'ball_post_onnx_alex_refined'),
                                               ('onnx_alex_polyfit', 'ball_post_onnx_alex_polyfit'),
                                               ('onnx_0426', 'ball_post_onnx_0426')]:
                            _bp = getattr(event, _attr, None)
                            arrays[f'vx_post_{_suffix}'].append(_bp.vx if _bp is not None else float('nan'))
                            arrays[f'vy_post_{_suffix}'].append(_bp.vy if _bp is not None else float('nan'))
                            arrays[f'vz_post_{_suffix}'].append(_bp.vz if _bp is not None else float('nan'))
                            arrays[f'wx_post_{_suffix}'].append(_bp.wx if _bp is not None else float('nan'))
                            arrays[f'wy_post_{_suffix}'].append(_bp.wy if _bp is not None else float('nan'))
                            arrays[f'wz_post_{_suffix}'].append(_bp.wz if _bp is not None else float('nan'))

                        player = 1 if event.type == 'shot_p1' else 2
                        player_list.append(player)

                        # Classify shot type: 0=serve, 1=rally, 2=last shot
                        # Serve = rally's actual first shot (not first accepted).
                        _is_serve = (
                            _rally_first_rc_ts is not None
                            and abs(event.timestamp - _rally_first_rc_ts) < 1e-6
                        )
                        _is_last = (
                            event.timestamp == _last_rc_ts
                            and _last_accepted_is_truly_last
                        )
                        if _is_serve:
                            _shot_type = 0
                        elif _is_last:
                            _shot_type = 2
                        else:
                            _shot_type = 1
                        shot_type_list.append(_shot_type)

                        metadata_list.append({
                            'match_date': match_date_str,
                            'match_player': match_player,
                            'match_policy': match_policy,
                            'game_id': game.game_id,
                            'game_name': game.game_name,
                            'rally_id': rally.rally_id,
                            'contact_type': event.type,
                            'contact_time': event.timestamp,
                            'match_file': match_file,
                            'base_folder': match_collection.base_folder,
                            'log_identifier': getattr(rally, 'log_identifier', None) or '',
                        })

                        # Find matching shot and flight segments around contact
                        _fs_pre, _fs_post, _shot = None, None, None
                        for _si in range(1, len(rally.shots)):
                            _sp = rally.shots[_si]
                            if len(_sp.flight_segments) == 0:
                                continue
                            _fsp = _sp.flight_segments[0]
                            if _fsp.trigger_event is not None and abs(_fsp.trigger_event.timestamp - event.timestamp) < 1e-6:
                                _shot = _sp
                                _fs_post = _fsp
                                _sp_prev = rally.shots[_si - 1]
                                if len(_sp_prev.flight_segments) > 0:
                                    _fs_pre = _sp_prev.flight_segments[-1]
                                break
                        shot_list.append(_shot)
                        rally_list.append(rally)
                        fs_pre_list.append(_fs_pre)
                        fs_post_list.append(_fs_post)

                        # Confidence: min across both adjacent segments
                        min_conf = 1.0
                        for _fs in [_fs_pre, _fs_post]:
                            if _fs is not None and all(col in _fs.data.columns for col in confidence_cols):
                                _conf_vals = _fs.data[confidence_cols].values
                                _seg_min = np.nanmin(_conf_vals)
                                if not np.isnan(_seg_min):
                                    min_conf = min(min_conf, _seg_min)
                        confidence_list.append(min_conf)

                        # Max RMSE (x_aps vs x_opt)
                        _max_rmse = np.nan
                        for _fs in [_fs_pre, _fs_post]:
                            if _fs is None:
                                continue
                            _has_opt = all(c in _fs.data.columns for c in ['x_opt', 'y_opt', 'z_opt'])
                            _has_aps = all(c in _fs.data.columns for c in ['x_aps', 'y_aps', 'z_aps'])
                            if _has_opt and _has_aps:
                                _xo = _fs.data['x_opt'].values
                                _yo = _fs.data['y_opt'].values
                                _zo = _fs.data['z_opt'].values
                                _xa = _fs.data['x_aps'].values
                                _ya = _fs.data['y_aps'].values
                                _za = _fs.data['z_aps'].values
                                _v = ~(np.isnan(_xa) | np.isnan(_xo))
                                if np.sum(_v) > 0:
                                    _dx = _xa[_v] - _xo[_v]
                                    _dy = _ya[_v] - _yo[_v]
                                    _dz = _za[_v] - _zo[_v]
                                    _seg_rmse = np.sqrt(np.mean(_dx**2 + _dy**2 + _dz**2))
                                    if np.isnan(_max_rmse) or _seg_rmse > _max_rmse:
                                        _max_rmse = _seg_rmse
                        max_rmse_opt_list.append(_max_rmse)

        _diag_accepted = len(arrays['vx_pre'])
        print(f"[RCM-diag] extract_rcm_data: total_events={_diag_total_events}, "
              f"racket_events={_diag_racket_events}, no_pre_post={_diag_no_pre_post}, "
              f"nan_values={_diag_nan_values}, accepted={_diag_accepted}")
        if _diag_racket_events > 0 and _diag_no_pre_post == _diag_racket_events:
            # ALL events are un-enriched — try to diagnose why
            print("[RCM-diag] ALL racket events are un-enriched (ball_pre=None).")
            # Check if CSV exists and show a sample key comparison
            if _csv_exists:
                try:
                    import pandas as _pd
                    _csv = _pd.read_csv(_csv_path, nrows=5)
                    print(f"[RCM-diag] CSV first rows columns: {list(_csv.columns)}")
                    if 'sequence_name' in _csv.columns:
                        _csv_seqs = list(_csv['sequence_name'].head(3))
                        print(f"[RCM-diag] CSV sample sequence_names: {_csv_seqs}")
                    else:
                        print(f"[RCM-diag] WARNING: 'sequence_name' column NOT in CSV!")
                except Exception as _e:
                    print(f"[RCM-diag] Could not read CSV for diag: {_e}")
            # Show what the expected sequence names look like
            _sample_seqs = set()
            for _m in match_collection.matches[:3]:
                _pn = _m.file.parent.name
                for _g in _m.games[:1]:
                    for _r in _g.rallies[:1]:
                        _sample_seqs.add(f"{_pn}_game_{_g.game_id}_rally_{_r.rally_id}")
            print(f"[RCM-diag] Expected seq_name pattern (from data): {list(_sample_seqs)[:3]}")

        result = {k: np.array(v) for k, v in arrays.items()}
        result['player'] = np.array(player_list, dtype=int)
        result['shot_type'] = np.array(shot_type_list, dtype=int)  # 0=serve, 1=rally, 2=last
        result['metadata_list'] = metadata_list
        result['shot_list'] = shot_list
        result['rally_list'] = rally_list
        result['fs_pre_list'] = fs_pre_list
        result['fs_post_list'] = fs_post_list
        result['confidence'] = np.array(confidence_list)
        result['max_rmse_opt'] = np.array(max_rmse_opt_list)
        return result

    def extract_table_contact_data(
        self, match_collection: MatchCollection, player_type: Optional[str] = "robot", enabled_indices: List[int] = None,
        use_extracted_tcm: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract table contact data from consecutive flight segments.

        Table contacts occur when a ball bounces on the table, creating a transition between
        two flight segments. This function extracts pre-contact and post-contact velocity/spin
        from the aerodynamics data stored in HDF5.

        Args:
            match_collection: MatchCollection object
            player_type: "robot" or "player" to filter by player type
            enabled_indices: List of match indices to include (None = all matches)
            use_extracted_tcm: When True (default), prefer enriched TableContactEvent
                data originating from extracted_TCM.csv.  When False, always fall
                back to the iloc-based extraction from HDF5 aerodynamics columns.

        Returns:
            Dictionary with arrays:
                - vx_pre, vy_pre, vz_pre: Pre-contact velocity (end of prev segment)
                - wx_pre, wy_pre, wz_pre: Pre-contact spin (end of prev segment)
                - vx_post, vy_post, vz_post: Post-contact velocity (start of next segment)
                - wx_post, wy_post, wz_post: Post-contact spin (start of next segment)
                - metadata_list: List of metadata dicts
                - shot_list, rally_list: Object references
        """
        # ── Fast cache path ──
        if self._cache is not None and self._cache.table_contact_data is not None:
            tcd = self._cache.table_contact_data
            # Build filter: enabled_indices + player_type
            # Note: table contacts don't store player directly in the cache dict,
            # but the shot_list has the player attribute.
            keep_indices = list(range(len(tcd['metadata_list'])))
            if enabled_indices is not None:
                enabled_set = set(enabled_indices)
                keep_indices = [i for i in keep_indices
                                if tcd['metadata_list'][i].get('match_idx') in enabled_set]
            if player_type is not None:
                pf = 1 if player_type == "robot" else 2
                keep_indices = [i for i in keep_indices
                                if tcd['shot_list'][i] is not None and tcd['shot_list'][i].player == pf]
            keep = np.array(keep_indices, dtype=int)
            if len(keep) == 0:
                return {k: (np.array([]) if isinstance(v, np.ndarray) else [])
                        for k, v in tcd.items()}
            result = {}
            for k, v in tcd.items():
                if isinstance(v, np.ndarray):
                    result[k] = v[keep]
                elif isinstance(v, list):
                    result[k] = [v[i] for i in keep]
                else:
                    result[k] = v
            return result

        # ── Fallback: full traversal ──
        # Data lists
        vx_pre_list, vy_pre_list, vz_pre_list = [], [], []
        wx_pre_list, wy_pre_list, wz_pre_list = [], [], []
        vx_post_list, vy_post_list, vz_post_list = [], [], []
        wx_post_list, wy_post_list, wz_post_list = [], [], []
        metadata_list = []
        shot_list = []
        rally_list = []
        fs_pre_list = []  # Track pre-contact flight segments
        fs_post_list = []  # Track post-contact flight segments
        segment_index_list = []  # Track segment index within shot
        confidence_list = []  # Track minimum confidence of both segments
        max_rmse_opt_list = []  # Track max RMSE error (x_aps vs x_opt) between both segments
        # Model prediction accumulators
        _tc_model_keys = [
            'vx_post_ittf', 'vy_post_ittf', 'vz_post_ittf', 'wx_post_ittf', 'wy_post_ittf', 'wz_post_ittf',
            'vx_post_paper', 'vy_post_paper', 'vz_post_paper', 'wx_post_paper', 'wy_post_paper', 'wz_post_paper',
            'vx_post_rcn', 'vy_post_rcn', 'vz_post_rcn', 'wx_post_rcn', 'wy_post_rcn', 'wz_post_rcn',
            'vx_post_res0805', 'vy_post_res0805', 'vz_post_res0805', 'wx_post_res0805', 'wy_post_res0805', 'wz_post_res0805',
            'vx_post_pysr', 'vy_post_pysr', 'vz_post_pysr', 'wx_post_pysr', 'wy_post_pysr', 'wz_post_pysr',
            'vx_post_0426', 'vy_post_0426', 'vz_post_0426', 'wx_post_0426', 'wy_post_0426', 'wz_post_0426',
        ]
        _tc_model_lists = {k: [] for k in _tc_model_keys}

        confidence_cols = SPIN_CONFIDENCE_COLS

        # player_filter: 1=robot, 2=player, None=all
        player_filter = None if player_type is None else (1 if player_type == "robot" else 2)

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            match_date_str = str(match.date)
            match_player = match.player
            match_file = match.file

            for game in match.games:
                game_name = game.game_name
                game_id = game.game_id

                for rally in game.rallies:
                    rally_id = rally.rally_id

                    for shot in rally.shots:
                        if player_filter is not None and shot.player != player_filter:
                            continue

                        # Look for consecutive flight segments with table bounce
                        segments = shot.flight_segments
                        for i in range(len(segments) - 1):
                            fs_pre = segments[i]
                            fs_post = segments[i + 1]

                            # Check if post segment starts with a table bounce event
                            if fs_post.trigger_event is None:
                                continue
                            if fs_post.trigger_event.type not in ("bounce_p1", "bounce_p2"):
                                continue

                            try:
                                # ── Prefer enriched TableContactEvent (from extracted_TCM.csv) ──
                                _tc_evt = fs_post.trigger_event
                                _tc_enriched = (
                                    use_extracted_tcm
                                    and isinstance(_tc_evt, TableContactEvent)
                                    and _tc_evt.ball_pre is not None
                                    and _tc_evt.ball_post is not None
                                    and not np.isnan(_tc_evt.ball_pre.vx)
                                )
                                if _tc_enriched:
                                    _bp = _tc_evt.ball_pre
                                    _ba = _tc_evt.ball_post
                                    vx_pre, vy_pre, vz_pre = _bp.vx, _bp.vy, _bp.vz
                                    wx_pre, wy_pre, wz_pre = _bp.wx, _bp.wy, _bp.wz
                                    vx_post, vy_post, vz_post = _ba.vx, _ba.vy, _ba.vz
                                    wx_post, wy_post, wz_post = _ba.wx, _ba.wy, _ba.wz
                                else:
                                    # ── Fallback: iloc-based extraction ──
                                    # Check both segments have aerodynamics data
                                    cols_pre = fs_pre.data.columns
                                    cols_post = fs_post.data.columns

                                    has_aero_pre = "vx_opt" in cols_pre or "vel_opt_x" in cols_pre
                                    has_aero_post = "vx_opt" in cols_post or "vel_opt_x" in cols_post

                                    if not has_aero_pre or not has_aero_post:
                                        continue

                                    # Get column names (handle different naming conventions)
                                    if "vx_opt" in cols_pre:
                                        vx_col, vy_col, vz_col = "vx_opt", "vy_opt", "vz_opt"
                                    else:
                                        vx_col, vy_col, vz_col = "vel_opt_x", "vel_opt_y", "vel_opt_z"

                                    # Get spin columns
                                    if "wx_opt" in cols_pre:
                                        wx_col, wy_col, wz_col = "wx_opt", "wy_opt", "wz_opt"
                                    elif "wx" in cols_pre:
                                        wx_col, wy_col, wz_col = "wx", "wy", "wz"
                                    elif "spin_x" in cols_pre:
                                        wx_col, wy_col, wz_col = "spin_x", "spin_y", "spin_z"
                                    else:
                                        continue

                                    # Pre-contact: end of previous segment
                                    pre_row = fs_pre.data.iloc[-1]
                                    vx_pre = pre_row[vx_col]
                                    vy_pre = pre_row[vy_col]
                                    vz_pre = pre_row[vz_col]
                                    wx_pre = pre_row[wx_col]
                                    wy_pre = pre_row[wy_col]
                                    wz_pre = pre_row[wz_col]

                                    # Post-contact: 2nd sample of next segment (skip 1st to account for label offset)
                                    if len(fs_post.data) < 2:
                                        continue
                                    post_row = fs_post.data.iloc[1]
                                    vx_post = post_row[vx_col if vx_col in cols_post else "vx_opt"]
                                    vy_post = post_row[vy_col if vy_col in cols_post else "vy_opt"]
                                    vz_post = post_row[vz_col if vz_col in cols_post else "vz_opt"]
                                    wx_post = post_row[wx_col if wx_col in cols_post else "wx_opt"]
                                    wy_post = post_row[wy_col if wy_col in cols_post else "wy_opt"]
                                    wz_post = post_row[wz_col if wz_col in cols_post else "wz_opt"]

                                # Skip if any NaN values
                                vals = [vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre,
                                       vx_post, vy_post, vz_post, wx_post, wy_post, wz_post]
                                if any(np.isnan(v) for v in vals):
                                    continue

                                vx_pre_list.append(vx_pre)
                                vy_pre_list.append(vy_pre)
                                vz_pre_list.append(vz_pre)
                                wx_pre_list.append(wx_pre)
                                wy_pre_list.append(wy_pre)
                                wz_pre_list.append(wz_pre)
                                vx_post_list.append(vx_post)
                                vy_post_list.append(vy_post)
                                vz_post_list.append(vz_post)
                                wx_post_list.append(wx_post)
                                wy_post_list.append(wy_post)
                                wz_post_list.append(wz_post)

                                metadata = {
                                    "match_date": match_date_str,
                                    "match_player": match_player,
                                    "game_name": game_name,
                                    "game_id": game_id,
                                    "rally_id": rally_id,
                                    "contact_time": fs_post.trigger_event.timestamp,
                                    "contact_type": fs_post.trigger_event.type,
                                    "match_file": match_file,
                                    "base_folder": match_collection.base_folder,
                                    "segment_index": i,  # Index of pre-contact segment
                                    "log_identifier": getattr(rally, 'log_identifier', None) or '',
                                }
                                metadata_list.append(metadata)
                                shot_list.append(shot)
                                rally_list.append(rally)
                                fs_pre_list.append(fs_pre)
                                fs_post_list.append(fs_post)
                                segment_index_list.append(i)

                                # Compute minimum confidence across both segments
                                min_conf = 1.0
                                for fs in [fs_pre, fs_post]:
                                    if all(col in fs.data.columns for col in confidence_cols):
                                        conf_vals = fs.data[confidence_cols].values
                                        seg_min_conf = np.nanmin(conf_vals)
                                        if not np.isnan(seg_min_conf):
                                            min_conf = min(min_conf, seg_min_conf)
                                confidence_list.append(min_conf)

                                # Compute max RMSE error (x_aps vs x_opt) between both segments
                                max_rmse = np.nan
                                for fs in [fs_pre, fs_post]:
                                    has_opt = all(c in fs.data.columns for c in ['x_opt', 'y_opt', 'z_opt'])
                                    has_aps = all(c in fs.data.columns for c in ['x_aps', 'y_aps', 'z_aps'])
                                    if has_opt and has_aps:
                                        x_opt = fs.data['x_opt'].values
                                        y_opt = fs.data['y_opt'].values
                                        z_opt = fs.data['z_opt'].values
                                        x_aps = fs.data['x_aps'].values
                                        y_aps = fs.data['y_aps'].values
                                        z_aps = fs.data['z_aps'].values
                                        valid = ~(np.isnan(x_aps) | np.isnan(x_opt))
                                        if np.sum(valid) > 0:
                                            dx = x_aps[valid] - x_opt[valid]
                                            dy = y_aps[valid] - y_opt[valid]
                                            dz = z_aps[valid] - z_opt[valid]
                                            seg_rmse = np.sqrt(np.mean(dx**2 + dy**2 + dz**2))
                                            if np.isnan(max_rmse) or seg_rmse > max_rmse:
                                                max_rmse = seg_rmse
                                max_rmse_opt_list.append(max_rmse)

                                # Model predictions from enriched TableContactEvent
                                if _tc_enriched and isinstance(_tc_evt, TableContactEvent):
                                    for _comp, _attr in [
                                        ('ittf', 'ball_post_nakashima_ittf'),
                                        ('paper', 'ball_post_nakashima_paper'),
                                        ('rcn', 'ball_post_residual_corrected'),
                                        ('res0805', 'ball_post_residual0805'),
                                        ('pysr', 'ball_post_pysr'),
                                        ('0426', 'ball_post_0426'),
                                    ]:
                                        _src = getattr(_tc_evt, _attr, None)
                                        if _src is not None:
                                            _tc_model_lists[f'vx_post_{_comp}'].append(_src.vx)
                                            _tc_model_lists[f'vy_post_{_comp}'].append(_src.vy)
                                            _tc_model_lists[f'vz_post_{_comp}'].append(_src.vz)
                                            _tc_model_lists[f'wx_post_{_comp}'].append(_src.wx)
                                            _tc_model_lists[f'wy_post_{_comp}'].append(_src.wy)
                                            _tc_model_lists[f'wz_post_{_comp}'].append(_src.wz)
                                        else:
                                            for _ax in ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']:
                                                _tc_model_lists[f'{_ax}_post_{_comp}'].append(np.nan)
                                else:
                                    for _mk in _tc_model_keys:
                                        _tc_model_lists[_mk].append(np.nan)

                            except Exception:
                                continue

        return {
            'vx_pre': np.array(vx_pre_list),
            'vy_pre': np.array(vy_pre_list),
            'vz_pre': np.array(vz_pre_list),
            'wx_pre': np.array(wx_pre_list),
            'wy_pre': np.array(wy_pre_list),
            'wz_pre': np.array(wz_pre_list),
            'vx_post': np.array(vx_post_list),
            'vy_post': np.array(vy_post_list),
            'vz_post': np.array(vz_post_list),
            'wx_post': np.array(wx_post_list),
            'wy_post': np.array(wy_post_list),
            'wz_post': np.array(wz_post_list),
            'metadata_list': metadata_list,
            'shot_list': shot_list,
            'rally_list': rally_list,
            'fs_pre_list': fs_pre_list,
            'fs_post_list': fs_post_list,
            'segment_index_list': segment_index_list,
            'confidence': np.array(confidence_list),
            'max_rmse_opt': np.array(max_rmse_opt_list),
            **{k: np.array(v) for k, v in _tc_model_lists.items()},
        }

    def extract_contact_residual_errors(
        self, match_collection: MatchCollection, player_type: Optional[str] = "robot", enabled_indices: List[int] = None, epsilon: float = 0.9
    ) -> Dict[str, Any]:
        """
        Extract table contact residual errors for confidence vs error plot.

        Computes velocity and spin magnitude errors between the residual model prediction
        and the actual post-contact values from the pseudoGT data.

        Args:
            match_collection: MatchCollection object
            player_type: "robot" or "player" to filter by player type
            enabled_indices: List of match indices to include (None = all matches)
            epsilon: Coefficient of restitution for the residual model

        Returns:
            Dictionary with:
                - velocity_error: Magnitude of velocity prediction error (m/s)
                - spin_error: Magnitude of spin prediction error (rad/s)
                - confidence: Minimum confidence of both contact segments
                - metadata_list: List of metadata dicts
                - shot_list, rally_list: Object references
        """
        # First, extract the table contact data
        contact_data = self.extract_table_contact_data(match_collection, player_type, enabled_indices)

        if len(contact_data['vx_pre']) == 0:
            return {
                'velocity_error': np.array([]),
                'spin_error': np.array([]),
                'confidence': np.array([]),
                'metadata_list': [],
                'shot_list': [],
                'rally_list': [],
            }

        # Import the residual model function (try multiple import paths)
        try:
            from .plot_widgets.table_contact_plot import contact_model_residual_vectorized
        except ImportError:
            try:
                from plot_widgets.table_contact_plot import contact_model_residual_vectorized  # type: ignore[no-redef]
            except ImportError:
                from data_plotter.plot_widgets.table_contact_plot import contact_model_residual_vectorized  # type: ignore[no-redef]

        # Get pre-contact values
        vx_pre = contact_data['vx_pre']
        vy_pre = contact_data['vy_pre']
        vz_pre = contact_data['vz_pre']
        wx_pre = contact_data['wx_pre']
        wy_pre = contact_data['wy_pre']
        wz_pre = contact_data['wz_pre']

        # Get actual post-contact values (ground truth)
        vx_post_gt = contact_data['vx_post']
        vy_post_gt = contact_data['vy_post']
        vz_post_gt = contact_data['vz_post']
        wx_post_gt = contact_data['wx_post']
        wy_post_gt = contact_data['wy_post']
        wz_post_gt = contact_data['wz_post']

        # Compute model predictions
        epsilon_arr = np.full(len(vx_pre), epsilon)
        vx_pred, vy_pred, vz_pred, wx_pred, wy_pred, wz_pred = contact_model_residual_vectorized(
            vx_pre, vy_pre, vz_pre, wx_pre, wy_pre, wz_pre, epsilon_arr
        )

        # Compute velocity error magnitude (difference between predicted and actual)
        dvx = vx_pred - vx_post_gt
        dvy = vy_pred - vy_post_gt
        dvz = vz_pred - vz_post_gt
        velocity_error = np.sqrt(dvx**2 + dvy**2 + dvz**2)

        # Compute spin error magnitude
        dwx = wx_pred - wx_post_gt
        dwy = wy_pred - wy_post_gt
        dwz = wz_pred - wz_post_gt
        spin_error = np.sqrt(dwx**2 + dwy**2 + dwz**2)

        # Compute GT200 vs CURR spin angle for each contact
        # Use pre-computed values from cache if available
        n_contacts = len(vx_pre)
        if 'spin_angles' in contact_data:
            spin_angles = contact_data['spin_angles']
        else:
            # Fallback: compute per-contact spin angles (slow path)
            spin_angles = np.full(n_contacts, np.nan)

            fs_pre_list = contact_data['fs_pre_list']
            fs_post_list = contact_data['fs_post_list']

            for i in range(n_contacts):
                max_angle = np.nan

                for fs in [fs_pre_list[i], fs_post_list[i]]:
                    data = fs.data

                    # Check for required columns
                    has_gt200_spin = all(col in data.columns for col in ['wx', 'wy', 'wz'])
                    has_curr_spin = all(col in data.columns for col in ['wx_curr', 'wy_curr', 'wz_curr'])

                    if not (has_gt200_spin and has_curr_spin):
                        continue

                    # Get spin vectors using mode (most frequent value)
                    wx_gt200 = _fast_mode(data['wx'])
                    wy_gt200 = _fast_mode(data['wy'])
                    wz_gt200 = _fast_mode(data['wz'])

                    wx_curr = _fast_mode(data['wx_curr'])
                    wy_curr = _fast_mode(data['wy_curr'])
                    wz_curr = _fast_mode(data['wz_curr'])

                    # Skip if any NaN
                    if (np.isnan(wx_gt200) or np.isnan(wy_gt200) or np.isnan(wz_gt200) or
                        np.isnan(wx_curr) or np.isnan(wy_curr) or np.isnan(wz_curr)):
                        continue

                    # Compute magnitudes
                    gt200_mag = np.sqrt(wx_gt200**2 + wy_gt200**2 + wz_gt200**2)
                    curr_mag = np.sqrt(wx_curr**2 + wy_curr**2 + wz_curr**2)

                    # Skip near-zero spin
                    if gt200_mag < 1e-6 or curr_mag < 1e-6:
                        continue

                    # Compute angle between vectors
                    dot_product = wx_gt200 * wx_curr + wy_gt200 * wy_curr + wz_gt200 * wz_curr
                    cos_angle = np.clip(dot_product / (gt200_mag * curr_mag), -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    # Take acute angle (0-90 degrees)
                    angle_deg = min(angle_deg, 180.0 - angle_deg)

                    # Keep max angle
                    if np.isnan(max_angle) or angle_deg > max_angle:
                        max_angle = angle_deg

                spin_angles[i] = max_angle

        return {
            'velocity_error': velocity_error,
            'spin_error': spin_error,
            'spin_angles': spin_angles,
            'confidence': contact_data['confidence'],
            'max_rmse_opt': contact_data.get('max_rmse_opt', np.full(n_contacts, np.nan)),
            'metadata_list': contact_data['metadata_list'],
            'shot_list': contact_data['shot_list'],
            'rally_list': contact_data['rally_list'],
            'fs_pre_list': contact_data['fs_pre_list'],
            'fs_post_list': contact_data['fs_post_list'],
        }

    def extract_confidence_distribution_data(
        self, match_collection: MatchCollection, enabled_indices: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract velocity and spin magnitude data grouped by confidence level

        Args:
            match_collection: MatchCollection object
            enabled_indices: List of match indices to include (None = all matches)

        Returns:
            Tuple of (vel_high_conf, spin_high_conf, vel_low_conf, spin_low_conf, duration_high_conf, duration_low_conf, dates_high_conf, dates_low_conf)
            vel_high_conf: Velocity magnitudes with min confidence >= 0.5
            spin_high_conf: Spin magnitudes with min confidence >= 0.5
            vel_low_conf: Velocity magnitudes with min confidence < 0.5
            spin_low_conf: Spin magnitudes with min confidence < 0.5
            duration_high_conf: Segment durations (s) with min confidence >= 0.5
            duration_low_conf: Segment durations (s) with min confidence < 0.5
            dates_high_conf: Match dates for segments with min confidence >= 0.5
            dates_low_conf: Match dates for segments with min confidence < 0.5
        """
        # Use cache if available
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache

            # Build filter mask: has confidence + has GT200 data
            mask = cache.has_confidence & cache.has_gt200

            if enabled_indices is not None:
                # Use Numba-accelerated mask building
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                match_mask = build_enabled_mask(cache.match_indices, enabled_arr)
                mask = mask & match_mask

            # Extract data using cache and Numba-accelerated magnitudes
            vx_masked = cache.vx_gt200[mask]
            vy_masked = cache.vy_gt200[mask]
            vz_masked = cache.vz_gt200[mask]
            wx_masked = cache.wx[mask]
            wy_masked = cache.wy[mask]
            wz_masked = cache.wz[mask]

            vel_last = compute_magnitudes_3d(vx_masked, vy_masked, vz_masked)
            spin_last = compute_magnitudes_3d(wx_masked, wy_masked, wz_masked)
            conf_last = cache.conf_last_point[mask]
            durations = cache.durations[mask]

            # Filter valid (non-NaN and positive velocity) using Numba
            valid = ~np.isnan(vel_last) & ~np.isnan(spin_last) & ~np.isnan(conf_last) & filter_valid_velocities(vel_last)
            vel_last = vel_last[valid]
            spin_last = spin_last[valid]
            conf_last = conf_last[valid]
            durations = durations[valid]

            # Get match dates for valid segments
            match_dates_arr = np.array(cache.match_dates)[mask][valid]

            # Split by confidence threshold (0.5)
            high_conf_mask = conf_last >= 0.5
            low_conf_mask = ~high_conf_mask

            vel_high_conf = vel_last[high_conf_mask]
            spin_high_conf = spin_last[high_conf_mask]
            duration_high_conf = durations[high_conf_mask]
            dates_high_conf = match_dates_arr[high_conf_mask]

            vel_low_conf = vel_last[low_conf_mask]
            spin_low_conf = spin_last[low_conf_mask]
            duration_low_conf = durations[low_conf_mask]
            dates_low_conf = match_dates_arr[low_conf_mask]

            return (
                vel_high_conf,
                spin_high_conf,
                vel_low_conf,
                spin_low_conf,
                duration_high_conf,
                duration_low_conf,
                dates_high_conf,
                dates_low_conf,
            )

        # Fallback to original implementation
        data_high = []  # [speed_mag, spin_mag, duration, match_date]
        data_low = []   # [speed_mag, spin_mag, duration, match_date]

        files_without_confidence = set()
        files_with_confidence = set()

        confidence_cols = SPIN_CONFIDENCE_COLS
        vel_cols = ['vx_gt200', 'vy_gt200', 'vz_gt200']
        spin_cols = ['wx', 'wy', 'wz']

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            match_has_confidence = False
            match_date_str = str(match.date)

            for game in match.games:
                for rally in game.rallies:
                    for shot in rally.shots:
                        for fs in shot.flight_segments:
                            if len(fs.data) <= 2:
                                continue

                            cols = fs.data.columns

                            if "vx_gt200" not in cols or "wx" not in cols:
                                continue

                            if not all(col in cols for col in confidence_cols):
                                continue

                            match_has_confidence = True

                            try:
                                vel_indices = [cols.get_loc(c) for c in vel_cols]
                                spin_indices = [cols.get_loc(c) for c in spin_cols]
                                conf_indices = [cols.get_loc(c) for c in confidence_cols]
                                time_idx = cols.get_loc('time') if 'time' in cols else None

                                last_row = fs.data.values[-1]

                                vel_values = last_row[vel_indices]
                                spin_values = last_row[spin_indices]
                                conf_values = last_row[conf_indices]

                                all_values = np.concatenate([vel_values, spin_values, conf_values])
                                if np.any(np.isnan(all_values)):
                                    continue

                                speed_mag = np.sqrt(np.sum(vel_values**2))
                                spin_mag = np.sqrt(np.sum(spin_values**2))

                                if speed_mag <= 1e-9:
                                    continue

                                min_conf: float = np.min(conf_values)

                                if time_idx is not None:
                                    duration = fs.data.values[-1, time_idx] - fs.data.values[0, time_idx]
                                else:
                                    duration = 0.0

                                if min_conf >= 0.5:
                                    data_high.append([speed_mag, spin_mag, duration, match_date_str])
                                else:
                                    data_low.append([speed_mag, spin_mag, duration, match_date_str])

                            except (KeyError, IndexError):
                                continue

            if match_has_confidence:
                files_with_confidence.add(pathlib.Path(match.file).parent.name)
            else:
                files_without_confidence.add(pathlib.Path(match.file).parent.name)

        if files_without_confidence:
            print("\n" + "="*80)
            print("CONFIDENCE DISTRIBUTION DATA REPORT")
            print("="*80)
            print(f"\nFiles WITH confidence data: {len(files_with_confidence)}")
            for folder in sorted(files_with_confidence):
                print(f"  ✓ {folder}")

            print(f"\nFiles WITHOUT confidence data (skipped): {len(files_without_confidence)}")
            for folder in sorted(files_without_confidence):
                print(f"  ✗ {folder}")
            print("="*80 + "\n")

        if data_high:
            data_high_arr = np.array(data_high, dtype=object)
            vel_high_conf = data_high_arr[:, 0].astype(float)
            spin_high_conf = data_high_arr[:, 1].astype(float)
            duration_high_conf = data_high_arr[:, 2].astype(float)
            dates_high_conf = data_high_arr[:, 3]
        else:
            vel_high_conf = spin_high_conf = duration_high_conf = np.array([])
            dates_high_conf = np.array([])

        if data_low:
            data_low_arr = np.array(data_low, dtype=object)
            vel_low_conf = data_low_arr[:, 0].astype(float)
            spin_low_conf = data_low_arr[:, 1].astype(float)
            duration_low_conf = data_low_arr[:, 2].astype(float)
            dates_low_conf = data_low_arr[:, 3]
        else:
            vel_low_conf = spin_low_conf = duration_low_conf = np.array([])
            dates_low_conf = np.array([])

        return (
            vel_high_conf,
            spin_high_conf,
            vel_low_conf,
            spin_low_conf,
            duration_high_conf,
            duration_low_conf,
            dates_high_conf,
            dates_low_conf,
        )

    def extract_gcs_spin_confidence_data(
        self, match_collection: MatchCollection, enabled_indices: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract median w_confidence and wx_confidence values per trajectory for GCS spin confidence histogram.

        For each flight segment, computes the median of all w_confidence values
        (from GCS offline spin estimation) and wx_confidence values (from GT200).

        Args:
            match_collection: MatchCollection object
            enabled_indices: List of match indices to include (None = all matches)

        Returns:
            Tuple of (median_w_confidences, median_wx_confidences) arrays (one value per trajectory each)
        """
        # ── Fast path: use pre-computed cache arrays ──
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache
            mask = np.ones(cache.n_segments, dtype=bool)
            if enabled_indices is not None:
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                mask = build_enabled_mask(cache.match_indices, enabled_arr)
            # Filter out NaN entries (segments without the column)
            w_vals = cache.median_w_confidence[mask]
            wx_vals = cache.median_wx_confidence[mask]
            return w_vals[~np.isnan(w_vals)], wx_vals[~np.isnan(wx_vals)]

        # ── Fallback: traverse hierarchy ──
        median_w_confidences = []
        median_wx_confidences = []

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            for game in match.games:
                for rally in game.rallies:
                    for shot in rally.shots:
                        for fs in shot.flight_segments:
                            if len(fs.data) <= 2:
                                continue

                            cols = fs.data.columns

                            # Extract w_confidence (GCS) if available
                            if "w_confidence" in cols:
                                try:
                                    w_conf = fs.data["w_confidence"].values
                                    w_conf_valid = w_conf[~np.isnan(w_conf)]

                                    if len(w_conf_valid) > 0:
                                        median_w_confidences.append(np.median(w_conf_valid))
                                except (KeyError, IndexError):
                                    pass

                            # Extract wx_confidence (GT200) if available
                            if "wx_confidence" in cols:
                                try:
                                    wx_conf = fs.data["wx_confidence"].values
                                    wx_conf_valid = wx_conf[~np.isnan(wx_conf)]

                                    if len(wx_conf_valid) > 0:
                                        median_wx_confidences.append(np.median(wx_conf_valid))
                                except (KeyError, IndexError):
                                    pass

        return np.array(median_w_confidences), np.array(median_wx_confidences)

    def extract_aero_summary_error_data(
        self, match_collection: MatchCollection, enabled_indices: List[int] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Optional[str]]:
        """
        Extract aerodynamic error data comparing APS observations to GT200, CPP, and OPT trajectories.

        Computes position RMSE for each flight segment between:
        - APS observations vs GT200 (ground truth)
        - APS observations vs CPP (C++ forward simulation)
        - APS observations vs OPT (optimized trajectory)

        Args:
            match_collection: MatchCollection object
            enabled_indices: List of match indices to include (None = all matches)

        Returns:
            Tuple of (error_dict, confidence_array, duration_array, base_folder)
            where error_dict has keys 'gt200', 'opt' with RMSE arrays
        """
        # ── Fast path: use pre-computed cache arrays ──
        # Column order in cache: gt200=0, opt=1, 0226=2, nakashima=3, 0426=4
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache
            mask = np.ones(cache.n_segments, dtype=bool)
            if enabled_indices is not None:
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                mask = build_enabled_mask(cache.match_indices, enabled_arr)

            _n_cached_traj = cache.pos_rmse_all.shape[1] if cache.pos_rmse_all.ndim == 2 else 0
            _nan_col = np.full(mask.sum(), np.nan)

            def _col(arr, idx):
                """Safe column access — returns NaN array if index out of range."""
                if arr.ndim == 2 and idx < arr.shape[1]:
                    return arr[mask, idx]
                return _nan_col

            error_dict = {
                'gt200':      _col(cache.pos_rmse_all, 0),
                'opt':        _col(cache.pos_rmse_all, 1),
                '0226':       _col(cache.pos_rmse_all, 2),
                'nakashima':  _col(cache.pos_rmse_all, 3),
                '0426':       _col(cache.pos_rmse_all, 4),
                'gt200_max':  _col(cache.pos_max_err_all, 0),
                'opt_max':    _col(cache.pos_max_err_all, 1),
                '0226_max':   _col(cache.pos_max_err_all, 2),
                'nakashima_max': _col(cache.pos_max_err_all, 3),
                '0426_max':   _col(cache.pos_max_err_all, 4),
                'gt200_mean': _col(cache.pos_mean_err_all, 0),
                'opt_mean':   _col(cache.pos_mean_err_all, 1),
                '0226_mean':  _col(cache.pos_mean_err_all, 2),
                'nakashima_mean': _col(cache.pos_mean_err_all, 3),
                '0426_mean':  _col(cache.pos_mean_err_all, 4),
                'gt200_vel':  _col(cache.vel_rmse_all, 0),
                'opt_vel':    _col(cache.vel_rmse_all, 1),
                '0226_vel':   _col(cache.vel_rmse_all, 2),
                'nakashima_vel': _col(cache.vel_rmse_all, 3),
                '0426_vel':   _col(cache.vel_rmse_all, 4),
            }

            # Build match_dates and shot_types for masked segments
            mask_indices = np.where(mask)[0]
            match_dates_out = [cache.match_dates[i] for i in mask_indices]
            shot_types_out = np.zeros(len(mask_indices), dtype=np.int32)
            for out_idx, seg_idx in enumerate(mask_indices):
                shot = cache.shots[seg_idx]
                rally = cache.rallies[seg_idx]
                n_shots = len(rally.shots)
                shot_idx_in_rally = next((j for j, s in enumerate(rally.shots) if s is shot), -1)
                if shot_idx_in_rally == 0:
                    shot_types_out[out_idx] = 0  # serve
                elif shot_idx_in_rally == n_shots - 1:
                    shot_types_out[out_idx] = 2  # last shot
                else:
                    shot_types_out[out_idx] = 1  # rally

            return (
                error_dict,
                cache.min_confidence[mask],
                cache.durations[mask],
                str(cache.base_folder) if cache.base_folder else None,
                match_dates_out,
                shot_types_out,
            )

        # ── Fallback: traverse hierarchy ──
        error_gt200_rmse = []
        error_opt_rmse = []
        error_0226_rmse = []
        error_nakashima_rmse = []
        error_0426_rmse = []
        error_gt200_max = []
        error_opt_max = []
        error_0226_max = []
        error_nakashima_max = []
        error_0426_max = []
        error_gt200_mean = []
        error_opt_mean = []
        error_0226_mean = []
        error_nakashima_mean = []
        error_0426_mean = []
        # Velocity RMSE arrays
        error_gt200_vel = []
        error_opt_vel = []
        error_0226_vel = []
        error_nakashima_vel = []
        error_0426_vel = []
        confidences = []
        durations = []
        match_dates_list = []
        shot_types_list = []
        base_folder = None

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            if base_folder is None:
                base_folder = str(pathlib.Path(match.file).parent.parent)

            match_date_str = str(match.date)

            for game in match.games:
                for rally in game.rallies:
                    _n_shots = len(rally.shots)
                    for shot_idx, shot in enumerate(rally.shots):
                        # Classify shot type: 0=serve, 1=rally, 2=last shot
                        if _n_shots <= 1:
                            _fs_shot_type = 0  # only one shot → serve
                        elif shot_idx == 0:
                            _fs_shot_type = 0  # serve
                        elif shot_idx == _n_shots - 1:
                            _fs_shot_type = 2  # last shot
                        else:
                            _fs_shot_type = 1  # rally
                        for fs in shot.flight_segments:
                            # TEMPORARY FIX: Filter to majority confidence before processing
                            data = filter_to_majority_confidence(fs.data, print_warning=True)

                            if len(data) <= 2:
                                continue

                            # Check for required columns
                            has_aps = all(col in data.columns for col in ['x_aps', 'y_aps', 'z_aps'])
                            has_gt200 = all(col in data.columns for col in ['x_gt200', 'y_gt200', 'z_gt200'])
                            has_opt = all(col in data.columns for col in ['x_opt', 'y_opt', 'z_opt'])
                            has_0226 = all(col in data.columns for col in ['x_0226', 'y_0226', 'z_0226'])
                            has_nakashima = all(col in data.columns for col in ['x_nakashima', 'y_nakashima', 'z_nakashima'])
                            has_0426 = all(col in data.columns for col in ['x_0426', 'y_0426', 'z_0426'])

                            if not has_aps:
                                continue

                            # Get APS positions (observations)
                            x_aps = data['x_aps'].values
                            y_aps = data['y_aps'].values
                            z_aps = data['z_aps'].values

                            # Compute duration
                            if 'time' in data.columns:
                                time_vals = data['time'].values
                                duration = time_vals[-1] - time_vals[0] if len(time_vals) > 1 else 0
                            else:
                                duration = 0

                            # Compute confidence (min of spin confidence values at last point)
                            confidence = 1.0  # default
                            conf_cols = SPIN_CONFIDENCE_COLS
                            if all(col in data.columns for col in conf_cols):
                                last_idx = len(data) - 1
                                conf_values = [data[col].iloc[last_idx] for col in conf_cols]
                                confidence = min(conf_values)

                            # Compute RMSE and max error for GT200
                            if has_gt200:
                                x_gt = data['x_gt200'].values
                                y_gt = data['y_gt200'].values
                                z_gt = data['z_gt200'].values

                                # Only compute where both have valid data
                                valid = ~(np.isnan(x_aps) | np.isnan(x_gt))
                                if np.sum(valid) > 0:
                                    dx = x_aps[valid] - x_gt[valid]
                                    dy = y_aps[valid] - y_gt[valid]
                                    dz = z_aps[valid] - z_gt[valid]
                                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                    error_gt200_rmse.append(np.sqrt(np.mean(dist**2)))
                                    error_gt200_max.append(np.max(dist))
                                    error_gt200_mean.append(np.mean(dist))
                                else:
                                    error_gt200_rmse.append(np.nan)
                                    error_gt200_max.append(np.nan)
                                    error_gt200_mean.append(np.nan)
                            else:
                                error_gt200_rmse.append(np.nan)
                                error_gt200_max.append(np.nan)
                                error_gt200_mean.append(np.nan)

                            # Compute RMSE and max error for OPT
                            if has_opt:
                                x_opt = data['x_opt'].values
                                y_opt = data['y_opt'].values
                                z_opt = data['z_opt'].values

                                valid = ~(np.isnan(x_aps) | np.isnan(x_opt))
                                if np.sum(valid) > 0:
                                    dx = x_aps[valid] - x_opt[valid]
                                    dy = y_aps[valid] - y_opt[valid]
                                    dz = z_aps[valid] - z_opt[valid]
                                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                    error_opt_rmse.append(np.sqrt(np.mean(dist**2)))
                                    error_opt_max.append(np.max(dist))
                                    error_opt_mean.append(np.mean(dist))
                                else:
                                    error_opt_rmse.append(np.nan)
                                    error_opt_max.append(np.nan)
                                    error_opt_mean.append(np.nan)
                            else:
                                error_opt_rmse.append(np.nan)
                                error_opt_max.append(np.nan)
                                error_opt_mean.append(np.nan)

                            # Compute RMSE and max error for 0226
                            if has_0226:
                                x_0226 = data['x_0226'].values
                                y_0226 = data['y_0226'].values
                                z_0226 = data['z_0226'].values

                                valid = ~(np.isnan(x_aps) | np.isnan(x_0226))
                                if np.sum(valid) > 0:
                                    dx = x_aps[valid] - x_0226[valid]
                                    dy = y_aps[valid] - y_0226[valid]
                                    dz = z_aps[valid] - z_0226[valid]
                                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                    error_0226_rmse.append(np.sqrt(np.mean(dist**2)))
                                    error_0226_max.append(np.max(dist))
                                    error_0226_mean.append(np.mean(dist))
                                else:
                                    error_0226_rmse.append(np.nan)
                                    error_0226_max.append(np.nan)
                                    error_0226_mean.append(np.nan)
                            else:
                                error_0226_rmse.append(np.nan)
                                error_0226_max.append(np.nan)
                                error_0226_mean.append(np.nan)

                            # Compute RMSE and max error for Nakashima
                            if has_nakashima:
                                x_nak = data['x_nakashima'].values
                                y_nak = data['y_nakashima'].values
                                z_nak = data['z_nakashima'].values

                                valid = ~(np.isnan(x_aps) | np.isnan(x_nak))
                                if np.sum(valid) > 0:
                                    dx = x_aps[valid] - x_nak[valid]
                                    dy = y_aps[valid] - y_nak[valid]
                                    dz = z_aps[valid] - z_nak[valid]
                                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                    error_nakashima_rmse.append(np.sqrt(np.mean(dist**2)))
                                    error_nakashima_max.append(np.max(dist))
                                    error_nakashima_mean.append(np.mean(dist))
                                else:
                                    error_nakashima_rmse.append(np.nan)
                                    error_nakashima_max.append(np.nan)
                                    error_nakashima_mean.append(np.nan)
                            else:
                                error_nakashima_rmse.append(np.nan)
                                error_nakashima_max.append(np.nan)
                                error_nakashima_mean.append(np.nan)

                            # Compute RMSE and max error for 0426
                            if has_0426:
                                x_0426 = data['x_0426'].values
                                y_0426 = data['y_0426'].values
                                z_0426 = data['z_0426'].values

                                valid = ~(np.isnan(x_aps) | np.isnan(x_0426))
                                if np.sum(valid) > 0:
                                    dx = x_aps[valid] - x_0426[valid]
                                    dy = y_aps[valid] - y_0426[valid]
                                    dz = z_aps[valid] - z_0426[valid]
                                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                    error_0426_rmse.append(np.sqrt(np.mean(dist**2)))
                                    error_0426_max.append(np.max(dist))
                                    error_0426_mean.append(np.mean(dist))
                                else:
                                    error_0426_rmse.append(np.nan)
                                    error_0426_max.append(np.nan)
                                    error_0426_mean.append(np.nan)
                            else:
                                error_0426_rmse.append(np.nan)
                                error_0426_max.append(np.nan)
                                error_0426_mean.append(np.nan)

                            # ==================== VELOCITY RMSE COMPUTATION ====================
                            # Check for velocity observation columns (APS)
                            has_vel_aps = all(col in data.columns for col in ['vx_aps', 'vy_aps', 'vz_aps'])
                            # Check for velocity model columns
                            has_vel_gt200 = all(col in data.columns for col in ['vx_gt200', 'vy_gt200', 'vz_gt200'])
                            has_vel_opt = all(col in data.columns for col in ['vx_opt', 'vy_opt', 'vz_opt'])
                            has_vel_0226 = all(col in data.columns for col in ['vx_0226', 'vy_0226', 'vz_0226'])
                            has_vel_nakashima = all(col in data.columns for col in ['vx_nakashima', 'vy_nakashima', 'vz_nakashima'])
                            has_vel_0426 = all(col in data.columns for col in ['vx_0426', 'vy_0426', 'vz_0426'])

                            # For velocity RMSE, compare model velocities to APS velocity (observations)
                            if has_vel_aps:
                                vx_obs = data['vx_aps'].values
                                vy_obs = data['vy_aps'].values
                                vz_obs = data['vz_aps'].values
                            else:
                                vx_obs = vy_obs = vz_obs = None

                            # GT200 velocity RMSE vs observations
                            if has_vel_aps and has_vel_gt200:
                                vx_gt = data['vx_gt200'].values
                                vy_gt = data['vy_gt200'].values
                                vz_gt = data['vz_gt200'].values
                                valid = ~(np.isnan(vx_obs) | np.isnan(vx_gt))
                                if np.sum(valid) > 0:
                                    dvx = vx_obs[valid] - vx_gt[valid]
                                    dvy = vy_obs[valid] - vy_gt[valid]
                                    dvz = vz_obs[valid] - vz_gt[valid]
                                    vel_dist = np.sqrt(dvx**2 + dvy**2 + dvz**2)
                                    error_gt200_vel.append(np.sqrt(np.mean(vel_dist**2)))
                                else:
                                    error_gt200_vel.append(np.nan)
                            else:
                                error_gt200_vel.append(np.nan)

                            # OPT velocity RMSE vs observations
                            if has_vel_aps and has_vel_opt:
                                vx_opt = data['vx_opt'].values
                                vy_opt = data['vy_opt'].values
                                vz_opt = data['vz_opt'].values
                                valid = ~(np.isnan(vx_obs) | np.isnan(vx_opt))
                                if np.sum(valid) > 0:
                                    dvx = vx_obs[valid] - vx_opt[valid]
                                    dvy = vy_obs[valid] - vy_opt[valid]
                                    dvz = vz_obs[valid] - vz_opt[valid]
                                    vel_dist = np.sqrt(dvx**2 + dvy**2 + dvz**2)
                                    error_opt_vel.append(np.sqrt(np.mean(vel_dist**2)))
                                else:
                                    error_opt_vel.append(np.nan)
                            else:
                                error_opt_vel.append(np.nan)

                            # 0226 velocity RMSE vs observations
                            if has_vel_aps and has_vel_0226:
                                vx_0226 = data['vx_0226'].values
                                vy_0226 = data['vy_0226'].values
                                vz_0226 = data['vz_0226'].values
                                valid = ~(np.isnan(vx_obs) | np.isnan(vx_0226))
                                if np.sum(valid) > 0:
                                    dvx = vx_obs[valid] - vx_0226[valid]
                                    dvy = vy_obs[valid] - vy_0226[valid]
                                    dvz = vz_obs[valid] - vz_0226[valid]
                                    vel_dist = np.sqrt(dvx**2 + dvy**2 + dvz**2)
                                    error_0226_vel.append(np.sqrt(np.mean(vel_dist**2)))
                                else:
                                    error_0226_vel.append(np.nan)
                            else:
                                error_0226_vel.append(np.nan)

                            # Nakashima velocity RMSE vs observations
                            if has_vel_aps and has_vel_nakashima:
                                vx_nak = data['vx_nakashima'].values
                                vy_nak = data['vy_nakashima'].values
                                vz_nak = data['vz_nakashima'].values
                                valid = ~(np.isnan(vx_obs) | np.isnan(vx_nak))
                                if np.sum(valid) > 0:
                                    dvx = vx_obs[valid] - vx_nak[valid]
                                    dvy = vy_obs[valid] - vy_nak[valid]
                                    dvz = vz_obs[valid] - vz_nak[valid]
                                    vel_dist = np.sqrt(dvx**2 + dvy**2 + dvz**2)
                                    error_nakashima_vel.append(np.sqrt(np.mean(vel_dist**2)))
                                else:
                                    error_nakashima_vel.append(np.nan)
                            else:
                                error_nakashima_vel.append(np.nan)

                            # 0426 velocity RMSE vs observations
                            if has_vel_aps and has_vel_0426:
                                vx_0426 = data['vx_0426'].values
                                vy_0426 = data['vy_0426'].values
                                vz_0426 = data['vz_0426'].values
                                valid = ~(np.isnan(vx_obs) | np.isnan(vx_0426))
                                if np.sum(valid) > 0:
                                    dvx = vx_obs[valid] - vx_0426[valid]
                                    dvy = vy_obs[valid] - vy_0426[valid]
                                    dvz = vz_obs[valid] - vz_0426[valid]
                                    vel_dist = np.sqrt(dvx**2 + dvy**2 + dvz**2)
                                    error_0426_vel.append(np.sqrt(np.mean(vel_dist**2)))
                                else:
                                    error_0426_vel.append(np.nan)
                            else:
                                error_0426_vel.append(np.nan)
                            # ==================== END VELOCITY RMSE COMPUTATION ====================

                            confidences.append(confidence)
                            durations.append(duration)
                            match_dates_list.append(match_date_str)
                            shot_types_list.append(_fs_shot_type)

        error_dict = {
            'gt200': np.array(error_gt200_rmse),
            'opt': np.array(error_opt_rmse),
            '0226': np.array(error_0226_rmse),
            'nakashima': np.array(error_nakashima_rmse),
            '0426': np.array(error_0426_rmse),
            'gt200_max': np.array(error_gt200_max),
            'opt_max': np.array(error_opt_max),
            '0226_max': np.array(error_0226_max),
            'nakashima_max': np.array(error_nakashima_max),
            '0426_max': np.array(error_0426_max),
            'gt200_mean': np.array(error_gt200_mean),
            'opt_mean': np.array(error_opt_mean),
            '0226_mean': np.array(error_0226_mean),
            'nakashima_mean': np.array(error_nakashima_mean),
            '0426_mean': np.array(error_0426_mean),
            # Velocity RMSE data
            'gt200_vel': np.array(error_gt200_vel),
            'opt_vel': np.array(error_opt_vel),
            '0226_vel': np.array(error_0226_vel),
            'nakashima_vel': np.array(error_nakashima_vel),
            '0426_vel': np.array(error_0426_vel),
        }

        return error_dict, np.array(confidences), np.array(durations), base_folder, match_dates_list, np.array(shot_types_list, dtype=np.int32)

    def extract_confidence_vs_error_data(
        self, match_collection: MatchCollection, enabled_indices: List[int] = None
    ) -> Dict[str, Any]:
        """
        Extract confidence vs error data for scatter plot visualization.

        Similar to extract_aero_summary_error_data but also returns flight segments,
        shots, rallies, and metadata for interactive click handling.

        Args:
            match_collection: MatchCollection object
            enabled_indices: List of match indices to include (None = all matches)

        Returns:
            Dictionary with keys:
                - 'error_dict': {'gt200', 'opt'} -> RMSE arrays
                - 'confidence': confidence score array
                - 'duration': duration array
                - 'metadata_list': list of metadata dicts
                - 'flight_segments': list of FlightSegment objects
                - 'shots': list of Shot objects
                - 'rallies': list of Rally objects
                - 'base_folder': base folder path
        """
        # ── Fast path: use pre-computed cache arrays ──
        if self._cache is not None and self._cache.n_segments > 0:
            cache = self._cache
            mask = np.ones(cache.n_segments, dtype=bool)
            if enabled_indices is not None:
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                mask = build_enabled_mask(cache.match_indices, enabled_arr)

            indices = np.where(mask)[0]

            metadata_list = []
            for i in indices:
                metadata_list.append({
                    'match_date': cache.match_dates[i],
                    'match_player': cache.match_players[i],
                    'game_name': cache.game_names[i],
                    'game_id': int(cache.game_ids[i]),
                    'rally_id': int(cache.rally_ids[i]),
                    'segment_start_time': cache.segment_start_times[i],
                    'match_file': cache.match_files[i],
                    'base_folder': cache.base_folder,
                    'log_identifier': cache.log_identifiers[i],
                })

            _nan_col2 = np.full(mask.sum(), np.nan)
            def _col2(arr, idx):
                if arr.ndim == 2 and idx < arr.shape[1]:
                    return arr[mask, idx]
                return _nan_col2

            return {
                'error_dict': {
                    'gt200': _col2(cache.pos_rmse_mad_all, 0),
                    'opt': _col2(cache.pos_rmse_mad_all, 1),
                    '0226': _col2(cache.pos_rmse_mad_all, 2),
                    'nakashima': _col2(cache.pos_rmse_mad_all, 3),
                    '0426': _col2(cache.pos_rmse_mad_all, 4),
                },
                'confidence': cache.min_confidence[mask],
                'duration': cache.durations[mask],
                'magnus_sensitivity': cache.magnus_impulse_proxy[mask],
                'drag_sensitivity': cache.drag_impulse_proxy[mask],
                'spin_density': cache.spin_density[mask],
                'spin_variance': cache.spin_variance[mask],
                'gcs_source': [cache.gcs_sources[i] for i in indices],
                'metadata_list': metadata_list,
                'flight_segments': [cache.flight_segments[i] for i in indices],
                'shots': [cache.shots[i] for i in indices],
                'rallies': [cache.rallies[i] for i in indices],
                'base_folder': str(cache.base_folder) if cache.base_folder else None,
            }

        # ── Fallback: traverse hierarchy ──
        error_gt200 = []
        error_opt = []
        error_0226 = []
        error_0426 = []
        confidences = []
        durations = []
        magnus_sensitivities = []  # Magnus force sensitivity (v x w integral)
        drag_sensitivities = []  # Drag force sensitivity (v^3 integral)
        spin_densities = []  # Spin density (% of samples with spin conf > 0.5)
        spin_variances = []  # Spin RMSE of GCS vs GT200 (rad/s)
        gcs_source_list = []  # per-segment GCS source tag
        metadata_list = []
        flight_segments = []
        shots = []
        rallies = []
        base_folder = None

        # Build lookup from flight segment id to cache index for fast sensitivity retrieval
        cache = self.cache
        fs_to_cache_idx = {}
        if cache is not None and cache.n_segments > 0:
            for i, fs in enumerate(cache.flight_segments):
                fs_to_cache_idx[id(fs)] = i

        # Helper function to compute RMSE with outlier filtering (MAD-based)
        def compute_rmse_with_outlier_filtering(x_obs, y_obs, z_obs, x_pred, y_pred, z_pred):
            """Compute RMSE excluding outlier observations using MAD filtering."""
            valid = ~(np.isnan(x_obs) | np.isnan(y_obs) | np.isnan(z_obs) |
                     np.isnan(x_pred) | np.isnan(y_pred) | np.isnan(z_pred))
            if np.sum(valid) == 0:
                return np.nan
            dx = x_obs[valid] - x_pred[valid]
            dy = y_obs[valid] - y_pred[valid]
            dz = z_obs[valid] - z_pred[valid]
            sq_err = dx**2 + dy**2 + dz**2
            # MAD-based outlier filtering (3 * MAD from median)
            median_sq_err = np.median(sq_err)
            mad = np.median(np.abs(sq_err - median_sq_err))
            if mad > 0:
                threshold = median_sq_err + 3.0 * 1.4826 * mad
                inliers = sq_err <= threshold
                if np.sum(inliers) > 0:
                    return np.sqrt(np.mean(sq_err[inliers]))
            return np.sqrt(np.mean(sq_err))

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            if base_folder is None:
                base_folder = str(pathlib.Path(match.file).parent.parent)

            match_date_str = str(match.date)
            match_player = match.player
            match_file = match.file

            for game in match.games:
                game_name = game.game_name
                game_id = game.game_id

                for rally in game.rallies:
                    rally_id = rally.rally_id
                    rally_gcs_source = getattr(rally.rally, 'attrs', {}).get('gcs_source', 'unknown') if hasattr(rally.rally, 'attrs') else 'unknown'

                    for shot in rally.shots:
                        for fs in shot.flight_segments:
                            # TEMPORARY FIX: Filter to majority confidence before processing
                            data = filter_to_majority_confidence(fs.data, print_warning=True)

                            if len(data) <= 2:
                                continue

                            # Check for required columns
                            has_aps = all(col in data.columns for col in ['x_aps', 'y_aps', 'z_aps'])
                            has_gt200 = all(col in data.columns for col in ['x_gt200', 'y_gt200', 'z_gt200'])
                            has_opt = all(col in data.columns for col in ['x_opt', 'y_opt', 'z_opt'])
                            has_0226 = all(col in data.columns for col in ['x_0226', 'y_0226', 'z_0226'])

                            if not has_aps:
                                continue

                            # Get APS positions (observations)
                            x_aps = data['x_aps'].values
                            y_aps = data['y_aps'].values
                            z_aps = data['z_aps'].values

                            # Compute duration
                            if 'time' in data.columns:
                                time_vals = data['time'].values
                                duration = time_vals[-1] - time_vals[0] if len(time_vals) > 1 else 0
                                segment_start_time = time_vals[0]
                            else:
                                duration = 0
                                segment_start_time = 0

                            # Compute spin density: % of samples with spin confidence > 0.5 AND w_confidence >= 0.5
                            # Expected samples = duration * 200 Hz + 1 (includes both endpoints)
                            spin_density = 0.0
                            spin_conf_cols = ['wx_confidence', 'wy_confidence', 'wz_confidence']
                            if all(col in data.columns for col in spin_conf_cols):
                                # Compute min spin confidence for each sample
                                spin_conf = data[spin_conf_cols].min(axis=1).values
                                _wconf_sd = data['w_confidence'].values if 'w_confidence' in data.columns else np.ones(len(data))
                                # Count samples with confidence > 0.5 AND w_confidence >= 0.5
                                high_conf_samples = np.sum((spin_conf > 0.5) & (_wconf_sd >= 0.5))
                                # Expected total samples at 200 Hz (+1 for endpoint)
                                expected_samples = max(1, duration * 200.0 + 1)
                                spin_density = 100.0 * high_conf_samples / expected_samples

                            # Compute spin variance: std of observed spin magnitude
                            # Only consider samples where w_confidence >= 0.5
                            spin_variance = np.nan
                            gcs_spin_cols = ['wx_curr', 'wy_curr', 'wz_curr']
                            if not all(col in data.columns for col in gcs_spin_cols):
                                gcs_spin_cols = ['wx_gcs', 'wy_gcs', 'wz_gcs']  # fallback
                            if all(col in data.columns for col in gcs_spin_cols):
                                wx_gcs = data[gcs_spin_cols[0]].values
                                wy_gcs = data[gcs_spin_cols[1]].values
                                wz_gcs = data[gcs_spin_cols[2]].values
                                _wconf_sv = data['w_confidence'].values if 'w_confidence' in data.columns else np.ones(len(data))
                                valid = (~(np.isnan(wx_gcs) | np.isnan(wy_gcs) | np.isnan(wz_gcs)) &
                                         (_wconf_sv >= 0.5))
                                if np.sum(valid) > 1:
                                    wmag = np.sqrt(wx_gcs[valid]**2 + wy_gcs[valid]**2 + wz_gcs[valid]**2)
                                    spin_variance = np.std(wmag)

                            # Compute confidence (min of spin confidence values at last point)
                            confidence = 1.0  # default
                            if all(col in data.columns for col in SPIN_CONFIDENCE_COLS):
                                last_idx = len(data) - 1
                                conf_values = [data[col].iloc[last_idx] for col in SPIN_CONFIDENCE_COLS]
                                confidence = min(conf_values)

                            # Compute RMSE for GT200 (with outlier filtering)
                            rmse_gt200 = np.nan
                            if has_gt200:
                                x_gt = data['x_gt200'].values
                                y_gt = data['y_gt200'].values
                                z_gt = data['z_gt200'].values
                                rmse_gt200 = compute_rmse_with_outlier_filtering(
                                    x_aps, y_aps, z_aps, x_gt, y_gt, z_gt)

                            # Compute RMSE for OPT (with outlier filtering)
                            rmse_opt = np.nan
                            if has_opt:
                                x_opt = data['x_opt'].values
                                y_opt = data['y_opt'].values
                                z_opt = data['z_opt'].values
                                rmse_opt = compute_rmse_with_outlier_filtering(
                                    x_aps, y_aps, z_aps, x_opt, y_opt, z_opt)

                            # Compute RMSE for 0226 (with outlier filtering)
                            rmse_0226 = np.nan
                            if has_0226:
                                x_0226 = data['x_0226'].values
                                y_0226 = data['y_0226'].values
                                z_0226 = data['z_0226'].values
                                rmse_0226 = compute_rmse_with_outlier_filtering(
                                    x_aps, y_aps, z_aps, x_0226, y_0226, z_0226)

                            # Compute RMSE for 0426 (with outlier filtering)
                            rmse_0426 = np.nan
                            has_0426_cols = all(col in data.columns for col in ['x_0426', 'y_0426', 'z_0426'])
                            if has_0426_cols:
                                x_0426 = data['x_0426'].values
                                y_0426 = data['y_0426'].values
                                z_0426 = data['z_0426'].values
                                rmse_0426 = compute_rmse_with_outlier_filtering(
                                    x_aps, y_aps, z_aps, x_0426, y_0426, z_0426)

                            # Store data
                            error_gt200.append(rmse_gt200)
                            error_opt.append(rmse_opt)
                            error_0226.append(rmse_0226)
                            error_0426.append(rmse_0426)
                            confidences.append(confidence)
                            durations.append(duration)

                            # Get sensitivity values from cache
                            magnus_sens = np.nan
                            drag_sens = np.nan
                            cache_idx = fs_to_cache_idx.get(id(fs))
                            if cache_idx is not None and cache is not None:
                                magnus_sens = cache.magnus_impulse_proxy[cache_idx]
                                drag_sens = cache.drag_impulse_proxy[cache_idx]
                            magnus_sensitivities.append(magnus_sens)
                            drag_sensitivities.append(drag_sens)
                            spin_densities.append(spin_density)
                            spin_variances.append(spin_variance)
                            gcs_source_list.append(rally_gcs_source)

                            metadata = {
                                'match_date': match_date_str,
                                'match_player': match_player,
                                'game_name': game_name,
                                'game_id': game_id,
                                'rally_id': rally_id,
                                'segment_start_time': segment_start_time,
                                'match_file': match_file,
                                'base_folder': match_collection.base_folder,
                                'log_identifier': getattr(rally, 'log_identifier', None) or '',
                            }
                            metadata_list.append(metadata)
                            flight_segments.append(fs)
                            shots.append(shot)
                            rallies.append(rally)

        return {
            'error_dict': {
                'gt200': np.array(error_gt200),
                'opt': np.array(error_opt),
                '0226': np.array(error_0226),
                '0426': np.array(error_0426),
            },
            'confidence': np.array(confidences),
            'duration': np.array(durations),
            'magnus_sensitivity': np.array(magnus_sensitivities),
            'drag_sensitivity': np.array(drag_sensitivities),
            'spin_density': np.array(spin_densities),
            'spin_variance': np.array(spin_variances),
            'gcs_source': gcs_source_list,
            'metadata_list': metadata_list,
            'flight_segments': flight_segments,
            'shots': shots,
            'rallies': rallies,
            'base_folder': base_folder,
        }

    def extract_spin_observation_data(
        self, match_collection: MatchCollection, enabled_indices: List[int] = None
    ) -> Dict[str, Any]:
        """Extract per-segment spin observation quality data.

        For every flight segment computes:
            - velocity_mag:     GT200 velocity magnitude at first point via mode (m/s)
            - spin_mag_gt:      GT200 spin magnitude at first point via mode (rad/s)
            - mean_delta_spin:  mean of |‖w_GT200‖ − ‖w_gcs‖| over all timesteps
            - spin_obs_var:     variance of ‖w_gcs‖ over all timesteps
            - confidence:       min of 6 confidence columns at last point
            - duration:         segment duration (s)

        Returns dict with numpy arrays and metadata lists.
        """
        # ── Fast cache path ──
        cache = self._cache
        if cache is not None and cache.n_segments > 0:
            mask = np.ones(cache.n_segments, dtype=bool)
            if enabled_indices is not None:
                enabled_arr = np.array(list(enabled_indices), dtype=np.int32)
                mask = build_enabled_mask(cache.match_indices, enabled_arr)

            # Require: has OPT velocity (vx_opt_first not NaN) AND GT200 spin (wx_first not NaN)
            # AND GCS observation metrics computed (n_gcs_samples > 0)
            valid = (mask &
                     ~np.isnan(cache.vx_opt_first) &
                     ~np.isnan(cache.wx_first) &
                     (cache.n_gcs_samples > 0))
            idxs = np.where(valid)[0]

            # Use pre-computed magnitudes from cache
            velocity_mag = cache.vel_mag_first[idxs]
            spin_mag_gt = cache.spin_mag_first[idxs]

            # Build metadata
            metadata_list = []
            for i in idxs:
                metadata_list.append({
                    'match_date': cache.match_dates[i],
                    'match_player': cache.match_players[i],
                    'game_name': cache.game_names[i],
                    'game_id': int(cache.game_ids[i]),
                    'rally_id': int(cache.rally_ids[i]),
                    'segment_start_time': float(cache.segment_start_times[i]),
                    'match_file': cache.match_files[i],
                    'base_folder': str(cache.base_folder) if cache.base_folder else None,
                    'log_identifier': cache.log_identifiers[i],
                })

            return {
                'velocity_mag': velocity_mag,
                'spin_mag_gt': spin_mag_gt,
                'mean_delta_spin': cache.mean_delta_spin[idxs],
                'spin_obs_var': cache.spin_obs_var[idxs],
                'confidence': cache.min_confidence[idxs],
                'opt_rmse': cache.rmse_opt[idxs],
                'n_gcs_samples': cache.n_gcs_samples[idxs],
                'spin_density': cache.spin_density[idxs],
                'shot_position': cache.shot_position[idxs],
                'gcs_source': [cache.gcs_sources[i] for i in idxs],
                'vx_abs': np.abs(cache.vx_opt_first[idxs]),
                'shot_player': cache.player_types[idxs],
                'mean_x': cache.mean_x_aps[idxs],
                'mean_y': cache.mean_y_aps[idxs],
                'mean_z': cache.mean_z_aps[idxs],
                'contact_type': [cache.contact_types[i] for i in idxs],
                'metadata_list': metadata_list,
                'flight_segments': [cache.flight_segments[i] for i in idxs],
                'shots': [cache.shots[i] for i in idxs],
                'rallies': [cache.rallies[i] for i in idxs],
                'base_folder': str(cache.base_folder) if cache.base_folder else None,
            }

        # ── Fallback: full traversal ──
        velocity_mags = []
        spin_mag_gts = []
        mean_delta_spins = []
        spin_obs_vars = []
        confidences = []
        opt_rmses = []
        n_gcs_samples_list = []
        spin_densities = []
        shot_position_list = []  # 0=other, 1=second-to-last, 2=last
        gcs_source_list = []  # per-segment GCS source tag (gcs_offline, gcs_filtered, gcs, none)
        vx_abs_list = []  # per-segment |vx_opt| at first point (m/s)
        shot_player_list = []  # per-segment player id (1=robot, 2=human)
        mean_x_list = []  # per-segment mean x_aps position (m)
        mean_y_list = []  # per-segment mean y_aps position (m)
        mean_z_list = []  # per-segment mean z_aps position (m)
        contact_type_list = []  # per-segment trigger event type (e.g. start, shot_p1, shot_p2, end)
        metadata_list = []
        flight_segments = []
        shots_list = []
        rallies_list = []
        base_folder = None

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue

            if base_folder is None:
                base_folder = str(pathlib.Path(match.file).parent.parent)

            match_date_str = str(match.date)
            match_player = match.player
            match_file = match.file

            for game in match.games:
                game_name = game.game_name
                game_id = game.game_id

                for rally in game.rallies:
                    rally_id = rally.rally_id
                    # Get GCS source for this rally (stored as trajectory attr)
                    rally_gcs_source = getattr(rally.rally, 'attrs', {}).get('gcs_source', 'unknown') if hasattr(rally.rally, 'attrs') else 'unknown'

                    for shot_idx, shot in enumerate(rally.shots):
                        n_shots = len(rally.shots)
                        if shot_idx == n_shots - 1:
                            shot_position = 2  # last
                        elif shot_idx == n_shots - 2:
                            shot_position = 1  # second-to-last
                        else:
                            shot_position = 0  # other
                        for fs in shot.flight_segments:
                            data = filter_to_majority_confidence(fs.data, print_warning=True)

                            if len(data) <= 2:
                                continue

                            # Required columns
                            has_opt_vel = all(c in data.columns for c in ['vx_opt', 'vy_opt', 'vz_opt'])
                            has_gt200_spin = all(c in data.columns for c in ['wx', 'wy', 'wz'])
                            has_obs_spin = all(c in data.columns for c in ['wx_gcs', 'wy_gcs', 'wz_gcs'])

                            # DEBUG: print column diagnostics for first segment
                            if match_idx == 0 and not hasattr(self, '_spin_obs_debug_printed'):
                                self._spin_obs_debug_printed = True
                                spin_cols = [c for c in data.columns if 'w' in c.lower() and ('x' in c or 'y' in c or 'z' in c)]
                                print(f"[DEBUG spin_obs] columns with w+xyz: {sorted(spin_cols)}")
                                print(f"[DEBUG spin_obs] has_opt_vel={has_opt_vel}, has_gt200_spin={has_gt200_spin}, has_obs_spin={has_obs_spin}")
                                print(f"[DEBUG spin_obs] total columns: {len(data.columns)}")

                            if not (has_opt_vel and has_gt200_spin and has_obs_spin):
                                continue

                            # --- velocity magnitude (mode of first-point-like value) ---
                            vx_m = _fast_mode(data['vx_opt'])
                            vy_m = _fast_mode(data['vy_opt'])
                            vz_m = _fast_mode(data['vz_opt'])
                            if np.isnan(vx_m) or np.isnan(vy_m) or np.isnan(vz_m):
                                continue
                            vel_mag = np.sqrt(vx_m**2 + vy_m**2 + vz_m**2)

                            # --- GT200 spin magnitude (mode) ---
                            wx_gt = _fast_mode(data['wx'])
                            wy_gt = _fast_mode(data['wy'])
                            wz_gt = _fast_mode(data['wz'])
                            if np.isnan(wx_gt) or np.isnan(wy_gt) or np.isnan(wz_gt):
                                continue
                            spin_gt_mag = np.sqrt(wx_gt**2 + wy_gt**2 + wz_gt**2)

                            # --- Per-timestep spin magnitudes ---
                            wx_gt_arr = data['wx'].values
                            wy_gt_arr = data['wy'].values
                            wz_gt_arr = data['wz'].values
                            wx_c_arr = data['wx_gcs'].values
                            wy_c_arr = data['wy_gcs'].values
                            wz_c_arr = data['wz_gcs'].values

                            # Only consider GCS samples with w_confidence >= 0.5
                            w_conf_arr = data['w_confidence'].values if 'w_confidence' in data.columns else np.ones(len(data))
                            valid = (~(np.isnan(wx_gt_arr) | np.isnan(wy_gt_arr) | np.isnan(wz_gt_arr) |
                                       np.isnan(wx_c_arr) | np.isnan(wy_c_arr) | np.isnan(wz_c_arr)) &
                                     (w_conf_arr >= 0.5))
                            if np.sum(valid) == 0:
                                continue

                            spin_mag_gt_arr = np.sqrt(wx_gt_arr[valid]**2 + wy_gt_arr[valid]**2 + wz_gt_arr[valid]**2)
                            spin_mag_curr_arr = np.sqrt(wx_c_arr[valid]**2 + wy_c_arr[valid]**2 + wz_c_arr[valid]**2)

                            mean_delta = float(np.mean(np.abs(spin_mag_gt_arr - spin_mag_curr_arr)))
                            obs_var = float(np.var(spin_mag_curr_arr))

                            # --- OPT RMSE (position error x_opt vs x_aps in meters) ---
                            has_opt_pos = all(c in data.columns for c in ['x_opt', 'y_opt', 'z_opt'])
                            has_aps_pos = all(c in data.columns for c in ['x_aps', 'y_aps', 'z_aps'])
                            opt_rmse_val = np.nan
                            if has_opt_pos and has_aps_pos:
                                x_a = data['x_aps'].values
                                y_a = data['y_aps'].values
                                z_a = data['z_aps'].values
                                x_o = data['x_opt'].values
                                y_o = data['y_opt'].values
                                z_o = data['z_opt'].values
                                pos_valid = ~(np.isnan(x_a) | np.isnan(x_o))
                                if np.sum(pos_valid) > 0:
                                    dx = x_a[pos_valid] - x_o[pos_valid]
                                    dy = y_a[pos_valid] - y_o[pos_valid]
                                    dz = z_a[pos_valid] - z_o[pos_valid]
                                    opt_rmse_val = np.sqrt(np.mean(dx**2 + dy**2 + dz**2))

                            # --- Segment start time ---
                            if 'time' in data.columns:
                                segment_start_time = data['time'].values[0]
                            else:
                                segment_start_time = 0

                            # --- Confidence ---
                            confidence = 1.0
                            if all(col in data.columns for col in SPIN_CONFIDENCE_COLS):
                                last_idx = len(data) - 1
                                conf_values = [data[col].iloc[last_idx] for col in SPIN_CONFIDENCE_COLS]
                                confidence = min(conf_values)

                            # --- Number of non-NaN GCS spin samples with w_confidence >= 0.5 ---
                            _wconf = data['w_confidence'].values if 'w_confidence' in data.columns else np.ones(len(data))
                            n_gcs = int(np.sum(~np.isnan(data['wx_gcs'].values) & (_wconf >= 0.5)))

                            # --- Spin density: % of samples with spin conf > 0.5 AND w_confidence >= 0.5 ---
                            spin_density = 0.0
                            spin_conf_cols = ['wx_confidence', 'wy_confidence', 'wz_confidence']
                            if all(col in data.columns for col in spin_conf_cols):
                                spin_conf = data[spin_conf_cols].min(axis=1).values
                                _wconf_sd = data['w_confidence'].values if 'w_confidence' in data.columns else np.ones(len(data))
                                high_conf_samples = np.sum((spin_conf > 0.5) & (_wconf_sd >= 0.5))
                                if 'time' in data.columns:
                                    time_vals = data['time'].values
                                    seg_dur = time_vals[-1] - time_vals[0] if len(time_vals) > 1 else 0
                                else:
                                    seg_dur = 0
                                expected_samples = max(1, seg_dur * 200.0 + 1)
                                spin_density = 100.0 * high_conf_samples / expected_samples

                            # --- Store ---
                            velocity_mags.append(vel_mag)
                            spin_mag_gts.append(spin_gt_mag)
                            mean_delta_spins.append(mean_delta)
                            spin_obs_vars.append(obs_var)
                            confidences.append(confidence)
                            opt_rmses.append(opt_rmse_val)
                            n_gcs_samples_list.append(n_gcs)
                            spin_densities.append(spin_density)
                            shot_position_list.append(shot_position)
                            gcs_source_list.append(rally_gcs_source)
                            vx_abs_list.append(abs(float(vx_m)))
                            shot_player_list.append(shot.player if hasattr(shot, 'player') else 0)

                            # --- Mean APS position over the segment ---
                            _xa = data['x_aps'].values if 'x_aps' in data.columns else np.array([])
                            _ya = data['y_aps'].values if 'y_aps' in data.columns else np.array([])
                            _za = data['z_aps'].values if 'z_aps' in data.columns else np.array([])
                            mean_x_list.append(float(np.nanmean(_xa)) if len(_xa) > 0 else np.nan)
                            mean_y_list.append(float(np.nanmean(_ya)) if len(_ya) > 0 else np.nan)
                            mean_z_list.append(float(np.nanmean(_za)) if len(_za) > 0 else np.nan)

                            # --- Contact type at start of segment ---
                            ct = fs.trigger_event.type if (fs.trigger_event is not None) else 'unknown'
                            contact_type_list.append(ct)

                            metadata = {
                                'match_date': match_date_str,
                                'match_player': match_player,
                                'game_name': game_name,
                                'game_id': game_id,
                                'rally_id': rally_id,
                                'segment_start_time': segment_start_time,
                                'match_file': match_file,
                                'base_folder': match_collection.base_folder,
                                'log_identifier': getattr(rally, 'log_identifier', None) or '',
                            }
                            metadata_list.append(metadata)
                            flight_segments.append(fs)
                            shots_list.append(shot)
                            rallies_list.append(rally)

        return {
            'velocity_mag': np.array(velocity_mags),
            'spin_mag_gt': np.array(spin_mag_gts),
            'mean_delta_spin': np.array(mean_delta_spins),
            'spin_obs_var': np.array(spin_obs_vars),
            'confidence': np.array(confidences),
            'opt_rmse': np.array(opt_rmses),
            'n_gcs_samples': np.array(n_gcs_samples_list),
            'spin_density': np.array(spin_densities),
            'shot_position': np.array(shot_position_list, dtype=int),
            'gcs_source': gcs_source_list,
            'vx_abs': np.array(vx_abs_list),
            'shot_player': np.array(shot_player_list, dtype=int),
            'mean_x': np.array(mean_x_list),
            'mean_y': np.array(mean_y_list),
            'mean_z': np.array(mean_z_list),
            'contact_type': contact_type_list,
            'metadata_list': metadata_list,
            'flight_segments': flight_segments,
            'shots': shots_list,
            'rallies': rallies_list,
            'base_folder': base_folder,
        }

    def extract_net_contact_data(
        self,
        match_collection: MatchCollection,
        enabled_indices: List[int] = None,
    ) -> pd.DataFrame:
        """Extract net-contact counts per rally, split by who hit the ball.

        Each ``NetEvent`` is classified as *robot* (ball travelling with
        ``vx > 0`` at the moment of net crossing) or *player* (``vx < 0``).
        The velocity is read from ``NetEvent.ball_pre`` when available,
        otherwise from the closest sample in the rally trajectory.

        Returns:
            DataFrame with columns:
                date              – match date string  (YYYY-MM-DD)
                player            – opponent player name
                match_idx         – index in *match_collection.matches*
                game_id           – game id
                rally_id          – rally id
                net_contacts      – total net events in the rally
                net_robot         – net events from robot shots (vx > 0)
                net_player        – net events from player shots (vx < 0)
        """
        rows: list[dict] = []

        for match_idx, match in enumerate(match_collection.matches):
            if enabled_indices is not None and match_idx not in enabled_indices:
                continue
            match_date = str(match.date) if match.date else "unknown"
            match_player = match.player if match.player else "unknown"

            for game in match.games:
                for rally in game.rallies:
                    net_events = [ev for ev in rally.events if isinstance(ev, NetEvent)]
                    n_robot = 0
                    n_player = 0

                    # Get trajectory vx array once per rally for lookup
                    traj = rally.rally
                    has_traj = (
                        traj is not None
                        and isinstance(traj, pd.DataFrame)
                        and not traj.empty
                        and "time" in traj.columns
                    )
                    # The rally trajectory stores velocity as vx_gt200;
                    # fall back to vx if the column was renamed.
                    vx_col = None
                    if has_traj:
                        if "vx_gt200" in traj.columns:
                            vx_col = "vx_gt200"
                        elif "vx" in traj.columns:
                            vx_col = "vx"
                        else:
                            has_traj = False
                    traj_times = traj["time"].values if has_traj else None
                    traj_vx = traj[vx_col].values if has_traj else None

                    for ev in net_events:
                        vx = None
                        # Try ball_pre first
                        if ev.ball_pre is not None and not np.isnan(ev.ball_pre.vx):
                            vx = ev.ball_pre.vx
                        elif traj_times is not None:
                            idx = np.argmin(np.abs(traj_times - ev.timestamp))
                            vx = float(traj_vx[idx])

                        if vx is not None and vx > 0:
                            n_robot += 1
                        else:
                            n_player += 1

                    rows.append({
                        "date": match_date,
                        "player": match_player,
                        "match_idx": match_idx,
                        "game_id": game.game_id,
                        "rally_id": rally.rally_id,
                        "net_contacts": len(net_events),
                        "net_robot": n_robot,
                        "net_player": n_player,
                    })

        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=[
            "date", "player", "match_idx", "game_id", "rally_id",
            "net_contacts", "net_robot", "net_player",
        ])
