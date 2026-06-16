# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
# this script should substitute the various evaluate_* and extract_* and update the local hdf5 files instead of creating intermediate files

import csv
import pathlib
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import h5py
import sys

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from ace_evaluation.utilities.physics_params import get_params
from ace_evaluation.utilities.aerodynamics_integrator import (
    BallState,
    make_physics_params,
    predict_ball_state_new_model,
    predict_ball_state_old_model,
    get_drag_coefficient_new,
    get_magnus_coefficient_new,
)
from scipy import stats

from ace_evaluation.utilities.data_classes import (
    MatchCollection,
)

def aerodynamics_diff(_, s: np.ndarray, params: np.ndarray, const_params: np.ndarray) -> list[float]:
    """aerodynamics differential equations"""
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

def huber_loss(errors: np.ndarray, delta: float) -> np.ndarray:
    """Vectorized Huber loss computation."""
    squared = 0.5 * errors**2
    linear = delta * (errors - 0.5 * delta)
    return np.where(errors <= delta, squared, linear)


def compute_fitness(ref_pos: np.ndarray, sim_pos: np.ndarray, dt: float, delta: float = 0.05) -> float:
    """Compute fitness using Huber loss over time-aligned 3D positions."""
    # Extract timestamps and 3D positions
    ref_times = ref_pos[:, 0]
    sim_times = sim_pos[:, 0]

    ref_coords = ref_pos[:, 1:4]
    sim_coords = sim_pos[:, 1:4]

    # For each ref time, find sim samples within ±dt/2
    losses = []
    for i, t_ref in enumerate(ref_times):
        time_diff = np.abs(sim_times - t_ref)
        mask = time_diff < dt * 0.5
        if np.any(mask):
            diffs = sim_coords[mask] - ref_coords[i]
            dists = np.linalg.norm(diffs, axis=1)
            losses.extend(huber_loss(dists, delta))

    if not losses:
        return float("inf")  # no matches found

    return float(np.sqrt(np.mean(losses)))

def solve_aerodynamics(
        params: list[float],
        flight_segment: pd.DataFrame,
        tstart: float,
        tend: float,
        dt: float,
        const_params: list[float],
        pos_cols: tuple[str, str, str] = ("x_aps", "y_aps", "z_aps"),
) -> float:
    px, py, pz = pos_cols
    t_eval = np.arange(tstart, tend, dt)
    # np.arange with floats can overshoot `tend` by ~1e-13 due to
    # floating-point accumulation, causing solve_ivp to reject t_eval
    # with "Values in t_eval are not within t_span".  Clip to be safe.
    t_eval = t_eval[t_eval <= tend]
    aero_params = np.array([params[3], params[4]])
    result = solve_ivp(
        aerodynamics_diff,
        [tstart, tend],
        (flight_segment.data[px].iloc[0], flight_segment.data[py].iloc[0], flight_segment.data[pz].iloc[0], params[0], params[1], params[2]),
        t_eval=t_eval,
        method="RK45",
        args=(aero_params, const_params)
    )

    # Check if ODE solver succeeded
    if not result.success or len(result.y) < 3:
        # Return large penalty if solver failed
        return float("inf")

    ref_pos = np.stack(
        [
            flight_segment.data["time"].to_numpy(),
            flight_segment.data[px].to_numpy(),
            flight_segment.data[py].to_numpy(),
            flight_segment.data[pz].to_numpy(),
        ]
    ).T
    sim_pos = np.stack([result.t, result.y[0], result.y[1], result.y[2]]).T

    return compute_fitness(ref_pos, sim_pos, dt)

class AerodynamicsOptimizer:
    """
    Class to optimize aerodynamics parameters and estimate pseudoGT
    """

    def __init__(self, data_folder, data, recompute, use_gt200_positions=False):
        self.data_folder = data_folder
        self.data = data # Match collection
        self.recompute = recompute
        self.default_cd = 0.55
        self.default_cm = 0.08
        self.rho_air = 1.204
        self.rho_ball = 81.2
        self.radius_ball = 0.02
        self.dt = 0.005
        self.pos_cols = ("x_gt200", "y_gt200", "z_gt200") if use_gt200_positions else ("x_aps", "y_aps", "z_aps")

        print("=============================================")
        print("\t\tAerodynamics")
        print("=============================================")
        print(f"  Position source: {self.pos_cols[0].split('_', 1)[1]}")

    def clean_aerodynamics_data(self):
        """Remove all existing aerodynamics data from HDF5 files"""
        print("Cleaning existing aerodynamics data...")

        # Find all HDF5 files in the data folder
        hdf5_files = list(pathlib.Path(self.data_folder).rglob("data.h5"))

        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, "a") as h5f:
                    # Iterate through all games
                    game_keys = [k for k in h5f.keys() if k.startswith("game_")]
                    for game_key in game_keys:
                        game_group = h5f[game_key]
                        # Iterate through all rallies
                        rally_keys = [k for k in game_group.keys() if k.startswith("rally_")]
                        for rally_key in rally_keys:
                            rally_group = game_group[rally_key]
                            # Check if aerodynamics data exists and delete it
                            if "ground_truth_200" in rally_group and "aerodynamics" in rally_group["ground_truth_200"]:
                                del rally_group["ground_truth_200"]["aerodynamics"]
                print(f"  Cleaned: {file_path}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"  Warning: Failed to clean {file_path}: {e}")

        print("Aerodynamics data cleanup complete.\n")

    def update_hdf(self):
        # Clean existing aerodynamics data if recomputing
        if self.recompute:
            self.clean_aerodynamics_data()

        # Collect per-segment error stats for summary
        all_stats = []

        for match in self.data.matches:
            match_file = match.file
            for game in match.games:
                for rally in game.rallies:
                    print(f"Processing Match {match.match_id} ({match.file}), Game {game.game_id}, Rally {rally.rally_id}")

                    # Initialize as empty list instead of empty DataFrame to avoid dtype warnings
                    trajectory_physics_list = []
                    for shot_idx, shot in enumerate(rally.shots):
                        for fs_idx, flight_segment in enumerate(shot.flight_segments):
                            if len(flight_segment.data) < 2:
                                continue

                            # Determine the next event boundary so the simulation
                            # extends all the way to the contact.
                            if fs_idx + 1 < len(shot.flight_segments):
                                tend_event = shot.flight_segments[fs_idx + 1].trigger_event.timestamp
                            else:
                                tend_event = shot.end_time

                            try:
                                results_optimized, results_0226, results_nakashima, results_0426, seg_stats = self.compute_aerodynamics_pseudoGT(flight_segment, tend_event=tend_event)
                                seg_stats['match_id'] = match.match_id
                                seg_stats['game_id'] = game.game_id
                                seg_stats['rally_id'] = rally.rally_id
                                seg_stats['shot_idx'] = shot_idx
                                seg_stats['fs_idx'] = fs_idx
                                all_stats.append(seg_stats)
                            except Exception as e:  # pylint: disable=broad-exception-caught
                                print(f"  Warning: Failed to compute aerodynamics for Shot {shot_idx}, FS {fs_idx}: {e}")
                                continue
                            # merge results on "t" keeping only pos and vel, renamed accordingly
                            # Merge with tolerance for similar t values
                            tolerance = 0.0025

                            # Round t values to enable grouping
                            results_optimized['t_rounded'] = (results_optimized['t'] / tolerance).round() * tolerance
                            results_0226['t_rounded'] = (results_0226['t'] / tolerance).round() * tolerance
                            results_nakashima['t_rounded'] = (results_nakashima['t'] / tolerance).round() * tolerance
                            results_0426['t_rounded'] = (results_0426['t'] / tolerance).round() * tolerance

                            # Rename columns before merging to ensure proper suffixes
                            results_optimized_renamed = results_optimized.rename(columns={
                                't': 't_optimized',
                                'pos': 'pos_optimized',
                                'vel': 'vel_optimized',
                                'spin': 'spin_optimized',
                                'cd': 'cd_optimized',
                                'cm': 'cm_optimized',
                                'v_eff_drag': 'v_eff_drag_optimized',
                                'v_eff_magnus': 'v_eff_magnus_optimized',
                                'w_eff_perp_magnus': 'w_eff_perp_magnus_optimized'
                            })
                            results_0226_renamed = results_0226.rename(columns={
                                't': 't_0226',
                                'pos': 'pos_0226',
                                'vel': 'vel_0226',
                                'spin': 'spin_0226',
                                'cd': 'cd_0226',
                                'cm': 'cm_0226'
                            })
                            results_nakashima_renamed = results_nakashima.rename(columns={
                                't': 't_nakashima',
                                'pos': 'pos_nakashima',
                                'vel': 'vel_nakashima',
                                'cd': 'cd_nakashima',
                                'cm': 'cm_nakashima'
                            })
                            results_0426_renamed = results_0426.rename(columns={
                                't': 't_0426',
                                'pos': 'pos_0426',
                                'vel': 'vel_0426',
                            })

                            merged_results = pd.merge(
                                results_optimized_renamed,
                                results_0226_renamed,
                                on="t_rounded",
                                how="outer"
                            )
                            merged_results = pd.merge(
                                merged_results,
                                results_nakashima_renamed,
                                on="t_rounded",
                                how="outer"
                            )
                            merged_results = pd.merge(
                                merged_results,
                                results_0426_renamed,
                                on="t_rounded",
                                how="outer"
                            )

                            merged_results.drop(columns=["t_optimized", "t_0226", "t_nakashima", "t_0426", "spin_0226"], inplace=True)
                            merged_results.rename(columns={"t_rounded": "t", "spin_optimized": "spin"}, inplace=True)
                            merged_results.dropna(inplace=True)

                            trajectory_physics_list.append(merged_results)

                    # Skip saving if no data was collected
                    if len(trajectory_physics_list) == 0:
                        print(f"  No trajectory data to save for Rally {rally.rally_id}")
                        continue

                    # Concatenate all results at once
                    trajectory_physics = pd.concat(trajectory_physics_list, ignore_index=True)

                    # Save updated trajectories back to HDF5
                    with h5py.File(match_file, "a") as h5f:
                        rally_group = h5f[f"game_{game.game_id}/rally_{rally.rally_id}/ground_truth_200"]
                        if "aerodynamics" in rally_group:
                            del rally_group["aerodynamics"]
                        rally_group.create_group("aerodynamics")

                        # Define which columns contain lists (vectors) vs scalars
                        vector_columns = ["pos_optimized", "pos_0226", "pos_nakashima", "pos_0426",
                                        "vel_optimized", "vel_0226", "vel_nakashima", "vel_0426", "spin"]
                        scalar_columns = ["t", "cd_optimized", "cd_0226", "cd_nakashima",
                                        "cm_optimized", "cm_0226", "cm_nakashima",
                                        "v_eff_drag_optimized",
                                        "v_eff_magnus_optimized",
                                        "w_eff_perp_magnus_optimized"]

                        # Handle scalar columns - explicitly convert to float64
                        for col in scalar_columns:
                            if col in trajectory_physics.columns:
                                # Ensure data is numeric and convert to numpy array with explicit dtype
                                data = pd.to_numeric(trajectory_physics[col], errors='coerce').to_numpy(dtype=np.float64)
                                rally_group["aerodynamics"].create_dataset(
                                    col,
                                    data=data,
                                    dtype=np.float64
                                )

                        # Handle vector columns (pos, vel, spin) - convert list columns to 2D arrays
                        for col in vector_columns:
                            if col in trajectory_physics.columns:
                                # Convert list of lists to 2D numpy array with explicit dtype
                                try:
                                    data_array = np.array(trajectory_physics[col].tolist(), dtype=np.float64)
                                    rally_group["aerodynamics"].create_dataset(
                                        col,
                                        data=data_array,
                                        dtype=np.float64
                                    )
                                except (ValueError, TypeError) as e:
                                    print(f"Warning: Could not convert column {col} to array: {e}")
                                    # Skip this column if conversion fails
                                    continue


        # ── Print error summary ──
        self._print_error_summary(all_stats)

        # files_h5 = utilities.list_files(self.data_folder, ".h5")
        # for file_h5 in files_h5:
        #     with h5py.File(file_h5, "r+") as src:
        #         for game, game_data in src.items():
        #             for rally, rally_data in game_data.items():
        #                 if not "aerodynamics" in rally_data["ground_truth_200"]["ball_physics"] or self.recompute:
        #                     self.compute_aerodynamics_pseudoGT(rally, rally_data)

    def compute_aerodynamics_pseudoGT(self, flight_segment, tend_event=None):
        # Estimate velocity with finite differences
        flight_segment.data = flight_segment.estimate_velocity()
        vx_fd = flight_segment.data.vx.iloc[0]
        vy_fd = flight_segment.data.vy.iloc[0]
        vz_fd = flight_segment.data.vz.iloc[0]

        tstart = flight_segment.data["time"].iloc[0]
        tend = flight_segment.data["time"].iloc[-1] + self.dt

        # For result reconstruction, extend to the next contact boundary
        # so the velocity estimate reaches all the way to the contact.
        # The optimization still uses `tend` (fitted to measurements only).
        tend_sim = tend_event if tend_event is not None else tend

        if tend > tstart:
            params = (
                vx_fd,
                vy_fd,
                vz_fd,
                self.default_cd,
                self.default_cm
                )

            # Check if all spin values are NaN before computing mode
            if flight_segment.data.wx.isna().all() or flight_segment.data.wy.isna().all() or flight_segment.data.wz.isna().all():
                raise ValueError(
                    f"All spin values are NaN in flight segment. "
                    f"Time range: [{tstart:.4f}, {tend:.4f}], "
                    f"wx NaN: {flight_segment.data.wx.isna().sum()}/{len(flight_segment.data)}, "
                    f"wy NaN: {flight_segment.data.wy.isna().sum()}/{len(flight_segment.data)}, "
                    f"wz NaN: {flight_segment.data.wz.isna().sum()}/{len(flight_segment.data)}"
                )

            wx = stats.mode(flight_segment.data.wx, keepdims=False, nan_policy='omit')[0]
            wy = stats.mode(flight_segment.data.wy, keepdims=False, nan_policy='omit')[0]
            wz = stats.mode(flight_segment.data.wz, keepdims=False, nan_policy='omit')[0]

            const_params = (
                wx,
                wy,
                wz,
                self.rho_air,
                self.rho_ball,
                self.radius_ball
            )

            # Compute initial (pre-optimization) fitness
            fitness_fd_initial = solve_aerodynamics(
                [vx_fd, vy_fd, vz_fd, self.default_cd, self.default_cm],
                flight_segment, tstart, tend, self.dt, const_params, self.pos_cols
            )

            optimization_result = minimize(
                solve_aerodynamics,
                params,
                args=(flight_segment, tstart, tend, self.dt, const_params, self.pos_cols),
                method="SLSQP",
                tol=0.0001,
            )

            # Check if optimization succeeded
            if not optimization_result.success or optimization_result.fun == float("inf"):
                raise ValueError(f"Optimization failed: {optimization_result.message if hasattr(optimization_result, 'message') else 'Unknown error'}")

            # Collect error stats for this segment
            seg_stats = {
                'n_samples': len(flight_segment.data),
                'duration': tend - tstart - self.dt,
                'fd_initial': fitness_fd_initial,
                'fd_optimized': optimization_result.fun,
                'fd_cd': optimization_result.x[3],
                'fd_cm': optimization_result.x[4],
            }

            # Build the evaluation time grid.  Stop just before the next
            # contact boundary (tend_sim - dt/2) so that consecutive
            # segments don't both produce a sample at the contact time.
            # Without this guard the previous segment's extrapolated
            # pre-contact velocity ends up at the same timestamp as
            # the next segment's post-contact velocity; after concat
            # and merge the pre-contact row can shadow the post-contact
            # one, causing downstream code to read the wrong value.
            #
            # When tend_sim == tend (no next-event extension), this still
            # covers the last observation since tend = iloc[-1] + dt.
            t = np.arange(tstart, tend_sim - self.dt * 0.5, self.dt)
            if len(t) == 0:
                t = np.array([tstart])

            # Build results for FD velocity optimization
            aero_params = np.array([optimization_result.x[3], optimization_result.x[4]])

            px, py, pz = self.pos_cols
            result_optimized = solve_ivp(
                aerodynamics_diff,
                [tstart, t[-1] + self.dt],
                (
                    flight_segment.data[px].iloc[0],
                    flight_segment.data[py].iloc[0],
                    flight_segment.data[pz].iloc[0],
                    optimization_result.x[0],
                    optimization_result.x[1],
                    optimization_result.x[2],
                ),
                t_eval=t,
                method="RK45",
                args=(aero_params, const_params),
            )

            results_optimized_df = pd.DataFrame({
                "t": result_optimized.t,
                "pos": [list(result_optimized.y[0:3, i]) for i in range(len(result_optimized.t))],
                "vel": [list(result_optimized.y[3:6, i]) for i in range(len(result_optimized.t))],
                "spin": [[wx, wy, wz]] * len(result_optimized.t)
            })

            results_optimized_df["cd"] = [optimization_result.x[3]] * len(results_optimized_df)
            results_optimized_df["cm"] = [optimization_result.x[4]] * len(results_optimized_df)
            results_optimized_df["v_eff_drag"] = self.compute_v_eff_drag(results_optimized_df.vel)
            results_optimized_df["v_eff_magnus"] = self.compute_v_eff_magnus(results_optimized_df.vel, results_optimized_df.spin)
            results_optimized_df["w_eff_perp_magnus"] = self.compute_w_eff_perp_magnus(results_optimized_df.vel, results_optimized_df.spin)

            # compute trajectory with new model (test_new_model=True):
            parameters_0226 = get_params()
            pp_0226 = make_physics_params(parameters_0226)

            curr_state_0226 = BallState(
                timestamp=tstart,
                position=np.array([flight_segment.data[px].iloc[0], flight_segment.data[py].iloc[0], flight_segment.data[pz].iloc[0]]),
                linear_velocity=np.array(optimization_result.x[0:3], dtype=np.float64),
                angular_velocity=np.array(const_params[0:3], dtype=np.float64),
            )

            def _get_cd_0226(v, w):
                return get_drag_coefficient_new(v, w, self.radius_ball)

            v_0226 = np.linalg.norm(curr_state_0226.linear_velocity)
            w_0226 = np.linalg.norm(curr_state_0226.angular_velocity)
            results_0226 = pd.DataFrame({
                "t": [curr_state_0226.timestamp],
                "pos": [list(curr_state_0226.position)],
                "vel": [list(curr_state_0226.linear_velocity)],
                "spin": [list(curr_state_0226.angular_velocity)],
                "cd": [_get_cd_0226(v_0226, w_0226)],
                "cm": [get_magnus_coefficient_new(v_0226, w_0226)]
            })

            while curr_state_0226.timestamp < tend_sim:
                curr_state_0226 = predict_ball_state_new_model(
                    curr_state_0226, self.dt, pp_0226,
                    _get_cd_0226, get_magnus_coefficient_new,
                )
                v_0226 = np.linalg.norm(curr_state_0226.linear_velocity)
                w_0226 = np.linalg.norm(curr_state_0226.angular_velocity)
                results_0226 = pd.concat(
                    [
                        results_0226,
                        pd.DataFrame(
                            {
                                "t": curr_state_0226.timestamp,
                                "pos": [list(curr_state_0226.position)],
                                "vel": [list(curr_state_0226.linear_velocity)],
                                "spin": [list(curr_state_0226.angular_velocity)],
                                "cd": [_get_cd_0226(v_0226, w_0226)],
                                "cm": [get_magnus_coefficient_new(v_0226, w_0226)]
                            }
                        ),
                    ],
                    ignore_index=True,
                )

            # ── Nakashima model: same ODE, fixed coefficients ──
            nakashima_cd = 0.54
            nakashima_cm = 0.069
            nakashima_rho_air = 1.184
            nakashima_const_params = (
                wx, wy, wz,
                nakashima_rho_air,
                self.rho_ball,
                self.radius_ball,
            )
            nakashima_aero_params = np.array([nakashima_cd, nakashima_cm])

            result_nakashima = solve_ivp(
                aerodynamics_diff,
                [tstart, t[-1] + self.dt],
                (
                    flight_segment.data[px].iloc[0],
                    flight_segment.data[py].iloc[0],
                    flight_segment.data[pz].iloc[0],
                    optimization_result.x[0],
                    optimization_result.x[1],
                    optimization_result.x[2],
                ),
                t_eval=t,
                method="RK45",
                args=(nakashima_aero_params, nakashima_const_params),
            )

            results_nakashima_df = pd.DataFrame({
                "t": result_nakashima.t,
                "pos": [list(result_nakashima.y[0:3, i]) for i in range(len(result_nakashima.t))],
                "vel": [list(result_nakashima.y[3:6, i]) for i in range(len(result_nakashima.t))],
            })
            results_nakashima_df["cd"] = nakashima_cd
            results_nakashima_df["cm"] = nakashima_cm

            # ── 0426 model: old linear drag/magnus model ──
            # This reproduces the aerodynamics from commit 1757e3ec (2025-04-26)
            # which used the old linear drag/magnus model:
            #   Cd = drag_coeff + drag_coeff_linear * S = 0.55 (constant)
            #   Cm = magnus_coeff + magnus_coeff_linear / S, capped at magnus_coeff_max
            parameters_0426 = get_params(test_new_model=False)
            pp_0426 = make_physics_params(parameters_0426)

            curr_state_0426 = BallState(
                timestamp=tstart,
                position=np.array([flight_segment.data[px].iloc[0], flight_segment.data[py].iloc[0], flight_segment.data[pz].iloc[0]]),
                linear_velocity=np.array(optimization_result.x[0:3], dtype=np.float64),
                angular_velocity=np.array(const_params[0:3], dtype=np.float64),
            )

            results_0426 = pd.DataFrame({
                "t": [curr_state_0426.timestamp],
                "pos": [list(curr_state_0426.position)],
                "vel": [list(curr_state_0426.linear_velocity)],
            })

            while curr_state_0426.timestamp < tend_sim:
                curr_state_0426 = predict_ball_state_old_model(
                    curr_state_0426, self.dt, pp_0426, parameters_0426,
                )
                results_0426 = pd.concat(
                    [
                        results_0426,
                        pd.DataFrame(
                            {
                                "t": curr_state_0426.timestamp,
                                "pos": [list(curr_state_0426.position)],
                                "vel": [list(curr_state_0426.linear_velocity)],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

            return results_optimized_df, results_0226, results_nakashima_df, results_0426, seg_stats


        else:
            print("interval time incorrect - this should not happen!")
            exit()

    def _print_error_summary(self, all_stats):
        """Print a summary of aerodynamics optimization errors."""
        if not all_stats:
            print("\nNo segments to summarize.")
            return

        n = len(all_stats)
        fd_init  = np.array([s['fd_initial'] for s in all_stats])
        fd_opt   = np.array([s['fd_optimized'] for s in all_stats])

        header = "\n" + "=" * 80
        print(header)
        print("  AERODYNAMICS ERROR SUMMARY")
        print("=" * 80)

        print(f"\n  Total segments: {n}")
        print(f"  {'Metric':<35} {'Value':>16}")
        print("  " + "-" * 55)
        print(f"  {'Initial fitness (mean ± std)':<35} {np.mean(fd_init):>7.5f}±{np.std(fd_init):.5f}")
        print(f"  {'Initial fitness (median)':<35} {np.median(fd_init):>16.5f}")
        print(f"  {'Optimized fitness (mean ± std)':<35} {np.mean(fd_opt):>7.5f}±{np.std(fd_opt):.5f}")
        print(f"  {'Optimized fitness (median)':<35} {np.median(fd_opt):>16.5f}")
        print(f"  {'Optimized fitness (max)':<35} {np.max(fd_opt):>16.5f}")
        print(f"  {'Optimization improved (mean)':<35} {np.mean(fd_init - fd_opt):>16.5f}")

        # Show worst segments
        worst_idx = np.argsort(fd_opt)[::-1][:10]
        print(f"\n  Top 10 worst-fit segments:")
        print("  " + "-" * 85)
        print(f"  {'Match':<8} {'Game':<6} {'Rally':<7} {'Shot':<6} {'FS':<4} {'#Pts':<5} {'Dur(s)':<7} {'Initial':<9} {'Optimized':<9}")
        for i in worst_idx:
            s = all_stats[i]
            print(f"  {s['match_id']:<8} {s['game_id']:<6} {s['rally_id']:<7} {s['shot_idx']:<6} {s['fs_idx']:<4} {s['n_samples']:<5} {s['duration']:<7.3f} {fd_init[i]:<9.5f} {fd_opt[i]:<9.5f}")

        print("=" * 80 + "\n")

    def compute_v_eff_drag(self, velocities):
        # Compute effective drag velocity
        v3_sum = 0
        v2_sum = 0
        for v in velocities:
            v_norm = np.linalg.norm(v)
            v3_sum += v_norm**3
            v2_sum += v_norm**2
        return v3_sum/v2_sum

    def compute_v_eff_magnus(self, velocities, spins):
        # Compute effective magnus velocity
        v2s_sum = 0
        vs_sum = 0
        for v,w in zip(velocities, spins):
            v_norm = np.linalg.norm(v)
            w_norm = np.linalg.norm(w)
            # Compute sine of angle between v and w using cross product
            cross_prod = np.cross(v, w)
            sintheta = np.linalg.norm(cross_prod) / (v_norm * w_norm + 1e-9)
            v2s_sum += v_norm**2 * sintheta
            vs_sum += v_norm * sintheta
        return v2s_sum/(vs_sum + 1e-9)

    def compute_w_eff_perp_magnus(self, velocities, spins):
        """
        Compute effective perpendicular spin component for Magnus force.

        Under the assumption that |ω| is constant during flight:
        w_eff_perp = |ω| * sin(θ_eff)

        where sin(θ_eff) = Σ(v * sin²θ) / Σ(v * sinθ)

        This is force-weighted (F_M ∝ v * ω * sinθ), accounting for the
        changing angle between the constant spin axis and velocity direction
        as velocity changes due to gravity.
        """
        v_sin2_sum = 0
        v_sin_sum = 0
        w_norm_first = None

        for v, w in zip(velocities, spins):
            v_norm = np.linalg.norm(v)
            w_norm = np.linalg.norm(w)

            if w_norm_first is None:
                w_norm_first = w_norm  # Use first spin magnitude (assumed constant)

            # Compute sine of angle between v and w using cross product
            cross_prod = np.cross(v, w)
            sintheta = np.linalg.norm(cross_prod) / (v_norm * w_norm + 1e-9)

            v_sin2_sum += v_norm * sintheta**2
            v_sin_sum += v_norm * sintheta

        sin_theta_eff = v_sin2_sum / (v_sin_sum + 1e-9)
        return (w_norm_first if w_norm_first is not None else 0) * sin_theta_eff

    def plot(self):
        print("TODO")

# TableContactOptimizer and RacketContactOptimizer are imported from
# their dedicated modules so that update_hdf5.py remains the single
# entry-point while the heavy logic lives in separate files.
from ace_evaluation.update_hdf5.table_contacts import (
    TableContactOptimizer,
)
from ace_evaluation.update_hdf5.racket_contacts import (
    RacketContactOptimizer,
)


def resolve_csv_h5_files(data_folder, csv_file):
    """Read a CSV file (same format as copy_h5_to_local.py) and return the list
    of data.h5 file paths under *data_folder* that match the CSV rows.

    CSV columns: location, date, experiment, ... (>=4 columns, only first 3 used).
    Each row maps to ``<data_folder>/<location>/<date>/<experiment>/`` and we
    collect every ``data.h5`` found recursively under that directory.
    """
    csv_path = pathlib.Path(csv_file)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    base = pathlib.Path(data_folder)
    h5_files = []

    with open(csv_path, mode="r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                print(f"Skipping invalid CSV row: {row}")
                continue
            location, date, experiment = row[0], row[1], row[2]
            source_dir = base / location / date / experiment
            if not source_dir.is_dir():
                print(f"Warning: directory '{source_dir}' does not exist. Skipping.")
                continue
            found = list(source_dir.rglob("data.h5"))
            h5_files.extend(str(f) for f in found)

    if not h5_files:
        print("Warning: no data.h5 files matched the CSV entries.")
    else:
        print(f"CSV filter matched {len(h5_files)} data.h5 file(s).")

    return h5_files


def load_hdf5_files(data_folder, csv_file=None):
    print(f"\n=== LOADING HDF5 FILES FROM: {data_folder} ===")
    sys.stdout.flush()

    allowed_files = None
    if csv_file is not None:
        allowed_files = resolve_csv_h5_files(data_folder, csv_file)

    match_collection: MatchCollection = MatchCollection(data_folder, allowed_files=allowed_files)

    # Debug: Inspect what was loaded
    print(f"Loaded {len(match_collection.matches)} match(es)")
    for match_idx, match in enumerate(match_collection.matches):
        print(f"  Match {match_idx} (ID: {match.match_id}): file={match.file}")
        print(f"    Games: {len(match.games)}")
        for game in match.games:
            print(f"      Game {game.game_id}: {len(game.rallies)} rallies")
    print(f"=== FINISHED LOADING ===\n")
    sys.stdout.flush()

    return match_collection


def main(args):
    """main"""

    data_folder = args.data_folder
    assert data_folder.is_dir()

    data = load_hdf5_files(data_folder, csv_file=args.csv_file)

    aerodynamics = AerodynamicsOptimizer(data_folder, data, args.recompute_aero, use_gt200_positions=args.use_gt200_positions)
    aerodynamics.update_hdf()
    aerodynamics.plot()

    table_contact = TableContactOptimizer(
        data_folder, data, args.recompute_table_contacts,
        export_csv=args.export_csv,
        use_gt200_positions=args.use_gt200_positions,
    )
    table_contact.update_hdf()
    table_contact.plot()

    racket_contact = RacketContactOptimizer(
        data_folder, data, args.recompute_racket_contacts,
        export_csv=args.export_csv,
        zero_toss_spin=args.zero_toss_spin,
    )
    racket_contact.update_hdf()
    racket_contact.plot()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_folder",
        help="Directory containing yaml output of automated dataset labeling",
        type=pathlib.Path,
        default="../Data/hdf5_cmax/"
    )
    parser.add_argument(
        "--recompute_aero",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--recompute_table_contacts",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--recompute_racket_contacts",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--use_gt200_positions",
        action="store_true",
        default=False,
        help="Use gt200 (state estimator) positions instead of raw APS observations for initial conditions and fitness reference",
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        default=False,
        help="Also write extracted.csv / extracted_TCM.csv for backward compatibility",
    )
    parser.add_argument(
        "--zero_toss_spin",
        action="store_true",
        default=False,
        help="Set pre-contact spin to 0 for detected ball tosses (serve tosses)",
    )
    parser.add_argument(
        "--csv_file",
        type=pathlib.Path,
        default=None,
        help="CSV file (location,date,experiment,...) to restrict which data.h5 files are processed (same format as copy_h5_to_local.py)",
    )
    main(parser.parse_args())
