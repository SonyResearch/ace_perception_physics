# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
import argparse
import os
import numpy as np
import h5py
from dataclasses import dataclass
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Tuple, Optional
from enum import Enum

import logging as log

log.basicConfig(level=log.ERROR)

AXES_LABELS = ["X", "Y", "Z"]
BALL_RADIUS = 20e-3  # m
NET_HEIGHT = 152.5e-3  # m
TABLE_LENGTH = 2.74  # m
TABLE_WIDTH = 1.525  # m
LANDING_POS_Z_THRESH = 40e-3  # m The height of the plane at which point the ball crossing is evaluated as the landing point. This avoids issues with the bounce.

SKIP_SERVES = True

# Debugging plots
PLOT_2D = False
PLOT_3D = False

# Plotting settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,  # Base font size for labels, legends
        "axes.labelsize": 9,
        "xtick.labelsize": 7,  # Tick labels are often slightly smaller
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.titlesize": "large",  # Will be larger than the base font
        "figure.titlesize": "x-large",  # Will be even larger
    }
)


def get_table_landing_pos(ball_pos: np.ndarray):
    ball_vel = np.gradient(
        ball_pos, axis=0
    )  # Since we are comparing to zero, DT doesn't matter

    landing_indices = np.argwhere(
        (ball_pos[1:, 2] <= LANDING_POS_Z_THRESH)
        & (ball_pos[0:-1, 2] > LANDING_POS_Z_THRESH)
        & (ball_vel[0:-1, 2] < 0)
    )

    # TODO use a linear? interpolation step to get a more accurate estimate. Extrapolation?

    if landing_indices.size == 0:  # Check if array is empty
        return np.full(3, np.nan)

    landing_idx = landing_indices[0].flatten() + 1
    return ball_pos[landing_idx].flatten()


def check_traj_net_contact(ball_pos: np.ndarray):
    """Check if a trajectory will have touched the net"""

    # Find the index when the ball crosses X=0 plane
    ball_vel = np.gradient(
        ball_pos, axis=0
    )  # Since we are comparing to zero, DT doesn't matter

    net_indices = np.argwhere(
        (ball_pos[1:, 0] >= 0) & (ball_pos[0:-1, 0] < 0) & (ball_vel[0:-1, 0] > 0)
    )

    # Check the Z-height at that index, if Z>BALL_RADIUS and Z < NET_HEIGHT+BALL_RADIUS then we can classify it as a net contact
    if net_indices.size == 0:
        return False  # The ball didn't reach the net -> No net contact

    net_idx = net_indices[0].flatten() + 1
    ball_pos_at_net = ball_pos[net_idx]

    if BALL_RADIUS < ball_pos_at_net.flatten()[2] < (NET_HEIGHT + BALL_RADIUS):
        return True
    return False


class ShotOutcome(Enum):
    VALID = 0  # Perfectly valid shot
    NET = 1  # Touched net, shouldn't consider landing position
    INVALID = 2  # Clearly invalid shot, shouldn't be analyzed
    OTHER = 3  # All other cases, landing position can be analyzed


def check_label(
    labels: np.ndarray, shot_p1_label_idx: int, is_serve: bool
) -> ShotOutcome:
    """Check if the actual trajectory touched the net according to labels"""
    assert labels[shot_p1_label_idx] == b"shot_p1"
    if is_serve:
        # For a serve: the label after 'shot_p1' may be a 'net', 'bounce_p1' (valid serve), 'bounce_p2' (invalid serve), or 'end'
        if labels[shot_p1_label_idx + 1] == b"bounce_p1":
            if labels[shot_p1_label_idx + 2] == b"bounce_p2":
                return ShotOutcome.VALID
            elif labels[shot_p1_label_idx + 2] == b"net":
                return ShotOutcome.NET
            else:
                return ShotOutcome.OTHER
        else:
            return ShotOutcome.INVALID

    else:
        # For a rally: the label after 'shot_p1' may be a 'net', 'bounce_p1', 'bounce_p2' (valid return), or 'end'
        if labels[shot_p1_label_idx + 1] == b"bounce_p2":
            return ShotOutcome.VALID
        elif labels[shot_p1_label_idx + 1] == b"net":
            return ShotOutcome.NET
        else:  # "end" or "bounce_p1"
            return ShotOutcome.OTHER


def plot_net_contact_confusion_matrix(sim_net_contact, act_net_contact):
    """Plot a confusion matrix for net contacts"""
    LABELS = ["No Net Contact", "Net Contact"]
    assert sim_net_contact.shape == act_net_contact.shape

    # TODO remove?
    # TP = act_net_contact & sim_net_contact
    # FP = ~act_net_contact & sim_net_contact
    # TN = ~act_net_contact & ~sim_net_contact
    # FN = act_net_contact & ~sim_net_contact

    cm = confusion_matrix(act_net_contact, sim_net_contact)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)

    _fig, ax = plt.subplots(figsize=(3.5, 2))
    disp.plot(ax=ax, cmap=plt.cm.Blues)  # pylint: disable=no-member


def plot_landing_error_2D_hist(
    ax,
    landing_positions_sim,
    landing_positions_act,
    landing_velocity_sim=None,
    landing_velocity_act=None,
):
    # 2D Histogram of distances
    distances = (
        landing_positions_sim[:, :2] - landing_positions_act[:, :2]
    )  # In world frame

    if landing_velocity_sim is not None and landing_velocity_act is not None:
        assert np.shape(landing_velocity_sim) == np.shape(landing_velocity_act)
        assert np.shape(landing_velocity_sim)[0] == np.shape(landing_positions_sim)[0]

        # Align distances to the velocity vector of the simulated ball
        # Distances may contain NaNs
        for idx, (distance, velocity) in enumerate(
            zip(distances, landing_velocity_sim)
        ):
            if np.isnan(distance).any():
                continue
            theta = np.arctan2(velocity[1], velocity[0])
            R = np.array(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            distances[idx] = R @ distance

        ax.set_title(
            "Distance Between Sim and Act Landings (XY Oriented with Sim. Ball Vel.)"
        )
    else:
        # ax.set_title("Distance Between Sim and Act Landings (XY Aligned to Table Frame)")
        ax.set_title("Error Between Sim and Act Landings")  # Shortened title

    mean_dist, median_dist = np.nanmean(distances, axis=0), np.nanmedian(
        distances, axis=0
    )

    DELTA_BIN = 0.05
    X_MIN = -0.75
    X_MAX = 0.75
    xbins = np.arange(X_MIN, X_MAX + DELTA_BIN / 2, DELTA_BIN)
    xticks = xbins[::5]

    Y_MIN = -0.5
    Y_MAX = 0.5
    ybins = np.arange(Y_MIN, Y_MAX + DELTA_BIN / 2, DELTA_BIN)
    yticks = ybins[::5]

    h = ax.hist2d(
        distances[:, 0], distances[:, 1], bins=[xbins, ybins], cmap="gist_yarg"
    )
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Extract histogram data for contour
    counts = h[0]  # 2D array of counts
    xedges = h[1]  # x bin edges
    yedges = h[2]  # y bin edges

    # Calculate bin centers
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    # Find thresholds for 25%, 50%, and 75% of data
    total_counts = np.sum(counts)
    sorted_counts = np.sort(counts.flatten())[::-1]  # Sort descending
    cumsum = np.cumsum(sorted_counts)

    percentiles = [25, 50, 75]
    thresholds = []
    colors = ["lime", "cyan", "magenta"]

    for pct in percentiles:
        threshold_idx = np.argmax(cumsum >= (pct / 100.0) * total_counts)
        thresholds.append(sorted_counts[threshold_idx])

    handles = []  # List to store legend handles

    # Add contour lines
    for threshold, pct, color in zip(thresholds, percentiles, colors):
        cs = ax.contour(
            x_centers,
            y_centers,
            counts.T,
            levels=[threshold],
            colors=color,
            linewidths=0.75,
            linestyles="solid",
            alpha=0.8,
        )

        # Grab the first (and only) collection for this specific level
        handle = cs.legend_elements()[0][0]
        handle.set_label(f"{pct}%")
        handles.append(handle)

    # Scatter handles to the list
    mean_scat = ax.scatter(
        mean_dist[0], mean_dist[1], label="Mean", marker="x", color="blue"
    )
    med_scat = ax.scatter(
        median_dist[0], median_dist[1], label="Median", marker="x", color="green"
    )
    handles.extend([mean_scat, med_scat])

    ax.legend(handles=handles, ncol=2)
    ax.set_xlabel("X distance [m]")
    ax.set_ylabel("Y distance [m]")
    ax.set_aspect("equal")
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.2)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.2)
    ax.get_figure().colorbar(h[3], ax=ax)  # h[3] is the mappable object


def plot_landing_error_1D_hist(ax, landing_positions_sim, landing_positions_act):
    # Histogram of distances
    distances = np.linalg.norm(
        landing_positions_sim[:, :2] - landing_positions_act[:, :2], axis=1
    )
    mean_dist, median_dist = np.nanmean(distances), np.nanmedian(distances)
    p25_dist, p75_dist = np.nanpercentile(distances, 25), np.nanpercentile(
        distances, 75
    )

    # Cap distances at 1.0 for binning - values >1.0 go into last bin
    distances_capped = np.clip(distances, 0, 1.0)

    # Create 20 bins from 0 to 1.0
    n_bins = 20
    bins = np.linspace(0, 1.0, n_bins + 1)

    # Histogram with LOW zorder
    counts, _, _patches = ax.hist(
        distances_capped,
        bins=bins,
        color="steelblue",
        alpha=0.8,
        edgecolor="white",
        zorder=1,
    )

    # ax.set_title("Distance Between Sim and Act Landings (XY)")
    ax.set_title("Distance Between Sim and Act Landings")  # Short title
    ax.set_xlabel("distance [m]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2, zorder=0)

    # Show every 5th bin edge (0, 0.25, 0.5, ... 1.0)
    xticks = bins[::5]
    xticklabels = [f"{x:.2g}" for x in xticks[:-1]] + ["≥1.0"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Calculate nice tick spacing for count axis - always 6 ticks (including 0)
    max_count_target = counts.max() * 1.1  # Add 10% headroom

    # Choose nice intervals: 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, etc.
    nice_intervals = [0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]

    # Find first interval where 5 * interval >= max_count_target (6 ticks = 0 to 5*interval)
    for interval in nice_intervals:
        if 5 * interval >= max_count_target:
            break

    # Create exactly 6 ticks from 0 to 5*interval
    y_max = 5 * interval
    count_ticks = np.arange(0, y_max + interval, interval)

    ax.set_ylim(0, y_max)
    ax.set_yticks(count_ticks)

    ax.grid(True, alpha=0.2, zorder=0)

    # Axvlines with HIGH zorder
    ax.axvline(mean_dist, color="darkorange", linestyle="--", label="mean", zorder=3)
    ax.axvline(median_dist, color="green", linestyle="--", label="median", zorder=3)
    ax.axvline(p25_dist, color="purple", linestyle="--", label="p25", zorder=3)
    ax.axvline(p75_dist, color="blue", linestyle="--", label="p75", zorder=3)

    # Annotate axvlines (after setting ylim) with HIGH zorder
    ax.text(
        mean_dist,
        ax.get_ylim()[1] * 0.95,
        f"mean: {mean_dist:.3f} m",
        color="darkorange",
        rotation=90,
        va="top",
        ha="right",
        zorder=4,
    )
    ax.text(
        median_dist,
        ax.get_ylim()[1] * 0.95,
        f"median: {median_dist:.3f} m",
        color="green",
        rotation=90,
        va="top",
        ha="right",
        zorder=4,
    )
    ax.text(
        p25_dist + 0.005,
        ax.get_ylim()[1] * 0.05,
        f"25th: {p25_dist:.3f} m",
        color="purple",
        rotation=90,
        va="bottom",
        ha="left",
        zorder=4,
    )
    ax.text(
        p75_dist + 0.005,
        ax.get_ylim()[1] * 0.05,
        f"75th: {p75_dist:.3f} m",
        color="blue",
        rotation=90,
        va="bottom",
        ha="left",
        zorder=4,
    )

    # Add text showing count of values ≥1.0 with HIGHEST zorder
    count_above_1 = np.sum(distances >= 1.0)
    if count_above_1 > 0:
        ax.text(
            0.98,
            0.98,
            f"Values ≥1.0: {count_above_1}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            zorder=5,
        )

    ax.set_xlim(-0.05, 1.05)


def plot_2d(ball_traj_sim: np.ndarray, ball_traj_act: np.ndarray):
    fig, axs = plt.subplots(
        3, 1, figsize=(14, 10), constrained_layout=True, sharex=True
    )
    axs = axs.flatten()
    ax = None
    for idx, ax in enumerate(axs):
        ax.plot(ball_traj_act[:, idx], ".-", label="ball_traj_act")
        ax.plot(ball_traj_sim[:, idx], ".-", label="ball_traj_sim")
        ax.set_title(AXES_LABELS[idx])
        ax.set_ylabel("Ball Position [m]")
        ax.set_xlabel("Sample")
        ax.legend()
        ax.minorticks_on()
        ax.grid("on", "both")
        if idx == 0:
            ax.axhline(0, linestyle="-", c="k")
            ax.axhline(TABLE_LENGTH / 2, linestyle=":", c="k")
            ax.axhline(-TABLE_LENGTH / 2, linestyle=":", c="k")
        elif idx == 1:
            ax.axhline(TABLE_WIDTH / 2, linestyle=":", c="k")
            ax.axhline(-TABLE_WIDTH / 2, linestyle=":", c="k")
        elif idx == 2:
            ax.axhline(0, linestyle="-", c="k")
            ax.axhline(NET_HEIGHT + BALL_RADIUS, linestyle=":", c="k")
            ax.axhline(LANDING_POS_Z_THRESH, linestyle=":", c="r")

    return fig, ax


def draw_table(ax):
    table_vertices = [
        np.array(
            [
                [-TABLE_LENGTH / 2, -TABLE_WIDTH / 2, 0],
                [TABLE_LENGTH / 2, -TABLE_WIDTH / 2, 0],
                [TABLE_LENGTH / 2, TABLE_WIDTH / 2, 0],
                [-TABLE_LENGTH / 2, TABLE_WIDTH / 2, 0],
            ]
        )
    ]
    net_vertices = [
        np.array(
            [
                [0, -TABLE_WIDTH / 2, 0],
                [0, -TABLE_WIDTH / 2, NET_HEIGHT],
                [0, TABLE_WIDTH / 2, NET_HEIGHT],
                [0, TABLE_WIDTH / 2, 0],
            ]
        )
    ]

    # Create the 3D polygon and add it to the axes
    table = Poly3DCollection(
        table_vertices, facecolors="mediumblue", edgecolors="white", alpha=0.5
    )
    net = Poly3DCollection(
        net_vertices, facecolors="gray", edgecolors="black", alpha=0.5
    )
    ax.add_collection3d(table)
    ax.add_collection3d(net)


def plot_3d(
    ball_traj_sim: np.ndarray,
    ball_traj_act: np.ndarray,
    land_pos_sim: Optional[np.ndarray] = None,
    land_pos_act: Optional[np.ndarray] = None,
    net_contact_label_act: Optional[bool] = None,
    net_contact_traj_act: Optional[bool] = None,
    net_contact_traj_sim: Optional[bool] = None,
    zoom=1.0,
):
    """Create 3D interactive plot"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=True)

    # Plot data points
    ax.scatter(
        ball_traj_act[:, 0],
        ball_traj_act[:, 1],
        ball_traj_act[:, 2],
        marker=".",
        color="tab:blue",
        label="Actual Trajectory",
        s=50,
    )

    ax.scatter(
        ball_traj_sim[:, 0],
        ball_traj_sim[:, 1],
        ball_traj_sim[:, 2],
        marker=".",
        color="tab:orange",
        label="Simulated",
        s=50,
    )

    # Draw the landing positions
    if land_pos_act is not None:
        ax.scatter(
            land_pos_act[0],
            land_pos_act[1],
            land_pos_act[2],
            marker="x",
            color="tab:blue",
            label="Actual Landing Pos.",
            s=100,
        )

    if land_pos_sim is not None:
        ax.scatter(
            land_pos_sim[0],
            land_pos_sim[1],
            land_pos_sim[2],
            marker="x",
            color="tab:orange",
            label="Sim Landing Pos.",
            s=100,
        )

    draw_table(ax)

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])

    # Calculate equal axis limits based on data range
    all_points = np.vstack(
        [
            ball_traj_sim,
            ball_traj_act,
        ]
    )

    # Find the range for each axis
    x_range = [np.nanmin(all_points[:, 0]), np.nanmax(all_points[:, 0])]
    y_range = [np.nanmin(all_points[:, 1]), np.nanmax(all_points[:, 1])]
    z_range = [np.nanmin(all_points[:, 2]), np.nanmax(all_points[:, 2])]

    x_center = 0.0
    y_center = 0.0
    z_center = 0.0

    max_range = np.nanmax(
        [
            abs(x_center - x_range[1]),
            abs(x_center - x_range[0]),
            abs(y_center - y_range[1]),
            abs(y_center - y_range[0]),
            abs(z_center - z_range[1]),
            abs(z_center - z_range[0]),
        ]
    )

    ax.set_xlim([x_center - zoom * max_range, x_center + zoom * max_range])
    ax.set_ylim([y_center - zoom * max_range, y_center + zoom * max_range])
    ax.set_zlim([z_center - zoom * max_range, z_center + zoom * max_range])

    ax.set_xlabel(AXES_LABELS[0])
    ax.set_ylabel(AXES_LABELS[1])
    ax.set_zlabel(AXES_LABELS[2])
    ax.legend(fontsize="small", loc="lower right")

    # plt.tight_layout(pad=0.5)  # Minimize padding

    # Orthographic (no perspective)
    ax.set_proj_type("ortho")

    # Include info about net contacts
    if (
        (net_contact_label_act is not None)
        and (net_contact_traj_act is not None)
        and (net_contact_traj_sim is not None)
    ):
        net_contact_str = f"Labeled Net Contact in Act.: {net_contact_label_act}\nAct. Traj. Net Contact: {net_contact_traj_act}\nSim. Traj. Net Contact: {net_contact_traj_sim}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text2D(
            0.05,
            0.95,
            net_contact_str,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

    return fig, ax


# Dataclass that specifies which data from the h5 file should be extracted for each case
@dataclass(slots=True)
class FieldMapping:
    """Specifies which Fields of the h5 rally to use"""

    desc: str  # Descriptor/name
    ball_traj_sim_name: str  # str used to extract the simulated ball trajectory from the h5
    timestamp_sim_name: Optional[
        str
    ]  # str used to extract the timestamps for the ball trajectory from the h5


# Racket contact comparisons
NAKASHIMA = FieldMapping(
    "Nakashima", "ground_truth_200/racket_contacts/simulated_trajectories", None
)
ONNX_ALEX_FM = FieldMapping(
    "Residual NN",
    "ground_truth_200/racket_contacts/simulated_trajectories_latest",
    None,
)
ONNX_RCM_FM = FieldMapping(
    "ONNX RCM", "ground_truth_200/racket_contacts/simulated_trajectories_onnx_rcm", None
)
RCM_POLYFIT_FM = FieldMapping(
    "RCM Polyfit",
    "ground_truth_200/racket_contacts/simulated_trajectories_rcm_polyfit",
    None,
)
ONNX_0426_FM = FieldMapping(
    "Dürr et al. (RCM + Aero)",
    "ground_truth_200/racket_contacts/simulated_trajectories_onnx_0426",
    None,
)
# Aerodynamics comparisons
AERO_0226_FM = FieldMapping(
    "Aero 0226",
    "ground_truth_200/aerodynamics/pos_0226",
    "ground_truth_200/aerodynamics/t",
)
AERO_NAKASHIMA_FM = FieldMapping(
    "Aero Nakashima",
    "ground_truth_200/aerodynamics/pos_nakashima",
    "ground_truth_200/aerodynamics/t",
)
AERO_0426_FM = FieldMapping(
    "Aero 0426 (Dürr et al.)",
    "ground_truth_200/aerodynamics/pos_0426",
    "ground_truth_200/aerodynamics/t",
)
AERO_OPT_FM = FieldMapping(
    "Aero Optimized",
    "ground_truth_200/aerodynamics/pos_optimized",
    "ground_truth_200/aerodynamics/t",
)


def analyse_data(
    contact_h5_fpath: str, fm: FieldMapping
) -> Tuple[List, List, List, List]:
    OFFSET = 0  # ms

    START_DIST_THRESH = 0.1  # m

    match_date = contact_h5_fpath.split("/")[-3]
    match_name = contact_h5_fpath.split("/")[-2]
    log.info(f"Analyzing {match_name}")

    act_landing_pos_list = []
    sim_landing_pos_list = []

    act_net_contact_list = []
    sim_net_contact_list = []

    # Load an h5 file containing pre-computed post-contact ball states
    with h5py.File(contact_h5_fpath, "r") as file:
        for game_idx, game in enumerate(file.values()):
            for _rally_idx, rally in enumerate(game.values()):
                labels = rally["labels/type"][:]  # Get the real rally number
                rally_num = int(rally.name.split("_")[-1])
                shot_p1_label_idx = np.argwhere(labels == b"shot_p1")
                n_shots = len(shot_p1_label_idx)

                # Iterate over each shot
                for shot_idx, shot_label_idx in enumerate(shot_p1_label_idx):
                    log.info(
                        f"D:{match_date}, M:{match_name}, G:{game_idx}, R:{rally_num}, S:{shot_idx}. Processing"
                    )

                    shot_seq_num = rally["labels/sequence_number"][shot_label_idx]
                    next_label_seq_num = rally["labels/sequence_number"][
                        shot_label_idx + 1
                    ]

                    # Window of GT200 data to extract
                    start = shot_seq_num + OFFSET
                    end = next_label_seq_num

                    # Label serves as such
                    if shot_label_idx == 1:
                        is_serve = True
                        if SKIP_SERVES:
                            log.info(
                                f"D:{match_date}, M:{match_name}, G:{game_idx}, R:{rally_num}, S:{shot_idx}. Skipping, serve"
                            )
                            continue  # skip
                    else:
                        is_serve = False

                    # TODO filter based on shot outcome?
                    shot_outcome = check_label(labels, shot_label_idx, is_serve)

                    # Get the GT data
                    gt_timestamps = np.array(rally["ground_truth_200/ball/time"][:])
                    gt_seq_nums = (1000 * gt_timestamps).astype(int)
                    gt_mask = (gt_seq_nums >= start) & (gt_seq_nums < end)
                    _window_gt_seq_nums = gt_seq_nums[gt_mask]
                    window_gt_ball_position = rally["ground_truth_200/ball/position"][
                        gt_mask
                    ]

                    # Simulate the evolution of the trajectories until they cross the table plane (Z=0) with negative Z velocity?
                    # The pre-simulated trajectories in the h5 file don't have the same sequence number/timestamps
                    try:
                        if fm.timestamp_sim_name is None:
                            n_sim_shots = len(rally[fm.ball_traj_sim_name])
                            if n_shots != n_sim_shots:
                                log.error(
                                    f"D:{match_date}, M:{match_name}, G:{game_idx}, R:{rally_num}, S:{shot_idx}. The number of simulated trajectories ({n_sim_shots}) does not match the number of shot_p1 labels ({n_shots})"
                                )
                                continue
                            sim_traj = rally[fm.ball_traj_sim_name + f"/{shot_idx}"]
                            window2_sim_ball_position = np.vstack(
                                (sim_traj["x"], sim_traj["y"], sim_traj["z"])
                            ).T
                        else:
                            sim_traj = rally[fm.ball_traj_sim_name][:]
                            sim_timestamps = rally[fm.timestamp_sim_name][:]
                            sim_seq_nums = (1000 * sim_timestamps).astype(int)
                            sim_mask = (sim_seq_nums >= start) & (sim_seq_nums < end)
                            window2_sim_ball_position = sim_traj[sim_mask]

                        if (
                            np.linalg.norm(
                                window_gt_ball_position[0, :]
                                - window2_sim_ball_position[0, :]
                            )
                            > START_DIST_THRESH
                        ):
                            log.error(
                                f"D:{match_date}, M:{match_name}, G:{game_idx}, R:{rally_num}, S:{shot_idx}. The simulated and actual trajectories originate from points far from each other"
                            )
                            # continue #TODO Skip cases where the trajectories originate from different points.

                        # Determine the 'landing position' in sim and reality. Don't consider cases with labeled net contacts!
                        if shot_outcome != ShotOutcome.NET:
                            act_landing_pos = get_table_landing_pos(
                                window_gt_ball_position
                            )
                            if np.isnan(act_landing_pos).all():
                                log.error(
                                    f"D:{match_date}, M:{match_name}, G:{game_idx}, R:{rally_num}, S:{shot_idx}. Actual landing position could not be determined"
                                )
                            sim_landing_pos = get_table_landing_pos(
                                window2_sim_ball_position
                            )
                            if np.isnan(sim_landing_pos).all():
                                log.error(
                                    f"D:{match_date}, M:{match_name}, G:{game_idx}, R:{rally_num}, S:{shot_idx}. Simulated landing position could not be determined"
                                )

                            act_landing_pos_list.append(act_landing_pos)
                            sim_landing_pos_list.append(sim_landing_pos)
                        else:
                            act_landing_pos = None
                            sim_landing_pos = None
                            log.info(
                                f"D:{match_date}, M:{match_name}, G:{game_idx}, R:{rally_num}, S:{shot_idx}. Net contact. Skipping landing position detection."
                            )

                        # Determine net contacts
                        # Compare it to what actually occured from the h5 file -> Net contact discrepancy
                        # TODO should we check for other failure modes too? e.g: bounce_p1 before bounce_p2
                        act_net_contact_label = shot_outcome == ShotOutcome.NET
                        act_net_contact_traj = check_traj_net_contact(
                            window_gt_ball_position
                        )
                        act_net_contact = act_net_contact_traj or act_net_contact_label
                        sim_net_contact = check_traj_net_contact(
                            window2_sim_ball_position
                        )

                        act_net_contact_list.append(act_net_contact)
                        sim_net_contact_list.append(sim_net_contact)

                        # Plots for debugging
                        if PLOT_2D:  # Time-domain plot for debugging
                            fig_2d, _ax_2d = plot_2d(
                                window2_sim_ball_position, window_gt_ball_position
                            )
                            fig_2d.suptitle(
                                f"Game {game_idx}, Rally {rally_num}, Shot {shot_idx}"
                            )

                        if (sim_landing_pos is not None) and (
                            act_landing_pos is not None
                        ):
                            if (
                                np.linalg.norm(sim_landing_pos - act_landing_pos) > 1.0
                                and PLOT_3D
                            ):  # 3D plot for debugging
                                _fig_3d, ax_3d = plot_3d(
                                    window2_sim_ball_position,
                                    window_gt_ball_position,
                                    sim_landing_pos,
                                    act_landing_pos,
                                    act_net_contact_label,
                                    act_net_contact_traj,
                                    sim_net_contact,
                                )
                                ax_3d.set_title(
                                    f"Date: {match_date}, {match_name}, Game {game_idx}, Rally {rally_num}, Shot {shot_idx}"
                                )

                        if PLOT_2D or PLOT_3D:
                            plt.show()

                    except KeyError:
                        log.error(
                            f"D:{match_date}, M:{match_name}, G:{game_idx}, R:{rally_num}, S:{shot_idx}. Missing simulated trajectory?"
                        )

    return (
        act_landing_pos_list,
        sim_landing_pos_list,
        act_net_contact_list,
        sim_net_contact_list,
    )


def generate_h5_file_list_partial(data_dir: str) -> List[str]:
    h5_files = [
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260403/seedy_wash_vs_kawamata_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260313/one_haul_vs_kawamata_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260313/now_role_vs_kawamata_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260313/tin_band_vs_kawamata_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251202/ace_vs_maehara_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251202/ace_vs_maehara_match_2/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251202/ace_vs_saeki_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251202/ace_vs_saeki_match_1/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260420/ace_r_vs_hirano_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260420/ace_tt_vs_hirano_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260420/ace_r2_vs_hirano_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260420/ace_t_vs_hirano_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251126/new_physics_vs_takenaka_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251126/proficiency_t5_vs_takenaka_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251126/rl_dd_eff_tactics_vs_takenaka_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251126/baseline_vs_takenaka_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251126/lobs_vs_takenaka_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260226/ace_r2_vs_shiomi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260226/ace_r_vs_shiomi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260226/ace_t2_vs_shiomi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260226/ace_t_vs_shiomi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260226/ace_t_vs_shiomi_match_1/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260311/bogus_yard_vs_nagao_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260311/one_haul_vs_nagao_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260311/baseline_vs_nagao_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260311/shut_robe_vs_nagao_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260311/top_chef_vs_nagao_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260409/baseline_vs_igarashi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260409/lead_buck_vs_igarashi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260409/mean_pact_2_vs_igarashi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260409/mean_pact_vs_igarashi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20260409/noble_wig_vs_igarashi_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251112/tt_ace_farshad_vs_murano_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251112/tt_ace_knuckle_vs_murano_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251112/tt_ace_rl_tactics_vs_hagii_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251112/tt_ace_rl_tactics_vs_murano_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251112/tt_ace_vs_hagii_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251112/tt_ace_vs_murano_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251201/ace2_vs_yoshimura_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251201/ace_vs_nozaki_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251201/ace_vs_nozaki_match_1/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251201/ace_vs_takenaka_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251201/ace_vs_takenaka_match_1/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251201/ace_vs_yoshimura_match_0/data.h5",
        "/RCM_evaluation/hdf5_physics_pseudogt/tyo01/20251201/ace_vs_yoshimura_match_1/data.h5",
    ]

    # TODO add the rest of the files

    return [data_dir + h5_file for h5_file in h5_files]


def find_data_h5_files(root_folder):
    """
    Search for all files named 'data.h5' in a folder and its subfolders.

    Args:
        root_folder (str): The root directory path to start searching from

    Returns:
        list: A list of file paths (as strings) to all 'data.h5' files found
    """
    h5_files = []

    for dirpath, _dirnames, filenames in os.walk(root_folder):
        if "data.h5" in filenames:
            full_path = os.path.join(dirpath, "data.h5")
            h5_files.append(full_path)

    return h5_files


def load_h5_file_list_from_json(json_path: str) -> List[str]:
    """Load a list of h5 file paths from a JSON manifest file.

    Supports the data_plotter format: a list of objects with keys
    ``lab``, ``date``, ``experiment``, ``game_id``, ``rally_id``.
    The HDF5 path is constructed as
    ``<json_parent>/<lab>/<date>/<experiment>/data.h5``.

    Entries may also provide ``hdf5_filepath`` directly.
    """
    import json
    import pathlib

    json_p = pathlib.Path(json_path).resolve()
    json_parent = json_p.parent

    with open(json_p, "r") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("JSON root must be a list of rally entries")

    seen = set()
    h5_files = []
    for entry in entries:
        lab = entry.get("lab") or entry.get("court")
        date = entry.get("date")
        experiment = entry.get("experiment")

        if lab and date and experiment:
            h5path = str(json_parent / lab / date / experiment / "data.h5")
        elif "hdf5_filepath" in entry:
            p = pathlib.Path(entry["hdf5_filepath"])
            if not p.is_absolute():
                p = json_parent / p
            h5path = str(p)
        else:
            continue

        if h5path not in seen:
            seen.add(h5path)
            h5_files.append(h5path)

    return h5_files


def analyze_multi_data(
    contact_h5_fpaths: List[str], fm: FieldMapping
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    act_landing_pos_list_full = []
    sim_landing_pos_list_full = []

    act_net_contact_list_full = []
    sim_net_contact_list_full = []

    for contact_h5_path in contact_h5_fpaths:
        (
            act_landing_pos_list,
            sim_landing_pos_list,
            act_net_contact_list,
            sim_net_contact_list,
        ) = analyse_data(contact_h5_path, fm)
        act_landing_pos_list_full.extend(act_landing_pos_list)
        sim_landing_pos_list_full.extend(sim_landing_pos_list)

        act_net_contact_list_full.extend(act_net_contact_list)
        sim_net_contact_list_full.extend(sim_net_contact_list)

    if not act_landing_pos_list_full:
        log.warn("No valid landing positions found across all H5 files")
        act_landing_pos_arr = np.empty((0, 3))
        sim_landing_pos_arr = np.empty((0, 3))
    else:
        act_landing_pos_arr = np.vstack(act_landing_pos_list_full)
        sim_landing_pos_arr = np.vstack(sim_landing_pos_list_full)

    act_net_contact_arr = np.array(act_net_contact_list_full)
    sim_net_contact_arr = np.array(sim_net_contact_list_full)

    return (
        act_landing_pos_arr,
        sim_landing_pos_arr,
        act_net_contact_arr,
        sim_net_contact_arr,
    )


def main():
    parser = argparse.ArgumentParser(description="Physics paper evaluation script")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing the HDF5 data (e.g. /path/to/Data)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--h5-file", help="Path to a single data.h5 file")
    group.add_argument("--h5-json", help="Path to a JSON manifest listing h5 files")
    group.add_argument(
        "--h5-list", action="store_true", help="Use the hardcoded partial file list"
    )
    group.add_argument(
        "--scan-dir", help="Scan a directory recursively for data.h5 files"
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    # Build a list of all the h5 files to iterate over
    if args.h5_file:
        contact_h5_fpaths = [args.h5_file]
    elif args.h5_list:
        contact_h5_fpaths = generate_h5_file_list_partial(data_dir)
    elif args.scan_dir:
        contact_h5_fpaths = find_data_h5_files(args.scan_dir)
    elif args.h5_json:
        contact_h5_fpaths = load_h5_file_list_from_json(args.h5_json)
    else:
        # Default: look for validation JSON manifest
        contact_h5_fpaths = load_h5_file_list_from_json(
            os.path.join(data_dir, "hdf5_physics_pseudogt", "validation_h5_files.json")
        )

    COMPARE_SIDE_BY_SIDE = False

    if COMPARE_SIDE_BY_SIDE:
        # Side-by-side comparison of two field mappings
        fms = [NAKASHIMA, ONNX_ALEX_FM]
        _fig, axs = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)

        for col, fm in enumerate(fms):
            (
                act_landing_pos_arr,
                sim_landing_pos_arr,
                act_net_contact_arr,
                sim_net_contact_arr,
            ) = analyze_multi_data(contact_h5_fpaths, fm)
            print(
                f"[{fm.desc}] Num shots act_landing: {act_landing_pos_arr.shape[0]}, sim_landing: {sim_landing_pos_arr.shape[0]}, act_net: {len(act_net_contact_arr)}, sim_net: {len(sim_net_contact_arr)}"
            )

            plot_landing_error_1D_hist(
                axs[0, col], sim_landing_pos_arr, act_landing_pos_arr
            )
            plot_landing_error_2D_hist(
                axs[1, col], sim_landing_pos_arr, act_landing_pos_arr
            )
            axs[0, col].set_title(fm.desc)

    else:
        # Single field mapping
        # fm = NAKASHIMA
        # fm = ONNX_ALEX_FM # proposed model
        fm = ONNX_0426_FM
        
        ## fm = ONNX_RCM_FM
        ## fm = RCM_POLYFIT_FM
        # fm = AERO_0226_FM
        # fm = AERO_0426_FM
        # fm = AERO_NAKASHIMA_FM
        # fm = AERO_OPT_FM

        (
            act_landing_pos_arr,
            sim_landing_pos_arr,
            act_net_contact_arr,
            sim_net_contact_arr,
        ) = analyze_multi_data(contact_h5_fpaths, fm)
        print(
            f"Num shots act_landing_pos_arr: {act_landing_pos_arr.shape[0]}\nNum shots sim_landing_pos_arr: {sim_landing_pos_arr.shape[0]}\nNum shots act_net_contact_arr: {len(act_net_contact_arr)}\nNum shots sim_net_contact_arr: {len(sim_net_contact_arr)}\n"
        )

        if act_landing_pos_arr.shape[0] > 0:
            _fig, axs = plt.subplots(2, 1, figsize=(3.5, 5), constrained_layout=True)
            axs = axs.flatten()
            plot_landing_error_1D_hist(axs[0], sim_landing_pos_arr, act_landing_pos_arr)
            plot_landing_error_2D_hist(axs[1], sim_landing_pos_arr, act_landing_pos_arr)
        if len(act_net_contact_arr) > 0:
            plot_net_contact_confusion_matrix(sim_net_contact_arr, act_net_contact_arr)

    plt.show()


if __name__ == "__main__":
    main()
