"""
@brief Main script for labeling proffesional player data.

@file label_pro_player_data.py
@author Asude Aydin (asude.aydin@sony.com)
@date 2026
@version 0.0
@copyright Confidential, Copyright 2026, Sony AI, All rights reserved.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from termcolor import cprint
import torch
from tools.h5_writer import H5WriterLabel
from tools.interpolate_ball_positions import (
    ball_vel_finite_diff,
    expand_with_nans,
    interpolate_ball_trajectory_polyfit,
    plot_ball_positions,
    plot_ball_positions_w_velocity,
    project_3d_to_2d,
)
from tqdm import tqdm


def parse_args():
    """
    Parse input arguments.

    @return: Parsed argument struct
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/media/chaydina/T71/asude_overhead_evs_recordings/Univ_2024-05-29_recording_upload/20240529_140358_sakamoto_vs_murano_set_1",
        help="Path to folder containing calibration file and evs, labels, and rosbag folder.",
        # required=True,
    )

    parser.add_argument(
        "--cam_name",
        type=str,
        default="evs00050026",
        help="Name of the EVS camera",
        # required=True,
    )

    parser.add_argument(
        "--ball_radius",
        type=float,
        help="Radius of the ball",
        required=False,
        default=0.02,
    )

    parser.add_argument(
        "--polyfit_degree",
        type=int,
        default=3,
        help="Degree of the polynomail fit for free flight trajectories",
        required=False,
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="Number of minimum APS samples to be processed",
        required=False,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the ball positions and velocities",
        required=False,
    )

    args = parser.parse_args()

    return args


def get_consecutive_non_nan_indices(label_array):
    """Return the indices of consecutive non-nan arrays.

    Args:
        label_array (np.array): Array

    Returns:
        indices (list): Appended list of indices with consecutively occuring non-nan values.
    """
    # Find NaN values in estimated ball positions
    nan_rows = np.any(np.isnan(label_array), axis=1)

    # Initialize variables to store sub-arrays
    indices = []
    start_idx = 0

    # Iterate through the array and split at NaN rows
    for idx, has_nan in enumerate(nan_rows):
        if has_nan:
            if start_idx != idx:  # To avoid appending empty arrays
                indices.append((start_idx, idx))
            start_idx = idx + 1

    # Append the last sub-array if the end was not NaN
    if start_idx < len(nan_rows):
        indices.append((start_idx, len(nan_rows)))

    return indices


def round_to_nearest_increment(value, increment=0.005):
    """Function to round a given value to the nearest increment.

    Args:
        value (float): Float value
        increment (float, optional): Increment value to round to. Defaults to 0.005.

    Returns:
        float: rounded value
    """
    scale = int(1 / increment)
    return round(value * scale) / scale


def split_sequence_indices_by_shot_times(sequence, events, deg):
    """Splits sequence given contact events and their corresponding timestamps.

    Args:
        sequence (list): List containing sequence of timestamps.
        events (dict): Dictionary containing event data
        deg (int): Degree of poylnomial fitting function for discarding short sequences

    Returns:
        list: Indices of split sequences stored in a list of lists.
    """
    rounded_ts = [round_to_nearest_increment(val) for val in sequence]

    shot_times = []
    for _, event in enumerate(events):
        # Check for edge cases
        if rounded_ts[0] <= event["timestamp"] <= rounded_ts[1]:
            shot_time = rounded_ts[0]

        elif rounded_ts[-2] <= event["timestamp"] <= rounded_ts[-1]:
            shot_time = rounded_ts[-1]

        else:
            shot_time = round_to_nearest_increment(event["timestamp"])

        shot_times.append(shot_time)

    split_indices = []
    current_index = 0
    for idx, timestamp in enumerate(rounded_ts):
        if timestamp in shot_times and current_index != idx:
            split_indices.append([current_index, idx])
            current_index = idx

    # Remove the first sequence of indices if the shot time split is too short.
    if len(split_indices) > 0:
        if split_indices[0][0] == 0 and (
            split_indices[0][1] - split_indices[0][0] <= deg + 1
        ):
            split_indices.pop(0)

    return split_indices


def save_to_csv(
    file_path: str,
    recording_name: str,
    sequence_id: int,
    metadata_value: list,
    time_orig: np.array,
    time: np.array,
    pos: np.array,
    vel: np.array,
    orien: np.array,
) -> None:
    "Given dictionary and filename, store data in pandas dataframe and save to a csv file."

    data_dict = {}

    ts_start = metadata_value[0]
    ts_end = metadata_value[1]

    idx_start = np.where(time == ts_start)[0][0]
    idx_end = np.where(time == ts_end)[0][0]

    data_dict["label_name"] = recording_name
    data_dict["sequence_id"] = sequence_id

    data_dict["timestamps_orig"] = time_orig[idx_start:idx_end]
    data_dict["timestamps"] = time[idx_start:idx_end]

    data_dict["first_label_timestamp"] = round_to_nearest_increment(ts_start)
    data_dict["last_label_timestamp"] = round_to_nearest_increment(ts_end)

    data_dict["pos_x_label"] = pos[idx_start:idx_end][:, 0]
    data_dict["pos_y_label"] = pos[idx_start:idx_end][:, 1]
    data_dict["pos_z_label"] = pos[idx_start:idx_end][:, 2]

    data_dict["vel_x_label"] = vel[idx_start:idx_end][:, 0]
    data_dict["vel_y_label"] = vel[idx_start:idx_end][:, 1]
    data_dict["vel_z_label"] = vel[idx_start:idx_end][:, 2]

    data_dict["orientation_x_label"] = orien[idx_start:idx_end][:, 0]
    data_dict["orientation_y_label"] = orien[idx_start:idx_end][:, 1]
    data_dict["orientation_z_label"] = orien[idx_start:idx_end][:, 2]
    data_dict["orientation_w_label"] = orien[idx_start:idx_end][:, 3]

    data_frame = pd.DataFrame(data_dict)
    data_frame.to_csv(file_path, index=False)


def interpolate_nan_timestamps(timestamps):
    """Interpolates timestamps where there is missing values.

    Args:
        timestamps (list): LIst containing timestamps.

    Returns:
        list: Interpolated timestamps for NaN values.
    """
    # Identify the NaN values
    nan_indices = np.isnan(timestamps)

    # Perform linear interpolation
    timestamps[nan_indices] = np.interp(
        np.flatnonzero(nan_indices),
        np.flatnonzero(~nan_indices),
        timestamps[~nan_indices],
    )

    return timestamps


def main():  # NOQA D103
    """Main script."""
    args = parse_args()

    root_dir = Path(args.root_dir)

    label_root_dir = Path(args.root_dir) / "labels"
    label_files = sorted(list(label_root_dir.glob("*.pt")))

    assert label_files

    config_files = list(root_dir.glob("*.yaml"))

    assert len(config_files) == 1

    metadata_root_dir = Path(args.root_dir) / "metadata" / args.cam_name
    metadata_root_dir.mkdir(parents=True, exist_ok=True)

    plot_root = Path.cwd()

    # Polynomial fitting degree
    deg = args.polyfit_degree

    ####################################### LABELS #########################################
    seq_id_counter = 0

    pbar = tqdm(label_files)
    for label_file in pbar:
        pbar.set_description("Processing %s" % label_file.name)

        labels = torch.load(label_file)

        # Resolve the estimated ball position source. When the labels do not
        # contain a separate "est_ball_position", fall back to the original
        # triangulated "ball_position".
        has_est_ball_position = "est_ball_position" in labels
        est_ball_position = (
            labels["est_ball_position"]
            if has_est_ball_position
            else labels["ball_position"]
        )

        # Check that the estimation and original triangulation doesn't diverge a lot
        # from each other. Only meaningful when a separate estimate is present;
        # otherwise the estimate IS the original and there is nothing to compare.
        if has_est_ball_position:
            est_mean = np.nanmean(
                np.linalg.norm(
                    est_ball_position[:] - labels["ball_position"][:], axis=1
                )
            )
            if est_mean > 0.01:
                continue

        assert (
            labels["ball_timestamps"].shape[0]
            == labels["ball_position"].shape[0]
            == est_ball_position.shape[0]
        )

        ############# Nan splitting ###############
        # Split the sequences into non nan sequences
        nan_split_indices = get_consecutive_non_nan_indices(est_ball_position)

        # Section the ball positions pre and post processed
        est_ball_position_arrays = [
            est_ball_position[start:end] for start, end in nan_split_indices
        ]
        orig_ball_position_arrays = [
            labels["ball_position"][start:end] for start, end in nan_split_indices
        ]
        timestamp_arrays = [
            labels["ball_timestamps"][start:end] for start, end in nan_split_indices
        ]

        ball_orientation_arrays = [
            labels["ball_orientation"][start:end] for start, end in nan_split_indices
        ]

        ############# Nan splitting ###############

        ############# Seperate by Events -> Fit polynomial + velocity -> H5 Saving #############
        event_split_est_ball_position_arrays = []
        event_split_orig_ball_position_arrays = []
        event_split_timestamp_arrays = []
        event_split_ball_orientation_arrays = []
        for ts_array_id, ts_array in enumerate(timestamp_arrays):
            if len(ts_array) <= max((deg + 1), args.min_samples):
                continue

            if np.isnan(ts_array).any():
                ts_array[:] = interpolate_nan_timestamps(ts_array)

            split_seq_indices = split_sequence_indices_by_shot_times(
                ts_array, labels["events"], deg=deg
            )

            if split_seq_indices == []:
                event_split_est_ball_position_arrays.append(
                    est_ball_position_arrays[ts_array_id]
                )
                event_split_orig_ball_position_arrays.append(
                    orig_ball_position_arrays[ts_array_id]
                )
                event_split_timestamp_arrays.append(ts_array)
                event_split_ball_orientation_arrays.append(
                    ball_orientation_arrays[ts_array_id]
                )
            else:
                for seq_ind in split_seq_indices:
                    event_split_est_ball_position_arrays.append(
                        est_ball_position_arrays[ts_array_id][
                            seq_ind[0] : seq_ind[1] + 1
                        ]
                    )
                    event_split_orig_ball_position_arrays.append(
                        orig_ball_position_arrays[ts_array_id][
                            seq_ind[0] : seq_ind[1] + 1
                        ]
                    )
                    event_split_timestamp_arrays.append(
                        ts_array[seq_ind[0] : seq_ind[1] + 1]
                    )
                    event_split_ball_orientation_arrays.append(
                        ball_orientation_arrays[ts_array_id][
                            seq_ind[0] : seq_ind[1] + 1
                        ]
                    )

            if args.plot:
                plot_out_dir = (
                    plot_root
                    / root_dir.stem
                    / args.cam_name
                    / f"polyfit_{deg}deg_stitch_sections"
                )
                plot_out_dir.mkdir(parents=True, exist_ok=True)
                cprint(
                    f"Plotting ball positions for each section to {plot_out_dir}",
                    color="green",
                )

            # Stitch segments back together seperated by events
            for i, ball_pos_arr in enumerate(event_split_est_ball_position_arrays):
                assert len(ball_pos_arr) >= (deg + 1)

                # Estimated mean between GT and post-processed ball positions should below 0.01 m threshold
                positions_1000hz, velocities_1000hz, timestamps_1000hz = (
                    interpolate_ball_trajectory_polyfit(
                        ball_pos_arr, event_split_timestamp_arrays[i], deg=deg
                    )
                )

                # Expand arrays
                timestamps_mask, timestamps_orig_expand = expand_with_nans(
                    event_split_timestamp_arrays[i]
                )
                _, positions_orig_expand = expand_with_nans(
                    event_split_orig_ball_position_arrays[i]
                )

                _, ball_orientation_expand = expand_with_nans(
                    event_split_ball_orientation_arrays[i]
                )

                velocities_200hz = ball_vel_finite_diff(
                    ball_pos_arr, event_split_timestamp_arrays[i]
                )

                if i == 0:
                    stitched_positions_1000hz = positions_1000hz
                    stitched_velocities_1000hz = velocities_1000hz
                    stitched_timestamps_1000hz = timestamps_1000hz

                    stitched_positions_original = positions_orig_expand
                    stitched_ball_orientation = ball_orientation_expand
                    stitched_timestamps_original = timestamps_orig_expand
                    stitched_timestamps_mask = timestamps_mask

                    stitched_positions_200hz = ball_pos_arr
                    stitched_timestamps_200hz = event_split_timestamp_arrays[i]

                else:
                    stitched_positions_1000hz = np.vstack(
                        (stitched_positions_1000hz, positions_1000hz[1:])
                    )
                    stitched_velocities_1000hz = np.vstack(
                        (stitched_velocities_1000hz, velocities_1000hz[1:])
                    )
                    stitched_timestamps_1000hz = np.hstack(
                        (stitched_timestamps_1000hz, timestamps_1000hz[1:])
                    )

                    stitched_positions_original = np.vstack(
                        (stitched_positions_original, positions_orig_expand[1:])
                    )
                    stitched_ball_orientation = np.vstack(
                        (stitched_ball_orientation, ball_orientation_expand[1:])
                    )
                    stitched_timestamps_original = np.hstack(
                        (stitched_timestamps_original, timestamps_orig_expand[1:])
                    )
                    stitched_timestamps_mask = np.hstack(
                        (stitched_timestamps_mask, timestamps_mask[1:])
                    )

                    stitched_timestamps_200hz = np.hstack(
                        (stitched_timestamps_200hz, event_split_timestamp_arrays[i][1:])
                    )
                    stitched_positions_200hz = np.vstack(
                        (stitched_positions_200hz, ball_pos_arr[1:])
                    )

                if args.plot:
                    # plotting ball positions
                    plot_ball_positions(
                        ball_pos_arr,
                        event_split_timestamp_arrays[i],
                        positions_1000hz,
                        timestamps_1000hz,
                        plot_out_dir / f"{seq_id_counter:03d}_{i}_ball_positions.png",
                    )

                    # Plotting interpolated and raw ball trajectory with estimated velocity
                    plot_ball_positions_w_velocity(
                        ball_pos_arr,
                        event_split_timestamp_arrays[i],
                        positions_1000hz,
                        timestamps_1000hz,
                        velocities_1000hz,
                        velocities_200hz,
                        title=plot_out_dir
                        / f"{seq_id_counter:03d}_{i}_ball_positions_w_velocity.png",
                    )

            if args.plot:
                # Plot for analysis
                plot_out_dir = (
                    plot_root
                    / root_dir.stem
                    / args.cam_name
                    / f"polyfit_{deg}deg_stitch"
                )
                plot_out_dir.mkdir(parents=True, exist_ok=True)

                # plotting ball positions
                plot_ball_positions(
                    stitched_positions_200hz,
                    stitched_timestamps_200hz,
                    stitched_positions_1000hz,
                    stitched_timestamps_1000hz,
                    plot_out_dir / f"{seq_id_counter:03d}_ball_positions.png",
                )

                # Plotting interpolated and raw ball trajectory with estimated velocity
                plot_ball_positions_w_velocity(
                    stitched_positions_200hz,
                    stitched_timestamps_200hz,
                    stitched_positions_1000hz,
                    stitched_timestamps_1000hz,
                    stitched_velocities_1000hz,
                    title=plot_out_dir
                    / f"{seq_id_counter:03d}_ball_positions_w_velocity.png",
                    events=labels["events"],
                )

            # Check if timestamps are correctly aligned and increasing order TODO make assert
            assert np.all(np.abs(np.diff(stitched_timestamps_200hz) - 0.005) < 1e-4)
            assert np.all(
                np.abs(np.diff(stitched_timestamps_1000hz) - 0.001) < 0.001 * 0.5
            )

            assert not np.isnan(stitched_timestamps_original[0])
            (
                positions_2d,
                velocity_2d,
                radius_2d,
                timestamps_2d,
                points_orig_2d,
                radius_orig_2d,
                _,
            ) = project_3d_to_2d(
                camera_name=args.cam_name,
                camera_params_file=config_files[0],
                positions=stitched_positions_1000hz,
                positions_orig=stitched_positions_original,
                velocities=stitched_velocities_1000hz,
                timestamps=stitched_timestamps_1000hz,
                timestamps_orig=stitched_timestamps_original,
                timestamps_mask=stitched_timestamps_mask,
                ball_radius=args.ball_radius,
            )

            assert (
                len(positions_2d)
                == len(velocity_2d)
                == len(radius_2d)
                == len(timestamps_2d)
            )

            # SKip saving if reprojection is out of camera view of total length is too short
            if len(positions_2d) <= 10:
                # reset event splitters
                event_split_est_ball_position_arrays = []
                event_split_orig_ball_position_arrays = []
                event_split_timestamp_arrays = []

                continue

            # Save labels into h5 files
            h5_folder_name = Path(args.root_dir) / "h5" / args.cam_name
            h5_folder_name.mkdir(parents=True, exist_ok=True)

            h5_label_file_name = (
                f"seq_{seq_id_counter:03d}_" + label_file.stem + "_label.h5"
            )

            h5_label_dir = h5_folder_name / h5_label_file_name
            h5_label = H5WriterLabel(h5_label_dir, len(positions_2d))

            label_dict = {
                "points": positions_2d,
                "radius": radius_2d,
                "velocities": velocity_2d,
                "timestamps": timestamps_2d,
                "points_orig": points_orig_2d,
                "radius_orig": radius_orig_2d,
            }
            h5_label.add_data(label_dict)

            metadata_file_name = metadata_root_dir / Path(
                f"seq_{seq_id_counter:03d}_" + str(label_file.stem) + ".csv"
            )

            save_to_csv(
                metadata_file_name,
                label_file.stem,
                seq_id_counter,
                (timestamps_2d[0], timestamps_2d[-1]),
                stitched_timestamps_original,
                stitched_timestamps_1000hz,
                stitched_positions_1000hz,
                stitched_velocities_1000hz,
                stitched_ball_orientation,
            )

            # reset event splitters
            event_split_est_ball_position_arrays = []
            event_split_orig_ball_position_arrays = []
            event_split_timestamp_arrays = []
            event_split_ball_orientation_arrays = []

            # Update unique sequence id
            seq_id_counter += 1
    ####################################### LABELS #########################################


if __name__ == "__main__":
    main()
