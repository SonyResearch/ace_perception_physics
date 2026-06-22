# pylint: disable=too-many-locals, too-many-statements, too-many-arguments
# TODO(asude): clean pylint

"""
@brief Supporting script for ball position interpolations in proplayer labeling.

@file interpolate_ball_positions.py
@author Asude Aydin (asude.aydin@sony.com)
@date August 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""


from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from calibration import python_module as calibration
from scipy.interpolate import interp1d


def interpolate_on_ellipse(focal_length_x, focal_length_y, angle):
    """
    Interpolate the focal length value on an ellipse based on an angle.

    Args:
        focal_length_x (float): Focal length in the x direction.
        focal_length_y (float): Focal length in the y direction.
        angle (float): The angle at which to interpolate (in rad).

    Returns:
        float: The interpolated focal length on the ellipse at the given angle.
    """
    # Calculate the parameters of the ellipse
    semi_major_axis = focal_length_x
    semi_minor_axis = focal_length_y

    # Calculate the focal length on the ellipse at the given angle
    interpolated_focal_length = np.sqrt((semi_major_axis * np.cos(angle)) ** 2 + (semi_minor_axis * np.sin(angle)) ** 2)

    return interpolated_focal_length


def calculate_radius_pixel(
    aps_position: List,
    projected_point: Tuple[float, float],
    t_co: np.ndarray,
    ball_radius: float,
    focal_lengths: Tuple[float, float],
) -> float:
    """
    Calculate rad with the similar triangle equivalency, for this we transform the ball point in the camera coordinates.

    Args:
        `aps_position`(List): 3D positions calibrated through the APS coordinates
        `projected_point` (Tuple[float]): projected center of the ball
        `t_co` (np.ndarray): transformation matrix from origin to camera
        `ball_radius` (float): ball radius in 3d
        `focal_lengths` (Tuple[float]): tuple of the x,y focal length of the camera
    returns:
        radius in pixel size
    """
    f_x, f_y = focal_lengths
    p_u, p_v = projected_point

    # Transform the point from origin back to camera
    x_o = np.concatenate([aps_position, [1]])
    x_c = np.dot(t_co, x_o)

    x_norm = np.sqrt((x_c[:3] ** 2).sum())
    x_z = x_c[2]

    # Angle in the projection place
    phi = np.arctan2(p_u, p_v)

    # slerped f
    f_new = interpolate_on_ellipse(f_x, f_y, phi)

    # Scale factor of the radius by what we can see of the ball
    scale_factor = np.sqrt(x_norm**2 - ball_radius**2) / x_norm

    # calculate projected radius using similar triangles
    projected_radius = f_new / x_z * ball_radius * scale_factor
    return projected_radius


def get_non_nan_mask(array: np.ndarray):
    """
    Creates a mask for the rows in the given array that do not contain NaN values.

    Args:
    array (np.ndarray): The input array.

    Returns:
    np.ndarray: A boolean mask where True indicates rows without NaN values.
    """
    return ~np.isnan(array).any(axis=1)


def find_largest_consecutive_non_nan_block(array: np.ndarray):
    """
    Finds the largest block of consecutive rows in the array that do not contain NaN values.

    Args:
    array (np.ndarray): The input array.

    Returns:
    np.ndarray: The largest block of consecutive rows without NaN values.
    """

    # Initialize variables to track the largest block
    max_start = max_end = start = end = None
    max_length = current_length = 0

    for i, value in enumerate(array):
        if not np.isnan(value).any():
            if current_length == 0:
                start = i
            current_length += 1
            end = i + 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start, max_end = start, end
            current_length = 0

    # Check the last block
    if current_length > max_length:
        max_start, max_end = start, end

    return (
        array[max_start:max_end] if max_start is not None else np.array([]),
        max_start,
        max_end,
    )


def expand_with_nans(input_array, num_nans=4):
    """Expand given array with NaNs

    Args:
        input_array (_type_): INput array to expand
        num_nans (int, optional): Number of NaN values to expand with. Defaults to 4.

    Returns:
        _type_: Non NaN mask and expanded array
    """
    # Get the shape of the input array
    input_shape = input_array.shape

    # Calculate the new shape for the expanded array
    expanded_shape = list(input_shape)
    expanded_shape[0] = input_shape[0] * (num_nans + 1) - num_nans

    # Create a new array with the expanded shape
    expanded_array = np.empty(expanded_shape)
    expanded_array[:] = np.nan

    # Fill in the original values in the expanded array
    expanded_array[:: num_nans + 1, ...] = input_array

    # Apply a mask to identify NaN values
    mask = np.isnan(expanded_array)

    return ~mask, expanded_array


def round_to_nearest_0_005(value):
    """Round value to nearest 0.005 value."""
    return round(value * 200) / 200


def plot_ball_positions(
    estimated_positions,
    timestamps,
    interpolated_positions,
    interpolated_timestamps,
    title=None,
):
    """Plot ball positions"""
    # Plot positions
    _, axs = plt.subplots(3, 1, figsize=(18, 10))

    for i, label in enumerate(["x", "y", "z"]):
        axs[i].plot(timestamps, estimated_positions[:, i], "o", label="200 Hz", markersize=3)
        axs[i].plot(
            interpolated_timestamps,
            interpolated_positions[:, i],
            "-",
            label="1 kHz",
            linewidth=1,
        )
        axs[i].set_title(f"Position - {label.upper()}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Position (m)")
        axs[i].legend()
    plt.tight_layout()
    if title is None:
        plt.show()
    else:
        plt.savefig(title)
    plt.close()


def plot_ball_positions_w_velocity(
    estimated_positions,
    timestamps,
    interpolated_positions,
    interpolated_timestamps,
    velocity,
    velocity_comp=None,
    comp_skip=5,
    title=None,
    events=None,
):
    """Plot ball positions and velocity EVS + APS."""
    event_dict = {
        "shot_p1": "r",
        "bounce_p1": "g",
        "bounce_p2": "cyan",
        "shot_p2": "orange",
        "net": "purple",
    }

    # Plotting TODO add velocity as quiver
    _, axs = plt.subplots(3, 1, figsize=(18, 10))

    quiver_scale = 10  # Adjust scale as needed for better visualization
    skip = 1  # Skip some points for quivers to make the plot cleaner

    # Plot positions
    for i, label in enumerate(["x", "y", "z"]):
        axs[i].plot(
            timestamps,
            estimated_positions[:, i],
            "o",
            label="200 Hz",
            markersize=3,
            color="b",
        )
        axs[i].plot(
            interpolated_timestamps,
            interpolated_positions[:, i],
            "-",
            label="1 kHz",
            linewidth=1,
            color="m",
        )
        # Add quivers for velocity
        axs[i].quiver(
            interpolated_timestamps[::skip],
            interpolated_positions[::skip, i],
            np.zeros_like(interpolated_timestamps[::skip]),
            velocity[::skip, i],
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            color="m",
            width=0.0005,
        )

        if velocity_comp is not None:
            axs[i].quiver(
                interpolated_timestamps[::comp_skip],
                interpolated_positions[::comp_skip, i],
                np.zeros_like(interpolated_timestamps[::comp_skip]),
                velocity_comp[::skip, i],
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                color="g",
                width=0.0005,
            )
        axs[i].set_title(f"Position - {label.upper()}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Position (m)")
        axs[i].legend()

        if events is not None:
            for event in events:
                assert event["type"] in event_dict.keys()
                if event["timestamp"] < timestamps.max() and event["timestamp"] > timestamps.min():
                    axs[i].axvline(event["timestamp"], color=event_dict[event["type"]])

    plt.tight_layout()
    if title is None:
        plt.show()
    else:
        plt.savefig(title)

    plt.close()


def ball_vel_finite_diff(positions, timestamp):
    """Estimate ball position finit differences."""
    velocities = np.ones((len(positions), 3)) * np.nan

    for i in range(3):
        velocities[1:, i] = np.diff(positions[:, i], axis=0) / np.diff(timestamp, axis=0)

    velocities[0, :] = velocities[1, :]

    return velocities


def interpolate_ball_trajectory_linear(estimated_positions: list, timestamps: list):
    """Linear ball position interpolation"""
    # Linearly interpolating positions to 500 Hz
    t_500hz = np.arange(
        round_to_nearest_0_005(timestamps[0]),
        round_to_nearest_0_005(timestamps[-1]) + 0.0005,
        0.001,
    )

    interp_func = interp1d(timestamps, estimated_positions, axis=0, kind="linear", fill_value="extrapolate")
    positions_500hz = interp_func(t_500hz)

    velocities_500hz = ball_vel_finite_diff(positions_500hz, t_500hz)

    mask, timestamps_original = expand_with_nans(timestamps)

    assert len(positions_500hz) == len(velocities_500hz) == len(t_500hz) == len(timestamps_original) == len(mask)

    return positions_500hz, velocities_500hz, t_500hz, timestamps_original, mask


def interpolate_ball_trajectory_polyfit(estimated_positions, timestamps, deg=3):
    """Interpolate ball positions with polynomial fit."""
    t_500hz = np.arange(
        round_to_nearest_0_005(timestamps[0]),
        round_to_nearest_0_005(timestamps[-1]) + 0.0005,
        0.001,
    )

    c_x = np.polyfit(timestamps - timestamps[0], estimated_positions[:, 0], deg)
    c_y = np.polyfit(timestamps - timestamps[0], estimated_positions[:, 1], deg)
    c_z = np.polyfit(timestamps - timestamps[0], estimated_positions[:, 2], deg)

    interp_x = np.poly1d(c_x)
    interp_x_positions = interp_x(t_500hz - t_500hz[0])

    interp_y = np.poly1d(c_y)
    interp_y_positions = interp_y(t_500hz - t_500hz[0])

    interp_z = np.poly1d(c_z)
    interp_z_positions = interp_z(t_500hz - t_500hz[0])

    positions_500hz = np.vstack((interp_x_positions, interp_y_positions, interp_z_positions)).T

    velocities_500hz = ball_vel_finite_diff(positions_500hz, t_500hz)

    assert len(positions_500hz) == len(velocities_500hz) == len(t_500hz)

    return positions_500hz, velocities_500hz, t_500hz


def project_3d_to_2d(
    camera_name: str,
    camera_params_file: str,
    positions: list,
    positions_orig: list,
    velocities: list,
    timestamps: list,
    timestamps_orig: list,
    timestamps_mask: list,
    ball_radius: float,
):
    """
    1. TODO
    2. Backproject the 3D points:
        Here we take the optimized 3D points into the predefined camera frame
    Finally we end up with a TODO file for every frame that has the representation, as well as the ground truths
    Args:
        TODO
        `camera_name` (str): name of the camera used for backprojection
        `camera_params_file` (str): parameter file of the cameras, in order to get the right backprojection
        `ball_radius`(float): radius of the ball [m]
    Returns:
        None
    """
    # Initialize calibration parameter for backprojection
    calib_params = calibration.CameraCalibrationParameters()
    calib_params.initialize(str(camera_params_file))
    resolution = np.array(calib_params.cameras[camera_name].resolution)

    # Extract origin->camera matrix
    evs_camera = calib_params.cameras[camera_name]

    # This matrix converts from the origin to the camera, careful about the naming convention
    t_co = evs_camera.T_camera_origin
    focal_lengths = (evs_camera.camera_matrix[0, 0], evs_camera.camera_matrix[1, 1])

    points_list, velocity_list, radius_list = [], [], []

    for idx, pos in enumerate(positions):
        # Project 3D positions into image space
        points_opt = np.zeros((2, 1), dtype=np.float32)
        dp_dx_opt = np.zeros((2, 3), dtype=np.float32, order="F")

        success = calib_params.cameras[camera_name].project_point_extended(np.float32(pos), points_opt, dp_dx_opt)
        if all(
            [
                success is True,
                round(points_opt[0][0]) >= 0,
                round(points_opt[0][0]) < resolution[0],
                round(points_opt[1][0]) >= 0,
                round(points_opt[1][0]) < resolution[1],
            ]
        ):
            # Reproject velocity in camera frame [m/s] --> [px/s]
            vel_2d_temp = np.dot(dp_dx_opt, velocities[idx])
            vel_2d = [vel_2d_temp[0], vel_2d_temp[1]]

            # Extract the radius in the image size
            radius_image_opt = calculate_radius_pixel(
                pos,
                (points_opt[0][0], points_opt[1][0]),
                t_co,
                ball_radius,
                focal_lengths,
            )

            points_opt = [round(points_opt[0][0]), round(points_opt[1][0])]

        else:
            radius_image_opt = np.nan
            points_opt = [np.nan, np.nan]
            vel_2d = [np.nan, np.nan]

        points_list.append(points_opt)
        radius_list.append(radius_image_opt)
        velocity_list.append(vel_2d)

    points_orig_list, radius_orig_list = [], []

    for idx, pos in enumerate(positions_orig):
        # Project 3D positions into image space
        points_opt = np.zeros((2, 1), dtype=np.float32)
        dp_dx_opt = np.zeros((2, 3), dtype=np.float32, order="F")

        if not np.any(np.isnan(pos)):
            success = calib_params.cameras[camera_name].project_point_extended(np.float32(pos), points_opt, dp_dx_opt)
            if all(
                [
                    success is True,
                    round(points_opt[0][0]) >= 0,
                    round(points_opt[0][0]) < resolution[0],
                    round(points_opt[1][0]) >= 0,
                    round(points_opt[1][0]) < resolution[1],
                ]
            ):
                # Extract the radius in the image size
                radius_image_opt = calculate_radius_pixel(
                    pos,
                    (points_opt[0][0], points_opt[1][0]),
                    t_co,
                    ball_radius,
                    focal_lengths,
                )

                points_opt = [round(points_opt[0][0]), round(points_opt[1][0])]

            else:
                radius_image_opt = np.nan
                points_opt = [np.nan, np.nan]
        else:
            radius_image_opt = np.nan
            points_opt = [np.nan, np.nan]

        points_orig_list.append(points_opt)
        radius_orig_list.append(radius_image_opt)

    points_final, non_nan_start, non_nan_end = find_largest_consecutive_non_nan_block(np.array(points_list))
    velocity_final, _, _ = find_largest_consecutive_non_nan_block(np.array(velocity_list))
    radius_final, _, _ = find_largest_consecutive_non_nan_block(np.array(radius_list))

    # Overwrite timesta
    if len(points_final) == 0:
        timestamps_final = np.array([])
        timestamps_orig_final = np.array([])
        radius_orig_final = np.array([])
        points_orig_final = np.array([])
    else:
        timestamps_final = timestamps[non_nan_start:non_nan_end]
        points_orig_final = points_orig_list[non_nan_start:non_nan_end]
        radius_orig_final = radius_orig_list[non_nan_start:non_nan_end]

        timestamps_orig_final = timestamps_orig[non_nan_start:non_nan_end]
        timestamps_mask_final = timestamps_mask[non_nan_start:non_nan_end]
        timestamps_orig_final = timestamps_orig_final[timestamps_mask_final]

        assert len(timestamps_final) == len(timestamps_mask_final) == len(points_orig_final) == len(radius_orig_final)

    assert len(points_final) == len(radius_final) == len(velocity_final) == len(timestamps_final)

    return (
        points_final,
        velocity_final,
        radius_final,
        timestamps_final,
        points_orig_final,
        radius_orig_final,
        timestamps_orig_final,
    )
