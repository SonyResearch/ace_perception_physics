# pylint: disable=too-many-locals, too-many-statements, too-many-branches, invalid-name, too-many-arguments
# TODO(asude): clean pylint

"""
@brief Create event representations (time surfaces and histograms) of the event stream.

@file create_event_represenations.py
@author Claudio Fanconi (claudioandrea.fanconi@sony.com)
@date 2023
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import os
from typing import List, Optional, Tuple

import hdf5storage
import numpy as np
import pandas as pd
from calibration import python_module as calibration
from evs.utilities.event_readers import FixedSizeTriggerEventReader

from data_generation.tools.e2frame import events_to_frame, events_to_timesurface


def merge_csv_files(folder_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Merge CSV files from a given folder into a single pandas DataFrame.

    Args:
        `folder_path` (str): The path to the folder containing the CSV files.
        `columns` (List[str], optional): A list of column names to select from the CSV files.
                                       If not provided, all columns will be included.
    Returns:
        pd.DataFrame: The merged DataFrame.
    Raises:
        ValueError: If the folder_path is not a valid directory.
        ValueError: If the folder does not contain any CSV files.
    Note:
        - All CSV files in the folder should have the same columns and structure.
        - The function assumes that the CSV files have a header row with column names.
    """
    # Check if folder_path is a valid directory
    if not os.path.isdir(folder_path):
        raise ValueError("Invalid directory path. Please provide a valid folder path.")

    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".csv")]

    # Check if the folder contains any CSV files
    if not csv_files:
        raise ValueError("The folder does not contain any CSV files.")

    dataframes = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        dataframe = pd.read_csv(file_path, usecols=columns)
        dataframes.append(dataframe)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    return merged_df


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


def create_event_representations(
    source_path_events: str,
    source_path_event_triggers: str,
    source_path_points: str,
    original_point_path: str,
    target_path: str,
    camera_name: str,
    camera_params_file: str,
    accumulation_time: float,
    ball_radius: float,
    offsets: Tuple[float, float, float],
    interpolated: bool,
    time_offset: int,
) -> None:
    """
    Create {time_surface, histogram} representations of the event stream with the given accumulation time.

    This step consists of two steps:
    1. Create the event representations:
        For this we need the raw files of the predefined camera, where we accumulate the events
    2. Backproject the 3D points:
        Here we take the optimized 3D points into the predefined camera frame
    Finally we end up with a .mat file for every frame that has the representation, as well as the ground truths
    Args:
        `source_path_events` (str): path of the source for the camera
        `source_path_event_triggers` (str): path to the event trigger file
        `source_path_points` (str): path of the source for the 3D optimized data
        `original_point_path` (str): path to the original points for visualisation
        `target_path` (str): target path, where the .mat files shall be saved
        `camera_name` (str): name of the camera used for backprojection
        `camera_params_file` (str): parameter file of the cameras, in order to get the right backprojection
        `accumulation_time` (float): accumulation time of events to create the representantion
        `ball_radius`(float): radius of the ball [m]
        `offsets`: (Tuple[float]): (x_offset, y_offset, z_offset)
        `interpolated` (bool): if true, uses the new frame ids, rather than the original ones
        `time_offset` (int): time offset in int, representing a millisecond
    Returns:
        None
    """
    # Read all the CSV files and merge into one dataframe:
    points_3d = merge_csv_files(source_path_points)
    points_3d_original = pd.read_csv(original_point_path)
    merged_df = points_3d.merge(points_3d_original, on="frame_id", suffixes=("", "_orig"), how="left")

    # Extract the trigger IDs to iterate throught the raw files
    gt_frameid = merged_df.frame_id.values.tolist()
    gt_frameid = [int(i) for i in gt_frameid if not np.isnan(i)]

    # Readjust the offset
    if not interpolated:
        merged_df.position_x -= offsets[0]
        merged_df.position_y -= offsets[1]
        merged_df.position_z -= offsets[2]

    # Initialize calibration parameter for backprojection
    calib_params = calibration.CameraCalibrationParameters()
    calib_params.initialize(str(camera_params_file))
    resolution = np.array(calib_params.cameras[camera_name].resolution)

    # Extract origin->camera matrix
    evs_camera = calib_params.cameras[camera_name]

    # This matrix converts from the origin to the camera, careful about the naming convention
    t_co = evs_camera.T_camera_origin
    focal_lengths = (evs_camera.camera_matrix[0, 0], evs_camera.camera_matrix[1, 1])

    # Initialize event reader
    event_window_iterator = FixedSizeTriggerEventReader(source_path_events, source_path_event_triggers)

    for idx, event_window in enumerate(event_window_iterator):
        if idx in gt_frameid:
            if interpolated:
                new_idx_filter = (idx - 1) * 10

                filtered_df = merged_df[
                    (merged_df.frame_id_new > new_idx_filter) & (merged_df.frame_id_new <= idx * 10)
                ].copy()

                # Reverse order to find the accumulated times
                filtered_df = filtered_df.iloc[::-1]

                for i, (index_frame, row) in enumerate(filtered_df.iterrows()):
                    frame_id_new = row.frame_id_new
                    # Extract the events within the accumulation time
                    event_ts = event_window[:, 0]
                    selected_idx = np.where(
                        (event_window[:, 0] <= event_ts[-1] - (i * float(accumulation_time)))
                        & (
                            event_window[:, 0]
                            >= event_ts[-1] - (i * float(accumulation_time)) - float(accumulation_time)
                        )
                    )

                    create_and_save_features(
                        merged_df=merged_df,
                        event_window=event_window[selected_idx],
                        index_frame=index_frame + time_offset,
                        calib_params=calib_params,
                        camera_name=camera_name,
                        resolution=resolution,
                        t_co=t_co,
                        focal_lengths=focal_lengths,
                        ball_radius=ball_radius,
                        target_path=target_path,
                        idx=frame_id_new,
                    )

            # Extract the ground truth position and velocity
            else:
                # Extract the events within the accumulation time
                event_ts = event_window[:, 0]
                first_idx = np.where(event_window[:, 0] >= event_ts[-1] - float(accumulation_time))
                event_window = event_window[first_idx[0][0] :]

                index_frame = gt_frameid.index(idx)
                create_and_save_features(
                    merged_df=merged_df,
                    event_window=event_window,
                    index_frame=index_frame + time_offset,
                    calib_params=calib_params,
                    camera_name=camera_name,
                    resolution=resolution,
                    t_co=t_co,
                    focal_lengths=focal_lengths,
                    ball_radius=ball_radius,
                    target_path=target_path,
                    idx=idx,
                )


def create_and_save_features(
    merged_df: pd.DataFrame,
    event_window: np.ndarray,
    index_frame: int,
    calib_params: calibration.CameraCalibrationParameters,
    camera_name: str,
    resolution: Tuple[float, float],
    t_co: np.ndarray,
    focal_lengths: Tuple[float, float],
    ball_radius: float,
    target_path: str,
    idx: int,
) -> None:
    """
    Create and save event representations and labels.

    Args:
        `merged_df` (pd.DataFrame): dataframe containing all the velocities and points of the recording
        `event_window` (np.ndarray): numpy array consisting of events
        `index_frame` (int): the index of the frame id in the large merged dataframe for iloc
        `calib_params` (calibration.CameraCalibrationParameters): calibration parameters for backprojection
        `camera_name` (str): name of the camera
        `resolution` (Tuple[float]): resolution of the camera tuple
        `t_co` (np.ndarray): matrix that converts from the origin to the camera space
        `focal_lengths` (Tuple[float]): (x focal lenght, y focal length)
        `ball_radius` (float): radius of the ball in [m]
        `target_path` (str): target path where the .mat files are stored
        `idx` (int): frame id (new or old, depending if interpolated)

    NOTE: optimised trajectories should be dense and not contain any nans, the original APS triangulations however do.

    returns:
        None
    """
    # Location and velocity:
    points_opt = np.zeros((2, 1), dtype=np.float32)
    dp_dx_opt = np.zeros((2, 3), dtype=np.float32, order="F")

    points_orig = np.zeros((2, 1), dtype=np.float32)

    not_nan_check_1 = (
        not np.isnan(float(merged_df.position_x.iloc[index_frame]))
        or not np.isnan(float(merged_df.position_y.iloc[index_frame]))
        or not np.isnan(float(merged_df.position_z.iloc[index_frame]))
    )
    if not_nan_check_1:
        positions_optimised = np.array(
            [
                float(merged_df.position_x.iloc[index_frame]),
                float(merged_df.position_y.iloc[index_frame]),
                float(merged_df.position_z.iloc[index_frame]),
            ]
        )
        success = calib_params.cameras[camera_name].project_point_extended(
            np.float32(positions_optimised), points_opt, dp_dx_opt
        )
        if all(
            [
                success is True,
                round(points_opt[0][0]) >= 0,
                round(points_opt[0][0]) < resolution[0],
                round(points_opt[1][0]) >= 0,
                round(points_opt[1][0]) < resolution[1],
            ]
        ):
            # Reproject original triangulation

            not_nan_check = (
                not np.isnan(float(merged_df.position_x_orig.iloc[index_frame]))
                or not np.isnan(float(merged_df.position_y_orig.iloc[index_frame]))
                or not np.isnan(float(merged_df.position_z_orig.iloc[index_frame]))
            )
            if not_nan_check:
                positions_original = np.array(
                    [
                        float(merged_df.position_x_orig.iloc[index_frame]),
                        float(merged_df.position_y_orig.iloc[index_frame]),
                        float(merged_df.position_z_orig.iloc[index_frame]),
                    ]
                )

            # Reproject velocity in camera frame [m/s] --> [px/s]
            vel_2d = np.dot(
                dp_dx_opt,
                np.array(
                    [
                        float(merged_df.vel_x.iloc[index_frame]),
                        float(merged_df.vel_y.iloc[index_frame]),
                        float(merged_df.vel_z.iloc[index_frame]),
                    ]
                ),
            )

            # Extract histogram represenation
            histogram_frame = events_to_frame(event_window, width=resolution[0], height=resolution[1])

            # Extract time surface representation
            time_surface_frame = events_to_timesurface(event_window, width=resolution[0], height=resolution[1])

            # Extract the radius in the image size
            radius_image_opt = calculate_radius_pixel(
                positions_optimised,
                (points_opt[0][0], points_opt[1][0]),
                t_co,
                ball_radius,
                focal_lengths,
            )

            if not_nan_check:
                radius_image_orig = calculate_radius_pixel(
                    positions_original,
                    (points_orig[0][0], points_orig[1][0]),
                    t_co,
                    ball_radius,
                    focal_lengths,
                )

            # Save to .mat files
            save_path = os.path.join(target_path, f"{int(idx)}.mat")
            hdf5storage.savemat(
                save_path,
                {
                    "idx": idx,
                    "points": np.array([round(points_opt[0][0]), round(points_opt[1][0])]),
                    "points_orig": np.array([round(points_orig[0][0]), round(points_orig[1][0])])
                    if not_nan_check
                    else np.nan,
                    "velocities": vel_2d,
                    "radius": radius_image_opt,
                    "radius_orig": radius_image_orig if not_nan_check else np.nan,
                    "events_histogram": histogram_frame,
                    "events_surface": time_surface_frame,
                },
            )
