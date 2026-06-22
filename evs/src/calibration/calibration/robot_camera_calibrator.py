#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Run extrinsics calibration on a multi-camera and robot system.

This is achieved by aligning trajectories of the ball from both the robot and each individual camera.
"""

import argparse
import copy
import datetime
import functools
import logging
import shutil

import ace_yaml as yaml
import evs.src.calibration.calibration.calibrator_utils as calibrator_utils
import click
import coloredlogs
import cv2
import numpy
import evs.src.calibration.calibration.recorder_utils as recorder_utils
import scipy.optimize
import evs.src.calibration.calibration.viz_utils as viz_utils
from ament_index_python.packages import get_package_share_directory

log = logging.getLogger(__name__)


def parse_args():
    """
    Provide script-specific arguments.

    @return: Parsed argument object
    """
    parser = argparse.ArgumentParser(description="Runs extrinscis calibration on a multi-camera and robot system")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Lab specific configuration, e.g. zrh00, zrh01, tyo02.",
    )
    parser.add_argument(
        "--ball_radius",
        type=float,
        default=0.02,
        help="Actual ball radius in meters (default: %(default)s)",
    )
    parser.add_argument(
        "--scale_ratio",
        type=float,
        default=0.0,
        help="Applied ratio of the detected scale, in the range [0, 1] (default: %(default)s)",
    )
    parser.add_argument(
        "--use_observations_file",
        action="store_true",
        default=False,
        help="Load observations file",
    )
    parser.add_argument("--log_level", default=logging.INFO, help="Logging level (default: %(default)s)")
    parser.add_argument(
        "--in_worldframe",
        action="store_true",
        default=False,
        help="Set this to change the camera basis into the world frame (default: %(default)s)",
    )
    parser.add_argument(
        "--robot_only",
        action="store_true",
        default=False,
        help="Set this to limit the changes to the robot transform only (default: %(default)s)",
    )
    parser.add_argument(
        "--save_graphics",
        action="store_true",
        default=False,
        help="Set this to store graphics to the disk (default: %(default)s)",
    )

    return parser.parse_args()


def configure_logger(log_level=logging.INFO):
    """Set the logging options (level, messages color, ... etc)."""
    coloredlogs.install(level=log_level)
    log.setLevel(log_level)

    viz_utils.configure()


def collect_or_load_observations(use_observations_file):
    """
    Collect (via ROS) or load observations (from a file).

    It either loads observations from a file (when use_observations_file is set),
    or otherwise collects new ones
    """
    if use_observations_file:
        observations_dict = numpy.load("data/tmp/observations.npz")
        observations_dict = {key: observations_dict[key] for key in observations_dict.files}
    else:
        observations_dict = recorder_utils.collect_observations(save_as="data/tmp/observations.npz")

    return observations_dict


def optimize_time_shift_and_transforms(calib_conf, observations_dict, initial_time_shift, params, scale_ratio=0.0):
    """
    Optimize both the time-shift and camera extrinsic (affine/rigid) transforms via error minimization.

    The time-shift is optimized via parameter space exploration (passed as external parameter),
    while transforms are updated with the corresponding optimal weighted least-squares solutions.
    There are three main steps:
     - The input time-delay is optimized (via some scipy.optimize) to match both robot and camera point sets
     - The transforms are optimized (closed loop formula) to minimize the point sets disagreement
     - The final error is computed after updating the point sets with the estimated transforms.
    """
    # Do not pass initial_time_shift in params0, as the optimizer will diverge when it has a large value
    time_shift = params[0] + initial_time_shift

    robot_time_stamps = observations_dict["robot_time_stamps"]
    robot_points_3d = observations_dict["robot_observations"].reshape((-1, 3))
    camera_time_stamps = observations_dict["camera_time_stamps"]

    matching_robot_points_3d = calibrator_utils.get_matching_points(
        robot_points_3d, robot_time_stamps - time_shift, camera_time_stamps
    )
    points_3d = compute_points_3d(calib_conf, observations_dict, matching_robot_points_3d=matching_robot_points_3d)
    try:
        update_transforms(calib_conf, observations_dict, points_3d, matching_robot_points_3d, scale_ratio=scale_ratio)
    except ValueError:
        return numpy.inf
    points_3d = compute_points_3d(calib_conf, observations_dict, matching_robot_points_3d=matching_robot_points_3d)

    error = calibrator_utils.get_root_mean_square_error(points_3d, matching_robot_points_3d, scale_factor=1e3)
    log.debug("Time shift of %.2f msec has an error of %.2f mm", 1e3 * time_shift, error)
    return error


def update_transforms(calib_conf, observations_dict, points_3d, matching_robot_points_3d, scale_ratio=0.0):
    """Update camera extrinsics based on the current 3D points."""
    for camera_index, camera_name in enumerate(observations_dict["camera_names"]):
        camera_conf = calib_conf[camera_name]

        valid_camera_mask = numpy.isfinite(observations_dict["camera_observations"][:, camera_index, 2])
        valid_camera_points_3d = points_3d[valid_camera_mask, camera_index]
        transform_cw = numpy.linalg.multi_dot(
            [
                camera_conf["T_CW"],
                calib_conf["T_WO"],
                calibrator_utils.compute_weighted_affine_transformation(
                    matching_robot_points_3d[valid_camera_mask],
                    valid_camera_points_3d,
                    scale_ratio=scale_ratio,
                    show_debug=log.isEnabledFor(logging.DEBUG),
                ),
                numpy.linalg.inv(calib_conf["T_WO"]),
            ]
        )

        # Update camera
        camera_conf["T_CW"] = transform_cw.tolist()


def compute_points_3d(
    calib_conf,
    observations_dict,
    matching_robot_points_3d=None,
    compute_triangulations=False,
    ball_radius=None,
):
    """Compute points of each camera based on robot points, observed ball_radius, or triangulation."""
    points_3d = numpy.empty(observations_dict["camera_observations"].shape[:2] + (3,)) * numpy.nan
    camera_transforms = numpy.empty((len(observations_dict["camera_names"]), 4, 4)) * numpy.nan

    for camera_index, camera_name in enumerate(observations_dict["camera_names"]):
        camera_conf = calib_conf[camera_name]
        camera_transforms[camera_index] = numpy.linalg.inv(numpy.matmul(camera_conf["T_CW"], calib_conf["T_WO"]))
        camera_matrix = numpy.array(camera_conf["camera_matrix"])
        distortion_vect = numpy.array(camera_conf["distortion_coeffs"])

        valid_camera_mask = numpy.isfinite(observations_dict["camera_observations"][:, camera_index, 2])
        valid_points_2d = observations_dict["camera_observations"][valid_camera_mask, camera_index, :2]
        if valid_points_2d.shape[0] == 0:
            raise ValueError("Camera {:s} has no observations!".format(camera_name))

        if camera_conf["distortion_model"] == "equidistant":
            valid_points_2d[:] = cv2.fisheye.undistortPoints(
                valid_points_2d.reshape((1, -1, 2)), camera_matrix, distortion_vect, None, camera_matrix
            ).reshape((-1, 2))
        elif camera_conf["distortion_model"] == "radtan":
            valid_points_2d[:] = cv2.undistortPoints(
                valid_points_2d.reshape((1, -1, 2)), camera_matrix, distortion_vect, None, camera_matrix
            ).reshape((-1, 2))
        else:
            raise ValueError(f"Unexpected distortion model '{camera_conf['distortion_model']}'")

        valid_points_2d_homo = calibrator_utils.transform_points(
            numpy.linalg.inv(camera_matrix), valid_points_2d, return_homo=True
        )

        if matching_robot_points_3d is not None:
            valid_depth = calibrator_utils.transform_points(
                numpy.linalg.inv(camera_transforms[camera_index]),
                matching_robot_points_3d[valid_camera_mask],
            )[:, 2:3]
        else:
            focal_length = numpy.prod(numpy.diag(camera_matrix[:2, :2])) ** 0.5
            valid_depth = (
                focal_length
                * ball_radius
                / observations_dict["camera_observations"][valid_camera_mask, camera_index, 2:3]
            )

        valid_points_3d = calibrator_utils.transform_points(
            camera_transforms[camera_index], valid_depth * valid_points_2d_homo
        )
        points_3d[valid_camera_mask, camera_index] = valid_points_3d

    if compute_triangulations:
        valid_point_indices = numpy.where(
            numpy.sum(numpy.isfinite(observations_dict["camera_observations"][..., 2]), axis=1) >= 2
        )[0]
        for point_index in valid_point_indices:
            camera_mask = numpy.isfinite(points_3d[point_index, :, 2])
            src_points = camera_transforms[camera_mask, :3, 3]
            dst_points = points_3d[point_index, camera_mask]
            points_3d[point_index, camera_mask] = calibrator_utils.triangulate_via_projection(src_points, dst_points)
    return points_3d


def estimate_coarse_time_shift_and_transforms(calib_conf, observations_dict, points_3d, scale_ratio=0.0):
    """Provide initial estimation of robot-camera time shift via a correlation-line mechanism."""
    always_correct_time_diff = True  # For now, we always correct the sync issues

    camera_time_stamps = observations_dict["camera_time_stamps"]
    robot_points_3d = observations_dict["robot_observations"].reshape((-1, 3))
    robot_time_stamps = observations_dict["robot_time_stamps"]

    valid_camera_mask = numpy.sum(numpy.isfinite(observations_dict["camera_observations"][..., 2]), axis=1) >= 2
    valid_camera_time_stamps = camera_time_stamps[valid_camera_mask]
    valid_camera_points_3d = numpy.nanmean(points_3d[valid_camera_mask], axis=1)

    actual_robot_frequency = 1.0 / numpy.mean(numpy.diff(robot_time_stamps))
    log.info("Robot observations sampled at {:.1f} Hz".format(actual_robot_frequency))

    actual_camera_frequency = 1.0 / numpy.mean(numpy.diff(valid_camera_time_stamps))
    log.info("Overall camera observations sampled at {:.1f} Hz".format(actual_camera_frequency))

    if actual_robot_frequency < actual_camera_frequency:
        raise ValueError("Robot observations should be more frequent than the camera ones!")

    time_diff = observations_dict["robot_time_stamps"][0] - observations_dict["camera_time_stamps"][0]
    is_time_diff_large = numpy.abs(time_diff) > 0.1
    if is_time_diff_large:
        log.warning(
            "Time difference between robot and camera observations is {:.2f} msec. Are clocks synced?".format(
                1e3 * time_diff
            )
        )

    if is_time_diff_large or always_correct_time_diff:
        observations_dict["camera_time_stamps"] += time_diff
        time_diff = 0

    # Initial guess
    time_shift = time_diff

    # Estimate initial alignment transform and time shift
    for iter_index in range(2):
        matching_robot_points_3d = calibrator_utils.get_matching_points(
            robot_points_3d, robot_time_stamps - time_shift, camera_time_stamps
        )
        transform_wo = numpy.matmul(
            calib_conf["T_WO"],
            calibrator_utils.compute_weighted_affine_transformation(
                matching_robot_points_3d[valid_camera_mask],
                valid_camera_points_3d,
                scale_ratio=scale_ratio,
                show_debug=log.isEnabledFor(logging.DEBUG),
            ),
        )
        calib_conf["T_WO"] = transform_wo.tolist()
        points_3d = compute_points_3d(
            calib_conf,
            observations_dict,
            matching_robot_points_3d=matching_robot_points_3d,
        )
        valid_camera_points_3d = numpy.nanmean(points_3d[valid_camera_mask], axis=1)

        # Following computations are not needed unless there is another iteration
        if iter_index == 1:
            break

        robot_align_feature = matching_robot_points_3d[valid_camera_mask, 2]
        camera_align_feature = valid_camera_points_3d[:, 2]
        time_shift = time_diff + (
            calibrator_utils.find_delay(robot_align_feature, camera_align_feature) / actual_camera_frequency
        )

    log.info(
        "Initially estimated time shift: {:.2f} msec (cost: {:.2f} mm)".format(
            time_shift * 1e3,
            calibrator_utils.get_root_mean_square_error(points_3d, matching_robot_points_3d, scale_factor=1e3),
        )
    )
    log.info(
        "Initial triangulation disagreement SD: {:.2f} mm".format(
            1e3 * numpy.nanmean(numpy.linalg.norm(numpy.std(points_3d, axis=1), axis=1))
        )
    )

    return (
        time_shift,
        points_3d,
        valid_camera_time_stamps,
        valid_camera_points_3d,
    )


def estimate_fine_time_shift_and_transforms(initial_time_shift, calib_conf, observations_dict, scale_ratio=0.0):
    """Provide final estimation of robot-camera time shift via error minimization."""
    camera_time_stamps = observations_dict["camera_time_stamps"]
    robot_points_3d = observations_dict["robot_observations"].reshape((-1, 3))
    robot_time_stamps = observations_dict["robot_time_stamps"]

    time_shift = (
        scipy.optimize.minimize(
            functools.partial(
                optimize_time_shift_and_transforms,
                calib_conf,
                observations_dict,
                initial_time_shift,
                scale_ratio=scale_ratio,
            ),
            [0],
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "adaptive": True, "disp": False},
        ).x[0]
        + initial_time_shift
    )

    matching_robot_points_3d = calibrator_utils.get_matching_points(
        robot_points_3d, robot_time_stamps - time_shift, camera_time_stamps
    )
    points_3d = compute_points_3d(calib_conf, observations_dict, matching_robot_points_3d=matching_robot_points_3d)
    log.info(
        "Optimized time shift: {:.2f} msec (cost: {:.2f} mm)".format(
            time_shift * 1e3,
            calibrator_utils.get_root_mean_square_error(points_3d, matching_robot_points_3d, scale_factor=1e3),
        )
    )

    final_disagreement = numpy.nanmean(numpy.linalg.norm(numpy.std(points_3d, axis=1), axis=1))
    log.info("Final triangulation disagreement SD: {:.2f} mm".format(1e3 * final_disagreement))
    if final_disagreement > 1e-2:
        raise ValueError(
            "Something went wrong! Triangulation disagreement of %.3f is too large ..." % final_disagreement
        )
    return time_shift, points_3d


def set_results_frame(calib_conf, camera_names, in_worldframe=False):
    """Set camera transforms in world frame, or otherwise in relative frame to primary camera."""
    primary_camera_name = next(iter(calib_conf.keys()))
    if in_worldframe:
        for camera_name in camera_names:
            calib_conf[camera_name]["T_CW"] = numpy.matmul(calib_conf[camera_name]["T_CW"], calib_conf["T_WO"]).tolist()
        calib_conf["T_WO"] = numpy.eye(4).tolist()

        log.info("Camera transforms are now expressed in world frame. Save results to apply!")
    else:
        calib_conf["T_WO"] = numpy.matmul(calib_conf[primary_camera_name]["T_CW"], calib_conf["T_WO"]).tolist()
        correction_transform = numpy.linalg.inv(calib_conf[primary_camera_name]["T_CW"])
        for camera_name in camera_names:
            calib_conf[camera_name]["T_CW"] = numpy.matmul(
                calib_conf[camera_name]["T_CW"], correction_transform
            ).tolist()

        log.info("Camera transforms are now expressed relative to the primary camera. Save results to apply!")


def save_results(calib_path, initial_calib_conf, calib_conf, camera_names, robot_name, robot_only=False):
    """Store calibration parameters to the specified config file."""
    is_change_required = False
    update_msgs = []

    # Check for robot transform
    transform_diff = numpy.matmul(calib_conf["T_WO"], numpy.linalg.inv(initial_calib_conf["T_WO"]))
    if not numpy.allclose(transform_diff, numpy.eye(4), atol=1e-3):
        is_change_required = True
        scale_vect = numpy.linalg.norm(transform_diff[:3, :3], axis=0)
        rotation_matrix = numpy.matmul(numpy.diag(1.0 / scale_vect), transform_diff[:3, :3])
        translation_vect = transform_diff[:3, 3]

        update_msgs.append(
            " - {:s}: {:.1f}% scale, {:.1f} deg rotation, and {:.1f} mm translation".format(
                robot_name,
                1e2 * numpy.linalg.norm(scale_vect) / (3**0.5),
                numpy.rad2deg(calibrator_utils.get_angle_from_matrix(rotation_matrix)),
                1e3 * numpy.linalg.norm(translation_vect),
            )
        )

    # Check for camera transforms
    for camera_name in camera_names:
        if robot_only:  # If in robot-only mode, then keep original camera transforms
            calib_conf[camera_name]["T_CW"] = copy.deepcopy(initial_calib_conf[camera_name]["T_CW"])

        transform_diff = numpy.matmul(
            calib_conf[camera_name]["T_CW"], numpy.linalg.inv(initial_calib_conf[camera_name]["T_CW"])
        )
        if numpy.allclose(transform_diff, numpy.eye(4), atol=1e-3):
            continue

        is_change_required = True
        scale_vect = numpy.linalg.norm(transform_diff[:3, :3], axis=0)
        rotation_matrix = numpy.matmul(numpy.diag(1.0 / scale_vect), transform_diff[:3, :3])
        translation_vect = transform_diff[:3, 3]

        update_msgs.append(
            " - {:s}: {:.1f}% scale, {:.1f} deg rotation, and {:.1f} mm translation".format(
                camera_name,
                1e2 * numpy.linalg.norm(scale_vect) / (3**0.5),
                numpy.rad2deg(calibrator_utils.get_angle_from_matrix(rotation_matrix)),
                1e3 * numpy.linalg.norm(translation_vect),
            )
        )

    if len(update_msgs) > 0:
        log.info("The following updates are required: \n{:s}".format("\n".join(update_msgs)))
    else:
        log.info("The calibration transforms are already up to date!")

    if is_change_required:
        if click.confirm("\nDo you want to update the configuration file?", default=True):
            shutil.copy(
                calib_path, "{:s}.backup_{:s}".format(calib_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            )
            with open(calib_path, "w", encoding="UTF-8") as file_handler:
                yaml.dump(calib_conf, file_handler, default_flow_style=None, sort_keys=False)
            log.info("Updated the configuration file.")
        else:
            log.warning("Configuration file is not updated.")


def run_calibrator(
    config,
    use_observations_file=False,
    ball_radius=0.02,
    scale_ratio=0.0,
    in_worldframe=False,
    robot_only=False,
    save_graphics=False,
):
    """Run the multi-camera and robot calibration routine."""
    if numpy.clip(scale_ratio, 0.0, 1.0) != scale_ratio:
        raise ValueError("Expects scale ratio within the range of [0, 1], got scale_ratio={:f}!".format(scale_ratio))
    if robot_only and in_worldframe:
        log.warning("Should not set world-frame when updating robot only.")
        in_worldframe = False

    calib_share_path = get_package_share_directory("calibration")
    calib_path = f"{calib_share_path}/parameters/camera_calibration/{config}.yaml"

    with open(calib_path, "r", encoding="UTF-8") as file_handler:
        calib_conf = yaml.full_load(file_handler)
    calib_conf["T_WO"] = calib_conf.get("T_WO", numpy.eye(4).tolist())
    initial_calib_conf = copy.deepcopy(calib_conf)

    observations_dict = collect_or_load_observations(use_observations_file)

    camera_names = observations_dict["camera_names"]
    camera_time_stamps = observations_dict["camera_time_stamps"]
    robot_names = observations_dict["robot_names"]
    robot_points_3d = observations_dict["robot_observations"].reshape((-1, 3))
    robot_time_stamps = observations_dict["robot_time_stamps"]

    # If observations are very sparse, then warn the user that something went wrong
    sparsity_ratio = (
        numpy.count_nonzero(numpy.isnan(observations_dict["camera_observations"]))
        / observations_dict["camera_observations"].size
    )
    if sparsity_ratio > 0.5:
        log.warning(
            "Camera observations has %.1f%% sparsity! Are the cameras synchronized,"
            " is the frame rate very high, or do ball detection parameters need tuning?",
            sparsity_ratio * 1e2,
        )
    else:
        log.info("Camera observations has {:.1f}% sparsity".format(sparsity_ratio * 1e2))

    # Computes tringulated points for initial alignment
    points_3d = compute_points_3d(
        calib_conf,
        observations_dict,
        compute_triangulations=True,
        ball_radius=ball_radius,
    )

    if log.isEnabledFor(logging.INFO):
        viz_utils.initialize()

    # Generate 3D plot for initial points
    viz_utils.generate_3d_plot(
        numpy.hsplit(points_3d, numpy.arange(1, points_3d.shape[1])) + [robot_points_3d],
        poses=[
            numpy.linalg.inv(numpy.matmul(calib_conf[camera_name]["T_CW"], calib_conf["T_WO"]))
            for camera_name in camera_names
        ]
        + [numpy.eye(4)],
        labels=list(camera_names) + [robot_names[0]],
        save_as="data/tmp/initial_points_3d.svg" if save_graphics else None,
        show_fig=log.isEnabledFor(logging.INFO),
        title="Initial points",
    )

    # Generate 2D plot for initial points
    viz_utils.generate_signals_plot(
        [camera_time_stamps] * points_3d.shape[1] + [robot_time_stamps],
        numpy.hsplit(points_3d, numpy.arange(1, points_3d.shape[1])) + [robot_points_3d],
        labels=list(camera_names) + [robot_names[0]],
        save_as="data/tmp/initial_points_signal.svg" if save_graphics else None,
        show_fig=log.isEnabledFor(logging.DEBUG),
        title="Initial points",
    )

    # Compute the initial time-shift
    (
        time_shift,
        points_3d,
        valid_camera_time_stamps,
        valid_camera_points_3d,
    ) = estimate_coarse_time_shift_and_transforms(calib_conf, observations_dict, points_3d, scale_ratio=scale_ratio)

    # Generate 2D plot for initial-alignment (cameras average)
    viz_utils.generate_signals_plot(
        [valid_camera_time_stamps] + [robot_time_stamps - time_shift],
        [valid_camera_points_3d] + [robot_points_3d],
        labels=["Cameras (avg)"] + [robot_names[0]],
        save_as="data/tmp/time_shift_signal.svg" if save_graphics else None,
        show_fig=log.isEnabledFor(logging.DEBUG),
        title="Robot-Camera time-shift alignment",
    )

    # Generate 2D plot for initial-alignment (individual cameras)
    viz_utils.generate_signals_plot(
        [camera_time_stamps] * points_3d.shape[1] + [robot_time_stamps - time_shift],
        numpy.hsplit(points_3d, numpy.arange(1, points_3d.shape[1])) + [robot_points_3d],
        labels=list(camera_names) + [robot_names[0]],
        save_as="data/tmp/updated_points_signal.svg" if save_graphics else None,
        show_fig=log.isEnabledFor(logging.DEBUG),
        title="Updated points (time shift: {:.2f} msec)".format(1e3 * time_shift),
    )

    del valid_camera_time_stamps, valid_camera_points_3d

    # Compute the final time-shift
    (
        time_shift,
        points_3d,
    ) = estimate_fine_time_shift_and_transforms(time_shift, calib_conf, observations_dict, scale_ratio=scale_ratio)

    # Show final visualization with computeTriangulations option
    points_3d = compute_points_3d(
        calib_conf,
        observations_dict,
        compute_triangulations=True,
        ball_radius=ball_radius,
    )

    # Generate 2D plot for final-alignment (individual cameras)
    viz_utils.generate_signals_plot(
        [camera_time_stamps] * points_3d.shape[1] + [robot_time_stamps - time_shift],
        numpy.hsplit(points_3d, numpy.arange(1, points_3d.shape[1])) + [robot_points_3d],
        labels=list(camera_names) + [robot_names[0]],
        save_as="data/tmp/optimized_points_signal.svg" if save_graphics else None,
        show_fig=log.isEnabledFor(logging.DEBUG),
        title="Optimized points (time shift: {:.2f} msec)".format(1e3 * time_shift),
    )

    # Generate 3D plot for final points
    viz_utils.generate_3d_plot(
        numpy.hsplit(points_3d, numpy.arange(1, points_3d.shape[1])) + [robot_points_3d],
        poses=[
            numpy.linalg.inv(numpy.matmul(calib_conf[camera_name]["T_CW"], calib_conf["T_WO"]))
            for camera_name in camera_names
        ]
        + [numpy.eye(4)],
        labels=list(camera_names) + [robot_names[0]],
        save_as="data/tmp/optimized_points_3d.svg" if save_graphics else None,
        show_fig=log.isEnabledFor(logging.INFO),
        title="Optimized points",
    )

    # Set results frame
    set_results_frame(calib_conf, camera_names, in_worldframe=in_worldframe)

    # Upon user acceptance of the new results, save the calibration results
    save_results(
        calib_path,
        initial_calib_conf,
        calib_conf,
        camera_names,
        robot_names[0],
        robot_only=robot_only,
    )

    # Keep plots running until the user ends the execution
    if log.isEnabledFor(logging.INFO):
        viz_utils.wait_for_user_input()


if __name__ == "__main__":
    args = parse_args()
    configure_logger(args.log_level)

    run_calibrator(
        args.config,
        use_observations_file=args.use_observations_file,
        ball_radius=args.ball_radius,
        scale_ratio=args.scale_ratio,
        in_worldframe=args.in_worldframe,
        robot_only=args.robot_only,
        save_graphics=args.save_graphics,
    )
