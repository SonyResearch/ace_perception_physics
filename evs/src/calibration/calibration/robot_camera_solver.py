#!/usr/bin/env python3
"""
Solver for camera-robot calibration.

Works by aligning prerecorded trajectories of the ball from both the robot and the cameras.
"""
# pylint: disable = line-too-long, redefined-outer-name, invalid-name
# Confidential, Copyright 2024, Sony AI, All rights reserved.

import ace_yaml as yaml
import evs.src.calibration.calibration.calibrator_utils as calibrator_utils
import click
import matplotlib.pyplot as plt
import numpy as np
from ament_index_python.packages import get_package_share_directory
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable = unused-import
from scipy.optimize import optimize
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def sum_total_cost(x, triangulation_data, robot_state_data, optimize_endeff_offset):
    """
    Calculate the alignment cost.

    Cost is calculated as the sum of squared errors of between the observed position from
    the robot and the camera for a given transformation and time delay.
    """
    R = Rot.from_rotvec(x[0:3]).as_matrix()
    t = x[3:6]

    cost = 0.0
    for i in range(triangulation_data.shape[0]):
        valid, robot_pos, robot_rot = get_robot_state(triangulation_data[i, 0] + x[6], robot_state_data)
        if not valid:
            continue

        triangulation_pos = np.matmul(triangulation_data[i, 1:4], R.T) + t
        if optimize_endeff_offset:
            triangulation_pos += robot_rot.as_matrix().dot(x[7:10])
        err = robot_pos - triangulation_pos
        cost += np.linalg.norm(err) ** 2

    return cost


def get_robot_state(t, robot_state_data):
    """Interpolate the position of the ball from the robot's perspective at a given time."""
    if t > robot_state_data[-1, 0]:
        return False, None, None

    idx_high = np.where(t <= robot_state_data[:, 0])[0][0]
    idx_low = idx_high - 1

    if idx_low < 0:
        return False, None, None

    alpha = (t - robot_state_data[idx_low, 0]) / (robot_state_data[idx_high, 0] - robot_state_data[idx_low, 0])
    if alpha < 0.0 or alpha > 1.0:
        print("Something went wrong at time", t)

    # interpolate position
    pos = robot_state_data[idx_low, 2:5] + alpha * (robot_state_data[idx_high, 2:5] - robot_state_data[idx_low, 2:5])

    # interpolate attitude
    rots = Rot.from_quat(robot_state_data[idx_low : idx_high + 1, 5:9])
    slerp = Slerp([0.0, 1.0], rots)
    rot = slerp(alpha)

    return True, pos, rot


iter_counter = 0


def cb(x):
    """Update iteration counter. Designed to be a callback function."""
    global iter_counter  # pylint: disable = global-statement
    iter_counter += 1
    print(str(iter_counter) + ":", x)


def load_data(calib_data_rob_csv_path, calib_data_ori_csv_path):
    """Load the pre-recorded data in csv format from both robot and camera."""
    robot_state_data = np.genfromtxt(calib_data_rob_csv_path, delimiter=",", skip_header=1)
    triangulation_data = np.genfromtxt(calib_data_ori_csv_path, delimiter=",", skip_header=1)

    # reset time
    t0 = min([triangulation_data[0, 0], robot_state_data[0, 0]])
    triangulation_data[:, 0] = (triangulation_data[:, 0] - t0) + triangulation_data[:, 1] * 1e-9
    robot_state_data[:, 0] = (robot_state_data[:, 0] - t0) + robot_state_data[:, 1] * 1e-9

    triangulation_data = triangulation_data[:, [0, 2, 3, 4]]
    return robot_state_data, triangulation_data


def get_initial_guess(robot_state_data, triangulation_data, optimize_endeff_offset):
    """Get initial guess of the parameters. Helps tremendously with computation time."""
    # x0 will be mostly overwritten below. Try adjusting the time delay to the expected delay
    x0 = [
        # Rot vector [0:3]
        0,
        0,
        0,
        # Translation vector [3:6]
        0,
        0,
        0,
        # Time delay [6]
        0.2,
    ]
    if optimize_endeff_offset:
        # Extra offset to compensate for the inaccurate length estimate of the calib. tool [6:9]
        x0.extend([0, 0, 0])
    x0 = np.array(x0)

    # Refine initial guess by finding the transformation assuming delay=0
    # sample n=10 points at random times
    samples_indices = np.random.choice(np.arange(triangulation_data.shape[0]), 10, replace=False)
    triangulation_samples = triangulation_data[samples_indices]
    triangulation_t = triangulation_samples[:, 0]
    triangulation_xyz = triangulation_samples[:, 1:]
    robot_state = [get_robot_state(t, robot_state_data) for t in triangulation_t]
    triangulation_xyz, robot_xyz = map(
        np.array, zip(*[(tri_pt, r_state[1]) for tri_pt, r_state in zip(triangulation_xyz, robot_state) if r_state[0]])
    )
    # Calc transform matrix between those n points
    transform_mat = calibrator_utils.compute_affine_transformation(triangulation_xyz, robot_xyz)

    x0[0:3] = Rot.from_matrix(transform_mat[:3, :3]).as_rotvec()  # pylint: disable = invalid-sequence-index
    x0[3:6] = transform_mat[:3, -1]  # pylint: disable = invalid-sequence-index
    return x0


@click.command()
@click.option("--robot_state_file", default="data/tmp/calib_data_robot.csv", help="File to load the saved robot states")
@click.option(
    "--triangulation_file",
    default="data/tmp/calib_data_triangulations.csv",
    help="File to load the saved triangulations",
)
@click.option(
    "--config",
    required=True,
    help="Lab specific configuration, e.g. zrh00, zrh01, tyo02.",
)
@click.option("--maxiter", default=1000, help="Max number of iterations for the solver")
@click.option("--downsample", default=50, help="Downsample the triangulations to improve convergence speed")
@click.option(
    "--optimize_endeff_offset",
    is_flag=True,
    show_default=True,
    default=False,
    help="If enabled, the script will try to compensate for any offsets not specified in kinematics",
)
@click.option(
    "--update_camera",
    is_flag=True,
    show_default=True,
    default=False,
    help="If enabled, the script will update the camera calibration instead of the robot calibration",
)
def main(robot_state_file, triangulation_file, config, maxiter, downsample, optimize_endeff_offset, update_camera):
    """Runs solver for camera-robot calibration."""
    # Solve here
    # 1. Load data
    robot_state_data, triangulation_data = load_data(robot_state_file, triangulation_file)

    # Initial guess
    x0 = get_initial_guess(robot_state_data, triangulation_data, optimize_endeff_offset)

    # 2. Run optimization
    triangulation_data_down = triangulation_data[::downsample, :]  # downsample for speed
    x = optimize.fmin(
        sum_total_cost,
        x0,
        args=(triangulation_data_down, robot_state_data, optimize_endeff_offset),
        maxiter=maxiter,
        disp=True,
        callback=cb,
    )

    cost = sum_total_cost(x, triangulation_data, robot_state_data, optimize_endeff_offset)
    rme = np.sqrt(cost / triangulation_data.shape[0])
    print(f"Average alignment error: {rme * 1e3:5.1f}mm")

    # Finish
    # Get transform from Origin (table center) to Robot, T_RO
    T_RO = np.eye(4)
    T_RO[:3, :3] = Rot.from_rotvec(x[0:3]).as_matrix()
    T_RO[:3, 3] = x[3:6]

    # Get common params path
    calib_share_path = get_package_share_directory("calibration")

    # Load previous calib params
    robot_calib_yaml_path = f"{calib_share_path}/parameters/robot/{config}.yaml"
    camera_calib_yaml_path = f"{calib_share_path}/parameters/camera_calibration/{config}.yaml"
    with open(robot_calib_yaml_path, "r", encoding="utf-8") as file:
        robot_calib_params = yaml.load(file, Loader=yaml.FullLoader)
    with open(camera_calib_yaml_path, "r", encoding="utf-8") as file:
        camera_calib_params = yaml.load(file, Loader=yaml.FullLoader)

    # Save the results
    if update_camera:
        camera_calib_params["T_WO"] = np.dot(camera_calib_params["T_WO"], np.linalg.inv(T_RO)).tolist()
        with open(camera_calib_yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(camera_calib_params, file, default_flow_style=None, sort_keys=False)
    else:
        robot_calib_params["T_RO"] = T_RO.tolist()
        with open(robot_calib_yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(robot_calib_params, file, default_flow_style=None, sort_keys=False)

    # Visualize results
    R = Rot.from_rotvec(x[0:3]).as_matrix()
    t = x[3:6]
    for i in range(triangulation_data.shape[0]):
        valid, _, robot_rot = get_robot_state(triangulation_data[i, 0] + x[6], robot_state_data)
        if valid:
            triangulation_data[i, 1:4] = R.dot(triangulation_data[i, 1:4]) + t
            if optimize_endeff_offset:
                triangulation_data[i, 1:4] += robot_rot.as_matrix().dot(x[7:10])

    # 3D plot
    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=30, azim=55)

    plt.plot(robot_state_data[:, 2], robot_state_data[:, 3], robot_state_data[:, 4], label="robot")
    plt.plot(triangulation_data[:, 1], triangulation_data[:, 2], triangulation_data[:, 3], label="vision")
    plt.legend()
    fig.savefig("data/tmp/robot_calib.svg", dpi=fig.dpi)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
