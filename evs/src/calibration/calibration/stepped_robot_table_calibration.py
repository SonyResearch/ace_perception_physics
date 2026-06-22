#!/usr/bin/env python3
# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""
Stepped robot table calibration.
1. Move the robot to a desired location, wait, record from robot and triangulation topic.
2. Repeat n >=3 times
3. Find transformation between the two sets of coordinates for the same ball location.
4. Move robot joint up and down
5. Fit sinusoid to z coordinate for both robot and triangulated values (helps with de-noising)
6. Get the phase difference between these two sine waves to calc. the delay

Note: The robot publishes the position of the ball after accounting for the calibration tool.
To run
python3 src/sensors/calibration/calibration/stepped_robot_table_calibration.py

"""

import ctypes
import multiprocessing
import os
import pathlib
import time
from yaml import dump
import ace_yaml as yaml
import arena_native_ros.python_module as arena_native_ros

import click
import colored_glog as log
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bullet_simulation.automatic_collision_detector import DetectorType
from evs.src.calibration.calibration.calibrator_utils import compute_affine_transformation
from closed_loop_execution.closed_loop_wrapper import (
    BallState,
    ClosedLoopWrapper,
    CollisionDetectorParameters,
    GeneralParameters,
    PerceptionParameters,
    RobotParameters,
    Sensor,
)
from sklearn.cluster import KMeans
import arena_native.python_module as arena_native


def load_yaml() -> dict:
    """Load the yaml file with the test settings

    Returns:
        dict: a dictionary with the test parameters
    """
    if "ACE_LAB" not in os.environ:
        raise ValueError("Environment varibale 'ACE_LAB' is missing.")
    filename = "robot_config/%s.yaml" % os.environ["ACE_LAB"]
    config = pathlib.Path(os.path.realpath(__file__)).parents[0] / filename
    with open(config, "r", encoding="utf8") as tuning_params_file:
        params = yaml.safe_load(tuning_params_file)
    return params


PARAMS = load_yaml()
N_JOINTS = PARAMS["number_of_joints"]
HOME_JOINT_POSE = PARAMS["home"]
ROBOT_MODEL = PARAMS["model"]
ROBOT_NAME = PARAMS["robot_name"]
TARGETS = PARAMS["targets"]
CONTROL_FREQ = PARAMS["control_frequency"]

SIMULATION_FLAG = False


# Example function to calculate the mean of close values
def calculate_mean_of_close_values(data_points: np.ndarray):
    """Calculating the mean values of the close cluster of the points.

    Args:
        data_points (Numpy array N*3): The input data points

    Returns:
        Numpy Array: the mean of the clusters
    """
    # Create a KMeans instance with 6 clusters
    kmeans = KMeans(n_clusters=np.array(TARGETS).shape[0])

    # Fit the KMeans model to your data
    kmeans.fit(data_points)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    # Define the maximum distance from the cluster center
    max_distance = 0.01

    # Initialize a list to store the filtered cluster points
    filtered_cluster_points = []

    # Iterate over each cluster center
    for center in cluster_centers:
        # Calculate the Euclidean distances from each point to the cluster center
        distances = np.linalg.norm(data_points - center, axis=1)
        # Filter out points that are within the maximum distance
        filtered_points = data_points[distances <= max_distance]
        # Append filtered points to the list
        filtered_cluster_points.append(filtered_points)

    cluster_means = np.array([np.mean(cluster_points, axis=0) for cluster_points in filtered_cluster_points])
    return cluster_means[np.argsort(cluster_means[:, 2])]


class RobotTableCalibration(ClosedLoopWrapper):
    """Calibration of the robot base position with respect to the table position."""

    def __init__(self):
        general_config = GeneralParameters(
            game_mode=False,
            simulation=SIMULATION_FLAG,
            log_level="info",
            plot=False,
        )
        robot_config = RobotParameters(
            robot_name=ROBOT_MODEL,
            control_freq=CONTROL_FREQ,
            cpu_affinity_robot_process={5},
        )
        collision_config = CollisionDetectorParameters(
            buffer_size=int(2 * 1000 / robot_config.control_freq),
            step_size=1,
            safety_distance=0.05,
            detector_type=DetectorType.SAFEIK,
            cpu_affinity={1, 2, 3, 4},
        )
        perception_config = PerceptionParameters(
            aps_freq=200,
            param_file=os.path.join(
                get_package_share_directory("trajectory_estimation"), "config", "parameters_mpc_tyo01.yaml"
            ),
            ekf_hor=1.0,
            ekf_dt=1 / 200,
            use_msg_timestamp=True,
        )

        super().__init__(
            general_config=general_config,
            collision_config=collision_config,
            robot_config=robot_config,
            perception_config=perception_config,
        )

        self._time_initial: float = time.time()
        self._full_data_set: list = []

        self._received_robot_states = []  # x, y, z,
        self._received_triang_states = []  # x, y, z,
        self._reference_trajectory = []

        self._received_robot_states_selected = []  # x, y, z

        self._flag_is_safe_to_dump_data = multiprocessing.Value(ctypes.c_bool)
        self._flag_is_safe_to_dump_data.value = False
        self._flag_is_safe_to_read_data = multiprocessing.Value(ctypes.c_bool)
        self._flag_is_safe_to_read_data.value = False

    # pylint: disable=unused-argument, no-self-use
    def _ball_state_callback(self, timestamp: float, ball: BallState, **kwargs) -> bool:
        """This is to prevent calling the EKF as we don't need it fot this calibration"""
        return False

    def _current_joint_callback(self, current_pos: np.ndarray, current_vel: np.ndarray, **kwargs) -> bool:
        # Construct the dictionary directly
        end_pos, _ = self._kinematics.calculate_end_effector_from_joints(joint_pos=current_pos, joint_vel=current_vel)
        self._received_robot_states.append(end_pos[0:3])

        if min(np.linalg.norm(np.array(TARGETS) - current_pos, axis=1)) < 0.01:
            self._received_robot_states_selected.append(end_pos[0:3])

        return super()._current_joint_callback(current_pos=current_pos, current_vel=current_vel, **kwargs)

    def _measurement_callback(self, meas: arena_native.sensor_measurement):
        """
        Perception callback
        @param msg: ball pose from the perception system
        """
        if self._flag_is_safe_to_dump_data.value and not self._flag_is_safe_to_read_data.value:
            self._add_perception_data_to_shared_mem()
            self._flag_is_safe_to_read_data.value = True
        super()._measurement_callback(meas=meas)

    def _compute_ruckig_trajectory(self, target_pos: np.ndarray, current_pos: np.ndarray):
        traj_p = []
        traj_v = []
        traj_a = []
        traj_j = []

        target_joint_pos: np.ndarray
        target_joint_pos = target_pos
        assert (
            max(target_joint_pos.shape) == self._kinematics.n_joints
        ), f"Size of target {target_joint_pos} does not match the expected number of joints {self._kinematics.n_joints}"

        # Each joint motion is created independently
        for joint_i in range(len(target_joint_pos)):
            p_go, v_go, a_go, j_go = self._ruckig_generator.make_point_to_point_trajectory(
                position_current=current_pos[joint_i, np.newaxis],
                position_target=np.array(target_joint_pos)[joint_i, np.newaxis],
                velocity_current=np.zeros_like(target_joint_pos)[joint_i, np.newaxis],
                velocity_target=np.zeros_like(target_joint_pos)[joint_i, np.newaxis],
                acceleration_current=np.zeros_like(target_joint_pos)[joint_i, np.newaxis],
                acceleration_target=np.zeros_like(target_joint_pos)[joint_i, np.newaxis],
                gain_vel=np.ones(self._kinematics.n_joints) * self._ruckig_v_max_ratio,
                gain_acc=np.ones(self._kinematics.n_joints) * self._ruckig_a_max_ratio,
                gain_jerk=np.ones(self._kinematics.n_joints) * self._ruckig_j_max_ratio,
            )

            traj_p += [p_go]
            traj_v += [v_go]
            traj_a += [a_go]
            traj_j += [j_go]

        return traj_p, traj_v, traj_a, traj_j

    def _compute_joint_trajectory(self, **kwargs):
        """
        Use ruckig to move the robot from the current_joint_pos to the target_joint_pos
        """

        full_traj_p, full_traj_v, full_traj_a, full_traj_j = self._compute_ruckig_trajectory(
            target_pos=np.array(HOME_JOINT_POSE), current_pos=self._last_u
        )

        for target_pose in np.array(TARGETS):
            current_pos = full_traj_p[-1]

            full_traj_p_2, full_traj_v_2, full_traj_a_2, full_traj_j_2 = self._compute_ruckig_trajectory(
                target_pos=target_pose, current_pos=current_pos
            )
            full_traj_p_2, full_traj_v_2, full_traj_a_2, full_traj_j_2 = [
                np.concatenate((full_traj, np.tile(full_traj[-1], (1000, 1))), axis=0)
                for full_traj in [full_traj_p_2, full_traj_v_2, full_traj_a_2, full_traj_j_2]
            ]
            full_traj_p = np.vstack((full_traj_p, full_traj_p_2))
            full_traj_v = np.vstack((full_traj_v, full_traj_v_2))
            full_traj_a = np.vstack((full_traj_a, full_traj_a_2))
            full_traj_j = np.vstack((full_traj_j, full_traj_j_2))

        current_pos = full_traj_p[-1]
        full_traj_p_2, full_traj_v_2, full_traj_a_2, full_traj_j_2 = self._compute_ruckig_trajectory(
            target_pos=np.array(HOME_JOINT_POSE), current_pos=current_pos
        )
        full_traj_p = np.vstack((full_traj_p, full_traj_p_2))
        full_traj_v = np.vstack((full_traj_v, full_traj_v_2))
        full_traj_a = np.vstack((full_traj_a, full_traj_a_2))
        full_traj_j = np.vstack((full_traj_j, full_traj_j_2))
        # pylint: disable=attribute-defined-outside-init
        self.command_trajectory = (full_traj_p, full_traj_v, full_traj_a, full_traj_j)
        self._reference_trajectory = (
            full_traj_p.copy(),
            full_traj_v.copy(),
            full_traj_a.copy(),
        )

    def _end_of_trajectory_callback(self, current_pos: np.ndarray, current_vel: np.ndarray, **kwargs):
        """Calculate the transformation matrix wrt to the values of the robot and the perception

        Args:
            current_pos (np.ndarray): we ignore it!
            current_vel (np.ndarray): we ignore it!
        """
        log.setLevel("ERROR")
        # Find the transformations
        self._flag_is_safe_to_dump_data.value = True
        robot_points = calculate_mean_of_close_values(np.array(self._received_robot_states_selected))
        print(f"robot_points:\n {robot_points}")
        # Recover full triangulations
        if self.using_pybind_interface():
            while not self._flag_is_safe_to_read_data.value:
                time.sleep(0.01)
                print("Waiting for the data to be saved!")
        else:
            self._add_perception_data_to_shared_mem()

        self._recover_perception_data_from_shared_mem()
        if len(self._p_data.sensor_data) != 0:
            self._received_triang_states = self._p_data.sensor_data[Sensor.APS.value].position

            triang_points = calculate_mean_of_close_values(np.array(self._received_triang_states))
            print(f"triang_points:\n {triang_points}")

            transform_robot_wrt_table = compute_affine_transformation(robot_points, triang_points)
            transform_table_wrt_robot = compute_affine_transformation(triang_points, robot_points)
            print(f"Transformation robot wrt table:\n {transform_robot_wrt_table}")
            print(f"Transformation table wrt robot:\n {transform_table_wrt_robot}")
            rot_mat = transform_robot_wrt_table[:3, :3]
            translation = transform_robot_wrt_table[:3, -1]
            transformed_points = np.dot(rot_mat, robot_points.T).T + translation
            errors = np.linalg.norm(transformed_points - triang_points, axis=1)
            print(f"errors:\n {errors}")
            print(f"max errors:\n {max(errors)}")

            assert np.mean(errors) < 0.01, f"Errors {errors} are too large! Failed to perform Robot-Cam calibration."

            print(f"Mean Error (m): {np.mean(errors)} | Errors (m): {errors}")

            # Print, create yaml, confirm.
            print("Saving transformation matrices and delay")
            filename = os.environ["ACE_LAB"] + ".yaml"
            filepath = os.path.join(get_package_share_directory("calibration"), "parameters", "robot", filename)

            with open(filepath, "w") as outfile:  # pylint: disable=unspecified-encoding)
                dump(
                    {"T_OR": transform_table_wrt_robot.tolist(), "T_RO": transform_robot_wrt_table.tolist()},
                    outfile,
                    default_flow_style=None,
                )
        else:
            log.error("[Error] There is no ball position logged!")
        return super()._end_of_trajectory_callback(current_pos=current_pos, current_vel=current_vel, **kwargs)


# pylint: disable=too-many-locals, invalid-name, too-many-statements, too-many-branches
def main():
    """Main."""
    rclpy.init()
    robot_node = RobotTableCalibration()
    if not robot_node.simulation:
        assert click.confirm(
            "Starting stepped robot table calibration."
            + "Is the calibration tool attached to the robot? Is the FK updated with its length?",
            default=False,
        )
        assert click.confirm(
            "Starting stepped robot table calibration." + "Confirm you are using " + ROBOT_MODEL + "?",
            default=False,
        )
        assert click.confirm(
            "Please run the triangulation on the other PC, by running:\n"
            + "ros2 launch ace_launch perception_spin.launch.py config:=zrh02 log_level:=error "
        )

    robot_node.set_ruckig_aggressiveness(0.3, 0.3, 0.3)
    if robot_node.using_pybind_interface():
        robot_node.start_pybind_interface()
    try:
        while arena_native_ros.ok():
            # tic = time.perf_counter()
            robot_node.update()
            rclpy.spin_once(robot_node, timeout_sec=0.001)
            # elapsed = time.perf_counter() - tic
            # if elapsed < 0.001:
            #     time.sleep(0.001 - elapsed)

    except:  # pylint: disable= bare-except
        robot_node.stop_robot_interface()
    robot_node.join_pybind_interface_process()


if __name__ == "__main__":
    main()
