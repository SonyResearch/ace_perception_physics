#!/usr/bin/env python3
"""
Recorder for camera-robot calibration.

Listens to the position of the ball as published by the robot and camera system. Saves to csv.
Note: stop the recorder before stopping any of the two publishers
"""
# pylint: disable = line-too-long, import-error, no-name-in-module
# SPDX-License-Identifier: MIT

import re
import time
import csv

import click
import rclpy
from rclpy.node import Node

from ace_interfaces.msg import PointsWithCovariance, RobotState


class RobotCameraCalibrationSubscriber(Node):
    """Subscriber ROS2 node."""

    ROBOT_TOPIC_TYPE = "ace_interfaces/msg/RobotState"
    ROBOT_TOPIC_NAME_REGEX = "/real_robot/.*/robot_state"

    def __init__(self, robot_state_topic_name, triangulation_topic_name, robot_save_file, triangulation_save_file):
        """Init."""
        super().__init__("vision_to_robot_calibration_sub")

        if robot_state_topic_name is None:
            total_trials = 10

            for trial_index in range(total_trials):
                robot_state_topic_names = [
                    topic_info[0]
                    for topic_info in self.get_topic_names_and_types()
                    if re.search(self.ROBOT_TOPIC_NAME_REGEX, topic_info[0]) and self.ROBOT_TOPIC_TYPE in topic_info[1]
                ]
                if len(robot_state_topic_names) == 0:
                    print(
                        "Trial {:d}/{:d}: Finding robots...".format(trial_index + 1, total_trials),
                        end="\r",
                    )
                    time.sleep(0.2)
                    continue

            if len(robot_state_topic_names) != 1:
                raise ValueError("Expects one robot, got {:d}!".format(len(robot_state_topic_names)))
            robot_state_topic_name = robot_state_topic_names[0]

        # pylint: disable = consider-using-with, unspecified-encoding
        self.calib_data_ori_csv = csv.DictWriter(
            open(triangulation_save_file, "w"),
            fieldnames=["sec", "nsec", "x", "y", "z"],
        )
        self.calib_data_ori_csv.writeheader()
        # pylint: disable = consider-using-with, unspecified-encoding
        self.calib_data_rob_csv = csv.DictWriter(
            open(robot_save_file, "w"),
            fieldnames=["sec", "nsec", "x", "y", "z", "qw", "qx", "qy", "qz"],
        )
        self.calib_data_rob_csv.writeheader()

        # Subscribers
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.create_subscription(
            PointsWithCovariance, triangulation_topic_name, self.ori_data_callback, qos_profile=qos_profile
        )
        self.create_subscription(
            RobotState,
            robot_state_topic_name,
            self.rob_data_callback,
            qos_profile=qos_profile,
        )

    def ori_data_callback(self, msg):
        """Save data from the camera system. Designed to be used as a callback function."""
        timestamp = self.get_clock().now()
        sec, nsec = timestamp.seconds_nanoseconds()
        if msg.points:
            point = msg.points[0].position
            row = {
                "sec": sec,
                "nsec": nsec,
                "x": point.x,
                "y": point.y,
                "z": point.z,
            }
            self.calib_data_ori_csv.writerow(row)

    def rob_data_callback(self, msg):
        """Save data from the robot. Designed to be used as a callback function."""
        timestamp = self.get_clock().now()
        sec, nsec = timestamp.seconds_nanoseconds()
        row = {
            "sec": sec,
            "nsec": nsec,
            "x": msg.end_effector_pose.position.x,
            "y": msg.end_effector_pose.position.y,
            "z": msg.end_effector_pose.position.z,
            "qw": msg.end_effector_pose.orientation.w,
            "qx": msg.end_effector_pose.orientation.x,
            "qy": msg.end_effector_pose.orientation.y,
            "qz": msg.end_effector_pose.orientation.z,
        }
        self.calib_data_rob_csv.writerow(row)


@click.command()
@click.option(
    "--robot_state_topic",
    default=None,
    help="Topic name for the robot states",
)
@click.option(
    "--triangulation_topic", default="/sensors/ball_triangulation/points", help="Topic name for the triangulations"
)
@click.option(
    "--triangulation_file", default="data/tmp/calib_data_triangulations.csv", help="File name for the triangulations"
)
@click.option("--robot_state_file", default="data/tmp/calib_data_robot.csv", help="File name for the robot states")
def main(robot_state_topic, triangulation_topic, triangulation_file, robot_state_file):
    """Spins the ROS2 node."""
    rclpy.init()
    robot_camera_calibration_sub = RobotCameraCalibrationSubscriber(
        robot_state_topic, triangulation_topic, robot_state_file, triangulation_file
    )
    rclpy.spin(robot_camera_calibration_sub)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
