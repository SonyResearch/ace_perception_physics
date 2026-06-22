# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""Provide observations collection utilities for multi-camera and robot."""

import functools
import logging
import re
import time

import numpy
import rclpy
import rclpy.node

import ace_interfaces.msg

log = logging.getLogger(__name__)


class ObservationsRecorder(rclpy.node.Node):
    """Listen to and record ROS topics of the robot and each camera."""

    valid_observations_count = None
    total_observations_count = None

    robot_is_reporting = None
    robot_last_observation_index = None
    robot_observations = None
    robot_time_stamps = None
    robot_names = None

    camera_is_reporting = None
    camera_start_offset = None
    camera_last_observation_index = None
    camera_observations = None
    camera_sequence_numbers = None
    camera_names = None

    camera_first_time_stamp = None
    camera_last_time_stamp = None

    BUFFER_SIZE = 1024

    ROBOT_TOPIC_TYPE = "ace_interfaces/msg/RobotState"
    ROBOT_TOPIC_NAME_REGEX = "/real_robot/.*/robot_state"
    CAMERA_TOPIC_TYPE = "ace_interfaces/msg/BallsWithAttributes"

    def __init__(self):
        """Prepare for multi-camera and robot observation collection."""
        super().__init__("observations_recorder")

        self.subscription = {}
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        total_trials = 10

        for trial_index in numpy.arange(total_trials):
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
        self.robot_names = [topic_name.split("/")[-2] for topic_name in robot_state_topic_names]
        log.info(
            "Detected {:d} robots:\n{:s}".format(
                len(self.robot_names),
                "\n".join(["  - {:s}".format(robot_name) for robot_name in self.robot_names]),
            )
        )

        self.robot_is_reporting = False
        self.robot_last_observation_index = -1
        self.robot_observations = numpy.empty((0, len(robot_state_topic_names), 3), numpy.float64) * numpy.nan
        self.robot_time_stamps = numpy.empty(0, numpy.float64)

        for robot_index, topic_name in enumerate(robot_state_topic_names):
            self.subscription[topic_name] = self.create_subscription(
                ace_interfaces.msg.RobotState,
                topic_name,
                functools.partial(self._robot_message_callback, robot_index),
                qos_profile=qos_profile,
            )

        for trial_index in numpy.arange(total_trials):
            ball_detection_topic_names = [
                topic_info[0]
                for topic_info in self.get_topic_names_and_types()
                if self.CAMERA_TOPIC_TYPE in topic_info[1]
            ]
            if len(ball_detection_topic_names) == 0:
                print(
                    "Trial {:d}/{:d}: Finding cameras...".format(trial_index + 1, total_trials),
                    end="\r",
                )
                time.sleep(0.2)
                continue

        if len(ball_detection_topic_names) < 2:
            raise ValueError("No enough cameras to calibrate!")

        self.camera_names = [topic_name.split("/")[-2] for topic_name in ball_detection_topic_names]
        log.info(
            "Detected {:d} cameras:\n{:s}".format(
                len(self.camera_names),
                "\n".join(["  - {:s}".format(camera_name) for camera_name in self.camera_names]),
            )
        )

        self.camera_is_reporting = False
        self.camera_last_observation_index = -1
        self.valid_observations_count = numpy.zeros(len(self.camera_names) + len(self.robot_names), numpy.int32)
        self.total_observations_count = numpy.zeros(len(self.camera_names) + len(self.robot_names), numpy.int32)
        self.camera_observations = numpy.empty((0, len(self.camera_names), 3), numpy.float64) * numpy.nan
        self.camera_sequence_numbers = numpy.empty(0, numpy.int64)

        self.camera_first_time_stamp = None
        self.camera_last_time_stamp = None

        log.info("Press Ctrl+c to stop collecting observations...\n")

        print(
            "{:20s}{:s}".format(
                "Device:",
                "".join(["{:^15s} ".format(device_name) for device_name in self.robot_names + self.camera_names]),
            )
        )
        print("Waiting for observations stream...", end="\r")

        for camera_index, topic_name in enumerate(ball_detection_topic_names):
            self.subscription[topic_name] = self.create_subscription(
                ace_interfaces.msg.BallsWithAttributes,
                topic_name,
                functools.partial(self._camera_message_callback, camera_index),
                qos_profile=qos_profile,
            )

    def _robot_message_callback(self, robot_index, incoming_msg):
        self.robot_is_reporting = True
        self.total_observations_count[robot_index] += 1
        if not self.camera_is_reporting:
            return

        self.robot_last_observation_index += 1
        while self.robot_last_observation_index >= self.robot_time_stamps.shape[0]:
            self.robot_observations = numpy.vstack(
                [
                    self.robot_observations,
                    numpy.empty(
                        (self.BUFFER_SIZE,) + self.robot_observations.shape[1:], dtype=self.robot_observations.dtype
                    )
                    * numpy.nan,
                ]
            )
            self.robot_time_stamps = numpy.hstack([self.robot_time_stamps, numpy.empty(self.BUFFER_SIZE, numpy.int64)])

        robot_observation = incoming_msg.end_effector_pose.position
        position_3d = [robot_observation.x, robot_observation.y, robot_observation.z]
        stamp = incoming_msg.header_robot.stamp
        self.robot_observations[self.robot_last_observation_index, robot_index] = position_3d
        self.robot_time_stamps[self.robot_last_observation_index] = stamp.sec + stamp.nanosec * 1e-9
        self.valid_observations_count[robot_index] += 1

    def _camera_message_callback(self, camera_index, incoming_msg):
        self.camera_is_reporting = True
        self.total_observations_count[len(self.robot_names) + camera_index] += 1
        self._report_progress()
        if not self.robot_is_reporting or len(incoming_msg.balls) != 1:
            return

        if self.camera_start_offset is None:
            self.camera_start_offset = incoming_msg.header.sequence_number
        curr_camera_observation_index = incoming_msg.header.sequence_number - self.camera_start_offset
        if curr_camera_observation_index > self.camera_last_observation_index:
            self.camera_last_observation_index = curr_camera_observation_index
            while self.camera_last_observation_index >= self.camera_sequence_numbers.shape[0]:
                self.camera_observations = numpy.vstack(
                    [
                        self.camera_observations,
                        numpy.empty(
                            (self.BUFFER_SIZE,) + self.camera_observations.shape[1:],
                            dtype=self.camera_observations.dtype,
                        )
                        * numpy.nan,
                    ]
                )
                self.camera_sequence_numbers = numpy.hstack(
                    [
                        self.camera_sequence_numbers,
                        numpy.empty(self.BUFFER_SIZE, numpy.int64),
                    ]
                )

        camera_observation = incoming_msg.balls[0]

        stamp = incoming_msg.header.stamp
        if self.camera_first_time_stamp is None:
            self.camera_first_time_stamp = stamp.sec + stamp.nanosec * 1e-9
        self.camera_last_time_stamp = stamp.sec + stamp.nanosec * 1e-9

        self.camera_observations[curr_camera_observation_index, camera_index] = [
            camera_observation.ball.center.x,
            camera_observation.ball.center.y,
            camera_observation.ball.radius,
        ]
        self.camera_sequence_numbers[curr_camera_observation_index] = curr_camera_observation_index
        self.valid_observations_count[len(self.robot_names) + camera_index] += 1

    def _report_progress(self):
        if self.camera_last_observation_index % 50 == 0:
            print(
                "{:20s}{:s}".format(
                    "Valid/Observations:",
                    "".join(
                        [
                            "{:^7d}/{:^7d} ".format(
                                self.valid_observations_count[device_index],
                                self.total_observations_count[device_index],
                            )
                            for device_index, deviceName in enumerate(self.robot_names + self.camera_names)
                        ]
                    ),
                ),
                end="\r",
            )

    def get_observations(self):
        """
        @return: Both camera and robot observations
        """

        if self.robot_last_observation_index <= 0:
            raise ValueError("No robot observations were captured!")
        if self.camera_last_observation_index <= 0:
            raise ValueError("No camera observations were captured!")

        camera_frame_rate = self.camera_last_observation_index / (
            self.camera_last_time_stamp - self.camera_first_time_stamp
        )
        return {
            "robot_observations": self.robot_observations[: self.robot_last_observation_index],
            "robot_time_stamps": self.robot_time_stamps[: self.robot_last_observation_index],
            "robot_names": self.robot_names,
            "camera_observations": self.camera_observations[: self.camera_last_observation_index],
            "camera_time_stamps": self.camera_first_time_stamp
            + self.camera_sequence_numbers[: self.camera_last_observation_index].astype(numpy.float64)
            / camera_frame_rate,
            "camera_names": self.camera_names,
        }


def collect_observations(save_as=None):
    """
    Create a ROS node for collecting robot and camera nodes.

    @return: Both camera and robot observations
    """
    rclpy.init()
    observations_recorder = ObservationsRecorder()

    try:
        rclpy.spin(observations_recorder)
    except (KeyboardInterrupt, RuntimeError):
        log.exception("Stopping observations collection")
    finally:
        print("")  # Avoid erasing previous line
        observations_recorder.destroy_node()
        rclpy.shutdown()

    observations_dict = observations_recorder.get_observations()

    # Save observations for future reference
    if save_as is not None:
        numpy.savez(save_as, **observations_dict)
        log.info("Observations saved at %s", save_as)

    return observations_dict
