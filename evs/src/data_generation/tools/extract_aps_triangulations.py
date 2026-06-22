# pylint: disable = line-too-long, c-extension-no-member
"""
@brief Extracts 3D ball position and rotation from ROSbag.

@file pos_triang_multi.py
@author Claudio Fanconi (claudioandrea.fanconi@sony.com), Asude Aydin (asude.aydin@sony.com)
@date 2024
@version 0.0
@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

from pathlib import Path

import numpy as np
import pandas
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from evs_ball_interfaces.msg import PosesWithCovariance


def get_rosbag_options(path, serialization_format="cdr"):
    """Get rosbag storage options."""
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    return storage_options, converter_options


def extract_first_aps_frameid(rosbag_path: str, target_path: str):
    """Extract ROS points and save them into a CSV file.

    Args:
        `rosbag_path` (str): path where the ROS bag is stored
        `target_path` (str): path where the CSV should be saved
    returns:
        None
    """
    # Load recorded rosbags
    store_options, conv_options = get_rosbag_options(rosbag_path)
    reader_detect = rosbag2_py.SequentialReader()
    reader_detect.open(store_options, conv_options)
    topic_types = reader_detect.get_all_topics_and_types()

    # Set filter for triangulations
    storage_filter = rosbag2_py.StorageFilter(
        topics=[
            "/sensors/ball_triangulation/points",
            "/sensors/ball_pose_estimation/poses",
        ]
    )
    reader_detect.set_filter(storage_filter)

    # Create a map for quicker lookup
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    while reader_detect.has_next():
        topic, data, _ = reader_detect.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        start_time_target_path = Path(target_path) / "start_time.txt"
        start_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1000000000
        with open(start_time_target_path, "w", encoding="utf-8") as file:
            file.write(str(start_time))

        return start_time


def extract_aps_points(rosbag_path: str, csv_target_path: str) -> None:
    """Extract ROS points and save them into a CSV file.

    Args:
        `rosbag_path` (str): path where the ROS bag is stored
        `csv_target_path` (str): path where the CSV should be saved
    returns:
        None
    """
    # Load recorded rosbags
    store_options, conv_options = get_rosbag_options(rosbag_path)
    reader_detect = rosbag2_py.SequentialReader()
    reader_detect.open(store_options, conv_options)
    topic_types = reader_detect.get_all_topics_and_types()

    # Set filter for triangulations
    storage_filter = rosbag2_py.StorageFilter(
        topics=[
            "/sensors/ball_triangulation/points",
            "/sensors/ball_pose_estimation/poses",
        ]
    )
    reader_detect.set_filter(storage_filter)

    # Create a map for quicker lookup
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    triangulated_points: dict = {
        "frame_id": [],
        "time": [],
        "position_x": [],
        "position_y": [],
        "position_z": [],
        "orientation_x": [],
        "orientation_y": [],
        "orientation_z": [],
        "orientation_w": [],
    }

    while reader_detect.has_next():
        topic, data, _ = reader_detect.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if isinstance(msg, PosesWithCovariance):
            if len(msg.poses) > 0:
                poses_positions = msg.poses[0].pose.position
                poses_orientation = msg.poses[0].pose.orientation
                time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1000000000
                triangulated_points["frame_id"].append(msg.header.sequence_number)
                triangulated_points["time"].append(time)
                triangulated_points["position_x"].append(poses_positions.x)
                triangulated_points["position_y"].append(poses_positions.y)
                triangulated_points["position_z"].append(poses_positions.z)
                triangulated_points["orientation_x"].append(
                    poses_orientation.x if poses_orientation.x < 1e38 else np.nan
                )
                triangulated_points["orientation_y"].append(
                    poses_orientation.y if poses_orientation.y < 1e38 else np.nan
                )
                triangulated_points["orientation_z"].append(
                    poses_orientation.z if poses_orientation.z < 1e38 else np.nan
                )
                triangulated_points["orientation_w"].append(
                    poses_orientation.w if poses_orientation.w < 1e38 else np.nan
                )

    dataframe = pandas.DataFrame.from_dict(triangulated_points)
    dataframe.to_csv(
        csv_target_path,
        encoding="utf-8",
        mode="a",
    )
