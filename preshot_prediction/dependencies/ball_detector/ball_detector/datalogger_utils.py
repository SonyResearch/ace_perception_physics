"""
Confidential, Copyright 2025, Sony AI, All rights reserved.
"""

from typing import Iterator

import os
import re
from ball_detector import datalogger_pybind as datalogger
from typing_extensions import override
from packaging import version
from rosbag_dataloaders import dataloaders

CHANGE_LOG = {
    "time_introduced": version.parse("1.0.1"),
}


def read_log_file(path):
    """
    Load a complete log file
    """
    with datalogger.DataReader(path) as reader:
        curr_version = version.parse(reader.log_version)
        while not reader.eof():
            sequence_number = reader.read_uint64()
            timestamp_sec, timestamp_nanosec = reader.read_timestamp()
            if curr_version < CHANGE_LOG["time_introduced"]:
                start_time, end_time = None, None
            else:
                start_time, end_time = reader.read_time(), reader.read_time()
            total_detections = reader.read_uint64()
            assert total_detections < 50
            ball_detections = []
            for _ in range(total_detections):
                reader.read_roi()
                reader.read_image()
                center = reader.read_ball_center()
                radius = reader.read_ball_radius()
                ball_detections.append({"center": center, "radius": radius})
            yield {
                "sequence_number": sequence_number,
                "timestamp": timestamp_sec + 1e-9 * timestamp_nanosec,
                "start_time": start_time,
                "end_time": end_time,
                "ball_detections": ball_detections,
            }


def read_log_metadata(path):
    """
    Load metadata from a log file
    """
    return datalogger.DataReader(path).get_metadata()


class RosLoader(dataloaders.BagFileParser):
    """
    Loader for ACE logger messages as ROS2 messages
    """

    def __init__(self, path, camera_name=None):
        super().__init__(path)
        if camera_name is None:
            with datalogger.DataReader(path) as reader:
                reader_metadata = reader.get_metadata()
                camera_name = reader_metadata.get("camera_name")
        if camera_name is None:
            match = re.match(".*ball_detector_(.*)\\.ace", os.path.basename(path))
            camera_name = "unknown" if match is None else match.group(1)
        self._path = path
        self._topics_with_types = {
            f"/sensors/{camera_name}/ball_detection": "ace_interfaces/msg/BallsWithAttributes",
        }

    @override
    def get_topics_with_types(self) -> list:
        return self._topics_with_types.items()

    @override
    def get_topics(self) -> list:
        return self._topics_with_types.keys()

    @override
    def get_messages(self, topic_name) -> Iterator[tuple]:
        topic_type = self._topics_with_types[topic_name]
        match topic_type:
            case "ace_interfaces/msg/BallsWithAttributes":
                # pylint: disable=import-outside-toplevel
                from ace_interfaces.msg import BallsWithAttributes, BallWithAttributes

                for msg in read_log_file(self._path):
                    ros_msg = BallsWithAttributes()
                    ros_msg.header.sequence_number = msg["sequence_number"]
                    ros_msg.header.stamp.sec = int(msg["timestamp"])
                    ros_msg.header.stamp.nanosec = int((msg["timestamp"] % 1) * 1e9)
                    for ball in msg["ball_detections"]:
                        ball_msg = BallWithAttributes()
                        ball_msg.ball.center.x = ball["center"][0]
                        ball_msg.ball.center.y = ball["center"][1]
                        ball_msg.ball.radius = ball["radius"]
                        ros_msg.balls.append(ball_msg)
                    yield msg["timestamp"], ros_msg
            case _:
                raise NotImplementedError

    @override
    def get_message(self, topic_name, message_index) -> tuple:
        raise NotImplementedError

    @override
    def get_messages_count(self, topic_name) -> int:
        raise NotImplementedError
