"""
Confidential, Copyright 2025, Sony AI, All rights reserved.
"""

from typing import Iterator

import numpy
from typing_extensions import override
from packaging import version
from rosbag_dataloaders import dataloaders
from triangulation import datalogger_pybind as datalogger


CHANGE_LOG = {
    "time_introduced": version.parse("1.0.1"),
}


def read_log_file(path):
    """
    Load a complete log file
    """
    total_cameras = None
    with datalogger.DataReader(path) as reader:
        curr_version = version.parse(reader.log_version)
        while not reader.eof():
            sequence_number = reader.read_uint64()
            timestamp_sec, timestamp_nanosec = reader.read_timestamp()
            if curr_version < CHANGE_LOG["time_introduced"]:
                start_time, end_time = None, None
            else:
                start_time, end_time = reader.read_time(), reader.read_time()
            assert reader.read_uint32() == 0
            input_points = reader.read_3d_input()
            if total_cameras is None:
                total_cameras = len(input_points)
            assert len(input_points) == total_cameras
            output_triangulations = reader.read_triangulation_output()

            output_positions = [ball.position for ball in output_triangulations]
            yield {
                "sequence_number": sequence_number,
                "timestamp": timestamp_sec + 1e-9 * timestamp_nanosec,
                "start_time": start_time,
                "end_time": end_time,
                "points_2d": input_points,
                "positions": output_positions,
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

    def __init__(self, path):
        super().__init__(path)
        self._path = path
        self._topics_with_types = {
            "/sensors/ball_triangulation/points": "ace_interfaces/msg/PointsWithCovariance",
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
            case "ace_interfaces/msg/PointsWithCovariance":
                # pylint: disable=import-outside-toplevel
                from ace_interfaces.msg import PointsWithCovariance, PointWithCovariance

                for msg in read_log_file(self._path):
                    ros_msg = PointsWithCovariance()
                    ros_msg.header.sequence_number = msg["sequence_number"]
                    ros_msg.header.stamp.sec = int(msg["timestamp"])
                    ros_msg.header.stamp.nanosec = int((msg["timestamp"] % 1) * 1e9)
                    for point in msg["positions"]:
                        point = point.astype(numpy.float64)
                        point_msg = PointWithCovariance()
                        point_msg.position.x = point[0]
                        point_msg.position.y = point[1]
                        point_msg.position.z = point[2]
                        ros_msg.points.append(point_msg)
                    yield msg["timestamp"], ros_msg
            case _:
                raise NotImplementedError

    @override
    def get_message(self, topic_name, message_index) -> tuple:
        raise NotImplementedError

    @override
    def get_messages_count(self, topic_name) -> int:
        raise NotImplementedError
