"""
Confidential, Copyright 2025, Sony AI, All rights reserved.
"""

from typing import Iterator

from typing_extensions import override
from packaging import version
from racket_pose_estimation import datalogger_pybind as datalogger
from rosbag_dataloaders import dataloaders


CHANGE_LOG = {
    "stable_version": version.parse("1.1.0"),
    "time_introduced": version.parse("1.2.0"),
}


def read_log_file(path):
    """
    Load a complete log file
    """
    with datalogger.DataReader(path) as reader:
        curr_version = version.parse(reader.log_version)
        while not reader.eof():
            if curr_version < CHANGE_LOG["time_introduced"]:
                start_time, end_time = None, None  # pylint: disable=unused-variable
            else:
                start_time, end_time = reader.read_time(), reader.read_time()  # pylint: disable=unused-variable
            data_type = reader.read_enum()
            match data_type:
                case datalogger.type.racket_pose_features:
                    yield reader.read_pose_features()
                case datalogger.type.racket_estimated_pose:
                    yield reader.read_estimated_pose()
                case datalogger.type.racket_frame_features:
                    data = reader.read_frame_features()
                    yield data
                case _:
                    assert False, f"data_type={int(data_type)} is not supported!"


def read_log_metadata(path):
    """
    Load metadata from a log file
    """
    return datalogger.DataReader(path).get_metadata()


class RosLoader(dataloaders.BagFileParser):
    """
    Loader for ACE logger messages as ROS2 messages
    """

    def __init__(self, path, racket_ids=None):
        super().__init__(path)
        if racket_ids is None:
            with datalogger.DataReader(path) as reader:
                reader_metadata = reader.get_metadata()
                racket_ids = reader_metadata.get("racket_ids")
        assert racket_ids is not None
        self._path = path
        self._topics_with_types = {
            f"/sensors/racket{racket_id}/pose": "ace_interfaces/msg/RacketPoseEstimate" for racket_id in racket_ids
        }
        self._topics_with_ids = {f"/sensors/racket{racket_id}/pose": racket_id for racket_id in racket_ids}

    @override
    def get_topics_with_types(self) -> list:
        """Get topics with types"""
        return self._topics_with_types.items()

    @override
    def get_topics(self) -> list:
        """Get topics"""
        return self._topics_with_types.keys()

    @override
    def get_messages(self, topic_name) -> Iterator[tuple]:
        """Get Messages iterator"""
        topic_type = self._topics_with_types[topic_name]
        racket_id = self._topics_with_ids[topic_name]
        match topic_type:
            case "ace_interfaces/msg/RacketPoseEstimate":
                # pylint: disable=import-outside-toplevel
                from ace_interfaces.msg import RacketPoseEstimate

                for msg in read_log_file(self._path):
                    timestamp = msg.sequence_number * 1e-3
                    ros_msg = RacketPoseEstimate()
                    ros_msg.pose.header.sequence_number = msg.sequence_number
                    ros_msg.pose.header.stamp.sec = int(timestamp)
                    ros_msg.pose.header.stamp.nanosec = int((timestamp % 1) * 1e9)
                    racket = next(filter(lambda racket: racket.racket_id == racket_id, msg.estimated_rackets), None)
                    if racket is None:
                        ros_msg.pose.tracked = False
                    else:
                        ros_msg.pose.tracked = True
                        ros_msg.pose.serial = str(racket.racket_id)
                        pos = racket.position.astype(float)
                        ori = racket.orientation.astype(float)
                        ros_msg.pose.position.x = pos[0]
                        ros_msg.pose.position.y = pos[1]
                        ros_msg.pose.position.z = pos[2]
                        ros_msg.pose.orientation.x = ori[0]
                        ros_msg.pose.orientation.y = ori[1]
                        ros_msg.pose.orientation.z = ori[2]
                        ros_msg.pose.orientation.w = ori[3]
                        ros_msg.projection_error = racket.reprojection_err
                        ros_msg.orientation_error = racket.orientation_error
                        ros_msg.confidence = racket.orientation_confidence
                    yield timestamp, ros_msg
            case _:
                raise NotImplementedError

    @override
    def get_message(self, topic_name, message_index) -> tuple:
        """Not implemented"""
        raise NotImplementedError

    @override
    def get_messages_count(self, topic_name) -> int:
        """Not implemented"""
        raise NotImplementedError
