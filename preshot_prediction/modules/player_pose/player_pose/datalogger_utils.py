"""
Confidential, Copyright 2025, Sony AI, All rights reserved.
"""

from typing import Iterator
import numpy as np

from typing_extensions import override
from packaging import version
from player_pose import datalogger_pybind as datalogger
from rosbag_dataloaders import dataloaders


CHANGE_LOG = {
    "stable_version": version.parse("1.0.0"),
    "time_introduced": version.parse("1.1.0"),
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
                case datalogger.type.player_bbox_detection:
                    yield reader.read_bbox_detection()
                case datalogger.type.player_pose_detection:
                    yield reader.read_pose_detection()
                case datalogger.type.player_pose_estimate:
                    yield reader.read_pose_estimate()
                case datalogger.type.player_frame_features:
                    yield reader.read_frame_features()
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

    def __init__(self, path, player_ids=None):
        super().__init__(path)
        if player_ids is None:
            with datalogger.DataReader(path) as reader:
                reader_metadata = reader.get_metadata()
                player_ids = reader_metadata.get("player_ids")
        assert player_ids is not None
        self._path = path
        self._topics_with_types = {
            f"/sensors/player{player_id}/pose": "ace_interfaces/msg/PlayerPose" for player_id in player_ids
        }
        self._topics_with_ids = {f"/sensors/player{player_id}/pose": player_id for player_id in player_ids}

    @override
    def get_topics_with_types(self) -> list:
        return self._topics_with_types.items()

    @override
    def get_topics(self) -> list:
        return self._topics_with_types.keys()

    @override
    def get_messages(self, topic_name) -> Iterator[tuple]:
        topic_type = self._topics_with_types[topic_name]
        player_id = self._topics_with_ids[topic_name]
        match topic_type:
            case "ace_interfaces/msg/PlayerPose":
                # pylint: disable=import-outside-toplevel
                from ace_interfaces.msg import PlayerPose

                for msg in read_log_file(self._path):
                    timestamp = msg.sequence_number * 1e-3
                    ros_msg = PlayerPose()
                    ros_msg.header.sequence_number = msg.sequence_number
                    ros_msg.header.stamp.sec = int(timestamp)
                    ros_msg.header.stamp.nanosec = int((timestamp % 1) * 1e9)
                    player = next(filter(lambda player: player.player_id == player_id, msg.estimated_players), None)
                    if player is not None:
                        # ros_msg.pose.player_id = player.player_id
                        kps = np.array(player.keypoints).astype(float)
                        for i in range(17):
                            ros_msg.keypoints[i].x = kps[i, 0]
                            ros_msg.keypoints[i].y = kps[i, 1]
                            ros_msg.keypoints[i].z = kps[i, 2]
                            ros_msg.projection_error[i] = player.projection_error[i]
                            ros_msg.confidences[i] = player.confidences[i]
                    yield timestamp, ros_msg
            case _:
                raise NotImplementedError

    @override
    def get_message(self, topic_name, message_index) -> tuple:
        raise NotImplementedError

    @override
    def get_messages_count(self, topic_name) -> int:
        raise NotImplementedError
