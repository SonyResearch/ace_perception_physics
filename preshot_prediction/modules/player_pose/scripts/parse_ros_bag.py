"""
Evaluate keypoints accuracy against vive tracker
Confidential, Copyright 2025, Sony AI, All rights reserved
"""
import argparse
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import rosbag_dataloaders.dataloaders
import ace_interfaces.msg


class RosbagParser:
    """Parse a rosbag and process it to extract human poses"""

    def __init__(self, path) -> None:
        rosbag_parser = rosbag_dataloaders.dataloaders.get_parser(path)

        def type_checker(topic_name):
            return (
                topic_name in "/sensors/player0/pose"
                or topic_name in "/sensors/racket0/pose"
                or topic_name in "/sensors/ball_triangulation/points"
            )

        curr_topics = [
            (topic_name) for topic_name, topic_type in rosbag_parser.get_topics_with_types() if type_checker(topic_name)
        ]
        messages = [
            {message.header.sequence_number: message for timestamp, message in rosbag_parser.get_messages(topic_name)}
            for topic_name in curr_topics
        ]
        self.num_topics = len(curr_topics)

        self.min_sequence_number = min([min(messages[topic_index].keys()) for topic_index in range(self.num_topics)])
        self.max_sequence_number = max([max(messages[topic_index].keys()) for topic_index in range(self.num_topics)])
        self.synced_messages = {
            sequence_number: [messages[topic_index].get(sequence_number) for topic_index in range(self.num_topics)]
            for sequence_number in range(self.min_sequence_number, 1 + self.max_sequence_number)
        }

        self.curr_sequence_number = self.min_sequence_number

    def is_done(self):
        """Check if the bag is done"""
        return self.max_sequence_number < self.curr_sequence_number

    def next_frame(self):
        """Get next frame"""
        if self.max_sequence_number < self.curr_sequence_number:
            raise Exception("Finished playback")

        messages = self.synced_messages[self.curr_sequence_number]

        result = {"sequence_id": self.curr_sequence_number, "player": None, "racket": None, "ball": None}

        for msg in messages:
            if msg is None:
                continue
            if isinstance(msg, ace_interfaces.msg.PlayerPose):
                result["player"] = msg
            if isinstance(msg, ace_interfaces.msg.RacketPose):
                result["racket"] = msg
            if isinstance(msg, ace_interfaces.msg.PosesWithCovariance):
                result["ball"] = msg

        self.curr_sequence_number += 1

        return result


def parse_args():
    """
    Provide script-specific arguments.

    @return: Parsed argument object
    """

    parser = argparse.ArgumentParser(description="Pose triangulation pipeline.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Parse a ros bag instead of real-time triangulation",
    )
    return parser.parse_args()


def calc_error(msg, keypoint, min_conf=0.3, max_err=20):
    """Calculate errors between keypoint and vive tracker"""

    def convert_point(point):
        """Convert a point to array"""
        return np.array([point.x, point.y, point.z])

    player_keypoint = convert_point(msg["player"].keypoints[keypoint])
    conf = msg["player"].confidences[keypoint]
    err = msg["player"].projection_error[keypoint]

    reference_pos = None
    if "racket" in msg and msg["racket"] is not None:
        reference_pos = convert_point(msg["racket"].position)
    elif "ball" in msg and msg["ball"] is not None and len(msg["ball"].poses) > 0:
        reference_pos = convert_point(msg["ball"].poses[0].pose.position)

    if reference_pos is None:
        return None, conf, msg["sequence_id"], keypoint, reference_pos

    diff = player_keypoint - reference_pos
    length = np.linalg.norm(diff)
    if conf < min_conf or err > max_err or np.isnan(err) or length > 0.7:
        return None, conf, msg["sequence_id"], keypoint, reference_pos
    return length, conf, msg["sequence_id"], keypoint, reference_pos


def process_errors(err, window=100):
    """Average errors along a sliding window"""
    mean = []
    std = []
    # for i in range(len(err) // window):
    #     sub_window = err[i * window : (i + 1) * window]
    for i in range(len(err)):
        sub_window = err[i : i + window]
        mean.append(np.mean(sub_window, axis=0))
        std.append(np.std(sub_window, axis=0))
    return np.array(mean), np.array(std)


def plot(axis, x_axis, data, label, color, window_size, std):
    """Plot error"""
    data = np.array(data)
    x_axis = x_axis[: len(data)]

    axis.plot(x_axis, data, label=label, color=color)
    axis.set_xlabel("time (sec)")
    axis.set_ylabel("error (cm)")
    axis.set_title(f"Window size: {window_size} samples ")
    if std is not None:
        std = np.array(std)
        axis.plot(x_axis, data - std, linestyle="--", color="orange")
        axis.plot(x_axis, data + std, linestyle="--", color="orange")
    axis.legend()


class KeypointName(Enum):
    """Enum for human keypoints"""

    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_HAND = 9
    R_HAND = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_FOOT = 15
    R_FOOT = 16


# pylint: disable=too-many-locals,too-many-statements
def main(args):
    """Main entry point"""
    print("Loading from a ros bag")
    reader = RosbagParser(args.path)
    distances = []
    sequences = []
    confidences = []

    min_conf = 0.3
    max_err = 10
    keypoint = KeypointName.R_HIP

    half_width = 0.76
    half_length = 1.37
    width = 4
    length = 4
    x_offset = 3
    y_offset = length / 2
    sampling_size = 0.1

    data = np.zeros((int(width / sampling_size), int(length / sampling_size)))
    count = np.zeros((int(width / sampling_size), int(length / sampling_size)))
    # r_wrist 10
    while not reader.is_done():
        msg = reader.next_frame()
        if msg is None or msg["player"] is None:
            # print(f"Dropping {msg['sequence_id']}")
            continue
        dist, conf, seq_id, _, reference = calc_error(msg, keypoint.value, min_conf, max_err)
        skip = False
        if dist is None:
            skip = True
        else:
            reference[0] += x_offset
            reference[1] += y_offset

            if reference[0] < 0 or reference[0] > length or reference[1] < 0 or reference[1] > width:
                skip = True

        if skip:
            # distances.append(0)
            # confidences.append(0)
            # sequences.append(seq_id * 0.005)
            continue

        data[int(reference[0] / sampling_size), int(reference[1] / sampling_size)] += dist
        count[int(reference[0] / sampling_size), int(reference[1] / sampling_size)] += 1
        distances.append(dist)
        confidences.append(conf)
        sequences.append(seq_id * 0.005)

    for x_index in range(data.shape[0]):
        for y_index in range(data.shape[0]):
            if count[x_index, y_index] == 0:
                continue
            data[x_index, y_index] /= count[x_index, y_index]

    start_from = int(7 * 2e2)
    end = int(-4 * 2e2)
    distances = distances[start_from:end]
    sequences = sequences[start_from:end]
    unit = 1e2
    data = data * unit
    distances = np.array(distances) * unit
    window_size = 100
    mean, std = process_errors(distances, window_size)

    fig, axis = plt.subplots()
    x_data = np.arange(data.shape[1]) * sampling_size - y_offset
    y_data = np.arange(data.shape[0]) * sampling_size - x_offset

    z_min, z_max = max(-1 * unit, -np.abs(data).max()), min(1 * unit, np.abs(data).max())
    colormesh = axis.pcolormesh(x_data, y_data, data, cmap="RdBu", vmin=z_min, vmax=z_max)
    plt.colorbar(colormesh, ax=axis)
    plt.plot([-half_width, -half_width], [-half_length, half_length / 2], marker="o", color="blue")
    plt.plot([-half_width, half_width], [-half_length, -half_length], marker="o", color="blue")
    plt.plot([-half_width, half_width], [0, 0], marker="o", color="red")
    plt.plot([half_width, half_width], [-half_length, half_length / 2], marker="o", color="blue")

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    plot(axis, np.array(list(range(len(mean)))) * 0.005, mean, keypoint.name + " Error", "blue", window_size, std)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
