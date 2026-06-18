"""
Helper tool to 3D visualized the published poses
Confidential, Copyright 2024, Sony AI, All rights reserved
"""

import functools
from collections import deque
from typing import List, Dict
import argparse
import csv
import numpy as np

# ros
import rclpy
from rclpy.node import Node, Subscription

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

from pose_utils import JOINT_PAIRS_ALT, KEYPOINTS_NAMES_MAP
import ace_interfaces.msg


COLOR_MAP = {
    "purple": (0.5, 0, 0.5, 1),
    "yellow": (0.5, 0.5, 0, 1),
    "blue": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "red": (0, 0, 1, 1),
    "skyblue": (0, 0.5, 0.5, 1),
}

RESOLUTION = (1080, 720)
EFFECTIVE_FPS = 200


def point_to_array(point):
    """Convert a point to array"""
    if isinstance(point, list):
        return np.array(point)
    if isinstance(point, np.ndarray):
        return point
    return np.array([point.x, point.y, point.z])


class PoseVisualizeContext(Node):
    """Visualize pose data"""

    def __init__(self, csv_path) -> None:
        super().__init__("pose_visualizer")

        self.subs: List[Subscription] = []
        self.player_pose: Dict[str, ace_interfaces.msg.PlayerPose] = {}
        self.player_index_map: Dict[str, int] = {}
        self.player_pose_updated: Dict[str, bool] = {}
        self.players_topics: List[str] = []
        self.csv_path = csv_path
        self.preloaded_keypoints: List[list] = []

        if csv_path is not None:
            self._load_csv(csv_path)
            self.csv_current_index = 0

        rclpy.spin_once(self, timeout_sec=0.1)
        self.populate_players()
        self._init_plot()

        pg.mkQApp().exec_()

    def _load_csv(self, path):
        with open(path, "rt", encoding="utf8") as file:
            reader = csv.reader(file, delimiter=",", quotechar="|")
            for row in reader:
                seq_id = int(row[0])
                keypoints_conf = np.array(row[1:], dtype=float).reshape((-1, 5))
                keypoints = keypoints_conf[:, :3]
                confidences = keypoints_conf[:, 3]
                reprojection = keypoints_conf[:, 4]
                self.preloaded_keypoints.append([seq_id, keypoints, confidences, reprojection])

            player_name = "/sensors/player0/pose"
            self.add_player(player_name, True)

    def populate_players(self):
        """populate topics"""
        new_players = []
        for topic_name, topic_types in self.get_topic_names_and_types():
            if "ace_interfaces/msg/PlayerPose" in topic_types:
                if topic_name not in self.players_topics:
                    print("Player discovered: " + topic_name)
                    new_players.append(topic_name)

        if len(new_players) > 0:
            # refresh rackets
            for topic_name in new_players:
                self.players_topics.append(topic_name)
                self.add_player(topic_name)

    def _init_plot(self):
        # Plot init
        self.app = pg.mkQApp("GLScatterPlotItem Example")
        widget = gl.GLViewWidget()
        widget.setFixedSize(*RESOLUTION)
        # widget.opts['distance'] = 10
        widget.show()
        widget.setWindowTitle("Ball & Pose triangulation")

        # Draw Floor
        grid = gl.GLGridItem()
        grid.translate(0, 0, -0.76)
        widget.addItem(grid)

        # Draw Table
        table_v_x = np.array([-1.37, 1.37])
        table_v_y = np.array([-0.7625, 0.7625])
        table_v_z = np.array([[0, 0], [0, 0]])
        table = gl.GLSurfacePlotItem(x=table_v_x, y=table_v_y, z=table_v_z, shader="normalColor")
        widget.addItem(table)

        # Draw Net
        net_v_x = np.array([0, 0])
        net_v_y = np.array([-0.7625, 0.7625])
        net_v_z = np.array([[0, 0], [0.1525, 0.1525]])
        net = gl.GLSurfacePlotItem(x=net_v_x, y=net_v_y, z=net_v_z, shader="normalColor")
        widget.addItem(net)

        # Init ball
        ball_xyz_history_size = [0.05, 0.03, 0.02, 0.01, 0.005][::-1]
        ball_xyz_history = deque(maxlen=len(ball_xyz_history_size))
        # pylint: disable = expression-not-assigned
        [ball_xyz_history.append([0, 0, 1]) for i in range(len(ball_xyz_history_size))]
        ball_xyz = np.array(ball_xyz_history)
        ball = gl.GLScatterPlotItem(pos=ball_xyz, color=(1, 1, 1, 1), size=ball_xyz_history_size, pxMode=False)
        widget.addItem(ball)

        # Init pose
        self.lines = []
        for i, _ in enumerate(self.subs):
            self.lines.append([])
            for _, joint_pair in enumerate(JOINT_PAIRS_ALT):
                pos1 = [0, 0, 0]
                pos2 = [1, 1, 1]
                pos = np.array([pos1, pos2])
                color = COLOR_MAP[joint_pair[2]]
                line = gl.GLLinePlotItem(pos=pos, color=color, width=1)
                widget.addItem(line)
                self.lines[i].append(line)

        self.q_timer = QtCore.QTimer()  # pylint: disable = no-member
        self.q_timer.timeout.connect(self.update)
        update_time_ms = int(1000 / EFFECTIVE_FPS)  # reduce for faster than realtime
        self.q_timer.start(update_time_ms)

    def add_player(self, topic_name, fake=False):
        """Add player details"""
        tmp = "/player"
        index = topic_name.index(tmp) + 1
        last_index = topic_name.index("/", index + 1)
        player_name = topic_name[index:last_index]
        if not fake:
            print("Subscribing to: ", topic_name, "/", player_name)
            sub = self.create_subscription(
                ace_interfaces.msg.PlayerPose, topic_name, functools.partial(self._on_player_pose, player_name), 1
            )
        else:
            sub = None
        self.subs.append(sub)
        player_index = len(self.player_pose)
        self.player_pose[player_name] = None
        self.player_index_map[player_name] = player_index
        self.player_pose_updated[player_index] = False

    def _on_player_pose(self, player_id, msg):
        self.player_pose[player_id] = msg
        player_index = self.player_index_map[player_id]
        self.player_pose_updated[player_index] = True

    def update_player_pose(
        self, player_index, keypoints, confidences, reprojection_err, confidence_threshould=0.2, max_error=20
    ):
        """Update player pose"""
        if not self.player_pose_updated[player_index]:
            return

        self.player_pose_updated[player_index] = False

        for link_index, joint_pair in enumerate(JOINT_PAIRS_ALT):
            skip_link = False
            pos1 = [0, 0, 0]
            pos2 = [0, 0, 0]
            try:
                joint1_index = KEYPOINTS_NAMES_MAP[joint_pair[0]]
                joint2_index = KEYPOINTS_NAMES_MAP[joint_pair[1]]

                joint1 = point_to_array(keypoints[joint1_index])
                joint2 = point_to_array(keypoints[joint2_index])
                joint1_conf = confidences[joint1_index]
                joint2_conf = confidences[joint2_index]
                joint1_err = reprojection_err[joint1_index]
                joint2_err = reprojection_err[joint2_index]

                # print("----")
                # print(joint1)
                # print(joint2)

                if joint1_conf < confidence_threshould or joint1_err > max_error:
                    skip_link = True
                else:
                    pos1 = joint1

                if joint2_conf < confidence_threshould or joint2_err > max_error:
                    skip_link = True
                else:
                    pos2 = joint2

            except Exception as err:  # pylint: disable = broad-except
                print(err)
                skip_link = True
            if np.any(np.abs(pos1) > 5) or np.any(np.abs(pos2) > 5):
                skip_link = True
            if skip_link:
                pos1 = [0, 0, 0]
                pos2 = [0, 0, 0]
            pos = np.array([pos1, pos2])
            self.lines[player_index][link_index].setData(pos=pos)

    def update(self):
        """Update ros and pose"""
        if self.csv_path is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not rclpy.ok():
            self.q_timer.stop()
            self.q_timer.deleteLater()
            return

        if self.csv_path is None:
            for player_name, pose in self.player_pose.items():
                player_index = self.player_index_map[player_name]
                if pose is None:
                    continue
                self.update_player_pose(
                    player_index,
                    pose.keypoints,
                    pose.confidences,
                    pose.projection_error,
                    confidence_threshould=0.2,
                    max_error=15,
                )
        else:
            if self.csv_current_index >= len(self.preloaded_keypoints):
                self.q_timer.stop()
                self.q_timer.deleteLater()
                print("Done with CSV")
                return
            player_index = 0
            self.player_pose_updated[player_index] = True
            current = self.preloaded_keypoints[self.csv_current_index]
            self.update_player_pose(player_index, current[1], current[2], current[3], confidence_threshould=0.2)
            self.csv_current_index = self.csv_current_index + 1


def parse_args():
    """
    Provide script-specific arguments.

    @return: Parsed argument object
    """

    parser = argparse.ArgumentParser(description="Pose triangulation pipeline.")
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        help="Path to csv file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        required=False,
        help="rendering framerate",
    )

    return parser.parse_args()


def main(args):
    """Main entry point"""

    rclpy.init()
    _ = PoseVisualizeContext(args.csv)
    rclpy.shutdown()


if __name__ == "__main__":
    args = parse_args()
    EFFECTIVE_FPS = args.fps
    main(args)
