# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Helper script to export racket topics to csv file"""

import time
import click
import rclpy
from rclpy.node import Node
import ace_interfaces.msg

rackets = {
    # {"name": "racket_gt_0.csv", "topic": "/sensors/racket_vive0/pose"},
    # {"name": "racket_gt_1.csv", "topic": "/sensors/racket_vive1/pose"},
    "_a": [
        {"name": "racket_est_2.csv", "topic": "/sensors/racket0/filtered"},
    ],
    # "_b": [{"name": "", "topic": "/sensors/racket0/pose"}],
}


class RacketSub:
    """RacketPose subscriber class"""

    def __init__(self, node, name, topic) -> None:
        self.name = name
        self.topic = topic
        self.pose = None
        qos_profile = rclpy.qos.QoSProfile(depth=1)
        if "vive/" in topic:
            self.msg_type = ace_interfaces.msg.RacketPose
        else:
            self.msg_type = ace_interfaces.msg.RacketPoseEstimate

        self.racket_subscriber = node.create_subscription(
            self.msg_type,
            self.topic,
            self.on_racket_pose_callback,
            qos_profile=qos_profile,
        )
        self.ball_subscriber = node.create_subscription(
            ace_interfaces.msg.PosesWithCovariance,
            "/sensors/ball_pose_estimation/poses",
            self.on_ball_pose_callback,
            qos_profile=qos_profile,
        )
        self.file = open(self.name, "wt", encoding="utf8")  # pylint: disable=consider-using-with
        self.ball_file = open("racket_gt_2.csv", "wt", encoding="utf8")  # pylint: disable=consider-using-with
        self.last_time = time.time()
        self.total = 0

    def on_ball_pose_callback(self, msg: ace_interfaces.msg.PosesWithCovariance):
        """On ball pose callback"""
        if len(msg.poses) == 0:
            return
        ball_pose: ace_interfaces.msg.PoseWithCovariance = msg.poses[0]

        pose = [
            msg.header.sequence_number,
            ball_pose.pose.position.x,
            ball_pose.pose.position.y,
            ball_pose.pose.position.z,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]

        pose = [str(x) for x in pose]
        self.ball_file.write("\t".join(pose) + "\n")
        self.file.flush()

    def on_racket_pose_callback(self, msg):
        """ROS callback for racket pose"""

        self.total += 1
        now = time.time()
        if now - self.last_time > 1:
            print(f"{self.topic}: {self.total}FPS")
            self.total = 0
            self.last_time = now

        est = None

        if hasattr(msg, "pose"):
            est = msg
            msg = msg.pose

        self.pose = [
            str(v)
            for v in [
                msg.header.sequence_number,
                msg.position.x,
                msg.position.y,
                msg.position.z,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
                est.projection_error if est is not None else 0,
                est.orientation_error if est is not None else 0,
                est.confidence if est is not None else 1,
            ]
        ]

        self.file.write("\t".join(self.pose) + "\n")
        self.file.flush()


@click.command()
@click.option("--clusters", help="Cluster name", required=True, type=str)
# @click.option("--name", help="record name", required=True, type=str)
def main(clusters):
    """Main entry point"""
    rclpy.init()
    dummy_node = Node("racket_sub")

    clusters = clusters.split(",")
    racket_subs = []
    for cluster in clusters:
        for racket in rackets[cluster]:
            # racket["name"] = name + cluster + ".csv"
            racket_subs.append(RacketSub(dummy_node, racket["name"], racket["topic"]))

    rclpy.spin(dummy_node)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
