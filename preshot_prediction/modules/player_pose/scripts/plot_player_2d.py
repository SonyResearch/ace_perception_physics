# Copyright Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Helper script to debug player pose 2D output"""

# system
import argparse

# opencv
import cv2

# ros
import rclpy
from rclpy.node import Node

from aps import python_module as aps
from ace_interfaces import msg
from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration
import os
import numpy as np

class PlotPlayer2D(Node):
    """Node to plot 2D player pose on images from APS camera."""

    # COCO17 keypoint skeleton and colors
    COCO17_SKELETON = [  # pylint: disable=invalid-name
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    COCO17_COLORS = [  # pylint: disable=invalid-name
        (255, 0, 0),  # Nose to left eye
        (0, 255, 0),  # Nose to right eye
        (255, 0, 255),  # left eye to left ear
        (0, 255, 255),  # right eye to right ear
        (255, 128, 0),  # left shoulder to left elbow
        (255, 255, 0),  # left elbow to left wrist
        (128, 0, 255),  # right shoulder to right elbow
        (0, 128, 255),  # right elbow to right wrist
        (0, 0, 255),  # left shoulder to right shoulder
        (128, 255, 0),  # left shoulder to left hip
        (0, 255, 128),  # right shoulder to right hip
        (128, 128, 128),  # left hip to right hip
        (255, 0, 128),  # left hip to left knee
        (128, 0, 0),  # left knee to left ankle
        (0, 128, 0),  # right hip to right knee
        (0, 0, 128),  # right knee to right ankle
    ]
    def __init__(self, camera_name,config):
        super().__init__("plot_player_2d")
        
        self.config=config
        self.camera_name = camera_name
        self.player_subscriber = self.create_subscription(
            msg.PlayerPose2d, f"/sensors/aps{camera_name}/player_pose_2d", self._on_player_msg, 1
        )
        self.pose_subscriber = self.create_subscription(
            msg.PlayerPose, f"/sensors/player0/pose", self._on_pose_msg, 1
        )
        
        calibration_params_path = get_package_share_directory("calibration")
        self.camera_calibration = calibration.CameraCalibrationParameters()
        if not self.camera_calibration.initialize(
            os.path.join(
                calibration_params_path,
                "parameters",
                "camera_calibration",
                self.config + ".yaml",
            )
        ):
            raise Exception("Failed to load configuration file")
        self.camera=self.camera_calibration.cameras["aps"+camera_name]
        self.sub = aps.ZeroCopySubscriber()  # pylint: disable=c-extension-no-member
        self.handles = self.sub.add_subscription(f"/sensors/aps{camera_name}/image")
        self.last_frame_id = -1
        
        self.output_video=cv2.VideoWriter(
            f"pose_analysis/player_pose_2d_{camera_name}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            5,
            (1440, 1080),
        )

        self.message_buffer={}
    def close(self):
        """Close resources"""
        self.output_video.release()
    def _on_player_msg(self, message: msg.PlayerPose2d):
        if message.header.sequence_number not in self.message_buffer:
            self.message_buffer[message.header.sequence_number] = {}
        self.message_buffer[message.header.sequence_number]["player_pose_2d"] = message

    def _on_pose_msg(self, message: msg.PlayerPose):
        if message.header.sequence_number not in self.message_buffer:
            self.message_buffer[message.header.sequence_number] = {}
        self.message_buffer[message.header.sequence_number]["player_pose_3d"] = message

    def update(self):
        """Update function"""
        (
            rets,
            frame_ids,
            _,
            imgs_raw,
        ) = self.sub.get_latest_message(self.handles)
        
        if frame_ids<self.last_frame_id:
            self.message_buffer.clear()

        if not rets or frame_ids == self.last_frame_id:
            self.sub.release_latest_message(self.handles)
            return
        self.last_frame_id = frame_ids

        img = cv2.cvtColor(imgs_raw, cv2.COLOR_BayerBG2BGR)
        if self.last_frame_id not in self.message_buffer:
            self.message_buffer[self.last_frame_id] = {}
        self.message_buffer[self.last_frame_id]["image"] = img

        self.sub.release_latest_message(self.handles)
        self.render()
    
    def render(self):
        """Render the latest image with player pose 2D overlayed."""
        if len(self.message_buffer)<3:
            return
        
        sequence_ids=list(self.message_buffer.keys())
        sequence_ids.sort()
        for i in range(len(sequence_ids)-3):
            del self.message_buffer[sequence_ids[i]]
        curr_seq=sequence_ids[-3]
        
        img=self.message_buffer[curr_seq].get("image",None)
        pose2d=self.message_buffer[curr_seq].get("player_pose_2d",None)
        pose3d:msg.PlayerPose=self.message_buffer[curr_seq].get("player_pose_3d",None)
        if img is None:
            return
        
        if pose2d is not None:
            cv2.rectangle(
                img,(pose2d.bbox[0], pose2d.bbox[1]),
                (pose2d.bbox[0]+pose2d.bbox[2], pose2d.bbox[1]+pose2d.bbox[3]),
                (255, 0, 0), 2
            )
            visibility_threshold=0.3
            # Draw skeleton links
            for idx, (i, j) in enumerate(PlotPlayer2D.COCO17_SKELETON):  # pylint: disable=invalid-name
                if i < len(pose2d.keypoints) and j < len(pose2d.keypoints):
                    p1 = pose2d.keypoints[i]  # pylint: disable=invalid-name
                    p2 = pose2d.keypoints[j]  # pylint: disable=invalid-name
                    color = PlotPlayer2D.COCO17_COLORS[idx % len(PlotPlayer2D.COCO17_COLORS)]
                    cv2.line(img, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), color, thickness=1)
            for point in pose2d.keypoints:
                cv2.circle(img, (int(point.x), int(point.y)), 3, (0, 255, 0), -1)

        # reproject pose3d to camera
        if pose3d is not None:
            keypoints_3d = []
            for point in pose3d.keypoints:
                keypoints_3d.append([point.x, point.y, point.z])
            keypoints_3d = np.array(keypoints_3d, dtype=np.float32)
            keypoints_2d, ret = self.camera.project_points(
                keypoints_3d
            )
            for point, valid in zip(keypoints_2d,ret):
                if not valid or np.isnan(point).any():
                    continue
                x2d, y2d = point
                cv2.circle(img, (int(x2d), int(y2d)), 3, (0, 0, 255), -1)
        self.output_video.write(img)
        cv2.imshow(f"camera {self.camera_name}", img)
        cv2.waitKey(1)



def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_name",
        type=str,
        default="21101998",
        help="camera name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tyo01",
        help="camera config",
    )
    parsed_args, _ = parser.parse_known_args()

    node = PlotPlayer2D(parsed_args.camera_name, parsed_args.config)
    print("Started Plot Player 2D. Press Ctrl+C to stop.")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.update()
    except KeyboardInterrupt:
        pass
    node.close()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
