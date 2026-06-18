# Copyright Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Helper script to debug player pose 2D output"""

# system
import argparse

# opencv
import cv2
import ace_yaml as yaml
# ros
import rclpy
from rclpy.node import Node

from aps import python_module as aps
from ace_interfaces import msg
from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration
import os
import numpy as np
from functools import partial

class PoseAnalyser(Node):
    COCO17_NAMES=[  # pylint: disable=invalid-name
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "l_foot",
        "r_foot"
    ]

    def __init__(self, config):
        super().__init__("pose_analyser")
        
        self.config=config
        player_params_path=get_package_share_directory("player_pose")
        player_conf_path=os.path.join(
                player_params_path,
                "parameters",
                self.config + ".yaml",
            )
        with open(player_conf_path,"r") as fp:
            self.player_params=yaml.load(fp)
            
        
        for player_id in self.player_params["players"]:
            player=self.player_params["players"][player_id]
            cameras=player["cameras"]
            break
        self.player_subscribers = {}
        for cam in cameras:
            sub=self.create_subscription(
                msg.PlayerPose2d, f"/sensors/{cam}/player_pose_2d", partial(self._on_player_msg, camera_name=cam), 1
            )
            self.player_subscribers[cam] = sub
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

        self.message_buffer={}

    def _on_player_msg(self,message: msg.PlayerPose2d,camera_name):
        if message.header.sequence_number not in self.message_buffer:
            self.message_buffer[message.header.sequence_number] = {}
        self.message_buffer[message.header.sequence_number][camera_name] = message

    def _on_pose_msg(self, message: msg.PlayerPose):
        if message.header.sequence_number not in self.message_buffer:
            self.message_buffer[message.header.sequence_number] = {}
        self.message_buffer[message.header.sequence_number]["player_pose_3d"] = message
        
        print(f"Received 3D pose for seq {message.header.sequence_number}")

    def analyse(self):
        header=[]
        header.append("sequence_number")
        for kp in PoseAnalyser.COCO17_NAMES:
            header.append(f"{kp}")
        csv_files={}
        for camera_name in self.player_subscribers:
            csv_file=f"pose_analysis/pose_analysis_{camera_name}.csv"
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            csv_file_handle=open(csv_file,"wt")
            csv_file_handle.write(",".join(header)+"\n")
            
            csv_files[camera_name]=csv_file_handle
            
        for seq_id in self.message_buffer:
            pose3d_msg=self.message_buffer[seq_id].get("player_pose_3d",None)
            for camera_name in self.message_buffer[seq_id]:
                if camera_name=="player_pose_3d":
                    continue
                row=[]
                row.append(f"{seq_id}")
                csv_file_handle=csv_files[camera_name]
                
                
                pose2d_msg=self.message_buffer[seq_id][camera_name]
                if pose3d_msg is None or pose2d_msg is None:
                    row.extend([f"{np.nan}"]*len(PoseAnalyser.COCO17_NAMES))
                    csv_file_handle.write(",".join(row)+"\n")
                    continue
                keypoints_3d=[]
                for point in pose3d_msg.keypoints:
                    keypoints_3d.append([point.x,point.y,point.z])
                keypoints_3d=np.array(keypoints_3d)
                camera=self.camera_calibration.cameras[camera_name]
                keypoints_2d, ret = camera.project_points(
                    keypoints_3d
                )
                for idx,(point,valid) in enumerate(zip(keypoints_2d,ret)):
                    if not valid or np.isnan(point).any():
                        row.append(np.nan)
                        continue
                    x2d, y2d = point
                    kp2d=pose2d_msg.keypoints[idx]
                    dist=np.sqrt((kp2d.x - x2d)**2 + (kp2d.y - y2d)**2)
                    row.append(f"{dist:.2f}")
                csv_file_handle.write(",".join([str(x) for x in row])+"\n")

def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="tyo01",
        help="camera config",
    )
    parsed_args, _ = parser.parse_known_args()

    node = PoseAnalyser( parsed_args.config)
    print("Started Pose Analyser. Press Ctrl+C to stop and analyse data.")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.analyse()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
