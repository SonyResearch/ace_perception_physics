# pylint: skip-file
"""
Example of using player pose pipeline
Confidential, Copyright 2024, Sony AI, All rights reserved
"""

import os
import click
import colored_glog as glog
import cv2
import numpy as np
import rclpy
from ace_loggers import logger_pybind as logger
from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration
from aps.image_subscriber import ManagedDisplayImagesContext
import ace_yaml as yaml

import player_pose.python_module as player_pose


def set_level(log_level: int = 0) -> None:
    """Parse C++ glog level to PyPI glog"""
    logger.initialize()
    logger.set_level(str(log_level))
    if log_level == 0:
        glog.setLevel("INFO")
    elif log_level == 1:
        glog.setLevel("WARNING")
    elif log_level == 2:
        glog.setLevel("ERROR")
    elif log_level == 3:
        glog.setLevel("FATAL")


set_level(3)


COLOR_DICT = {
    "purple": (255, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "skyblue": (235, 206, 135),
    "navyblue": (128, 0, 0),
    "azure": (255, 255, 240),
    "slate": (255, 0, 127),
    "choco": (30, 105, 210),
    "olive": (112, 255, 202),
    "orange": (0, 140, 255),
    "orchid": (255, 102, 224),
}
COLOR_LIST = list(COLOR_DICT.values())

# Media Pipe joint names
JOINT_NAMES = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_sho",
    "r_sho",
    "l_elb",
    "r_elb",
    "l_wri",
    "r_wri",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ank",
    "r_ank",
]

# Define pairs for drawing
JOINT_PAIRS = [
    ["nose", "r_eye", COLOR_DICT["purple"]],
    ["nose", "l_eye", COLOR_DICT["purple"]],
    ["nose", "r_sho", COLOR_DICT["yellow"]],
    ["nose", "l_sho", COLOR_DICT["yellow"]],
    ["r_sho", "l_sho", COLOR_DICT["blue"]],
    ["r_sho", "r_elb", COLOR_DICT["blue"]],
    ["r_elb", "r_wri", COLOR_DICT["green"]],
    ["l_sho", "l_elb", COLOR_DICT["blue"]],
    ["l_elb", "l_wri", COLOR_DICT["green"]],
    ["r_sho", "r_hip", COLOR_DICT["yellow"]],
    ["l_sho", "l_hip", COLOR_DICT["yellow"]],
    ["r_hip", "l_hip", COLOR_DICT["red"]],
    ["r_hip", "r_knee", COLOR_DICT["red"]],
    ["r_knee", "r_ank", COLOR_DICT["skyblue"]],
    ["l_hip", "l_knee", COLOR_DICT["red"]],
    ["l_knee", "l_ank", COLOR_DICT["skyblue"]],
]
SELECTED_KEYPOINTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class PoseContext(ManagedDisplayImagesContext):
    """
    Draw camera images helper class
    """

    def __init__(self, config, cluster):
        self.features = None
        # Load camera calibration.
        self.config = config

        calibration_params_path = get_package_share_directory("calibration")
        calibration_params_path = os.path.join(
            calibration_params_path,
            "parameters",
            "camera_calibration",
            config + ".yaml",
        )

        player_params_path = get_package_share_directory("player_pose")
        player_params_path = os.path.join(
            player_params_path,
            "parameters",
            config + cluster + ".yaml",
        )

        movenet_model_path = get_package_share_directory("player_pose")
        movenet_model_path = os.path.join(
            movenet_model_path,
            "models",
            "singlepose-lightning-tflite-float16" + ".onnx",  # movenet-lightning pose_landmark_heavy
        )

        person_detector_model_path = os.path.join(
            get_package_share_directory("player_pose"),
            "models",
            "yolov8n" + ".onnx",
        )

        self.camera_calibration = calibration.CameraCalibrationParameters()
        if not self.camera_calibration.initialize(calibration_params_path):
            raise Exception("Failed to load configuration file")

        self.detector = player_pose.player_pose_extractor("")
        self.detector.start(
            calibration_params_path,
            player_params_path,
            movenet_model_path,
            person_detector_model_path,
            player_pose.PlayerInferenceEngine.movenet,
        )

        self.last_frame = 0

        camera_mask = []
        with open(player_params_path, "r", encoding="utf8") as file:
            nodes = yaml.load(file)
            players = nodes["players"]
            for player in players:
                for cam in players[player]["cameras"]:
                    camera_mask.append(cam)

        super().__init__(config, 200, 200, camera_mask=camera_mask)
        self.node = player_pose.player_pose_ros_node()
        self.node.start("player_pose_ros_node", self.detector, True, True, True)

    def destroy(self):  # pylint: disable = useless-super-delegation
        """
        Destroy the object
        """
        super().destroy()
        self.detector.stop()

    def fetch_detections(self):
        """Fetch available detections from the detector"""
        frames = self.detector.get_detections()
        if frames and len(frames) > 0:
            self.features = frames[-1].features
            self.last_frame = frames[-1].sequence_number
            # print("Fetched: ",self.last_frame)

    def _on_image_arrived(self, index, image, topic_name):
        cam_index = 0
        for cam_index, camera in enumerate(self.camera_names):
            if camera in topic_name:
                break
        if cam_index >= len(self.camera_names):
            print("camera not found: ", cam_index)
            return
        if self.features is not None:
            cv2.putText(
                image,
                str(self.last_frame),
                (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.5,
                (85, 196, 196),
                1,
                cv2.LINE_AA,
            )
            detection = self.detector.get_last_camera_detection(cam_index)
            if detection is not None:
                # print(index," - Cam index-->",detection.camera_index)
                bbox = np.array(detection.bbox, dtype=int)
                kpt = np.array(detection.keypoints)
                # print(kpt_int)

                keypoints_map = {}
                keypoints_map = {JOINT_NAMES[i]: kpt[i] for i in range(len(kpt))}

                confidence = 0.3
                for kp_index, keypoint in enumerate(keypoints_map):
                    keypoint = keypoints_map[keypoint]
                    if keypoint[2] > confidence:
                        cv2.circle(image, keypoint[:2].astype(int), radius=3, thickness=3, color=(255, 255, 255))

                        cv2.putText(
                            image,
                            JOINT_NAMES[kp_index],
                            keypoint[:2].astype(int) + [4, 4],
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,
                            (85, 196, 196),
                            1,
                            cv2.LINE_AA,
                        )
                for pair in JOINT_PAIRS:
                    point_0 = keypoints_map[pair[0]]
                    point_1 = keypoints_map[pair[1]]
                    if point_0[2] > confidence and point_1[2] > confidence:
                        cv2.line(image, point_0[:2].astype(int), point_1[:2].astype(int), pair[2], 3)

            detection = self.detector.get_last_camera_bbox_detection(cam_index)
            if detection is not None:
                bbox = np.array(detection.bbox, dtype=int)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

        super()._on_image_arrived(index, image, topic_name)


@click.command()
@click.option("--config", help="Config name", required=True, type=str)
@click.option("--cluster", help="cluster name", required=True, type=str)
def main(config, cluster):
    """Main entry point"""
    rclpy.init()
    context: PoseContext = PoseContext(config, cluster)

    while rclpy.ok() and not context.is_done:
        context.render(force_update=False)
        context.fetch_detections()

    # Clean up.
    cv2.destroyAllWindows()
    context.destroy()
    print("Clean up successfully...")

    return 0


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
