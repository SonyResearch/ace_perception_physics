# pylint: skip-file
"""
Example of using human detection
Confidential, Copyright 2024, Sony AI, All rights reserved
"""

import os

import colored_glog as glog
import cv2
import numpy as np
import rclpy
import click
from ace_loggers import logger_pybind as logger
from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration

from aps.image_subscriber import ManagedDisplayImagesContext
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


class PoseContext(ManagedDisplayImagesContext):
    """
    Draw camera images helper class
    """

    def __init__(self, config):
        self.features = None
        super().__init__(config, 200, 200)
        # Load camera calibration.
        self.config = config

        player_params_path = get_package_share_directory("player_pose")
        player_params_path = os.path.join(
            player_params_path,
            "parameters",
            config + ".yaml",
        )

        calibration_params_path = get_package_share_directory("calibration")
        calibration_params_path = os.path.join(
            calibration_params_path,
            "parameters",
            "camera_calibration",
            config + ".yaml",
        )

        person_detector_model_path = os.path.join(
            get_package_share_directory("player_pose"),
            "models",
            "yolov8n" + ".onnx",
        )

        self.player_params = player_pose.PlayerParameters()
        if not self.player_params.initialize(player_params_path):
            raise Exception("Failed to load configuration file")

        self.camera_calibration = calibration.CameraCalibrationParameters()
        if not self.camera_calibration.initialize(calibration_params_path):
            raise Exception("Failed to load configuration file")

        self.player_params.initialize_cameras(self.camera_calibration)
        self.detector = player_pose.person_detector()
        self.detector.initialize(self.player_params, person_detector_model_path)
        self.detector.set_callback(self.on_detection)

        self.last_frame = 0
        self.detections = {}

    def on_detection(self, detection):
        """On player detection callback"""
        self.detections[detection.camera_index] = detection

    def destroy(self):  # pylint: disable = useless-super-delegation
        """
        Destroy the object
        """
        super().destroy()

    def _on_image_arrived(self, index, image, topic_name):
        cam_index = 0
        for cam_index, camera in enumerate(self.camera_names):
            if camera in topic_name:
                break
        if cam_index >= len(self.camera_names):
            return

        self.detector.add_detection_request(0, index, image, True)

        detection = self.detections[cam_index] if cam_index in self.detections else None
        if detection is not None:
            bbox = np.array(detection.bbox, dtype=int)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

        super()._on_image_arrived(index, image, topic_name)


@click.command()
@click.option("--config", help="Config name", required=True, type=str)
def main(config):
    """Main entry point"""
    rclpy.init()
    context: PoseContext = PoseContext(config)

    while rclpy.ok() and not context.is_done:
        context.render(force_update=False)

    # Clean up.
    cv2.destroyAllWindows()
    context.destroy()
    print("Clean up successfully...")

    return 0


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
