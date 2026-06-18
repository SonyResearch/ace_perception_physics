# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""An example to run racket estimator in python"""

import os
import sys
import argparse
import cv2
import ace_yaml as yaml

import rclpy
import numpy as np
import colored_glog as glog
from ace_loggers import logger_pybind as logger
from ament_index_python.packages import get_package_share_directory
from aps.image_subscriber import ManagedDisplayImagesContext
import racket_pose_estimation.python_module as racket_pose_estimation


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


set_level(0)


class PoseContext(ManagedDisplayImagesContext):
    """
    Draw camera images helper class
    """

    def __init__(self, config, cluster, version):
        # Load camera calibration.
        self.config = config

        calibration_params_path = get_package_share_directory("calibration")
        calibration_params_path = os.path.join(
            calibration_params_path,
            "parameters",
            "camera_calibration",
            config + ".yaml",
        )

        racket_params_path = get_package_share_directory("racket_pose_estimation")
        racket_params_path = os.path.join(
            racket_params_path,
            "parameters",
            config + cluster + ".yaml",
        )

        model_path = get_package_share_directory("racket_pose_estimation")
        model_path = os.path.join(
            model_path,
            "models",
            config + "_" + version + ".onnx",
        )
        if not os.path.isfile(model_path):
            model_path = os.path.join(
                model_path,
                "models",
                "racket_" + version + ".onnx",
            )
        camera_mask = []
        with open(racket_params_path, "r", encoding="utf8") as file:
            nodes = yaml.load(file)
            rackets = nodes["rackets"]
            for racket in rackets:
                for cam_roi in rackets[racket]["cameras"]:
                    camera_mask.append(cam_roi)

        self.features = None
        self.last_frame = 0
        super().__init__(config, 200, 200, camera_mask=camera_mask)

        self.detector = racket_pose_estimation.racket_pose_extractor(f"racket_pose_{config}{cluster}")
        self.detector.start(calibration_params_path, racket_params_path, model_path, True)

        self.node = racket_pose_estimation.racket_pose_ros_node()
        self.node.start("racket_pose_ros_node", self.detector)

    def destroy(self):  # pylint: disable = useless-super-delegation
        """
        Destroy the object
        """
        super().destroy()
        self.detector.stop()

    def fetch_detections(self):
        """Fetch detections from the detector"""
        frames = self.detector.get_detections()
        if frames and len(frames) > 0:
            self.features = frames[-1].features
            self.last_frame = frames[-1].sequence_number

    def _on_image_arrived(self, index, image, topic_name):
        cam_index = 0
        for cam_index, camera in enumerate(self.camera_calibration.cameras):
            if camera in topic_name:
                break
        if cam_index >= len(self.camera_calibration.cameras):
            return

        # print(f"Mapping: {topic_name} to {cam_index}")

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
                bbox = np.array(detection.bbox, dtype=int)
                kpt_int = np.array(detection.keypoints, dtype=int)
                center_int = np.array(detection.center, dtype=int)

                roi = self.detector.get_roi(cam_index)
                if roi is not None:
                    cv2.rectangle(image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 255), 4)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
                cv2.circle(image, (center_int[0], center_int[1]), radius=2, thickness=3, color=(255, 255, 255))
                print(kpt_int)
                cv2.line(
                    image,
                    kpt_int[0][:2],
                    kpt_int[1][:2],
                    (255, 0, 0),
                    thickness=2,
                )
                cv2.line(
                    image,
                    kpt_int[2][:2],
                    kpt_int[3][:2],
                    (0, 0, 255),
                    thickness=2,
                )
        super()._on_image_arrived(index, image, topic_name)


def main(config, cluster, version):
    """Main entry point"""
    rclpy.init()
    context: PoseContext = PoseContext(config, cluster, version)

    while rclpy.ok() and not context.is_done:
        context.render(force_update=False)
        context.fetch_detections()

    # Clean up.
    cv2.destroyAllWindows()
    context.destroy()
    print("Clean up successfully...")

    return 0


if __name__ == "__main__":
    # setup argument list
    parser = argparse.ArgumentParser(description="Evaluate APS calibration.", usage="")
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        help="Lab specific configuration, e.g. zrh00, zrh01, tyo01.",
        required=True,
    )
    parser.add_argument(
        "--cluster",
        dest="cluster",
        type=str,
        default="",
        help="cluster name.",
        required=False,
    )
    parser.add_argument(
        "--version",
        dest="version",
        type=str,
        default="v11n",
        help="network version",
        required=False,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    # parse argument list
    try:
        parsed = parser.parse_args()
    except:  # pylint: disable = bare-except
        sys.exit(0)
    main(parsed.config, parsed.cluster, parsed.version)
