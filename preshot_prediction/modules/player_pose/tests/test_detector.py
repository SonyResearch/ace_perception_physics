#!/usr/bin/env python3
"""
Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

import os
import sys
import time
import numpy as np

import pytest
import cv2
import colored_glog as glog
from scipy.spatial.transform import Rotation as R
from ace_loggers import logger_pybind as logger
from ament_index_python.packages import get_package_share_directory
import player_pose.python_module as player_pose
from calibration import python_module as calibration


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


@pytest.mark.skip(
    reason="disabling this unit test as it requires cuda capable gpu for testing, and CI servers don't have"
)
def test_detector():
    config = "tyo01"

    package_path = get_package_share_directory("player_pose")
    calibration_params_path = os.path.join(
        package_path,
        "tests",
        "config",
        config + "_calib.yaml",
    )

    player_params_path = os.path.join(
        package_path,
        "tests",
        "config",
        config + "_player.yaml",
    )
    model_path = os.path.join(
        package_path,
        "models",
        "yolov11n.onnx",
    )

    test_image_path = os.path.join(
        package_path,
        "tests",
        "images",
        # "000000007.jpg",
        "img_2.bayer8.npy",
    )

    camera_params = calibration.CameraCalibrationParameters()
    camera_params.initialize(calibration_params_path)

    player_params = player_pose.PlayerParameters()
    player_params.initialize(player_params_path)
    player_params.initialize_cameras(camera_params)

    detector = player_pose.person_detector()
    detector.initialize(player_params, model_path)
    # image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)

    image_raw = np.load(test_image_path)

    image = cv2.cvtColor(image_raw, cv2.COLOR_BayerBG2BGR)

    detection = detector.extract_person(image, 0, True)

    # print(detector.on_images(0, [[0, image_raw]], False))
    # for d in cb.detection.keypoints:
    #     cv2.circle(image, center=(int(d[0]), int(d[1])), radius=2, color=[0, 255, 0])
    cv2.rectangle(
        image,
        pt1=(int(detection.bbox[0]), int(detection.bbox[1])),
        pt2=(int(detection.bbox[0] + detection.bbox[2]), int(detection.bbox[1] + detection.bbox[3])),
        color=[0, 255, 0],
        thickness=1,
    )
    cv2.imshow("result", image)
    cv2.waitKey(0)
    # assert cb.detection is not None


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
