#!/usr/bin/env python3
"""
Confidential, Copyright 2025, Sony AI, All rights reserved.
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
import racket_pose_estimation.python_module as racket_pose_estimation
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


def rgb2bayer(bgr):
    """implementation for rgb2bayer
    # https://github.com/guochengqian/TENet/issues/5
    Converts from a RGB representation to Bayer8 representation
    """
    bayer = np.zeros((bgr.shape[0], bgr.shape[1]))
    bayer[0::2, 1::2] = bgr[0::2, 1::2, 1]
    bayer[0::2, 0::2] = bgr[0::2, 0::2, 2]
    bayer[1::2, 1::2] = bgr[1::2, 1::2, 0]
    bayer[1::2, 0::2] = bgr[1::2, 0::2, 1]
    return bayer.astype(np.uint8)


@pytest.mark.skip(
    reason="disabling this unit test as it requires cuda capable gpu for testing, and CI servers don't have"
)
def test_detector():
    config = "tyo01"
    version = "v11n"

    package_path = get_package_share_directory("racket_pose_estimation")
    calibration_params_path = os.path.join(
        package_path,
        "tests",
        "config",
        config + "_calib.yaml",
    )

    racket_params_path = os.path.join(
        package_path,
        "parameters",
        config + ".yaml",
    )

    model_path = os.path.join(
        package_path,
        "models",
        "racket_" + version + ".onnx",
    )

    test_image_path = os.path.join(
        package_path,
        "tests",
        "images",
        # "000000007.jpg",
        "img_2.bayer8.npy",
    )
    detector = racket_pose_estimation.racket_pose_extractor("")

    assert detector.start(calibration_params_path, racket_params_path, model_path, True)

    if test_image_path.endswith(".npy"):
        image_raw = np.load(test_image_path)
        image = cv2.cvtColor(image_raw, cv2.COLOR_BayerBG2BGR)
    else:
        image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
        image_raw = rgb2bayer(image)

    class Result:
        def __init__(self) -> None:
            self.detection = None

        def callback(self, cam, detection):
            self.detection = detection
            print(self.detection.keypoints[0])

    cb = Result()

    detector.set_racket_detection_callback(cb.callback)
    detector.set_roi(0, [100, 100, 640, 640])
    print(detector.on_images(0, [[0, image_raw]], False))
    for d in cb.detection.keypoints:
        cv2.circle(image, center=(int(d[0]), int(d[1])), radius=2, color=[0, 255, 0])
    cv2.imshow("result", image)
    cv2.waitKey(0)
    assert cb.detection is not None


def test_fitter():
    def get_keypoints(cameras, rotation, cams, cov=0):
        """Project main keypoints to selected cameras"""
        orientation = rotation.as_matrix()
        points = []
        racket_width = 0.3 / 2
        racket_length = 0.4 / 2
        model_points = np.array(
            [
                [0, 0, -racket_length - 0.1],
                [0, 0, +racket_length],
                [0, -racket_width, 0],
                [0, +racket_width, 0],
            ]
        )

        for cam in cams:
            kps = []
            for point in model_points:
                p3d = np.matmul(orientation, point)
                p2d = np.array((2, 1), np.float32)
                cameras[cam].project_point(p3d, p2d)
                kp = [0, 0, 100]
                kp[:2] = p2d + ((np.random.rand(2) * 2 - 1) * cov).astype(int)
                kps.append(kp)
            points.append(kps)

        return points

    def solve_for(pos, fitter, cameras, target, cameras_indicies, cov=0):
        """Solve orientation fitting"""
        points = get_keypoints(cameras, target, cameras_indicies, cov)
        fitter.set_cameras(cameras_indicies)
        return fitter.fit_points(pos, points)

    config = "tyo02"

    package_path = get_package_share_directory("racket_pose_estimation")
    calibration_params_path = os.path.join(
        package_path,
        "tests",
        "config",
        config + "_calib.yaml",
    )
    racket_params_path = os.path.join(
        package_path,
        "tests",
        "config",
        config + "_racket.yaml",
    )

    camera_calibration = calibration.CameraCalibrationParameters()
    if not camera_calibration.initialize(calibration_params_path):
        raise Exception("Failed to load configuration file")

    racket_params = racket_pose_estimation.RacketParameters()
    if not racket_params.initialize(racket_params_path):
        raise Exception("Failed to load configuration file")
    num_cameras = len(camera_calibration.cameras)
    cameras = [None] * num_cameras

    for i, camera in enumerate(camera_calibration.cameras):
        cameras[i] = camera_calibration.cameras[camera]

    fitter = racket_pose_estimation.RacketPoseFitter()
    fitter.initialize(racket_params)
    fitter.set_calibration(camera_calibration)

    cov = 0

    np.random.seed(0)

    ref = [
        [139, 17, -163],
        [111, 8, -153],
        [107, -1, -158],
        [110, -9, -163],
        [111, -8, -164],
        [113, -6, -164],
        [115, -5, -164],
        [116, -4, -164],
        [118, -3, -165],
        [119, -3, -166],
        [121, -3, -168],
        [123, -4, -169],
        [125, -5, -172],
        [127, -5, -174],
        [129, -6, -176],
        [130, -6, -178],
        [131, -7, -179],
    ]

    errors = [999] * len(ref)

    last_error = 9999

    for cam_count in range(len(cameras)):
        cameras_indicies = list(range(cam_count + 1))
        initial = [1, 0, 0, 0]
        fitter.reset(initial[0], initial[1], initial[2], initial[3])
        for index, target in enumerate(ref):
            target_rotation = R.from_euler("xyz", target, degrees=True)
            rot, iter, e1 = solve_for(
                pos=[0, 0, 0],
                fitter=fitter,
                cameras=cameras,
                cameras_indicies=cameras_indicies,
                target=target_rotation,
                cov=cov,
            )
            rot = R.from_quat(rot)

            err = np.linalg.norm((rot * target_rotation.inv()).as_rotvec(True))
            # print(rot.as_euler("xyz", degrees=True), e1, err)

            errors[index] = np.linalg.norm(err)

        current_err = np.average(errors)
        print(current_err)
        assert current_err < last_error * (1 + 1e-1)
        last_error = current_err


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
