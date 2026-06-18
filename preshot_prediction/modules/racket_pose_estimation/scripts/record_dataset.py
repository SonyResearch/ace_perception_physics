# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""
This script used to record images to be used for training racket pose detection model.
It requires to have images published along with a vive tracker based racket.
Make sure to use the correct configuration file for the cameras while running this script,
as it will project the racket to image space to construct the keypoints used in the dataset.
"""

import argparse
import sys
import os
import shutil
import json
import numpy as np
import cv2

from nvjpeg import NvJpeg
import colored_glog as glog
from ace_loggers import logger_pybind as logger
from ament_index_python.packages import get_package_share_directory
from helpers.racket_db_common import Context


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


class TrainingContext(Context):
    """Helper class"""

    def create_pose_detector(self, config):
        """Create a pose detector"""
        from calibration import python_module as calibration  # pylint: disable=import-outside-toplevel
        import player_pose.python_module as player_pose  # pylint: disable=import-outside-toplevel

        player_params_path = os.path.join(
            get_package_share_directory("player_pose"),
            "parameters",
            config + ".yaml",
        )
        calibration_params_path = os.path.join(
            get_package_share_directory("calibration"),
            "parameters",
            "camera_calibration",
            config + ".yaml",
        )

        person_detector_model_path = os.path.join(
            get_package_share_directory("player_pose"),
            "models",
            "yolov8n" + ".onnx",
        )
        print(f"Loading player parameters from: {player_params_path}")
        self.player_params = player_pose.PlayerParameters()
        if not self.player_params.initialize(player_params_path):
            raise Exception("Failed to load configuration file")

        self.camera_calibration = calibration.CameraCalibrationParameters()
        if not self.camera_calibration.initialize(calibration_params_path):
            raise Exception("Failed to load configuration file")
        print("Initializing player pose")
        self.player_params.initialize_cameras(self.camera_calibration)

        self.detector = player_pose.person_detector()
        self.detector.initialize(self.player_params, person_detector_model_path)

    def __init__(self, config, path, detection_threshold, rec_type, start_index, margin, remove_old=False):
        super().__init__(detection_threshold, config, margin)

        self.seq_id = start_index
        self.seq_counter = start_index
        self.rec_type = rec_type

        self.db_path = path
        self.jpeg_encoder = NvJpeg()

        os.makedirs(self.db_path, exist_ok=True)

        self.create_pose_detector(config)

        self.rcnn_format = False
        self.append_keypoints = True
        self.append_position = False
        self.append_angles = True
        if self.rcnn_format:
            self.img_path = os.path.join(self.db_path, self.rec_type, "images", "")
            self.label_path = os.path.join(self.db_path, self.rec_type, "labels", "")
            self.annotation_path = os.path.join(self.db_path, self.rec_type, "annotations", "")
            self.pose_path = os.path.join(self.db_path, self.rec_type, "pose", "")
        else:
            self.img_path = os.path.join(self.db_path, "images", self.rec_type, "")
            self.label_path = os.path.join(self.db_path, "labels", self.rec_type, "")
            self.annotation_path = os.path.join(self.db_path, "annotations", self.rec_type, "")
            self.pose_path = os.path.join(self.db_path, "pose", self.rec_type, "")

        print("Creating: ", self.img_path)
        print("Creating: ", self.annotation_path)
        print("Creating: ", self.label_path)
        print("Creating: ", self.pose_path)

        if remove_old:
            shutil.rmtree(os.path.dirname(self.img_path), ignore_errors=True)
            shutil.rmtree(os.path.dirname(self.annotation_path), ignore_errors=True)
            shutil.rmtree(os.path.dirname(self.label_path), ignore_errors=True)
            shutil.rmtree(os.path.dirname(self.pose_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.img_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.annotation_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.label_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.pose_path), exist_ok=True)

    def detect_person(self, camera_index, image):
        """Detect a person in an image"""
        det = self.detector.extract_person(image, camera_index, True)
        if det:
            return det.bbox
        return None

    @classmethod
    def prepare_image(cls, image):
        """Prepare an image for inference"""
        target = np.zeros((640, 640, 3), dtype=np.uint8)
        width = min(target.shape[0], image.shape[0])
        height = min(target.shape[1], image.shape[1])
        target[:width, :height, :] = image[:width, :height, :]
        return target

    # pylint: disable=too-many-branches,too-many-statements
    def write_data(self, seq_id, image, camera_index, racket_ifo):
        """Write data to files"""
        if np.random.rand() > 0.08 or len(racket_ifo) == 0 or image is None:
            return  # skip
        print(camera_index)
        detection_bbox = self.detect_person(camera_index, image)
        if detection_bbox is None:
            return
        img_path = os.path.join(self.img_path, f"{self.seq_counter:09d}.jpg")
        annotation_path = os.path.join(self.annotation_path, f"{self.seq_counter:09d}.json")
        label_path = os.path.join(self.label_path, f"{self.seq_counter:09d}.txt")
        pose_path = os.path.join(self.pose_path, f"{self.seq_counter:09d}.txt")

        print(f"Writing: {self.seq_counter}")
        self.seq_counter += 1

        annotation_results = {"bboxes": [], "keypoints": []}

        write_image = not self.rcnn_format

        if write_image:
            if detection_bbox is not None:
                image = image[
                    detection_bbox[1] : detection_bbox[1] + detection_bbox[3],
                    detection_bbox[0] : detection_bbox[0] + detection_bbox[2],
                ]
                image = TrainingContext.prepare_image(image)

        with open(label_path, "wt", encoding="utf8") as label_file:
            for racket in racket_ifo:
                racket_bbox = racket["bbox"]
                keypoints = racket["keypoints"]
                if racket_bbox is None:
                    continue

                racket_bbox[0] -= detection_bbox[0]
                racket_bbox[2] -= detection_bbox[0]
                racket_bbox[1] -= detection_bbox[1]
                racket_bbox[3] -= detection_bbox[1]

                racket_bbox[0] = max(0, racket_bbox[0])
                racket_bbox[1] = max(0, racket_bbox[1])
                racket_bbox[2] = min(image.shape[0] - 1, racket_bbox[2])
                racket_bbox[3] = min(image.shape[1] - 1, racket_bbox[3])

                center_x = (racket_bbox[0] + racket_bbox[2]) / 2
                center_y = (racket_bbox[1] + racket_bbox[3]) / 2
                width = racket_bbox[2] - racket_bbox[0]
                height = racket_bbox[3] - racket_bbox[1]

                center_x = center_x / image.shape[1]
                center_y = center_y / image.shape[0]
                width = width / image.shape[1]
                height = height / image.shape[0]

                for keypoint in keypoints:
                    keypoint[0] -= detection_bbox[0]
                    keypoint[1] -= detection_bbox[1]

                keypoints_bbox = cv2.boundingRect(
                    np.array([[x[0], x[1]] for x in keypoints], dtype=np.float32).astype(int)
                )

                # pylint: disable=too-many-boolean-expressions
                if (
                    center_x + width / 2 > 1
                    or center_y + height / 2 > 1
                    or width < 0
                    or height < 0
                    or racket_bbox[0] < 0
                    or racket_bbox[1] < 0
                    or keypoints_bbox[0] < 0
                    or keypoints_bbox[1] < 0
                    or (keypoints_bbox[0] + keypoints_bbox[2]) >= image.shape[1]
                    or (keypoints_bbox[1] + keypoints_bbox[3]) >= image.shape[0]
                ):
                    print(center_x, center_y, width, height, racket_bbox[0], racket_bbox, keypoints_bbox)
                    continue
                if self.rcnn_format:
                    annotation_results["bboxes"].append(racket_bbox)
                    annotation_results["keypoints"].append(keypoints)

                line = f"0 {center_x} {center_y} {width} {height} "
                if self.append_keypoints:
                    print(keypoints)
                    for keypoint in keypoints:
                        line += f"{keypoint[0]/image.shape[1]} {keypoint[1]/image.shape[0]} {keypoint[2]} "

                if self.append_position:
                    pos = racket["pos"]  # -racket_bbox[:2]
                    pos /= (image.shape[1], image.shape[0])
                    line += f"{pos[0]} {pos[1]} "
                if self.append_angles:
                    orientation = (racket["rotation"] + 180) / 360.0
                    line += f"{orientation[0]} {orientation[1]} {orientation[2]} "

                label_file.write(f"{line}\r\n")

        if write_image:
            jpg_data = self.jpeg_encoder.encode(image, 90)
            with open(img_path, "wb") as file:  # pylint: disable=unspecified-encoding
                file.write(jpg_data)

        if len(annotation_results["bboxes"]) > 0:
            with open(annotation_path, "wt", encoding="utf8") as annotation_file:
                annotation_file.write(json.dumps(annotation_results))
                write_image = True

        with open(pose_path, "wt", encoding="utf8") as label_file:
            for racket in racket_ifo:
                racket_bbox = racket["bbox"]
                if racket_bbox is None:
                    continue
                pos = racket["pos"] - racket_bbox[:2]
                pos /= np.subtract(racket_bbox[2:4], racket_bbox[0:2])
                orientation = (racket["rotation"] + 180) / 360.0
                label_file.write(f"{pos[0]}, {pos[1]}, {orientation[0]}, {orientation[1]}, {orientation[2]}\r\n")


def main(config, path, detection_threshold, rec_type, start_index, margin, remove_old):
    """Main entry point"""

    context: TrainingContext = TrainingContext(
        config, path, detection_threshold, rec_type, start_index, margin, remove_old
    )

    context.run()
    context.destroy()

    return 0


if __name__ == "__main__":
    USAGE = """
    """
    # setup argument list
    parser = argparse.ArgumentParser(description="Evaluate APS calibration.", usage=USAGE)
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        help="Lab specific configuration, e.g. zrh00, zrh01, tyo01.",
        required=True,
    )
    parser.add_argument(
        "--path",
        dest="path",
        type=str,
        help="Record path.",
        required=True,
    )
    parser.add_argument(
        "--type",
        dest="type",
        type=str,
        default="train",
        help="Type (train/val).",
        required=False,
    )
    parser.add_argument(
        "--detection_threshold",
        dest="detection_threshold",
        type=float,
        default=0.02,
        help="Detection Threshold for racket bbox.",
        required=False,
    )
    parser.add_argument(
        "--start_index",
        dest="start_index",
        type=int,
        default=0,
        help="Start index.",
        required=False,
    )
    parser.add_argument(
        "--margin",
        dest="margin",
        type=int,
        default=20,
        help="Margin in pixels.",
        required=False,
    )
    parser.add_argument(
        "--remove-old",
        dest="remove_old",
        action="store_true",
        help="Remove old files.",
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

    main(
        parsed.config,
        parsed.path,
        parsed.detection_threshold,
        parsed.type,
        parsed.start_index,
        parsed.margin,
        parsed.remove_old,
    )
