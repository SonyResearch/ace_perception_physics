"""
ROS bag processing o extract human poses
"""
# pylint: disable = line-too-long, fixme, invalid-name, too-many-branches
# Confidential, Copyright 2024, Sony AI, All rights reserved.

import os
import time
import argparse
import threading
import queue
from typing import List, Dict
from functools import partial
import numpy as np
import colored_glog as glog

import rclpy

import rosbag_dataloaders.dataloaders
from nvjpeg import NvJpeg

from ament_index_python.packages import get_package_share_directory
from ace_loggers import logger_pybind as logger
from calibration import python_module as calibration
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


def rgb2bayer(bgr) -> np.ndarray:
    """implementation for rgb2bayer
    # https://github.com/guochengqian/TENet/issues/5
    Converts from a RGB representation to Bayer8 representation
    """
    bayer = np.zeros((bgr.shape[0], bgr.shape[1]))
    bayer[0::2, 1::2] = bgr[0::2, 1::2, 1]
    bayer[0::2, 0::2] = bgr[0::2, 0::2, 2]
    bayer[1::2, 1::2] = bgr[1::2, 1::2, 0]
    bayer[1::2, 0::2] = bgr[1::2, 0::2, 1]
    bayer = bayer.astype(np.uint8)
    return bayer.reshape((bayer.shape[0], bayer.shape[1], 1))


class CameraConfigurations:
    """Camera configurations object"""

    def __init__(self, config):
        self.config = config
        calibration_params_path = get_package_share_directory("calibration")
        calibration_params_path = os.path.join(
            calibration_params_path,
            "parameters",
            "camera_calibration",
            config + ".yaml",
        )
        self.camera_calibration = calibration.CameraCalibrationParameters()
        if not self.camera_calibration.initialize(calibration_params_path):
            raise Exception("Failed to load configuration file")

        # Compute undistortion maps.
        self.num_cameras = len(self.camera_calibration.cameras)
        self.camera_names = [None] * self.num_cameras
        self.camera_matrix = [None] * self.num_cameras
        self.imgs_updated = []
        self.display_imgs = []
        for i, camera in enumerate(self.camera_calibration.cameras):
            self.camera_names[i] = camera


class ImageLoaderThread(threading.Thread):
    """Multhreaded image loader"""

    def __init__(self, picture_queue, processing_queue):
        self.picture_queue = picture_queue
        self.processing_queue = processing_queue
        self.url = None
        self.sequence_id = 0
        self.camera_id = 0
        self.is_done = False
        self.jpeg_encoder = NvJpeg()
        super().__init__()

    def run(self):
        while not self.is_done:
            if not self.url:
                time.sleep(0.001)
                continue
            self._get_picture()

    def is_ready(self):
        """Check if the job is ready to take new request"""
        return self.url is None

    def _get_picture(self):
        # --- get your picture --- #
        try:
            with open(self.url, "rb") as file:
                picture = self.jpeg_encoder.decode(file.read())
            # picture=rgb2bayer(picture)
            self.picture_queue.put([self.sequence_id, self.camera_id, picture])
        except:  # pylint: disable=bare-except
            pass
        self.processing_queue[self.sequence_id].remove(self.camera_id)
        self.url = None

        # print(f"Removing from processing queue: {self.sequence_id}:{self.camera_id}")

    def assign_url(self, sequence_id, camera_id, url):
        """Assign a url to this loader"""
        if self.url is not None:
            return False
        self.url = url
        self.camera_id = camera_id
        self.sequence_id = sequence_id

        if sequence_id not in self.processing_queue:
            self.processing_queue[sequence_id] = []
        self.processing_queue[sequence_id].append(camera_id)
        # print(f"Adding to processing queue: {self.sequence_id}:{self.camera_id}")
        return True


class RosbagParser:
    """Parse a rosbag and process it to extract human poses"""

    def __init__(self, path, image_path, config) -> None:
        self.image_path = image_path

        self.camera_config = CameraConfigurations(config)
        rosbag_parser = rosbag_dataloaders.dataloaders.get_parser(path)
        curr_topics = [
            (topic_name)
            for topic_name, topic_type in rosbag_parser.get_topics_with_types()
            if topic_type in "ace_interfaces/msg/FileImage"
        ]
        messages = [
            {message.header.sequence_number: message for timestamp, message in rosbag_parser.get_messages(topic_name)}
            for topic_name in curr_topics
        ]
        self.num_topics = len(curr_topics)

        self.min_sequence_number = min(
            [
                min(messages[topic_index].keys()) if len(messages[topic_index].keys()) > 0 else 9999
                for topic_index in range(self.num_topics)
            ]
        )
        self.max_sequence_number = max(
            [
                max(messages[topic_index].keys()) if len(messages[topic_index].keys()) > 0 else 0
                for topic_index in range(self.num_topics)
            ]
        )
        self.synced_messages = {
            sequence_number: [messages[topic_index].get(sequence_number) for topic_index in range(self.num_topics)]
            for sequence_number in range(self.min_sequence_number, 1 + self.max_sequence_number)
        }

        self.curr_sequence_number = self.min_sequence_number
        self.picture_queue: queue.Queue = queue.Queue(maxsize=0)
        self.picture_threads: List[threading.Thread] = []

        self.cached_images: Dict[int, list] = {}

        self.processing_queue: Dict[int, list] = {}
        self.sequence_id_list: List[int] = []

        self.last_percent = 0
        self.cache_size = 10

        for _ in range(20):
            thread = ImageLoaderThread(self.picture_queue, self.processing_queue)
            self.picture_threads.append(thread)
            thread.start()

    def load_image_async(self, sequence_id, camera_id, url):
        """Add a request to load image"""
        for thread in self.picture_threads:
            if thread.assign_url(sequence_id, camera_id, url):
                return True
        return False

    def destroy(self):
        """Destroy object"""
        # wait for threads to finish
        for picture_thread in self.picture_threads:
            picture_thread.is_done = True
            picture_thread.join()

    def is_done(self):
        """Check if the bag is done"""
        return self.max_sequence_number < self.curr_sequence_number

    def _load_image(self, seq_id, camera_id, path):
        try:
            img_path = os.path.join(self.image_path, path)
            if not os.path.isfile(img_path):
                return -1

            return 1 if self.load_image_async(seq_id, camera_id, img_path) else 0
        except:  # pylint: disable=bare-except
            return -1

    def _cache_frames(self):
        if self.max_sequence_number < self.curr_sequence_number:
            return False
        is_ok = True

        # pylint: disable= too-many-nested-blocks
        cached_count = 0
        while is_ok and cached_count < 10:
            appended_count = 0
            total_count = 0
            is_done = False
            if self.curr_sequence_number in self.synced_messages:
                messages = self.synced_messages[self.curr_sequence_number]
            else:
                messages = []
            for msg in messages:
                if msg is None:
                    continue
                for i, cam in enumerate(self.camera_config.camera_names):
                    if cam in msg.path:
                        total_count += 1
                        trials = 0
                        while trials < 10:
                            # print(msg.path)
                            ret = self._load_image(self.curr_sequence_number, i, msg.path)
                            if ret == 1:
                                # successfully appended
                                appended_count += 1
                                break
                            if ret == -1:  # image path is invalid
                                break
                            # otherwise, thread pool is full
                            # indicate its the last sequence to process and keep waiting until its added to the queue
                            is_done = True
                            trials += 1
                            time.sleep(0.005)
                        break
            if appended_count > 0:
                self.sequence_id_list.append(self.curr_sequence_number)
            elif total_count != 0:
                is_ok = False

            if is_done:
                is_ok = False
            self.curr_sequence_number += 1
            cached_count += 1
        return True

    def next_frame(self):
        """Get next frame"""
        if len(self.cached_images) < self.cache_size and not self._cache_frames():
            raise Exception("Finished playback")

        # print(f"Cached: {len(self.cached_images)}")

        while not self.picture_queue.empty():
            img = self.picture_queue.get()

            if img[0] not in self.cached_images:
                self.cached_images[img[0]] = []

            self.cached_images[img[0]].append((img[1], img[2]))

        if len(self.sequence_id_list) == 0:
            print("Empty seq list")
            return None, None

        seq_id = self.sequence_id_list[0]
        if len(self.processing_queue[seq_id]) != 0:
            # print("Processing not finished")
            return None, None

        percent = int(100 * (seq_id - self.min_sequence_number) / (self.max_sequence_number - self.min_sequence_number))
        if self.last_percent != percent and percent % 5 == 0:
            print(f"Processed: {percent}%")
        self.last_percent = percent
        # print(f"Next seq: {seq_id}={len(self.cached_images[seq_id])}")
        self.sequence_id_list.pop(0)

        imgs = self.cached_images[seq_id]
        self.cached_images.pop(seq_id)
        self.processing_queue.pop(seq_id)
        return seq_id, imgs


class PoseProcessor:
    """Pose Processor class"""

    def __init__(self, config, fast) -> None:
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
            config + ".yaml",
        )

        player_pose_package = get_package_share_directory("player_pose")
        model_name = "singlepose-lightning-tflite-float16.onnx" if fast else "singlepose-thunder-tflite-float16.onnx"
        movenet_model_path = os.path.join(
            player_pose_package,
            "models",
            model_name,  # movenet-lightning pose_landmark_heavy
        )
        person_detector_model_path = os.path.join(
            player_pose_package,
            "models",
            "yolov8n.onnx",
        )

        self.detector = player_pose.player_pose_extractor()
        self.detector.start(
            calibration_params_path,
            player_params_path,
            movenet_model_path,
            person_detector_model_path,
            player_pose.PlayerInferenceEngine.movenet,
        )

        self.total_processed = 0

        # def player_detection(seq_id, cam_idx, bbox, conf):
        #     print(f"Player detected: {seq_id}, {cam_idx}, {bbox}, {conf}")

        # def player_keypoints(seq_id, features):
        #     print(f"player keypoints: {seq_id}")

        def player_pose_callback(self: PoseProcessor, seq_id, pose):
            if pose.player_id not in self.poses:
                self.poses[pose.player_id] = []
            # print(f"Player pose: {seq_id}, {pose}")
            self.poses[pose.player_id].append([seq_id, pose])

        self.poses: Dict[int, list] = {}

        # self.detector.set_player_detection_callback(player_detection)
        # self.detector.set_player_keypoints_callback(player_keypoints)
        self.detector.set_player_pose_callback(partial(player_pose_callback, self))

    def process_frame(self, seq_id, imgs, timeout=0.002):
        """Add a request to process a frame"""
        if len(imgs) == 0:
            return False
        timeout_count = 0
        while timeout_count < 5:
            processed = self.detector.on_images(seq_id, imgs)
            if processed:
                break
            timeout_count += 1
            # print("workers not ready yet")
            time.sleep(timeout)
        if not processed:
            print("Failed to process: ", seq_id)
        else:
            self.total_processed += 1
        return processed

    def get_csv_line(self, seq_id, pose):  # pylint: disable = no-self-use
        """Get a csv line as an array"""
        line = [str(seq_id)]

        for key, conf, cov in zip(pose.keypoints, pose.confidences, pose.projection_error):
            line.append(str(key[0]))
            line.append(str(key[1]))
            line.append(str(key[2]))
            line.append(str(conf))
            line.append(str(cov))

        return line

    def export_csv(self, path):
        """Export to CSV file"""
        os.makedirs(path, exist_ok=True)
        print("Total processed: ", self.total_processed)
        for player, pose in self.poses.items():
            pose.sort(key=lambda x: x[0])  # sort by sequence id

            file_name = os.path.join(path, f"player_{player}.csv")
            print("Exporting to: ", file_name, " with frames count: ", len(pose))
            with open(file_name, "wt", encoding="utf8") as file:
                # header = ", ".join(self.csv_header)
                # file.write(header + "\n")

                for pos in pose:
                    seq_id = pos[0]
                    pose = pos[1]

                    line = self.get_csv_line(seq_id, pose)
                    file.write(", ".join(line) + "\n")

        print("CSV Exported")


def parse_args():
    """
    Provide script-specific arguments.

    @return: Parsed argument object
    """

    parser = argparse.ArgumentParser(description="Pose triangulation pipeline.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to ros bag",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to recorded images",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration name",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use fast models",
    )

    return parser.parse_args()


def main(args):
    """Main entry point"""
    rclpy.init()

    conf = args.config

    print("Loading from a ros bag")
    reader = RosbagParser(args.path, args.images, conf)

    processor = PoseProcessor(args.config, args.fast)

    success_list = 0
    failed_list = 0

    t1 = 0

    try:
        while rclpy.ok() and not reader.is_done():
            seq_id, imgs = reader.next_frame()
            if seq_id is None:
                # print("seq_id is none")
                time.sleep(0.001)
                continue
            if processor.process_frame(seq_id, imgs, timeout=0.02):
                success_list += 1
            else:
                failed_list += 1

            now = time.time()
            if now - t1 > 60:
                print(f"Success/Failed: {success_list}/{failed_list}")
                t1 = now
                success_list = failed_list = 0

    except Exception as e:  # pylint: disable = broad-except
        print(e)
    except:  # pylint: disable = bare-except
        pass

    print("Exporting..")
    time.sleep(3)  # wait until any buffered data is processed
    processor.export_csv("poses")
    reader.destroy()


if __name__ == "__main__":
    args = parse_args()
    main(args)
