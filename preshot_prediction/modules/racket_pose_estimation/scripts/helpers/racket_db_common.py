# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""
Helper class to subscribe to racket poses
"""

import os
import time
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration
from aps import python_module as aps
from helpers.racket_info import RacketInfo
from python_helpers.keyboard_input import KBHit


class Context:
    """Helper class"""

    def __init__(self, detection_threshold, config, margin):
        """Init"""
        self.detection_threshold = detection_threshold
        self.config = config
        self.margin = margin

        self.bbox_set = {}
        rclpy.init()
        self.dummy_node = Node("racket_db_builder")
        self.keyboard = KBHit()

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

        self.num_cameras = len(self.camera_calibration.cameras)
        self.camera_names = [None] * self.num_cameras
        self.camera_matrix = [None] * self.num_cameras
        self.cameras = [None] * self.num_cameras

        self.racket_states = [None] * self.num_cameras

        for i, camera in enumerate(self.camera_calibration.cameras):
            self.camera_names[i] = camera
            self.cameras[i] = self.camera_calibration.cameras[camera]
            self.camera_matrix[i] = self.cameras[i].T_world_camera

        self.sub = aps.ZeroCopySubscriber()
        self.handles = [None] * self.num_cameras
        for i in range(self.num_cameras):
            topic = "/sensors/" + self.camera_names[i] + "/image"
            self.handles[i] = self.sub.add_subscription(topic)
            print("Subscribing to: ", topic)
            if self.handles[i] < 0:
                print(f'Failed to subscribe to topic "{topic}"')

        self.rets = [None] * self.num_cameras
        self.frame_ids = [None] * self.num_cameras
        self.imgs_updated = [None] * self.num_cameras
        self.last_frame_ids = [None] * self.num_cameras
        self.encodings = [None] * self.num_cameras
        self.imgs_raw = [None] * self.num_cameras
        self.imgs_bgr = [None] * self.num_cameras
        self.exported = [True] * self.num_cameras

        self.rackets = []
        self.rackets_history = {}

        self.rackets.append(RacketInfo("/sensors/racket_vive0/pose", self))
        self.rackets.append(RacketInfo("/sensors/racket_vive1/pose", self))
        # self.rackets.append(RacketInfo("/sensors/racket0/pose", self))
        # self.rackets.append(RacketInfo("/sensors/racket1/pose", self))

        self.records = {}

        self.is_done = False
        self.image_queue = []

        self.seq_id = 0
        self.pause_recording = False

        self.thread = threading.Thread(target=self._working_thread)
        self.thread.start()

    def destroy(self):
        """
        Destroy the object
        """
        self.is_done = True
        self.handles.clear()
        self.keyboard.set_normal_term()

    def write_data(self, seq_id, image, camera_index, racket_ifo):  # pylint: disable=unused-argument, no-self-use
        """called when new data is available"""
        return

    def _working_thread(self):
        total_written = 0
        print("_working_thread")

        while not self.is_done or len(self.image_queue) > 0:
            if len(self.image_queue) == 0:
                time.sleep(0.005)
                continue
            meta = self.image_queue.pop(0)
            total_written += 1
            seq_id = meta[0]
            img = meta[1]
            racket = meta[2]
            camera_index = meta[3]
            if not self.pause_recording:
                self.write_data(seq_id, img, camera_index, racket)
            time.sleep(0.005)
        print(f"Thread finished. Total written images: {total_written}")

    def project_point(self, camera_idx, point, p2d):
        """Project point to camera space"""
        camera = self.camera_calibration.cameras[self.camera_names[camera_idx]]
        return camera.project_point(np.float32(point), p2d)

    @classmethod
    def get_bbox_size(cls, bbox):
        """Get bbox size"""
        if bbox is None:
            return 0
        size = bbox[2] * bbox[3]
        return size

    def update_racket(self, racket: RacketInfo, img_id, pose=None):
        """Update racket"""
        if racket.pose is None or self.imgs_bgr[img_id] is None:
            return False
        position = racket.pose[0]
        rotation = racket.pose[2]
        p3d = position + np.matmul(rotation, [0, 0, 0.1])
        p2d = np.zeros((2, 1), dtype=np.float32)

        if self.project_point(img_id, p3d, p2d):
            bbox = racket.project_key_points(
                self.camera_calibration.cameras[self.camera_names[img_id]],
                self.imgs_bgr[img_id],
                self.margin,
                pose=pose,
            )
            if (
                bbox is None
                or bbox[2] < 0
                or bbox[3] < 0
                or Context.get_bbox_size(bbox)
                < self.detection_threshold * self.imgs_bgr[img_id].shape[0] * self.imgs_bgr[img_id].shape[1]
            ):
                return False

            racket_dict = racket.convert_coords(self.cameras[img_id], pose=pose)
            if racket_dict is None:
                return False
            racket_dict["bbox"] = bbox
            self.racket_states[img_id].append(racket_dict)
            return True
        return False

    def on_racket_updated(self, racket: RacketInfo):
        """Refresh images when racket is updated"""
        if racket.topic not in self.rackets_history:
            self.rackets_history[racket.topic] = {}
        self.rackets_history[racket.topic][racket.sequence_id] = racket.pose
        self.update_images()

    def update_images(self):
        """Update images"""
        # Process all images
        for i in range(self.num_cameras):
            if self.imgs_updated[i]:
                continue

            self.imgs_updated[i] = True
            self.racket_states[i] = []
            racket: RacketInfo
            for racket in self.rackets:
                if racket.topic in self.rackets_history:
                    to_remove = []
                    for racket_id in self.rackets_history[racket.topic]:
                        if racket_id < self.frame_ids[i] - 10:
                            to_remove.append(racket_id)
                    for key in to_remove:
                        self.rackets_history[racket.topic].pop(key)

                if racket.topic in self.rackets_history and self.frame_ids[i] in self.rackets_history[racket.topic]:
                    self.update_racket(racket, i, self.rackets_history[racket.topic][self.frame_ids[i]])
                    if not self.exported[i]:
                        self.exported[i] = True
                        self.image_queue.append([self.seq_id, np.copy(self.imgs_bgr[i]), self.racket_states[i], i])
                        self.seq_id += 1

    def fetch_images(self):
        """
        Fetch the images
        """
        try:
            rclpy.spin_once(self.dummy_node, timeout_sec=0.1)
        except Exception as err:  # pylint: disable=broad-except
            print(err)
        for handle in self.handles:
            self.sub.release_latest_message(handle)
        # Get latest image from each camera.
        for i in range(self.num_cameras):
            (
                self.rets[i],
                self.frame_ids[i],
                self.encodings[i],
                self.imgs_raw[i],
            ) = self.sub.get_latest_message(self.handles[i])
        # Process all images
        for i in range(self.num_cameras):
            if self.rets[i] and self.frame_ids[i] != self.last_frame_ids[i]:
                self.exported[i] = False
                self.last_frame_ids[i] = self.frame_ids[i]
                if self.encodings[i] == "bayer_rggb8":
                    self.imgs_bgr[i] = cv2.cvtColor(self.imgs_raw[i], cv2.COLOR_BayerBG2BGR)
                elif self.encodings[i] == "mono8":
                    self.imgs_bgr[i] = cv2.cvtColor(self.imgs_raw[i], cv2.COLOR_GRAY2BGR)
                else:
                    print("Received unknown encoding: ", self.encodings[i])

                self.imgs_updated[i] = False

        self.update_images()

    def process_key(self, key):
        """Process KB key"""
        try:
            if key == " ":
                self.pause_recording = not self.pause_recording
                if self.pause_recording:
                    print("Pausing recording")
                else:
                    print("Resuming recording")
        except Exception as err:  # pylint: disable = broad-except
            print(err)

    def run(self):
        """Run renderer"""
        # Main loop.
        while rclpy.ok():
            if self.keyboard.kbhit():
                key = self.keyboard.getch()
                self.process_key(key)
            self.fetch_images()
        self.destroy()
        print("Clean up successfully...")
