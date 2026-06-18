# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""Image subscriber helper classes, used to subscribe to camera images"""

import os
import time
import threading
from math import ceil
from typing import List

import cv2
import numpy as np

# ros
import rclpy
from rclpy.node import Node


# camera
from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration
from aps import python_module as aps
from python_helpers.screen import (
    get_screen_resolution,
    compute_optimal_image_arrangement,
)


class FPSCalculator:
    """
    Calculates frame rate
    """

    def __init__(self) -> None:
        self.fps_ = 0
        self.last_time_ = time.time()
        self.frame_acc_ = 0

    def register(self):
        """Register a new frame"""
        self.frame_acc_ += 1

    def update(self):
        """Update FPS calculation"""
        now = time.time()
        delta = now - self.last_time_
        if delta > 1:
            self.fps_ = ceil(self.frame_acc_ / delta)
            self.last_time_ = now
            self.frame_acc_ = 0

    def fps(self):
        """Get latest FPS"""
        return self.fps_

    def delta_time(self):
        """Get latest delta-time"""
        return 1.0 / self.fps_ if self.fps_ > 0 else None


class ImageSubContext:
    """
    This class manages the context of show camera utility tool, and manages the population of image topics
    """

    def __init__(self, rate, decode_bgr, fps=200, node_name="record_images", camera_mask=None):
        self.ros_node = Node(node_name)
        self.topics = {}
        self.rate = rate
        self.decode_bgr = decode_bgr
        self.timeout = max(1, int(fps / rate))  # assuming cameras are running at 200Hz
        self.camera_mask = camera_mask
        print("Timeout: ", self.timeout)
        time.sleep(1)  # Wait to populate topics
        # Set up subscriber.
        self.sub = aps.ZeroCopySubscriber()  # pylint: disable=c-extension-no-member

        self.rets = []
        self.frame_ids = []
        self.last_frame_ids = []
        self.registered_frame_ids = []
        self.encodings = []
        self.imgs_raw = []
        self.imgs = []

        self.fps_: List[FPSCalculator] = []

        self.image_queue = []

        self.is_done = False
        self.updated = False

        self.processing_threads = threading.Thread(target=self._fetch_images)
        self.processing_threads.start()

    def _fetch_images(self):
        time.sleep(1)
        while not self.is_done:
            self.refresh_topics()
            self.update()
            time.sleep(0.001)

    def destroy(self):
        """
        Destroy the object
        """

        self.is_done = True
        self.ros_node.destroy_node()

    def _on_image_arrived(self, index, image, topic_name):
        pass

    def update(self, **kwargs):
        """
        Update is called when new topics were populated
        """

        for i, topic_name in enumerate(self.topics):
            self.fps_[i].update()
            topic = self.topics[topic_name]
            (
                self.rets[i],
                self.frame_ids[i],
                self.encodings[i],
                self.imgs_raw[i],
            ) = self.sub.get_latest_message(topic["handle"])
            if self.rets[i] and self.frame_ids[i] != self.registered_frame_ids[i]:
                self.registered_frame_ids[i] = self.frame_ids[i]
                self.fps_[i].register()

            if self.rets[i] and self.frame_ids[i] != self.last_frame_ids[i] and self.frame_ids[i] % self.timeout == 0:
                self.last_frame_ids[i] = self.frame_ids[i]
                if self.decode_bgr:
                    if self.encodings[i] == "bayer_rggb8":
                        self.imgs[i] = cv2.cvtColor(self.imgs_raw[i], cv2.COLOR_BayerBG2BGR)
                    elif self.encodings[i] == "mono8":
                        self.imgs[i] = cv2.cvtColor(self.imgs_raw[i], cv2.COLOR_GRAY2BGR)
                    else:
                        print("Received unknown encoding: ", self.encodings[i])
                else:
                    self.imgs[i] = self.imgs_raw[i]
                self.last_frame_ids[i] = self.frame_ids[i]
                self._on_image_arrived(i, self.imgs[i], topic_name)
                self.updated = True

            self.sub.release_latest_message(topic["handle"])

    def _add_image(self, image_topic_name):
        self.topics[image_topic_name] = {
            "topic": image_topic_name,
            "handle": self.sub.add_subscription(image_topic_name),
        }
        self.rets.append(False)
        self.frame_ids.append(0)
        self.last_frame_ids.append(0)
        self.registered_frame_ids.append(0)
        self.imgs_raw.append(None)
        self.imgs.append(None)
        self.encodings.append(None)
        self.fps_.append(FPSCalculator())

    def refresh_topics(self):
        """
        check for any new image topics available, and update the context if necessary
        """
        rclpy.spin_once(self.ros_node, timeout_sec=0)
        # populate topics
        # pylint: disable=too-many-nested-blocks
        for (
            topic_name,
            topic_types,
        ) in self.ros_node.get_topic_names_and_types():
            if "ace_interfaces/msg/ImageData" in topic_types:
                if topic_name not in self.topics:
                    if self.camera_mask is not None:
                        found = False
                        for cam in self.camera_mask:
                            if cam in topic_name:
                                found = True
                                break
                        if not found:
                            continue
                    print("Topic discovered: " + topic_name)
                    self._add_image(topic_name)


class DisplayImagesContext(ImageSubContext):

    """Helper class to display images captured"""

    def __init__(self, config, rate, decode_bgr, capture_fps=200, node_name="record_images", camera_mask=None):
        """Init"""
        # Load camera calibration.
        self.config = config

        self.window_name = "image"
        cv2.namedWindow(self.window_name)

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

        # Compute undistortion maps.
        self.num_cameras = len(self.camera_calibration.cameras) if camera_mask is None else len(camera_mask)
        self.camera_names = [None] * self.num_cameras
        self.camera_matrix = [None] * self.num_cameras
        self.imgs_updated = []
        self.display_imgs = []
        index = 0
        for camera in self.camera_calibration.cameras:
            if camera_mask is None or camera in camera_mask:
                self.camera_names[index] = camera
                index += 1

        # Set up window.
        self.screen_resolution = get_screen_resolution()

        self.imgs_updated = [False] * self.num_cameras
        self.display_imgs = [None] * self.num_cameras
        self.max_window_resolution = (
            0.9 * self.screen_resolution[0],
            0.9 * self.screen_resolution[1],
        )
        (
            self.scaling,
            self.grid,
            self.image_resolution,
            self.window_resolution,
        ) = compute_optimal_image_arrangement(self.num_cameras, (1440, 1080), self.max_window_resolution)

        self.img_zero = np.zeros((self.image_resolution[1], self.image_resolution[0], 3), dtype=np.uint8)
        self.img_combined = np.zeros((self.window_resolution[1], self.window_resolution[0], 3), dtype=np.uint8)
        self.img_combined_height = self.window_resolution[1]
        self.img_combined_width = self.window_resolution[0]
        super().__init__(rate, decode_bgr, capture_fps, node_name=node_name, camera_mask=camera_mask)

    def destroy(self):
        """on destroy"""
        super().destroy()
        cv2.destroyWindow(self.window_name)

    def _on_image_arrived(self, index, image, topic_name):
        index = -1
        i = 0
        for camera in self.camera_calibration.cameras:
            if self.camera_mask is not None and camera not in self.camera_mask:
                continue
            if camera in topic_name:
                index = i
                break
            i = i + 1
        if index == -1:
            return
        self.imgs_updated[index] = True
        self.display_imgs[index] = image

    def prepare_image_for_display(self, image, index):  # pylint: disable = unused-argument, no-self-use
        """Internally called when an image is preparing for display, can override"""
        return

    def prepare_display_images(self, force_update):
        """
        Prepare CV image containing the images, and markers overlayed.
        Return an OpenCV image that can be used to display
        """
        for i in range(len(self.imgs)):
            grid_m = i // self.grid[1]
            grid_n = i % self.grid[1]

            if self.rets[i] and (self.imgs_updated[i] or force_update):
                self.imgs_updated[i] = False
                self.img_combined[
                    grid_m * self.image_resolution[1] : (grid_m + 1) * self.image_resolution[1],
                    grid_n * self.image_resolution[0] : (grid_n + 1) * self.image_resolution[0],
                    :,
                ] = cv2.resize(self.display_imgs[i], (0, 0), fx=self.scaling, fy=self.scaling)
                img_roi = self.img_combined[
                    grid_m * self.image_resolution[1] : (grid_m + 1) * self.image_resolution[1],
                    grid_n * self.image_resolution[0] : (grid_n + 1) * self.image_resolution[0],
                    :,
                ]

                self.prepare_image_for_display(img_roi, i)

                cv2.putText(
                    img_roi,
                    self.camera_names[i],
                    (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (85, 196, 196),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img_roi,
                    "frame_id: " + str(self.frame_ids[i]),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (85, 196, 196),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img_roi,
                    "fps: " + str(self.fps_[i].fps()),
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (85, 196, 196),
                    1,
                    cv2.LINE_AA,
                )
        return self.img_combined

    def on_key_pressed(self, key):  # pylint: disable = unused-argument, no-self-use
        """can override"""
        return

    def render(self, force_update=False):
        """render images, can override"""
        if self.updated or force_update:
            image = self.prepare_display_images(force_update)
            cv2.imshow(self.window_name, image)
            self.updated = False

        key = cv2.waitKey(1) & 0xFF
        self.on_key_pressed(key)


class ManagedDisplayImagesContext(DisplayImagesContext):
    """High level context manager for displaying images"""

    def __init__(self, config, display_rate=30, capture_fps=200, node_name="record_images", camera_mask=None):
        super().__init__(
            config=config,
            rate=display_rate,
            decode_bgr=True,
            capture_fps=capture_fps,
            node_name=node_name,
            camera_mask=camera_mask,
        )
        self.is_done = False

    def on_key_pressed(self, key):
        if key == ord("q"):
            self.is_done = True


if __name__ == "__main__":
    rclpy.init()
    context = ManagedDisplayImagesContext("tyo00", 200, 200)
    while rclpy.ok() and not context.is_done:
        context.render(force_update=False)

    context.destroy()
    cv2.destroyAllWindows()
    rclpy.shutdown()
