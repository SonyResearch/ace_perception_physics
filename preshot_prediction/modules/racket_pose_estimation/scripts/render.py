# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""
Sample script to use vive module
"""

import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import click

from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration
from aps import python_module as aps
from python_helpers.screen import (
    get_screen_resolution,
    compute_optimal_image_arrangement,
)
import ace_interfaces.msg


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


class RacketInfo:
    """Class for racket information"""

    def __init__(self, topic, owner, name="", color=(128, 0, 0), msg_type=ace_interfaces.msg.RacketPose):
        self.owner = owner
        self.pose = None
        self.topic = topic
        self.name = name
        self.color = color
        qos_profile = rclpy.qos.QoSProfile(depth=2)
        callbacks = {}
        callbacks[ace_interfaces.msg.RacketPose] = self.on_racket_pose_callback
        callbacks[ace_interfaces.msg.RacketPoseEstimate] = self.on_racket_pose_est_callback
        self.subscriber = owner.dummy_node.create_subscription(
            msg_type,
            topic,
            callbacks[msg_type],
            qos_profile=qos_profile,
        )

        self.key_points = []
        self.racket_width = 0.15
        self.racket_length = 0.2

        self.sequence_id = 0

        self.is_tracked = False

        num_points = 20
        for alpha in range(num_points + 1):
            angle = np.deg2rad(360 * alpha / num_points)
            y_pos = np.cos(angle) * self.racket_width / 2
            z_pos = np.sin(angle) * self.racket_length / 2
            self.key_points.append((0, y_pos, z_pos))

    def on_racket_pose_callback(self, msg: ace_interfaces.msg.RacketPose):
        """Racket callback"""
        try:
            self.is_tracked = msg.tracked
            if self.is_tracked:
                self.sequence_id = msg.header.sequence_number
                pos = msg.position
                ori = msg.orientation
                pos = [pos.x, pos.y, pos.z]
                ori = [ori.x, ori.y, ori.z, ori.w]
                rot = R.from_quat(ori)
                transform = rot.as_matrix()
                pos = pos + np.matmul(transform, [0, 0, 0.03])
                self.pose = [pos, ori, transform]

                self.owner.on_racket_updated(self)
        except Exception as err:  # pylint: disable=broad-except
            print(err)

    def on_racket_pose_est_callback(self, msg: ace_interfaces.msg.RacketPoseEstimate):
        """On racket pose estimate message"""
        self.on_racket_pose_callback(msg.pose)

    def get_bbox(self, points, img_shape, margin=80):  # pylint: disable=no-self-use
        """
        Get bounding box for the racket.

        Return a bounding box (bbox_x1, bbox_y1, bbox_x2, bbox_y2) tight to the keypoints and within the image.
        Return None if keypoints are nan or if the derived bbox is malformed.
        """
        # Ignore unclear detections
        if np.isnan(points).any() or len(points) == 0:
            return None

        bbox_x1, bbox_y1, bbox_w, bbox_h = cv2.boundingRect(np.array(points).astype(int))
        bbox_x1 -= margin // 2
        bbox_y1 -= margin // 2
        bbox_w += margin
        bbox_h += margin

        # bbox_x1, bbox_y1 = max([bbox_x1, 0]), max([bbox_y1, 0])
        # Adjust coords to be within the image
        bbox_x2, bbox_y2 = min([bbox_x1 + bbox_w, img_shape[1] - 1]), min([bbox_y1 + bbox_h, img_shape[0] - 1])

        if (
            bbox_x2 < 0 or bbox_y2 < 0 or bbox_x1 > bbox_x2 or bbox_y1 > bbox_y2
        ):  # Shouldn't happen but we have seen it, so just in case
            return None

        # bbox_x1/=img_shape[1]
        # bbox_y1/=img_shape[0]
        # bbox_x2/=img_shape[1]
        # bbox_y2/=img_shape[0]

        return [bbox_x1, bbox_y1, bbox_x2, bbox_y2]

    def project_key_points(self, camera, image, margin, pose=None):
        """project key points"""
        pose = self.pose if pose is None else pose
        if pose is None or image is None:
            return None
        key_points_2d = []

        for point in self.key_points:
            p2d = np.zeros((2, 1), dtype=np.float32)
            p3d = np.matmul(pose[2], point) + pose[0]
            if camera.project_point(np.float32(p3d), p2d):
                key_points_2d.append((int(p2d[0]), int(p2d[1])))

        bbox = self.get_bbox(key_points_2d, image.shape, margin)
        # print(bbox)
        return bbox

    def convert_coords(self, camera, pose=None):
        """convert coordinates to a dictionary"""
        pose = self.pose if pose is None else pose
        # project position to camera space (2D coords)
        # convert orientation to camera space (3-axis)

        pos_2d = np.zeros(2, dtype=np.float32)
        visible = camera.project_point(np.float32(pose[0]), pos_2d)
        camera_rotation = R.from_matrix(camera.T_camera_world[:3, :3])
        rot = R.from_quat(pose[1])
        mat = rot.as_matrix()

        camera_space_ori = (camera_rotation * rot).as_euler("xyz", degrees=True)

        keypoints_3d = [
            [0, 0, -self.racket_length / 2],
            [0, 0, +self.racket_length / 2],
            [0, -self.racket_width / 2, 0],
            [0, +self.racket_width / 2, 0],
        ]
        keypoints_2d = []
        for keypoint in keypoints_3d:
            p2d = np.zeros((2), dtype=np.float32)
            p3d = np.matmul(mat, keypoint) + pose[0]
            visible = camera.project_point(p3d, p2d)
            if not 0 < p2d[0] < 1440 or not 0 < p2d[1] < 1080:
                visible = False
            keypoints_2d.append([int(p2d[0]), int(p2d[1]), 2 if visible else 0])

        if not visible:
            return None
        return {"pos": pos_2d, "rotation": camera_space_ori, "visible": visible, "keypoints": keypoints_2d}

    def draw_key_points(self, camera, image, pose, margin, transform=None):  # pylint: disable=unused-argument
        """draw key points of the racket into an image"""
        if pose is None or image is None:
            return
        key_points_2d = []

        for point in self.key_points:
            p2d = np.zeros((2, 1), dtype=np.float32)
            p3d = np.matmul(pose[2], point) + pose[0]
            if transform is not None:
                p3d = np.matmul(transform, [p3d[0], p3d[1], p3d[2], 1])[:3]
            if camera.project_point(np.float32(p3d), p2d):
                key_points_2d.append((int(p2d[0]), int(p2d[1])))

        for i, _ in enumerate(key_points_2d):
            point1 = key_points_2d[i]
            point2 = key_points_2d[(i + 1) % len(key_points_2d)]
            try:
                cv2.line(image, pt1=point1, pt2=point2, color=self.color, thickness=2)
            except Exception as err:  # pylint: disable=broad-except
                print("exception: ", err)
                print(point1)
                print(point2)
        # bbox = self.get_bbox(key_points_2d, image.shape, margin)
        # if bbox is not None:
        #     bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        #     cv2.rectangle(image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 0, 0), 2)

        self.racket_width = 0.15
        self.racket_length = 0.2
        # draw axis
        points = np.array(
            [
                [0, 0, -self.racket_length / 2 - 0.05],
                [0, 0, self.racket_length / 2],
                [0, -self.racket_width / 2, 0],
                [0, self.racket_width / 2, 0],
            ]
        )
        points_3d = []
        skip = False
        for point in points:
            p2d = np.zeros((2, 1), dtype=np.float32)
            p3d = np.matmul(pose[2], point) + pose[0]
            skip |= not camera.project_point(np.float32(p3d), p2d)
            points_3d.append((int(p2d[0]), int(p2d[1])))
        if not skip:
            cv2.line(image, pt1=points_3d[0], pt2=points_3d[1], color=self.color, thickness=1)
            cv2.line(image, pt1=points_3d[2], pt2=points_3d[3], color=self.color, thickness=1)

    def draw_coords(self, img, camera):
        """Draw racket coordinates into an image"""
        p3d = self.pose[0]
        rot = self.pose[2]
        p2d = np.zeros((2, 1), dtype=np.float32)

        axis_scale = 0.1
        p3d_x = p3d + np.matmul(rot, [1, 0, 0]) * axis_scale
        p3d_y = p3d + np.matmul(rot, [0, 1, 0]) * axis_scale
        p3d_z = p3d + np.matmul(rot, [0, 0, 1]) * axis_scale
        p2d_x = np.zeros((2, 1), dtype=np.float32)
        p2d_y = np.zeros((2, 1), dtype=np.float32)
        p2d_z = np.zeros((2, 1), dtype=np.float32)
        if camera.project_point(p3d, p2d):
            cv2.circle(
                img,
                center=(int(p2d[0]), int(p2d[1])),
                color=(255, 0, 0),
                radius=3,
                thickness=4,
            )
            camera.project_point(p3d_x, p2d_x)
            camera.project_point(p3d_y, p2d_y)
            camera.project_point(p3d_z, p2d_z)
            cv2.line(
                img,
                (int(p2d[0]), int(p2d[1])),
                (int(p2d_x[0]), int(p2d_x[1])),
                color=(0, 0, 255),
                thickness=4,
            )
            cv2.line(
                img,
                (int(p2d[0]), int(p2d[1])),
                (int(p2d_y[0]), int(p2d_y[1])),
                color=(0, 255, 0),
                thickness=4,
            )
            cv2.line(
                img,
                (int(p2d[0]), int(p2d[1])),
                (int(p2d_z[0]), int(p2d_z[1])),
                color=(255, 0, 0),
                thickness=4,
            )

    def render(self, image, camera, pose=None, margin=30, transform=None):
        """Project racket to an image using a camera"""

        pose = self.pose if pose is None else pose
        if pose is None or image is None:
            return
        p3d = pose[0]
        if transform is not None:
            p3d = np.matmul(transform, [p3d[0], p3d[1], p3d[2], 1])[:3]
        p2d = np.zeros((2, 1), dtype=np.float32)

        if camera.project_point(p3d, p2d):
            cv2.circle(
                image,
                center=(int(p2d[0]), int(p2d[1])),
                color=(255, 0, 0),
                radius=3,
                thickness=4,
            )
            self.draw_key_points(camera, image, pose, margin, transform=transform)

            cv2.putText(
                image, self.name, (int(p2d[0]), int(p2d[1])), cv2.FONT_HERSHEY_DUPLEX, 1.5, self.color, 1, cv2.LINE_AA
            )


class BallInfo:
    """Holds ball information"""

    def __init__(self, owner):
        self.owner = owner
        self.pose = None
        qos_profile = rclpy.qos.QoSProfile(depth=1)
        self.subscriber = owner.dummy_node.create_subscription(
            ace_interfaces.msg.PosesWithCovariance,
            "/sensors/ball_pose_estimation/poses",
            self.on_ball_pose_callback,
            qos_profile=qos_profile,
        )

    def on_ball_pose_callback(self, msg):
        """Racket callback"""
        try:
            if len(msg.poses) > 0:
                pos = msg.poses[0].pose.position
                ori = msg.poses[0].pose.orientation
                self.pose = [[pos.x, pos.y, pos.z], [ori.x, ori.y, ori.z, ori.w]]
        except Exception as err:  # pylint: disable=broad-except
            print(err)


class Player2DKeypointseInfo:
    """Helper class for player pose 2d"""

    def __init__(self, owner, topic, topic_detection) -> None:
        """Initializer"""
        self.pose = None
        self.owner = owner
        node = owner.dummy_node
        self.topic = topic
        self.topic_detection = topic_detection
        self.keypoints = None
        self.bbox = None

        qos_profile = rclpy.qos.QoSProfile(depth=1)
        self.player_2d_subscriber = node.create_subscription(
            ace_interfaces.msg.PlayerPose2d,
            self.topic,
            self.on_player_pose_callback,
            qos_profile=qos_profile,
        )
        self.player_detection = node.create_subscription(
            ace_interfaces.msg.PlayerDetection,
            self.topic_detection,
            self.on_player_detection_callback,
            qos_profile=qos_profile,
        )

    def on_player_pose_callback(self, msg):
        """ROS callback for player pose"""
        self.keypoints = np.array(msg.keypoints)
        # self.bbox = np.array(msg.bbox, dtype=int)

    def on_player_detection_callback(self, msg):
        """ROS callback for player pose"""
        self.bbox = np.array(msg.bbox, dtype=int)

    @classmethod
    def point_to_array(cls, point):
        """Convert a point to array"""
        if isinstance(point, list):
            return np.array(point)
        if isinstance(point, np.ndarray):
            return point
        return np.array([point.x, point.y, point.z])

    def draw(self, image):
        """Draw keypoints"""
        if self.bbox is not None:
            cv2.rectangle(
                image,
                (self.bbox[0], self.bbox[1]),
                (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]),
                (255, 0, 0),
                5,
            )
        if self.keypoints is None:
            return
        keypoints_map = {
            JOINT_NAMES[i]: Player2DKeypointseInfo.point_to_array(self.keypoints[i]) for i in range(len(self.keypoints))
        }

        confidence = 0.3
        for index, kpt in enumerate(keypoints_map):  # pylint: disable=unused-variable
            keypoint = keypoints_map[kpt]
            if keypoint[2] > confidence:
                cv2.circle(image, keypoint[:2].astype(int), radius=3, thickness=2, color=(255, 255, 255))

                # cv2.putText(
                #     image,
                #     JOINT_NAMES[index],
                #     keypoint[:2].astype(int) + [4, 4],
                #     cv2.FONT_HERSHEY_DUPLEX,
                #     1.5,
                #     (85, 196, 196),
                #     1,
                #     cv2.LINE_AA,
                # )
        for pair in JOINT_PAIRS:
            pt0 = keypoints_map[pair[0]]
            pt1 = keypoints_map[pair[1]]
            if pt0[2] > confidence and pt1[2] > confidence:
                cv2.line(image, pt0[:2].astype(int), pt1[:2].astype(int), pair[2], 2)


# pylint: disable=too-many-instance-attributes
class Context:
    """Helper class"""

    # pylint: disable=too-many-statements
    def __init__(self, config):
        """Init"""
        # Load camera calibration.
        self.config = config

        rclpy.init()
        self.dummy_node = Node("racket_ball_visualizer")

        self.camera_filter = []

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
        self.selected_cameras = list(set(list(self.camera_calibration.cameras)) | set(self.camera_filter))
        self.num_cameras = len(self.selected_cameras)
        print(self.selected_cameras)
        self.camera_names = [None] * self.num_cameras
        self.camera_matrix = [None] * self.num_cameras
        self.cameras = [None] * self.num_cameras

        self.racket_states = [None] * self.num_cameras

        for i, camera in enumerate(self.selected_cameras):
            self.camera_names[i] = camera
            self.cameras[i] = self.camera_calibration.cameras[camera]

        self.sub = aps.ZeroCopySubscriber()
        self.handles = [None] * self.num_cameras
        self.player_pose_2d = [None] * self.num_cameras
        for i in range(self.num_cameras):
            topic = "/sensors/" + self.camera_names[i] + "/image"
            self.handles[i] = self.sub.add_subscription(topic)
            if self.handles[i] < 0:
                print(f'Failed to subscribe to topic "{topic}"')

            topic = "/sensors/" + self.camera_names[i] + "/player_pose_2d"
            topic_detection = "/sensors/" + self.camera_names[i] + "/player_detection"
            self.player_pose_2d[i] = Player2DKeypointseInfo(self, topic, topic_detection)

        # Set up window.
        self.screen_resolution = get_screen_resolution()

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

        self.rets = [None] * self.num_cameras
        self.frame_ids = [None] * self.num_cameras
        self.imgs_updated = [None] * self.num_cameras
        self.last_frame_ids = [None] * self.num_cameras
        self.encodings = [None] * self.num_cameras
        self.imgs_raw = [None] * self.num_cameras
        self.imgs_bgr = [None] * self.num_cameras

        self.rackets = []
        self.rackets_history = {}

        self.rackets.append(RacketInfo("/sensors/racket_0/racket_pose_vive/pose", self, "vive", (128, 128, 128)))
        self.rackets.append(RacketInfo("/sensors/racket_1/racket_pose_vive/pose", self, "vive", (128, 128, 128)))

        self.rackets.append(RacketInfo("/sensors/racket/old", self, "old", (255, 0, 0)))
        self.rackets.append(RacketInfo("/sensors/racket/new", self, "new", (0, 0, 255)))
        self.rackets.append(
            RacketInfo("/sensors/racket0/pose", self, "est0", (0, 255, 0), ace_interfaces.msg.RacketPoseEstimate)
        )
        self.rackets.append(
            RacketInfo("/sensors/racket1/pose", self, "est1", (0, 255, 0), ace_interfaces.msg.RacketPoseEstimate)
        )
        self.rackets.append(
            RacketInfo(
                "/sensors/racket0/filtered", self, "filtered", (255, 255, 0), ace_interfaces.msg.RacketPoseEstimate
            )
        )

        self.ball = BallInfo(self)

        self.transform_pos = [0, 0, 0]
        self.transform_ori = [0, 0, 0]
        self.pose_transform = None

    def update_transform(self):
        """Update pose transform"""
        self.pose_transform = np.identity(4, dtype=np.float32)
        rot = R.from_euler("xyz", self.transform_ori, degrees=True)
        self.pose_transform[:3, :3] = rot.as_matrix()
        self.pose_transform[:3, 3] = np.array(self.transform_pos).transpose()

    def on_racket_updated(self, racket: RacketInfo):
        """callback from rackets"""
        self.rackets_history[racket.topic] = racket.pose

    def estimate_orientation(self):
        """using project racket orientation, estimate the final orientation"""
        if self.racket_states is None:
            return None
        projected = []
        final = [0, 0, 0]
        for i, state in enumerate(self.racket_states):
            if state is None or not state["visible"]:
                continue
            camera_rotation = R.from_matrix(self.cameras[i].T_world_camera[:3, :3])
            rot = R.from_euler("xyz", state["rotation"], degrees=True)
            world_rot = (camera_rotation * rot).as_euler("xyz", degrees=True)
            final += world_rot
            projected.append(world_rot)

        if len(projected) == 0:
            return None

        return final / len(projected)

    def update_racket(self, racket: RacketInfo, img_id):
        """Update racket pose"""
        if racket.pose is None:
            return
        self.racket_states[img_id] = racket.convert_coords(self.cameras[img_id])

    def draw_ball(self, img_id):
        """Draw ball circle into the image"""
        if self.ball.pose is None:
            return
        p3d = self.ball.pose[0]
        p2d = np.zeros((2, 1), dtype=np.float32)

        if self.cameras[img_id].project_point(p3d, p2d):
            cv2.circle(
                self.imgs_bgr[img_id],
                center=(int(p2d[0]), int(p2d[1])),
                color=(255, 0, 0),
                radius=10,
                thickness=4,
            )

    def draw_table(self, img_id):
        """Draw table"""
        table_width = 1.525
        table_length = 2.738
        table_height = 0.0
        table_corners = np.array(
            [
                [-0.5 * table_length, -0.5 * table_width, table_height],
                [-0.5 * table_length, +0.5 * table_width, table_height],
                [+0.5 * table_length, +0.5 * table_width, table_height],
                [+0.5 * table_length, -0.5 * table_width, table_height],
            ]
        )
        if self.pose_transform is not None:
            for i, _ in enumerate(table_corners):
                table_corners[i] = np.matmul(
                    self.pose_transform, [table_corners[i][0], table_corners[i][1], table_corners[i][2], 1]
                )[:3]
        indicies = [[0, 1], [1, 2], [2, 3], [3, 0]]
        for idx in indicies:
            p2d = [np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)]

            self.cameras[img_id].project_point(table_corners[idx[0]], p2d[0])
            self.cameras[img_id].project_point(table_corners[idx[1]], p2d[1])
            cv2.line(
                self.imgs_bgr[img_id],
                (int(p2d[0][0][0]), int(p2d[0][1][0])),
                (int(p2d[1][0][0]), int(p2d[1][1][0])),
                (255, 0, 0),
                thickness=4,
            )

    def update_images(self):
        """Update images"""
        # Process all images
        for index in range(self.num_cameras):
            if self.imgs_updated[index]:
                continue
            racket: RacketInfo
            for racket in self.rackets:
                self.update_racket(racket, index)
                if (
                    racket.topic in self.rackets_history
                ):  # and self.frame_ids[index] in self.rackets_history[racket.topic]:
                    racket.render(
                        self.imgs_bgr[index],
                        self.cameras[index],
                        self.rackets_history[racket.topic],
                        transform=self.pose_transform,
                    )

                    self.draw_ball(index)
                    self.imgs_updated[index] = True

            self.player_pose_2d[index].draw(self.imgs_bgr[index])

    def destroy(self):
        """
        Destroy the object
        """
        self.handles.clear()

    def fetch_images(self):
        """
        Fetch the images
        """
        try:
            rclpy.spin_once(self.dummy_node, timeout_sec=0.02)
        except Exception as err:  # pylint: disable = broad-except
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
                self.last_frame_ids[i] = self.frame_ids[i]
                if self.encodings[i] == "bayer_rggb8":
                    self.imgs_bgr[i] = cv2.cvtColor(self.imgs_raw[i], cv2.COLOR_BayerBG2BGR)
                elif self.encodings[i] == "mono8":
                    self.imgs_bgr[i] = cv2.cvtColor(self.imgs_raw[i], cv2.COLOR_GRAY2BGR)
                else:
                    print("Received unknown encoding: ", self.encodings[i])

                self.draw_table(i)
                self.imgs_updated[i] = False

    def prepare_display_images(self):
        """
        Prepare CV image containing the images, and markers overlayed.
        Return an OpenCV image that can be used to display
        """
        self.update_images()
        for i in range(self.num_cameras):
            grid_m = i // self.grid[1]
            grid_n = i % self.grid[1]

            if self.rets[i]:
                self.img_combined[
                    grid_m * self.image_resolution[1] : (grid_m + 1) * self.image_resolution[1],
                    grid_n * self.image_resolution[0] : (grid_n + 1) * self.image_resolution[0],
                    :,
                ] = cv2.resize(self.imgs_bgr[i], (0, 0), fx=self.scaling, fy=self.scaling)
                img_roi = self.img_combined[
                    grid_m * self.image_resolution[1] : (grid_m + 1) * self.image_resolution[1],
                    grid_n * self.image_resolution[0] : (grid_n + 1) * self.image_resolution[0],
                    :,
                ]
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
                offset = 50
                for racket in self.rackets:
                    cv2.putText(
                        img_roi,
                        f"racket_id: {racket.name}: {racket.sequence_id}",
                        (10, offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (85, 196, 196),
                        1,
                        cv2.LINE_AA,
                    )
                    offset += 20
            else:
                self.img_combined[
                    grid_m * self.image_resolution[1] : (grid_m + 1) * self.image_resolution[1],
                    grid_n * self.image_resolution[0] : (grid_n + 1) * self.image_resolution[0],
                    :,
                ] = self.img_zero
                img_roi = self.img_combined[
                    grid_m * self.image_resolution[1] : (grid_m + 1) * self.image_resolution[1],
                    grid_n * self.image_resolution[0] : (grid_n + 1) * self.image_resolution[0],
                    :,
                ]
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

        return self.img_combined


@click.command()
@click.option("--config", help="Config", required=True, type=str)
def main(config):
    """Main entry point"""

    context: Context = Context(config)

    # Main loop.
    should_exit = False
    window_name = "APS Camera Images"
    commands = [
        "",
        "'q' to exit",
    ]
    while not should_exit:
        context.fetch_images()

        img_combined = context.prepare_display_images()

        for i, command in enumerate(commands):
            cv2.putText(
                img_combined,
                command,
                (
                    context.img_combined_width - 420,
                    context.img_combined_height - 30 * len(commands) + 20 * (i + 1),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (85, 196, 196),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img_combined,
            f"Pos: {context.transform_pos}\nRot: {context.transform_ori}",
            (
                context.img_combined_width - 420,
                context.img_combined_height - 50,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (85, 196, 196),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, img_combined)

        # Process Input
        k = chr(cv2.waitKey(20) % 256)
        if k == "q":
            should_exit = True
        elif k == "w":
            context.transform_pos[0] += 0.01
            context.update_transform()
        elif k == "s":
            context.transform_pos[0] -= 0.01
            context.update_transform()
        elif k == "a":
            context.transform_pos[1] += 0.01
            context.update_transform()
        elif k == "d":
            context.transform_pos[1] -= 0.01
            context.update_transform()
        elif k == "z":
            context.transform_pos[2] += 0.01
            context.update_transform()
        elif k == "x":
            context.transform_pos[2] -= 0.01
            context.update_transform()
    # Clean up.
    cv2.destroyAllWindows()
    context.destroy()
    print("Clean up successfully...")

    return 0


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
