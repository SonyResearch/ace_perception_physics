# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""
Racket information
"""

import cv2
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
import ace_interfaces.msg


class RacketInfo:
    """Class for racket information"""

    def __init__(self, topic, owner, name=""):
        self.owner = owner
        self.pose = None
        self.topic = topic
        self.name = name
        qos_profile = rclpy.qos.QoSProfile(depth=2)
        self.subscriber = owner.dummy_node.create_subscription(
            ace_interfaces.msg.RacketPose,
            topic,
            self.on_racket_pose_callback,
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

    def on_racket_pose_callback(self, msg):
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

    def draw_key_points(self, camera, image, pose, margin, transform=None):
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
                cv2.line(image, pt1=point1, pt2=point2, color=(255, 0, 0), thickness=2)
            except Exception as err:  # pylint: disable=broad-except
                print("exception: ", err)
                print(point1)
                print(point2)
        bbox = self.get_bbox(key_points_2d, image.shape, margin)
        if bbox is not None:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            cv2.rectangle(image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 0, 0), 2)

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
            cv2.line(image, pt1=points_3d[0], pt2=points_3d[1], color=(255, 255, 0), thickness=2)
            cv2.line(image, pt1=points_3d[2], pt2=points_3d[3], color=(255, 255, 0), thickness=2)

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
