# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""An example of using orientation fitting via racket keypoints"""

import time
import os
from typing import List
import numpy as np
import cv2

import colored_glog as glog
from ace_loggers import logger_pybind as logger
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
from calibration import python_module as calibration
from rotations import rotations_playback
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


class UIContext:
    """Handles UI interface"""

    def __init__(self, camera_count):
        self.timeout = 30
        self.iterations = 200
        self.error = 100
        self.cov = 0
        self.rate = 1000
        self.cameras = 2

        self.controls_window_name = "controls"
        self._prepare_ui(camera_count)

    def _prepare_ui(self, camera_count):
        cv2.namedWindow(self.controls_window_name)
        cv2.createTrackbar(
            "timeout",
            self.controls_window_name,
            int(self.timeout),
            500,
            lambda value: self._on_params_changed(0, value),
        )
        cv2.createTrackbar(
            "iterations",
            self.controls_window_name,
            self.iterations,
            500,
            lambda value: self._on_params_changed(1, value),
        )
        cv2.createTrackbar(
            "error",
            self.controls_window_name,
            self.error,
            500,
            lambda value: self._on_params_changed(2, value),
        )
        cv2.createTrackbar(
            "rate",
            self.controls_window_name,
            self.rate,
            1000,
            lambda value: self._on_params_changed(3, value),
        )
        cv2.createTrackbar(
            "cov",
            self.controls_window_name,
            self.cov,
            50,
            lambda value: self._on_params_changed(4, value),
        )
        cv2.createTrackbar(
            "cameras",
            self.controls_window_name,
            self.cameras,
            camera_count - 1,
            lambda value: self._on_params_changed(5, value),
        )

    def _on_params_changed(self, index, value):
        if index == 0:
            self.timeout = value + 1
        elif index == 1:
            self.iterations = value + 1
        elif index == 2:
            self.error = value
        elif index == 3:
            self.rate = value
        elif index == 4:
            self.cov = value
        elif index == 5:
            self.cameras = value + 1


class RacketRenderer:
    """Helper class to render racket"""

    def __init__(self) -> None:
        self.key_points = []

        self.racket_width = 0.3 / 2
        self.racket_length = 0.4 / 2

        num_points = 20
        for alpha in range(num_points + 1):
            angle = np.deg2rad(360 * alpha / num_points)
            y_pos = np.cos(angle) * self.racket_width
            z_pos = np.sin(angle) * self.racket_length
            self.key_points.append((0, y_pos, z_pos))

    def draw_key_points(self, camera, image, pose, transform=None):
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
                cv2.line(image, pt1=point1, pt2=point2, color=(255, 0, 0), thickness=3)
            except Exception as err:  # pylint: disable=broad-except
                print("exception: ", err)
                print(point1)
                print(point2)

        # draw axis
        points = np.array(
            [
                [0, 0, -self.racket_length - 0.05],
                [0, 0, self.racket_length],
                [0, -self.racket_width, 0],
                [0, self.racket_width, 0],
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
            cv2.line(image, pt1=points_3d[0], pt2=points_3d[1], color=(0, 255, 0), thickness=3)
            cv2.line(image, pt1=points_3d[2], pt2=points_3d[3], color=(0, 255, 0), thickness=3)

    def render(self, pose, image, camera, transform=None):
        """Project racket to an image using a camera"""

        if pose is None or image is None:
            return
        p3d = pose[0]
        transform = None
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
            self.draw_key_points(camera, image, pose, transform=transform)

    def get_keypoints(self, cameras, rotation, cams, cov=0):
        """Project main keypoints to selected cameras"""
        orientation = rotation.as_matrix()
        points = []
        model_points = np.array(
            [
                [0, 0, -self.racket_length - 0.1],
                [0, 0, +self.racket_length],
                [0, -self.racket_width, 0],
                [0, +self.racket_width, 0],
            ]
        )

        for cam in cams:
            kps = []
            for point in model_points:
                p3d = np.matmul(orientation, point)
                p2d = np.array((2, 1), np.float32)
                cameras[cam].project_point(p3d, p2d)
                keypoint = [0, 0, 100]
                keypoint[:2] = p2d + ((np.random.rand(2) * 2 - 1) * cov).astype(int)
                kps.append(keypoint)
            points.append(kps)

        return points


class Tuner:
    """Main tuner class"""

    def __init__(self) -> None:
        calibration_params_path = get_package_share_directory("calibration")
        calib_path = os.path.join(
            calibration_params_path,
            "parameters",
            "camera_calibration",
            "tyo02.yaml",
        )
        racket_params_path = get_package_share_directory("racket_pose_estimation")
        racket_params_path = os.path.join(
            racket_params_path,
            "parameters",
            "tyo02.yaml",
        )
        self.camera_calibration = calibration.CameraCalibrationParameters()
        if not self.camera_calibration.initialize(calib_path):
            raise Exception("Failed to load configuration file")

        self.racket_params = racket_pose_estimation.RacketParameters()
        if not self.racket_params.initialize(racket_params_path):
            raise Exception("Failed to load configuration file")

        self.racket_params.fitter.min_axis_length = 100

        # Compute undistortion maps.
        num_cameras = len(self.camera_calibration.cameras)
        self.camera_names = [None] * num_cameras
        self.camera_matrix = [None] * num_cameras
        self.cameras = [None] * num_cameras

        for i, camera in enumerate(self.camera_calibration.cameras):
            self.camera_names[i] = camera
            self.cameras[i] = self.camera_calibration.cameras[camera]
            self.camera_matrix[i] = self.cameras[i].T_world_camera

        self.ui_context = UIContext(num_cameras)
        self.racket_renderer = RacketRenderer()
        self.time_per_call: List[float] = []
        self.last_time = time.time()

        self.average_time_per_call = 0

        self.fitter = racket_pose_estimation.RacketPoseFitter()
        self.fitter.initialize(self.racket_params)
        self.fitter.set_calibration(self.camera_calibration)

    def run(self):  # pylint: disable=too-many-statements, too-many-locals
        """Run main loop"""
        width = 1440
        height = 1080
        frame_width = int(1280)
        frame_height = int(720)
        subframe_width = frame_width // 3
        subframe_height = frame_height // 2

        initial = [1, 0, 0, 0]
        errors = []
        angular_vel = np.random.rand((3)) * 10
        angular_vel[2] *= 0.1

        is_paused = False

        cv2.imshow("Results", np.zeros((frame_height, frame_width, 3)))

        index = 0
        while True:
            target_rotation = R.from_euler("xyz", rotations_playback[index], degrees=True)
            if not is_paused:
                index = index + 1
            if index >= len(rotations_playback):
                index = 0
            cameras_indicies = list(range(self.ui_context.cameras))
            self.racket_params.fitter.max_error = self.ui_context.error * 1e-3
            self.racket_params.fitter.max_iterations = self.ui_context.iterations
            self.racket_params.fitter.initial_rate = self.ui_context.rate * 1e-3
            self.racket_params.fitter.min_axis_length = 10
            gt_points, rotations, iter_count = self.solve_for(
                target_rotation,
                initial,
                cameras_indicies,
                cov=self.ui_context.cov,
            )
            rotation = rotations[-1]
            err = target_rotation * rotation.inv()
            err_vec = np.array(err.as_rotvec() * 180 / np.pi, dtype=int)
            err = np.linalg.norm(err_vec)
            # print(f"{index}, {err_vec[0]}, {err_vec[1]}, {err_vec[2]}, {err}, {iter_count}, {len(cameras_indicies)}")

            errors.append([target_rotation.as_euler("xyz", degrees=True), rotation.as_euler("xyz", degrees=True), err])
            position = [0, 0, 0]

            initial = rotation

            points = self.racket_renderer.get_keypoints(self.cameras, rotation, cameras_indicies, 0)
            points = np.array(points, int)
            target = target_rotation
            full_frame = np.zeros((frame_height, frame_width, 3), np.uint8)
            for cam_index in range(len(cameras_indicies)):
                gt_point = np.array(gt_points[cam_index]).astype(int)
                img = np.zeros((height, width, 3), np.uint8)

                self.racket_renderer.render([position, rotation, rotation.as_matrix()], img, self.cameras[cam_index])

                cv2.rectangle(img, (0, 0), (width, height), (255, 255, 255), thickness=5, lineType=cv2.LINE_AA)
                cv2.circle(
                    img,
                    (gt_point[0][0], gt_point[0][1]),
                    10,
                    (0, 1040, 255),
                    thickness=4,
                )
                cv2.line(
                    img,
                    (gt_point[0][0], gt_point[0][1]),
                    (gt_point[1][0], gt_point[1][1]),
                    (0, 0, 255),
                    thickness=4,
                )
                cv2.line(
                    img,
                    (gt_point[2][0], gt_point[2][1]),
                    (gt_point[3][0], gt_point[3][1]),
                    (0, 0, 255),
                    thickness=4,
                )

                p2d = np.array((2, 1), np.float32)
                self.cameras[cam_index].project_point(np.array([0, 0, 0], np.float32), p2d)
                p2d = np.array(p2d, int)

                cv2.circle(
                    img,
                    (points[cam_index][0][0], points[cam_index][0][1]),
                    5,
                    (0, 0, 255),
                    thickness=2,
                )
                cv2.line(
                    img,
                    (points[cam_index][0][0], points[cam_index][0][1]),
                    (points[cam_index][1][0], points[cam_index][1][1]),
                    (255, 0, 0),
                    thickness=3,
                )
                cv2.line(
                    img,
                    (points[cam_index][2][0], points[cam_index][2][1]),
                    (points[cam_index][3][0], points[cam_index][3][1]),
                    (255, 0, 0),
                    thickness=3,
                )

                row = (cam_index // 3) * subframe_height
                col = (cam_index % 3) * subframe_width
                full_frame[row : row + subframe_height, col : col + subframe_width] = cv2.resize(
                    img, (subframe_width, subframe_height)
                )

            stats = [
                f"Total Iterations: {iter_count}",
                f"Estimated Rotation: {rotation.as_euler('xyz',degrees=True).astype(int)}",
                f"Target Rotation: {target.as_euler('xyz', degrees=True).astype(int)}",
                f"Average Error: {int(np.mean(np.array(errors)[:, 2]))} +/-[{int(np.std(np.array(errors)[:, 2]))}] deg",
                f"Average Time: {self.average_time_per_call:.3f}ms",
                "",
                f"Observations count: {self.ui_context.cameras}",
                f"Max Iterations   : {self.ui_context.iterations}",
                f"Gradient Rate    : {self.ui_context.rate*1e-3:.2f}",
                f"Target projection error: {self.ui_context.error*1e-3:.2f}",
                f"Keypoints error [px]: {self.ui_context.cov}",
            ]
            for i, stat in enumerate(stats):
                cv2.putText(
                    full_frame,
                    stat,
                    (10, 30 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (85, 196, 196),
                    1,
                    cv2.LINE_AA,
                )

            errors = errors[-20:]
            cv2.imshow("Results", full_frame)
            k = cv2.waitKey(self.ui_context.timeout)
            if k == ord("q"):
                break
            if k == ord(" "):
                is_paused = not is_paused
            if k == ord("r"):
                initial = [1, 0, 0, 0]
                self.fitter.reset(initial[0], initial[1], initial[2], initial[3])

        while k != ord("q"):
            k = cv2.waitKey(0)

        with open(f"results_{len(cameras_indicies)}.txt", "wt", encoding="utf8") as file:
            for err in errors:
                file.write(f"{err}\n")

    def solve_for(self, target, initial, cameras_indicies, cov=0):
        """Solve orientation fitting"""
        points = self.racket_renderer.get_keypoints(self.cameras, target, cameras_indicies, cov)
        self.fitter.set_cameras(cameras_indicies)
        rots = []
        iters = []

        initial = [1, 0, 0, 0]
        # self.fitter.reset(initial[0],initial[1],initial[2],initial[3])
        rots.append(R.from_quat(initial))
        time1 = time.time()
        rot, it_count, _ = self.fitter.fit_points([0, 0, 0], points)
        self.time_per_call.append(time.time() - time1)
        if time.time() - self.last_time > 1:
            self.average_time_per_call = np.mean(self.time_per_call) * 1e3 / (time.time() - self.last_time)
            self.time_per_call = []
            self.last_time = time.time()
        rots.append(R.from_quat(rot))
        iters.append(it_count)

        return points, rots, np.sum(iters)


def main():
    """Main entry point"""
    tuner = Tuner()
    tuner.run()


if __name__ == "__main__":
    main()
