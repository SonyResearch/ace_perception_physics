# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Helper script to correct racket pose using Kalman filtering.
This script implements a Kalman filter for 6DoF pose estimation, allowing for confidence-based adaptive noise handling.
It provides a RacketPoseCorrector class that can be used to correct racket poses based on measurements.
author: Yamen Saraiji
email: yamen.saraiji@sony.com
"""

import logging
from typing import List
import numpy as np
from racket_pose_estimation.pose_correction.estimators.config import CorrectorConfig
from racket_pose_estimation.pose_correction.estimators.position_estimator import PositionKalmanFilter
from racket_pose_estimation.pose_correction.estimators.orientation_estimator import (
    QuaternionKF,
    QuaternionMEKF,
    QuaternionHelpers,
)


class RacketPoseCorrector:

    """Racket pose corrector using separate position and quaternion estimators."""

    def __init__(self, config: CorrectorConfig):
        self.config = config
        self.position_filter = PositionKalmanFilter(config)
        self.quaternion_estimator_type = RacketPoseCorrector._create_quaternion_estimator(
            getattr(config, "orientation_estimator_type", None)
        )

        self.orientation_filter = self.quaternion_estimator_type(config)
        self.missing_count = 0
        self.quaternion_history: List[np.ndarray] = []

    @staticmethod
    def _create_quaternion_estimator(est_type):
        if est_type == "QuaternionKF":
            return QuaternionKF
        return QuaternionMEKF

    @staticmethod
    def correct_sign_quaternion(quaternion):
        """correct the sign of quatanion to remove jumps"""
        # x: (t,4)
        # change sign to have lower distances
        nan_idx = np.isnan(quaternion[:, 0])
        quat_copy = np.copy(quaternion)
        valid_quat = np.copy(quaternion[~nan_idx])
        for i in range(1, valid_quat.shape[0]):
            if np.dot(valid_quat[i], valid_quat[i - 1]) < 0:
                valid_quat[i] = -valid_quat[i]
        quat_copy[~nan_idx] = valid_quat
        return quat_copy

    @staticmethod
    def best_flip(prev, quat):
        """Find the best flip quaternion to minimize the dot product with the previous quaternion."""
        # 180° flips about principal axes:
        flip_quats = [
            np.array([1, 0, 0, 0]),  # 180° around X
            np.array([0, 1, 0, 0]),  # 180° around Y
            np.array([0, 0, 1, 0]),  # 180° around Z
        ]
        # for a,b in combinations(flip_quats, 2):
        #     flip_quats.append(quat_multiply(a,b))

        best = quat
        best_dot = np.dot(prev, best)

        # Try 180° flips around X, Y, Z
        for flip_quat in flip_quats[1:]:
            candidate = QuaternionHelpers.quaternion_multiply(flip_quat, quat)
            cos_distance = np.dot(prev, candidate)
            if cos_distance > best_dot:
                best_dot = cos_distance
                best = candidate

        # Also check whole quaternion sign (since quat and -quat are equivalent)
        if np.dot(prev, -best) > best_dot:
            best = -best

        return best / np.linalg.norm(best)

    @staticmethod
    def minimize_quaternion_jumps(quaternion):
        """
        Minimize quaternion jumps by correcting the sign of quaternions to ensure continuity.

        Args:
            quaternion: numpy array of shape (N, 4) representing a series of quaternions.

        Returns:
            Corrected quaternion array with minimized jumps.
        """
        # Ensure input is a numpy array
        quaternion = np.array(quaternion)

        # Check if the input is empty
        if quaternion.size == 0:
            return quaternion

        # Correct the sign of the quaternion to minimize jumps
        fixed = RacketPoseCorrector.correct_sign_quaternion(quaternion)
        fixed /= np.linalg.norm(fixed, axis=1, keepdims=True)

        # # Fix flips
        for i in range(1, len(fixed)):
            while not np.any(np.isnan(fixed[i])):
                # Keep flipping until we stabilize
                flipped = RacketPoseCorrector.best_flip(fixed[i - 1], fixed[i])
                if (
                    np.dot(flipped, fixed[i]) >= 1 - 1e-3
                ):  # Check if the flipped quaternion is close enough to the current one
                    break
                fixed[i] = flipped
        return fixed

    def correct_pose(self, pose):
        """
        Corrects the pose using separate position and quaternion estimators.
        :param pose: dict with keys 'delta_time', 'position', 'orientation',
                     optional 'pos_confidence', 'quat_confidence'
        :return: dict with corrected pose
        """
        if "position" not in pose or "orientation" not in pose or "delta_time" not in pose:
            raise ValueError("Pose must contain 'position', 'orientation', and 'delta_time' keys")
        delta_time = pose["delta_time"]
        pos = np.array(pose["position"])
        quat = np.array(pose["orientation"])
        # Quaternion sign correction for continuity
        if quat[3] < 0:
            quat = -quat
        self.quaternion_history.append(quat)
        if len(self.quaternion_history) > 10:
            self.quaternion_history.pop(0)
        self.quaternion_history = RacketPoseCorrector.minimize_quaternion_jumps(
            np.array(self.quaternion_history)
        ).tolist()
        quat = np.array(self.quaternion_history[-1])
        # Prepare measurement dicts
        pos_meas = {"position": pos}
        quat_meas = {"orientation": quat}
        if "pos_confidence" in pose:
            pos_meas["pos_confidence"] = pose["pos_confidence"]
        if "quat_confidence" in pose:
            quat_meas["quat_confidence"] = pose["quat_confidence"]
        if np.any(np.isnan(pos)) or np.any(np.isnan(quat)):
            return self.predict(delta_time)
        self.missing_count = 0

        # print(f"Quaternion: {R.from_quat(quat_meas['orientation'],).as_rotvec(True)}")
        self.position_filter.predict(delta_time)
        self.orientation_filter.predict(delta_time)
        self.position_filter.update(pos_meas)
        self.orientation_filter.update(quat_meas)
        pos_out, vel_out = self.position_filter.get_position()
        quat_out, omega_out = self.orientation_filter.get_orientation()
        return {
            "position": pos_out,
            "orientation": quat_out,
            "velocity": vel_out,
            "angular_velocity": omega_out,
        }

    def _add_missing_observation(self):
        if not self.is_tracking():
            return
        self.missing_count += 1
        if self.config.reset_after > 0 and self.missing_count >= self.config.reset_after:
            logging.info("Too many missing poses, resetting filter")
            self.reset()

    def predict(self, delta_time):
        """Predict the next state."""
        self._add_missing_observation()
        if not self.is_tracking():
            return None
        self.position_filter.predict(delta_time)
        self.orientation_filter.predict(delta_time)
        pos_out, vel_out = self.position_filter.get_position()
        quat_out, omega_out = self.orientation_filter.get_orientation()
        return {
            "position": pos_out,
            "orientation": quat_out,
            "velocity": vel_out,
            "angular_velocity": omega_out,
        }

    def reset(self):
        """Reset the state of the corrector."""
        self.position_filter = PositionKalmanFilter(self.config)
        self.orientation_filter = self.quaternion_estimator_type(self.config)
        self.quaternion_history.clear()

    def is_tracking(self):
        """Check if the corrector is actively tracking."""
        return self.position_filter.initialized and self.orientation_filter.initialized


def test_racket_pose_corrector():
    """Test function."""
    # Example usage of the RacketPoseCorrector

    import yaml  # pylint: disable = import-outside-toplevel
    import os  # pylint: disable = import-outside-toplevel

    with open(os.path.dirname(__file__) + "/config/config00.yaml", "r", encoding="utf8") as file_handle:
        params = yaml.load(file_handle)

    config = CorrectorConfig.from_dict(params)
    corrector = RacketPoseCorrector(config)
    corrector = RacketPoseCorrector(config)

    # Generate a series of quaternion samples representing rotation around a fixed axis
    def quaternion_from_axis_angle(axis, angle):
        axis = np.asarray(axis)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        half_angle = angle / 2.0
        return np.array(
            [
                np.sin(half_angle) * axis[0],
                np.sin(half_angle) * axis[1],
                np.sin(half_angle) * axis[2],
                np.cos(half_angle),
            ]
        )

    print("Quaternion series (rotation around Z axis) - Conservative UKF:")
    axis = [0, 0, 1]
    pose_input = {}
    np.random.seed(42)  # Reproducible results
    np.set_printoptions(precision=3, suppress=True)

    for i in range(10):
        angle = np.deg2rad(i * 10)  # 0, 10, 20, 30, 40 degrees
        q_true = quaternion_from_axis_angle(axis, angle)

        # Add small random noise to the quaternion
        noise = np.random.normal(0, 0.1, size=4)
        q_noisy = q_true + noise
        q_noisy = q_noisy / np.linalg.norm(q_noisy)

        pose_input["delta_time"] = 0.01
        pose_input["position"] = [1.0, 2.0, 3.0]
        pose_input["orientation"] = q_noisy
        # Use proper confidence values (0 to 1 range)
        pose_input["quat_confidence"] = 0.8 - abs(noise[0]) * 5  # Convert noise to confidence
        pose_input["quat_confidence"] = np.clip(pose_input["quat_confidence"], 0.1, 1.0)

        print(f"\nSample {i+1}:")
        print(f"  True quat:      {q_true}")
        print(f"  Noisy quat:     {q_noisy}")
        print(f"  Quat confidence: {pose_input['quat_confidence']:.3f}")

        # Correct the pose
        corrected_pose = corrector.correct_pose(pose_input)

        print(f"  Corrected quat: {corrected_pose['orientation']}")

        # Compute rotation errors
        true_angle = np.rad2deg(angle)
        corrected_quat = corrected_pose["orientation"]
        corrected_angle = 2 * np.rad2deg(np.arccos(np.clip(corrected_quat[3], -1, 1)))

        print(f"  True angle: {true_angle:.1f}°, Corrected angle: {corrected_angle:.1f}°")
        print(f"  Quat norm: {np.linalg.norm(corrected_pose['orientation']):.6f}")


if __name__ == "__main__":
    test_racket_pose_corrector()
