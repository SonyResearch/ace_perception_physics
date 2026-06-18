# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Kalman filters for orientation estimation"""

import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from .config import CorrectorConfig


class QuaternionHelpers:
    """Helper functions for quaternion operations."""

    @staticmethod
    def quaternion_conjugate(quat):
        """Compute the conjugate of a quaternion."""
        # (x, y, z, w) format
        return np.array([-quat[0], -quat[1], -quat[2], quat[3]])

    @staticmethod
    def normalize_quaternion(quat):
        """Normalize a quaternion."""
        norm = np.linalg.norm(quat)
        if norm > 1e-8:
            return quat / norm
        return np.array([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def quaternion_multiply(quat_1, quat_2):
        """Multiply two quaternions."""
        x_1, y_1, z_1, w_1 = quat_1
        x_2, y_2, z_2, w_2 = quat_2
        x_val = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
        y_val = w_1 * y_2 - x_1 * z_2 + y_1 * w_2 + z_1 * x_2
        z_val = w_1 * z_2 + x_1 * y_2 - y_1 * x_2 + z_1 * w_2
        w_val = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
        return np.array([x_val, y_val, z_val, w_val])

    @staticmethod
    def angular_velocity_to_quaternion_derivative(quat, omega):
        """Compute the quaternion derivative from angular velocity."""
        omega_quat = np.array([omega[0], omega[1], omega[2], 0.0])
        dq_dt = 0.5 * QuaternionHelpers.quaternion_multiply(quat, omega_quat)
        return dq_dt


class BaseQuaternionEstimator:
    """Base class for quaternion estimators."""

    def __init__(self, quat_dim, config: CorrectorConfig):
        self.config = config
        self.quat_dim = quat_dim
        self.r_base = np.eye(quat_dim) * self.config.measurement_noise_quat  # Measurement noise covariance

    def compute_adaptive_noise(self, quat_confidence):
        """Compute adaptive noise covariance based on quaternion confidence."""
        if not self.config.use_confidence:
            return self.r_base.copy()
        quat_confidence = np.clip(quat_confidence, self.config.min_confidence, 1.0)
        quat_noise_multiplier = min(1.0 / quat_confidence, self.config.max_noise_multiplier)
        return np.eye(self.quat_dim) * self.config.measurement_noise_quat * quat_noise_multiplier


# --- Quaternion Multiplicative Extended Kalman Filter (MEKF) ---
class QuaternionMEKF(BaseQuaternionEstimator):
    """
    Multiplicative Extended Kalman Filter for quaternion (rotation) estimation.
    State: [qx, qy, qz, qw, wx, wy, wz] (quaternion + angular velocity)
    """

    def __init__(self, config: CorrectorConfig):
        super().__init__(3, config)
        self.measurement_dim = 4
        self.initialized = False
        self.quat = np.zeros(4)
        self.omega = np.zeros(3)
        self.covariance = np.eye(6) * 10.0  # Initial error covariance
        self.process_noise = np.eye(6)  # Process noise covariance
        self.process_noise[0:3, 0:3] *= self.config.process_noise_quat  # Quaternion noise
        self.process_noise[3:6, 3:6] *= self.config.process_noise_omega  # Angular velocity noise
        self.measurement_matrix = np.zeros((3, 6))
        self.measurement_matrix[:, :3] = np.eye(3)  # quat as rotvec representation

    def predict(self, delta_time):
        """Predict the next state."""
        if not self.initialized or delta_time <= 0:
            return
        delta_theta = self.omega * delta_time
        # Quaternion propagation using estimated omega
        delta_quat = R.from_rotvec(delta_theta)  # (x, y, z, w)
        quat_pred = delta_quat * R.from_quat(self.quat)
        quat_pred = quat_pred.as_quat()

        self.quat = quat_pred
        # Jacobian (linearized)
        state_transition = np.eye(6)  # State transition matrix
        state_transition[0:3, 3:6] = np.eye(3, 3) * delta_time
        self.covariance = state_transition @ self.covariance @ state_transition.T + self.process_noise

    def update(self, measurement):
        """Update the filter with a new measurement."""
        quat_meas = np.array(measurement["orientation"])
        quat_meas = QuaternionHelpers.normalize_quaternion(quat_meas)
        quat_confidence = measurement.get("quat_confidence", 1.0)
        if not self.initialized:
            self.quat = quat_meas
            self.omega = np.zeros(3)
            self.initialized = True
            return

        # Handle quaternion wraparound
        if np.dot(self.quat, quat_meas) < 0:
            quat_meas = -quat_meas

        quat_pred = R.from_quat(self.quat)
        rotvec_error = R.from_quat(quat_meas) * quat_pred.inv()
        rotvec_error = rotvec_error.as_rotvec()  # Convert quaternion error to rotation vector

        measurement_noise = self.compute_adaptive_noise(quat_confidence)
        innovation_cov = self.measurement_matrix @ self.covariance @ self.measurement_matrix.T + measurement_noise
        kalman_gain = self.covariance @ self.measurement_matrix.T @ np.linalg.inv(innovation_cov)

        self.covariance = (np.eye(6) - kalman_gain @ self.measurement_matrix) @ self.covariance
        # Enforce symmetry for numerical stability
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        # apply correction to state
        delta_state = kalman_gain @ rotvec_error
        # Cap the correction step to avoid instability
        max_angle = self.config.max_angle * np.pi / 180.0
        angle_norm = np.linalg.norm(delta_state[:3])
        if angle_norm > max_angle:
            delta_state[:3] = delta_state[:3] * (max_angle / angle_norm)
            logging.info(  # pylint: disable=logging-fstring-interpolation
                f"MEKF correction step capped: {angle_norm:.3f} -> {max_angle}"
            )

        delta_quat = R.from_rotvec(delta_state[:3])  # Convert rotation vector to quaternion
        self.quat = (delta_quat * R.from_quat(self.quat)).as_quat()
        self.omega += delta_state[3:]

    def get_orientation(self):
        """Get the current orientation and angular velocity."""
        return self.quat.copy(), self.omega.copy()


# --- Quaternion Kalman Filter (KF) ---
class QuaternionKF(BaseQuaternionEstimator):
    """
    Kalman Filter for quaternion (rotation) estimation.
    State: [qx, qy, qz, qw, wx, wy, wz] (quaternion + angular velocity)
    """

    def __init__(self, config: CorrectorConfig):
        super().__init__(4, config)
        self.state_dim = 7
        self.measurement_dim = 4
        self.initialized = False
        self.state = np.zeros(self.state_dim)
        self.state[3] = 1.0  # Identity quaternion
        self.covariance = np.eye(self.state_dim) * 10.0
        self.process_noise = np.zeros((self.state_dim, self.state_dim))
        self.process_noise[0:4, 0:4] = np.eye(4) * self.config.process_noise_quat
        self.process_noise[4:7, 4:7] = np.eye(3) * self.config.process_noise_omega
        self.measurement_matrix = np.zeros((self.measurement_dim, self.state_dim))
        self.measurement_matrix[:, 0:4] = np.eye(4)

    def predict(self, delta_time):
        """Predict the next state."""
        if not self.initialized or delta_time <= 0:
            return
        # Quaternion prediction
        quat = self.state[0:4]
        omega = self.state[4:7]
        delta_theta = omega * delta_time
        # Quaternion propagation using estimated omega
        delta_quat = R.from_rotvec(delta_theta).as_quat()  # (x, y, z, w)
        quat_pred = QuaternionHelpers.quaternion_multiply(quat, delta_quat)
        quat_pred = QuaternionHelpers.normalize_quaternion(quat_pred)
        self.state[0:4] = quat_pred
        # Jacobian (linearized)
        state_transition = np.eye(self.state_dim)
        state_transition[0:4, 4:7] = np.eye(4, 3) * delta_time
        self.covariance = state_transition @ self.covariance @ state_transition.T + self.process_noise

    def update(self, measurement):
        """Update the filter with a new measurement."""
        quat_meas = np.array(measurement["orientation"])
        quat_meas = QuaternionHelpers.normalize_quaternion(quat_meas)
        quat_confidence = measurement.get("quat_confidence", 1.0)
        if not self.initialized:
            self.state[0:4] = quat_meas
            self.state[4:7] = np.zeros(3)
            self.initialized = True
            return

        # Handle quaternion wraparound
        quat_pred = self.state[0:4]
        if np.dot(quat_pred, quat_meas) < 0:
            quat_meas = -quat_meas

        residual = quat_meas - quat_pred  # residual

        measurement_noise = self.compute_adaptive_noise(quat_confidence)
        innovation_cov = self.measurement_matrix @ self.covariance @ self.measurement_matrix.T + measurement_noise
        kalman_gain = self.covariance @ self.measurement_matrix.T @ np.linalg.inv(innovation_cov)
        self.covariance = (np.eye(self.state_dim) - kalman_gain @ self.measurement_matrix) @ self.covariance
        # Enforce symmetry for numerical stability
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        # apply correction to state
        delta_state = kalman_gain @ residual
        self.state = self.state + delta_state
        self.state[0:4] = QuaternionHelpers.normalize_quaternion(self.state[0:4])

    def get_orientation(self):
        """Get the current orientation and angular velocity."""
        return self.state[0:4].copy(), self.state[4:7].copy()
