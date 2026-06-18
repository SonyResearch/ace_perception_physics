# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Kalman filters for position estimation"""
import numpy as np
from .config import CorrectorConfig


# --- Position Kalman Filter ---
class PositionKalmanFilter:
    """
    Standard Kalman filter for position and velocity estimation.
    State: [x, y, z, vx, vy, vz]
    """

    def __init__(self, config: CorrectorConfig):
        self.config = config
        self.state_dim = 6
        self.measurement_dim = 3
        self.initialized = False
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim) * 10.0
        self.process_noise = np.zeros((self.state_dim, self.state_dim))
        self.process_noise[0:3, 0:3] = np.eye(3) * self.config.process_noise_pos
        self.process_noise[3:6, 3:6] = np.eye(3) * self.config.process_noise_vel
        self.measurement_noise = np.eye(3) * self.config.measurement_noise_pos
        self.measurement_matrix = np.zeros((self.measurement_dim, self.state_dim))  # measurement matrix
        self.measurement_matrix[0:3, 0:3] = np.eye(3)

    def compute_adaptive_noise(self, pos_confidence):
        """Compute adaptive measurement noise based on position confidence."""
        if not self.config.use_confidence:
            return self.measurement_noise.copy()
        pos_confidence = np.clip(pos_confidence, self.config.min_confidence, 1.0)
        pos_noise_multiplier = min(1.0 / pos_confidence, self.config.max_noise_multiplier)
        return np.eye(3) * self.config.measurement_noise_pos * pos_noise_multiplier

    def predict(self, delta_time):
        """Predict the next state."""
        if not self.initialized or delta_time <= 0:
            return
        state_transition = np.eye(self.state_dim)  # State transition matrix
        state_transition[0:3, 3:6] = np.eye(3) * delta_time
        self.state = state_transition @ self.state
        self.covariance = state_transition @ self.covariance @ state_transition.T + self.process_noise

    def update(self, measurement):
        """Update the filter with a new measurement."""
        position_measured = np.array(measurement["position"])
        pos_confidence = measurement.get("pos_confidence", 1.0)
        if not self.initialized:
            self.state[0:3] = position_measured
            self.state[3:6] = np.zeros(3)
            self.initialized = True
            return
        measurement_noise = self.compute_adaptive_noise(pos_confidence)
        position_predicted = self.measurement_matrix @ self.state
        residual = position_measured - position_predicted
        innovation_cov = self.measurement_matrix @ self.covariance @ self.measurement_matrix.T + measurement_noise
        kalman_gain = self.covariance @ self.measurement_matrix.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + kalman_gain @ residual
        self.covariance = (np.eye(self.state_dim) - kalman_gain @ self.measurement_matrix) @ self.covariance

    def get_position(self):
        """Get the current position and velocity estimates."""
        return self.state[0:3].copy(), self.state[3:6].copy()
