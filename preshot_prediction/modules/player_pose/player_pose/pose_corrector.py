
import logging
import numpy as np
import ace_interfaces.msg

log = logging.getLogger(__name__)

# --- Position Kalman Filter ---
class PositionKalmanFilter:
    """
    Standard Kalman filter for position and velocity estimation.
    State: [x, y, z, vx, vy, vz]
    """

    def __init__(self, config):
        self.state_dim = 6
        self.config=config
        self.measurement_dim = 3
        self.initialized = False
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim) * 10.0
        self.process_noise = np.zeros((self.state_dim, self.state_dim))
        self.process_noise[0:3, 0:3] = np.eye(3) * config["process_noise_pos"]
        self.process_noise[3:6, 3:6] = np.eye(3) * config["process_noise_vel"]
        self.measurement_noise = np.eye(3) * config["measurement_noise_pos"]
        self.measurement_matrix = np.zeros((self.measurement_dim, self.state_dim))  # measurement matrix
        self.measurement_matrix[0:3, 0:3] = np.eye(3)

    def compute_adaptive_noise(self, pos_confidence):
        """Compute adaptive measurement noise based on position confidence."""
        if not self.config.get("use_confidence", True):
            return self.measurement_noise.copy()
        pos_confidence = np.clip(pos_confidence, self.config.get("min_confidence", 0.0), 1.0)
        pos_noise_multiplier = min(1.0 / pos_confidence, self.config.get("max_noise_multiplier", 1.0), 1.0)
        return np.eye(3) * self.config["measurement_noise_pos"] * pos_noise_multiplier
    def predict(self, delta_time):
        """Predict the next state."""
        if not self.initialized or delta_time <= 0:
            return
        state_transition = np.eye(self.state_dim)  # State transition matrix
        state_transition[0:3, 3:6] = np.eye(3) * delta_time
        self.state = state_transition @ self.state
        self.covariance = state_transition @ self.covariance @ state_transition.T + self.process_noise

    def update(self, position_measured, pos_confidence=1.0):
        """Update the filter with a new measurement."""
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


class PlayerPoseCorrector:
    """Placeholder for PlayerPoseCorrector implementation."""

    KEYPOINT_MAP={
        "nose":0,
        "l_eye":1,
        "r_eye":2,
        "l_ear":3,
        "r_ear":4,
        "l_shoulder":5,
        "r_shoulder":6,
        "l_elbow":7,
        "r_elbow":8,
        "l_hand":9,
        "r_hand":10,
        "l_hip":11,
        "r_hip":12,
        "l_knee":13,
        "r_knee":14,
        "l_foot":15,
        "r_foot":16,
    }
    def __init__(self, config):
        self.keypoint_filters = []
        for _ in range(17):  # Assuming 17 keypoints
            self.keypoint_filters.append(PositionKalmanFilter(config))

    def predict(self, delta_time):
        for i, kf in enumerate(self.keypoint_filters):
            kf.predict(delta_time)

    def correct_pose(self, pose:ace_interfaces.msg.PlayerPose):
        kf:PositionKalmanFilter=None
        for i, kf in enumerate(self.keypoint_filters):
            keypoint = pose.keypoints[i]
            confidence = pose.confidences[i]
            projection_error_pct = 1-min(pose.projection_error[i]/30.0,1)
            kp=np.array([keypoint.x, keypoint.y, keypoint.z])
            if np.any(np.isnan(kp)) or np.any(np.isinf(kp)) or np.any(np.abs(kp)>1e2):
                confidence = 0.0
            confidence *= projection_error_pct
            if confidence > 0:
                kf.update(kp, confidence)
                # pose.projection_error[i]=1
            keypoint.x=kf.state[0]
            keypoint.y=kf.state[1]
            keypoint.z=kf.state[2]
            # If no confidence, we can choose to not update the filter
    def get_filtered_pose(self, msg):
        for i, kf in enumerate(self.keypoint_filters):
            msg.keypoints[i].x = float(kf.state[0])
            msg.keypoints[i].y = float(kf.state[1])
            msg.keypoints[i].z = float(kf.state[2])
        return msg