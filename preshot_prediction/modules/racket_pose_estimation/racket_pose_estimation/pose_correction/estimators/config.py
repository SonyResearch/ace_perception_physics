# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Configuration for the pose corrector."""


class CorrectorConfig:
    """Configuration for the pose corrector."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        max_dt=0.01,
        use_confidence=True,
        min_confidence=0.1,
        max_noise_multiplier=10.0,
        reset_after=10,
        process_noise_pos=0.1,
        process_noise_vel=0.1,
        measurement_noise_pos=0.5,
        max_angle=10,  # max angle before capping (MEKF)
        orientation_estimator_type="QuaternionMEKF",
        process_noise_quat=0.01,
        process_noise_omega=0.1,
        measurement_noise_quat=0.1,
    ):
        """
        Args:
            max_dt: max time step between measurements before doing multiple predictions
            process_noise_pos: Process noise for position
            process_noise_vel: Process noise for velocity
            process_noise_quat: Process noise for quaternion
            process_noise_omega: Process noise for angular velocity
            measurement_noise_pos: Base measurement noise for position
            measurement_noise_quat: Base measurement noise for quaternion
            use_confidence: Whether to use confidence-based adaptive noise
            min_confidence: Minimum confidence value to prevent division by zero
            max_noise_multiplier: Maximum multiplier for noise when confidence is very low
            reset_after: Number of consecutive missing poses before resetting the filter
        """
        self.max_dt = max_dt  # pylint: disable=invalid-name
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.measurement_noise_pos = measurement_noise_pos

        self.orientation_estimator_type = orientation_estimator_type
        self.process_noise_quat = process_noise_quat
        self.process_noise_omega = process_noise_omega
        self.measurement_noise_quat = measurement_noise_quat
        self.max_angle = max_angle

        self.use_confidence = use_confidence
        self.min_confidence = min_confidence
        self.max_noise_multiplier = max_noise_multiplier
        self.reset_after = reset_after

    @classmethod
    def from_dict(cls, config_dict):
        """Create a CorrectorConfig from a dictionary."""
        return cls(**config_dict)
