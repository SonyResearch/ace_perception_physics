# Confidential, Copyright 2025, Sony AI, All rights reserved.
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from racket_pose_estimation.pose_correction.racket_pose_corrector import CorrectorConfig, QuaternionKF, QuaternionMEKF


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


def quat_distance(q1, q2):
    # Returns the angle (in radians) between two quaternions
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot))


@pytest.mark.parametrize("noise_std", [0.01, 0.1, 0.2])
def test_quaternionkf_vs_mekf(noise_std):
    config = CorrectorConfig(
        process_noise_quat=0.01,
        process_noise_omega=0.1,
        measurement_noise_quat=0.05,
        use_confidence=True,
        min_confidence=0.1,
        max_noise_multiplier=10.0,
        max_angle=10.0,  # degrees
    )
    kf = QuaternionKF(config)
    mekf = QuaternionMEKF(config)
    axis = [0, 1, 1]
    axis = np.array(axis) / np.linalg.norm(axis)
    dt = 0.001
    np.random.seed(123)
    n_steps = 300
    errors_kf = []
    errors_mekf = []
    for i in range(n_steps):
        angle = np.deg2rad(i * 5) * dt
        q_true = quaternion_from_axis_angle(axis, angle)
        noise = np.random.normal(0, noise_std, size=4)
        q_noisy = q_true + noise
        q_noisy = q_noisy / np.linalg.norm(q_noisy)
        # Confidence score: higher noise -> lower confidence, clipped to [0.1, 1.0]
        confidence = 0.8 - np.abs(noise[0]) * 5
        confidence = float(np.clip(confidence, 0.1, 1.0))
        meas = {"orientation": q_noisy, "quat_confidence": confidence}
        kf.predict(dt)
        mekf.predict(dt)
        kf.update(meas)
        mekf.update(meas)
        q_kf, _ = kf.get_orientation()
        q_mekf, _ = mekf.get_orientation()
        err_kf = quat_distance(q_kf, q_true)
        err_mekf = quat_distance(q_mekf, q_true)
        errors_kf.append(err_kf)
        errors_mekf.append(err_mekf)
    mean_err_kf = np.mean(errors_kf)
    mean_err_mekf = np.mean(errors_mekf)
    print(f"Mean error KF: {mean_err_kf:.4f}, MEKF: {mean_err_mekf:.4f}")
    # Both filters should track the true quaternion with similar accuracy
    assert abs(mean_err_kf - mean_err_mekf) < 0.2
    assert mean_err_kf < 0.5 and mean_err_mekf < 0.5


if __name__ == "__main__":
    test_quaternionkf_vs_mekf(0.1)
