"""
Confidential, Copyright 2024, Sony AI, All rights reserved
"""
import vision_common.python_module as vision_common
import time
import ace_yaml as yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from spin_estimation import SpinEstimator


def test_native_python():
    def random_vector(x, y, z):
        vec = np.array([(np.random.rand() - 0.5) * x, (np.random.rand() - 0.5) * y, (np.random.rand() - 0.5) * z])
        # print("random vector: ", vec)
        return vec

    np.random.seed(100)
    np.set_printoptions(suppress=True, precision=2)
    poses_frequency = 200
    min_history_length = 3
    max_history_length = 10
    min_inliers = 3
    inliers_quality = 0.95
    estimator = vision_common.spin_estimator()
    settings = {
        "poses_frequency": poses_frequency,
        "min_history_length": min_history_length,
        "max_history_length": max_history_length,
        "inliers_quality": inliers_quality,
        "min_inliers": min_inliers,
    }
    estimator.initialize_yaml(yaml.load_dict(settings))

    python_estimator = SpinEstimator(
        poses_frequency, min_history_length, max_history_length, min_inliers, inliers_quality
    )

    rotation = R.from_euler("xyz", [0, 0, 0], degrees=False)
    spin = random_vector(100, 100, 100)
    dt = 1 / 200.0
    drot = R.from_euler("xyz", spin * dt, degrees=False)
    timestamp = 0.0
    counter = 0
    total_error_python = np.array([0, 0, 0])
    total_error_native = np.array([0, 0, 0])

    N = 1000
    for _ in range(N):
        success = estimator.estimate_spin(timestamp, rotation.as_quat())
        p_est = python_estimator.estimate_ball_spin(rotation.as_quat(), timestamp)
        if success:
            # print(f"Native: {estimator.get_last_spin()} + {estimator.get_last_covariance()}")
            total_error_native = total_error_native + np.abs(np.array(spin - estimator.get_last_spin()))

        if not np.isnan(p_est[0]).any():
            total_error_python = total_error_python + np.abs(np.array(spin - p_est[0]))
            # print(f"Python: {p_est[0]} + {p_est[1]}")
        # else:
        #     print("Failed")
        timestamp += dt
        rotation = drot * rotation
        counter += 1
        if counter > 5 * max_history_length:
            counter = 0
            spin = random_vector(100, 100, 100)
            drot = R.from_euler("xyz", spin * dt, degrees=False)

    print(f"Native average error: {total_error_native/ N}")
    print(f"Python average error: {total_error_python/ N}")
    assert np.less_equal(total_error_native, total_error_python).all()
    assert np.all(total_error_python / N < 8e0)
    assert np.all(total_error_native / N < 8e0)
