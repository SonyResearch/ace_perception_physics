"""
Confidential, Copyright 2024, Sony AI, All rights reserved
"""
import vision_common.python_module as vision_common
import time
import ace_yaml as yaml
import numpy as np
from scipy.spatial.transform import Rotation as R


def test_native_estimator():
    def random_vector(x, y, z):
        vec = np.array([(np.random.rand() - 0.5) * x, (np.random.rand() - 0.5) * y, (np.random.rand() - 0.5) * z])
        # print("random vector: ", vec)
        return vec

    np.random.seed(100)
    np.set_printoptions(suppress=True, precision=2)
    poses_frequency = 200
    min_history_length = 3
    max_history_length = 10
    estimator = vision_common.velocity_estimator()
    settings = {
        "poses_frequency": poses_frequency,
        "min_history_length": min_history_length,
        "max_history_length": max_history_length,
    }
    estimator.initialize_yaml(yaml.load_dict(settings))

    position = np.array([0, 0, 0])
    vel = random_vector(100, 100, 100)
    dt = 1 / 200.0
    dvel = vel * dt
    timestamp = 0.0
    counter = 0
    total_error_native = np.array([0, 0, 0])

    N = 1000
    for _ in range(N):
        success = estimator.estimate_velocity(timestamp, position)
        if success:
            # print(f"Native: {estimator.get_last_spin()} + {estimator.get_last_covariance()}")
            total_error_native = total_error_native + np.abs(np.array(vel - estimator.get_last_velocity()))

        timestamp += dt
        position = position + dvel
        counter += 1
        if counter > 5 * max_history_length:
            counter = 0
            vel = random_vector(100, 100, 100)
            dvel = vel * dt

    print(f"Native average error: {total_error_native/ N}")
    assert np.all(total_error_native / N < 5e0)


def runtime_test_subscriber():
    def random_vector(x, y, z):
        vec = np.array([(np.random.rand() - 0.5) * x, (np.random.rand() - 0.5) * y, (np.random.rand() - 0.5) * z])
        print("random vector: ", vec)
        return vec

    try:
        estimator = vision_common.velocity_estimator()
        poses_frequency = 200
        min_history_length = 5
        max_history_length = 10
        settings = {
            "poses_frequency": poses_frequency,
            "min_history_length": min_history_length,
            "max_history_length": max_history_length,
            "inliers_cos_distance": 0.95,
            "min_inliers": 3,
        }
        estimator.initialize_yaml(yaml.load_dict(settings))

        position = np.array([0, 0, 0])
        dt = 1 / 200.0
        velocity = random_vector(10, 10, 10)
        dpos = velocity * dt
        timestamp = 0.0
        counter = 0
        for i in range(100):
            success = estimator.estimate_velocity(timestamp, position)
            if success:
                print(f"Estimated: {estimator.get_last_velocity()} + {estimator.get_last_covariance()}")
            timestamp += dt
            position = position + dpos
            counter += 1
            if np.random.rand() > 0.9:
                position = position + np.array([0.1, 0.1, 0.1]) * 0.1
            if counter > 5 * max_history_length:
                velocity = random_vector(100, 100, 100)
                dpos = velocity * dt
                counter = 0

    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    runtime_test_subscriber()
