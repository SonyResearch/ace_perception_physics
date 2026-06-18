# type: ignore
# pylint: skip-file

"""
Spin estimation functionalities for the policy execution node.
"""

__copyright__ = "Confidential, Copyright 2024, Sony AI, All rights reserved."
__author__ = "Etienne Walther"
__maintainer__ = "Etienne Walther"
__email__ = "etienne.walther@sony.com"


from typing import Tuple, Union, List
import numpy as np
from scipy.spatial.transform import Rotation


class SpinEstimator:
    """Implementation of a class estimating spin from consecutive ball poses."""

    def __init__(
        self,
        ball_poses_frequency: int = 200,
        min_spin_history_length: int = 6,
        max_spin_history_length: int = 20,
        spin_obs_cov_scaling: float = 1.0e-3,
        min_inliers: int = 3,
        inliers_cos_distance: float = 0.95,
    ) -> None:
        """Implementation of a class estimating spin from consecutive ball poses.

        Args:
            ball_poses_frequency (float): Frequency of expected camera triangulations

            min_spin_history_length (int, optional): Min length of the spin-history required for the first valid spin
            estimation. Defaults to 10.

            max_spin_history_length (int, optional): Max length of the spin-history taken into account. Defaults to 20.

            spin_obs_cov_scaling (float, optional): Custom scaling factor for estimated spin observation covariance.
            Defaults to 1.0e-3.
        """
        self._ball_poses_frequency = ball_poses_frequency
        self._spin_obs_cov_scaling = spin_obs_cov_scaling

        self._min_inliers = min_inliers
        self._inliers_cos_distance = inliers_cos_distance

        self._ball_spin_history: List = []
        self._prev_rot_matrix: np.ndarray = np.eye(3)
        self._prev_ball_time: float = 0.0
        self._min_spin_history_length: int = min_spin_history_length
        self._max_spin_history_length: int = max_spin_history_length

    def estimate_ball_spin(self, ball_quat: np.ndarray, ball_time: float) -> Tuple[np.ndarray, Union[float, None]]:
        """Estimates the current spin from the last ball poses.

        Args:
            ball_quat (np.ndarray): Current ball orientation as quat (x,y,z,w).

            ball_time (float): Current time of the measurement.

        Returns:
            Tuple[np.ndarray, Union[float, None]]: Returns a tuple of spin expressed as a rotation vector [rad/s] and
            the related spin observation covariance.
        """
        ball_spin = np.array([np.nan, np.nan, np.nan])
        spin_obs_cov = None
        if ball_quat[0] < 1.0e3 and not np.all((ball_quat[:3] == 0.0)):
            ball_quat = ball_quat / np.linalg.norm(ball_quat)
            current_rot_matrix = Rotation.from_quat(ball_quat).as_matrix()

            if 0.0 < (ball_time - self._prev_ball_time) <= (1.0 / self._ball_poses_frequency + 1.0e-4):
                rot_diff = current_rot_matrix @ self._prev_rot_matrix.transpose()

                current_spin = Rotation.from_matrix(rot_diff).as_rotvec() / (ball_time - self._prev_ball_time)

                self._ball_spin_history = [current_spin] + self._ball_spin_history[: self._max_spin_history_length - 1]
                if len(self._ball_spin_history) >= self._min_spin_history_length:
                    spin_history = np.vstack(self._ball_spin_history)
                    spin_history = spin_history[self._get_history_inliers(), :]
                    if spin_history.shape[0] >= self._min_inliers:
                        ball_spin = np.mean(spin_history, axis=0)
                        spin_obs_cov = np.trace(np.cov(spin_history.transpose())) / 3 * self._spin_obs_cov_scaling

            self._prev_rot_matrix = current_rot_matrix.copy()
            self._prev_ball_time = ball_time
        else:
            # print("NO VALID MARKERS ON BALL OBSERVED!")
            # ball_spin = np.zeros(3)
            # spin_obs_cov = 0.0
            pass

        return ball_spin, spin_obs_cov

    def clear_spin_history(self):
        """Clear the history of previous spin measurements. This must be called whenever a change in spin is assumed
        (eg. at table- and racket-contacts)."""
        self._ball_spin_history = []
        self._prev_rot_matrix = np.eye(3)
        self._prev_ball_time = 0.0

    def _get_history_inliers(self):
        """Get all inliers which fit the current best estimate of the spin axis."""
        spin_history = np.vstack(self._ball_spin_history)
        median_spin = np.median(spin_history, axis=0)

        # if maginuted of median spin == 0.0 we don't find any inliers
        if np.linalg.norm(median_spin) == 0.0:
            return np.zeros(spin_history.shape[0]).astype(bool)

        median_spin_axis = median_spin / np.linalg.norm(median_spin)

        # if maginuted of spin of any element in the history == 0.0 we assume this as an outlier by setting the respective entries to nan
        spin_history_nom = np.linalg.norm(spin_history, axis=1)
        spin_history_axes = np.where(spin_history_nom != 0.0, spin_history.transpose() / spin_history_nom, np.nan)

        # calc dot products to find outliers
        dot_p = median_spin_axis @ spin_history_axes

        return dot_p > self._inliers_cos_distance
