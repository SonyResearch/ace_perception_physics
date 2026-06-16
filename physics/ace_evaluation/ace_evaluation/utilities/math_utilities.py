# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.

import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def q_to_rot_mat(quat: np.ndarray, q_format: str = "wxyz") -> np.ndarray:
    """
    Transform a quaternion to the equivalent 3x3 rotation matrix.

    Parameters
    ----------
    quat : np.ndarray
        4-element quaternion array.
    q_format : str
        ``"wxyz"`` (default) or ``"xyzw"``.
    """
    if q_format == "wxyz":
        q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]
    elif q_format == "xyzw":
        q_x, q_y, q_z, q_w = quat[0], quat[1], quat[2], quat[3]
    else:
        raise ValueError(
            f"Unknown quaternion format {q_format!r}, expected 'xyzw' or 'wxyz'"
        )

    return np.array(
        [
            [
                1 - 2 * (q_y**2 + q_z**2),
                2 * (q_x * q_y - q_w * q_z),
                2 * (q_x * q_z + q_w * q_y),
            ],
            [
                2 * (q_x * q_y + q_w * q_z),
                1 - 2 * (q_x**2 + q_z**2),
                2 * (q_y * q_z - q_w * q_x),
            ],
            [
                2 * (q_x * q_z - q_w * q_y),
                2 * (q_y * q_z + q_w * q_x),
                1 - 2 * (q_x**2 + q_y**2),
            ],
        ]
    )


def quat_quat_rot(
    q_1: np.ndarray, q_2: np.ndarray, q_format: str = "wxyz"
) -> np.ndarray:
    """
    Apply the rotation of quaternion *q_2* to quaternion *q_1*.

    Parameters
    ----------
    q_1, q_2 : np.ndarray
        4-element quaternion arrays.
    q_format : str
        ``"wxyz"`` (default) or ``"xyzw"``.
    """
    if q_format == "wxyz":
        q_w, q_x, q_y, q_z = q_1[0], q_1[1], q_1[2], q_1[3]
        r_w, r_x, r_y, r_z = q_2[0], q_2[1], q_2[2], q_2[3]
    elif q_format == "xyzw":
        q_x, q_y, q_z, q_w = q_1[0], q_1[1], q_1[2], q_1[3]
        r_x, r_y, r_z, r_w = q_2[0], q_2[1], q_2[2], q_2[3]
    else:
        raise ValueError(
            f"Unknown quaternion format {q_format!r}, expected 'xyzw' or 'wxyz'"
        )

    t_0 = r_w * q_w - r_x * q_x - r_y * q_y - r_z * q_z  # w
    t_1 = r_w * q_x + r_x * q_w - r_y * q_z + r_z * q_y  # x
    t_2 = r_w * q_y + r_x * q_z + r_y * q_w - r_z * q_x  # y
    t_3 = r_w * q_z - r_x * q_y + r_y * q_x + r_z * q_w  # z

    if q_format == "wxyz":
        return np.array([t_0, t_1, t_2, t_3])
    return np.array([t_1, t_2, t_3, t_0])


def quaternion_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w])


def quaternion_conjugate(q):
    """Return the conjugate of quaternion [x, y, z, w]"""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_normalize(q):
    """
    Normalize a quaternion to unit length.

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])  # Default to identity
    return q / norm


def calculate_spin(input_subset: pd.DataFrame) -> np.ndarray:
    """calculate spin"""
    # if this is still needed, consider having the full length instead of cutting 1 value
    if "ball_gt200_wx" in input_subset:
        return input_subset[["ball_gt200_wx", "ball_gt200_wy", "ball_gt200_wz"]][
            :-1
        ].values

    print("no spin available")
    exit()
    if "ball_spin_x" in input_subset:
        return input_subset[["ball_spin_x", "ball_spin_y", "ball_spin_z"]][:-1].values
    if "ball_orientation_x" in input_subset:
        orientations = input_subset[
            [
                "ball_orientation_x",
                "ball_orientation_y",
                "ball_orientation_z",
                "ball_orientation_w",
            ]
        ].values
        orientations_normalized = (
            orientations / np.linalg.norm(orientations, axis=1)[:, np.newaxis]
        )

        rot_matrices = [R.from_quat(r).as_matrix() for r in orientations_normalized]
        assert len(rot_matrices) - 1 > 1

        spin = [
            R.from_matrix(rot_matrices[i + 1] @ rot_matrices[i].transpose()).as_rotvec()
            / np.diff(input_subset["time"])[i]
            for i in range(len(rot_matrices) - 1)
        ]
        return np.stack(spin, axis=0)

    print("unrecognized type in calculate_spin")
    sys.exit()
    return 0


def compute_angular_velocity(q1, q2, delta_t):
    # Convert quaternions to scipy Rotation objects
    r1 = R.from_quat(q1)  # Assume input quaternions are [x, y, z, w]
    r2 = R.from_quat(q2)

    # Compute the relative rotation
    q_rel = r2 * r1.inv()

    # Extract the rotation vector (axis * angle)
    rot_vec = q_rel.as_rotvec()  # Rotation vector (angle * axis)

    # Compute angular velocity
    angular_velocity = rot_vec / delta_t
    return angular_velocity


def project_to_racket(v_ref, v, rot):
    # rotate in the racket frame of reference
    rotation = rot.inv().as_matrix()
    tmp_v_ref = rotation @ v_ref
    tmp_v = rotation @ v

    # rotate racket plane to align with velocity
    _, vry, vrz = tmp_v_ref
    norm_yz = np.sqrt(vry**2 + vrz**2)
    u_x = np.array([1.0, 0.0, 0.0])
    u_y = np.array([0.0, vry / norm_yz, vrz / norm_yz])
    u_z = np.array([0.0, -vrz / norm_yz, vry / norm_yz])

    rotation_matrix = np.vstack([u_x, u_y, u_z])
    return rotation_matrix @ tmp_v
