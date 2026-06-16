# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Column rename mapping: MatchCollection → flat DataFrame names.

The MatchCollection rally.rally DataFrame uses short, prefixed column names,
while the extraction scripts (extract_racket_contacts.py, extract_table_contacts.py)
expect the legacy flat DataFrame column names from ``utilities.load_data()``.

This module provides a mapping dict and a helper that renames a rally DataFrame
*in-place* so extraction functions can work with MatchCollection data unchanged.
"""

import pandas as pd

# MatchCollection column name → legacy flat DataFrame column name
MATCHCOLLECTION_TO_FLAT: dict[str, str] = {
    # Ball APS position
    "x_aps":  "ball_position_x",
    "y_aps":  "ball_position_y",
    "z_aps":  "ball_position_z",
    # Ball GT200 position
    "x_gt200": "ball_gt200_x",
    "y_gt200": "ball_gt200_y",
    "z_gt200": "ball_gt200_z",
    # Ball GT200 velocity
    "vx_gt200": "ball_gt200_vx",
    "vy_gt200": "ball_gt200_vy",
    "vz_gt200": "ball_gt200_vz",
    # Ball GT200 spin (angular velocity)
    "wx":  "ball_gt200_wx",
    "wy":  "ball_gt200_wy",
    "wz":  "ball_gt200_wz",
    # Ball spin confidence
    "wx_confidence": "ball_gt200_wx_confidence",
    "wy_confidence": "ball_gt200_wy_confidence",
    "wz_confidence": "ball_gt200_wz_confidence",
    # Racket position
    "robot_racket_x":  "racket_1_position_x",
    "robot_racket_y":  "racket_1_position_y",
    "robot_racket_z":  "racket_1_position_z",
    # Racket velocity
    "robot_racket_vx": "racket_1_velocity_x",
    "robot_racket_vy": "racket_1_velocity_y",
    "robot_racket_vz": "racket_1_velocity_z",
    # Racket orientation (quaternion)
    "robot_racket_qx": "racket_1_orientation_x",
    "robot_racket_qy": "racket_1_orientation_y",
    "robot_racket_qz": "racket_1_orientation_z",
    "robot_racket_qw": "racket_1_orientation_w",
}


def rename_to_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Rename MatchCollection columns to flat DataFrame names.

    Returns a **copy** with renamed columns (the original is not modified).
    Only columns present in the DataFrame are renamed; others are left as-is.
    """
    mapping = {k: v for k, v in MATCHCOLLECTION_TO_FLAT.items() if k in df.columns}
    return df.rename(columns=mapping)
