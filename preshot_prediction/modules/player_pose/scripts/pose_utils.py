"""Utilities for pose tracking."""
# pylint: disable = line-too-long
# Confidential, Copyright 2024, Sony AI, All rights reserved.


def get_main_bbox(bboxes):
    """Return the first bbox of a list of bboxes if any. Else returns None."""
    if not bboxes:
        return None
    return bboxes[0]


KEYPOINTS_NAMES_MAP = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_sho": 5,
    "r_sho": 6,
    "l_elb": 7,
    "r_elb": 8,
    "l_hand": 9,
    "r_hand": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_foot": 15,
    "r_foot": 16,
}

# To compute the length of some body part
JOINTS_LENGTH_IDX = {
    "r_eye_nose": [3, 1],
    "l_eye_nose": [2, 1],
    "r_sho_nose": [6, 1],
    "l_sho_nose": [5, 1],
    "r_sho_elb": [6, 8],
    "r_elb_hand": [8, 10],
    "l_sho_elb": [5, 7],
    "l_elb_hand": [7, 9],
    "r_sho_hip": [6, 12],
    "l_sho_hip": [5, 11],
    "r_hip_knee": [12, 14],
    "r_knee_foot": [14, 16],
    "l_hip_knee": [11, 13],
    "l_knee_foot": [13, 15],
    "r_sho_l_sho": [6, 5],
    "r_hip_l_hip": [12, 11],
    "r_eye_l_eye": [3, 2],
}

CONF_THR = 0.2

JOINT_PAIRS_ALT = [
    ["nose", "r_eye", "purple"],
    ["nose", "l_eye", "purple"],
    ["nose", "r_sho", "yellow"],
    ["nose", "l_sho", "yellow"],
    ["r_sho", "l_sho", "blue"],
    ["r_sho", "r_elb", "blue"],
    ["r_elb", "r_hand", "green"],
    ["l_sho", "l_elb", "blue"],
    ["l_elb", "l_hand", "green"],
    ["r_sho", "r_hip", "yellow"],
    ["l_sho", "l_hip", "yellow"],
    ["r_hip", "l_hip", "blue"],
    ["r_hip", "r_knee", "red"],
    ["r_knee", "r_foot", "skyblue"],
    ["l_hip", "l_knee", "red"],
    ["l_knee", "l_foot", "skyblue"],
]
