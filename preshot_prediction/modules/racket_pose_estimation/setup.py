# Confidential, Copyright 2026, Sony AI, All rights reserved.
"""
racket_pose_estimation
"""

from ace_setuptools import get_data_mapping, setup

setup(
    data_files=get_data_mapping(share_files=["parameters/*", "models/*", "tests/*"]),
    entry_points={
        "console_scripts": [
            "racket_pose_filter = racket_pose_estimation.racket_pose_filter:main",
        ],
    },
)
