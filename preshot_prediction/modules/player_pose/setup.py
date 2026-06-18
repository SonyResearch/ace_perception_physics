"""
Package installation script
Confidential, Copyright 2024, Sony AI, All rights reserved
"""
from ace_setuptools import get_data_mapping, setup

setup(
    data_files=get_data_mapping(share_files=["parameters/*", "launch/*"]),
    entry_points={
        "console_scripts": [
            "player_pose_filter = player_pose.player_pose_filter:main",
        ],
    },
)
