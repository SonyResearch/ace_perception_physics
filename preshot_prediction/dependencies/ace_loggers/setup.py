"""
Package installation script

@copyright Confidential, Copyright 2025, Sony AI, All rights reserved.
"""

from ace_setuptools import setup

setup(
    install_requires=["numpy>=1.23.5,<2.0.0", "orjson>=3.0.0"],
    scripts=[
        "tools/log_control",
        "tools/log_convert",
        "tools/log_play",
    ],
)
