"""
Package installation script

@copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
"""

from ace_setuptools import setup

setup(
    install_requires=["numpy>=1.23.5,<2.0.0"],
    tests_require=["pytest"],
)
