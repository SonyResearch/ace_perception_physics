#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Collects multi-camera and robot observations for calibration purposes.

This is achieved by subscribing to ROS topics of both the robot and each individual camera
"""

import argparse
import logging
import os

import coloredlogs
import evs.src.calibration.calibration.recorder_utils as recorder_utils

log = logging.getLogger(__name__)


def parse_args():
    """
    Provide script-specific arguments.

    @return: Parsed argument object
    """
    parser = argparse.ArgumentParser(description="Collects multi-camera and robot observations")
    parser.add_argument(
        "--result_path",
        type=str,
        default="data/tmp",
        help="Path to store observations file to (default: %(default)s)",
    )
    parser.add_argument("--log_level", default=logging.INFO, help="Logging level (default: %(default)s)")

    return parser.parse_args()


def configure_logger(log_level=logging.INFO):
    """Set the logging options (level, messages color, ... etc)."""
    coloredlogs.install(level=log_level)
    log.setLevel(log_level)


if __name__ == "__main__":
    args = parse_args()
    configure_logger(args.log_level)

    recorder_utils.collect_observations(
        save_as=os.path.join(args.result_path, "observations.npz") if len(args.result_path) > 0 else None
    )
