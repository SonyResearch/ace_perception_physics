"""Normalize calibration by removing the T_WO transformation."""
#!/usr/bin/env python3
# Confidential, Copyright 2025, Sony AI, All rights reserved.
# pylint: disable = line-too-long, import-error, no-name-in-module, no-member

import argparse
import sys
import numpy as np
import ace_yaml as yaml


def calc_rotation(angle):  # pylint: disable = redefined-outer-name
    """Round angle to [-90, 0, 90, 180] degress."""
    while abs(angle) > 180:
        angle -= np.sign(angle) * 360

    if angle < -135 or angle >= 135:
        return 180

    if -135 <= angle < -45:
        return -90

    if -45 <= angle < 45:
        return 0

    if 45 <= angle < 135:
        return 90

    return "invalid"


if __name__ == "__main__":
    # setup argument list
    parser = argparse.ArgumentParser(description="Normalize calibration.")

    parser.add_argument(
        "--config", type=str, required=True, help="Lab specific configuration, e.g. zrh00, zrh01, tyo02."
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # parse argument list
    try:
        parsed = parser.parse_args()
    except Exception:  # pylint: disable = broad-except
        sys.exit(0)

    # load camera calibration
    try:
        # read yaml file for storage
        FILE_PATH = "src/sensors/calibration/parameters/camera_calibration/" + parsed.config + ".yaml"
        FILE = open(FILE_PATH, "r")  # pylint: disable = unspecified-encoding, consider-using-with

        skip_line_counter = 0  # pylint: disable = invalid-name

        FILE_CONTENT = ""
        for line in FILE:
            if skip_line_counter > 0:
                skip_line_counter -= 1
                continue

            if line.startswith("T_WO:"):
                skip_line_counter = 4  # pylint: disable = invalid-name
                continue

            FILE_CONTENT += line

        FILE.close()

        # read yaml file
        FILE = open(FILE_PATH, "r")  # pylint: disable = unspecified-encoding, consider-using-with
        params = yaml.full_load(FILE)
        FILE.close()

    except IOError:
        print("Failed to open file.")
        sys.exit()

    T_WO = np.array(params["T_WO"])
    for camera in params:
        if not camera.startswith("aps"):
            continue
        params[camera]["T_CW"] = (params[camera]["T_CW"] @ T_WO).tolist()
    params["T_WO"] = np.identity(4).tolist()

    with open(FILE_PATH, "w") as yaml_file:  # pylint: disable = unspecified-encoding
        yaml_file.write(yaml.dump(params, default_flow_style=None, sort_keys=False))
    print("Normalization complete.")
