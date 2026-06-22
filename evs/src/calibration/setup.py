"""Package installation script"""

from ace_setuptools import get_data_mapping, setup

setup(
    data_files=get_data_mapping(share_files=["parameters/*", "launch/*"]),
    scripts=[
        "tools/calibration_calibrate_table",
        "tools/calibration_evaluate",
        "tools/calibration_run_kalibr",
        "tools/calibration_set_origin",
    ],
)
