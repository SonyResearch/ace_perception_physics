"""Package installation script"""

from ace_setuptools import get_data_mapping, setup

setup(
    data_files=get_data_mapping(share_files=["parameters/*", "launch/*"]),
    scripts=[
        "tools/evs_calibrate_evs",
        "tools/evs_trigger_to_txt",
        "tools/evs_run_reconstruction_recording",
        "tools/evs_run_reconstruction",
        "tools/evs_check_sync",
    ],
)
