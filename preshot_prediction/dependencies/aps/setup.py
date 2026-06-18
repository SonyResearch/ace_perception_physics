"""Package installation script"""

from ace_setuptools import get_data_mapping, setup

setup(
    data_files=get_data_mapping(share_files=["parameters/*", "launch/*"]),
    scripts=[
        "tools/aps_multi_camera_cpu_setup",
        "tools/aps_play_publisher",
        "tools/aps_rec_publisher",
        "tools/aps_show_camera_images",
    ],
)
