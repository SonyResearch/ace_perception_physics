# Confidential, Copyright 2025, Sony AI, All rights reserved
"""Package installation script"""

from ace_setuptools import setup, get_data_mapping

setup(
    data_files=get_data_mapping(share_files=["config/*"]),
    install_requires=["argparse", "colored_glog", "onnxruntime"],
    scripts=["scripts/cat_ace_rt_profiles"],
)
