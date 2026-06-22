"""Package installation script"""
import os
from setuptools import setup

package_name = 'calibration'
share = 'share/' + package_name

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    (share, ['package.xml']),
]
for d in ['parameters', 'launch']:
    for root, _, files in os.walk(d):
        if files:
            data_files.append((os.path.join(share, root), [os.path.join(root, f) for f in files]))

setup(
    name=package_name,
    data_files=data_files,
    scripts=[
        "tools/calibration_calibrate_table",
        "tools/calibration_evaluate",
        "tools/calibration_run_kalibr",
        "tools/calibration_set_origin",
    ],
)
