"""Package installation script"""
import os
from setuptools import setup

package_name = 'evs'
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
        "tools/evs_calibrate_evs",
        "tools/evs_trigger_to_txt",
        "tools/evs_run_reconstruction_recording",
        "tools/evs_run_reconstruction",
        "tools/evs_check_sync",
    ],
)
