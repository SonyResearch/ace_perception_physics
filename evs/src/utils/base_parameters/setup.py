"""Package installation script"""
from setuptools import setup

package_name = 'base_parameters'

setup(
    name=package_name,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
)
