"""Package installation script"""
from setuptools import setup, find_packages

package_name = 'data_generation'

setup(
    name=package_name,
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
)
