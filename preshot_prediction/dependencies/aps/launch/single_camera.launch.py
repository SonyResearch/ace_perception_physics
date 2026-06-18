# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""This file will launch a single APS camera."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ace_loggers.glog_helper import roslog_to_glog_level


def generate_launch_description():
    """Resolve parameter paths and launch a single APS camera."""

    return LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info", description="Set the log level of the launchfile"),
            DeclareLaunchArgument("config", description="Lab specific configuration, e.g. zrh00, zrh01, tyo02."),
            DeclareLaunchArgument("serial_number", description="Serial number of camera."),
            DeclareLaunchArgument("config", description="Lab specific configuration, e.g. zrh00, zrh01, tyo02."),
            OpaqueFunction(function=launch_setup),
        ]
    )


def launch_setup(context):
    """
    Calls all relevant ROS nodes
    """
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))

    share_path = get_package_share_directory("aps")

    return [
        Node(
            package="aps",
            executable="single_camera_standalone",
            parameters=[
                {
                    "serial_number": ParameterValue(LaunchConfiguration("serial_number"), value_type=str),
                    "camera_params_path": [
                        share_path + "/parameters/camera_parameters/",
                        LaunchConfiguration("config"),
                        ".yaml",
                    ],
                }
            ],
            arguments=[
                "--colorlogtostderr=1",
                "--logtostderr=1",
                f"--minloglevel={glog_level}",
            ],
            name=["aps", LaunchConfiguration("serial_number")],
            namespace="sensors",
            output="screen",
        ),
    ]
