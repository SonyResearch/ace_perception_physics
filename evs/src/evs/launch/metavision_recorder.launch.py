# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""This file will launch a Metavision camera recorder."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ace_loggers.glog_helper import roslog_to_glog_level


def generate_launch_description():
    """Launch a Metavision camera recorder."""

    return LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info", description="Set the log level of the launchfile"),
            DeclareLaunchArgument("config", description="Lab specific configuration, e.g. zrh00, zrh01, tyo02."),
            DeclareLaunchArgument(
                "camera_type",
                default_value="gen4",
                description="Type of camera, either gen3 or gen4 (optional, default is gen4).",
            ),
            DeclareLaunchArgument("name", default_value="", description="Name of recording (optional)."),
            OpaqueFunction(function=launch_setup),
        ]
    )


def launch_setup(context):
    """
    Calls all relevant ROS nodes
    """
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))

    share_path = get_package_share_directory("evs")

    return [
        Node(
            package="evs",
            executable="metavision_recorder_standalone",
            parameters=[
                {
                    "multi_camera_params_path": [
                        share_path + "/parameters/multi_camera_parameters/",
                        LaunchConfiguration("config"),
                        ".yaml",
                    ],
                    "camera_params_path": [
                        share_path + "/parameters/camera_parameters/",
                        LaunchConfiguration("config"),
                        "_",
                        LaunchConfiguration("camera_type"),
                        ".yaml",
                    ],
                    "name": ParameterValue(LaunchConfiguration("name"), value_type=str),
                }
            ],
            arguments=[
                "--colorlogtostderr=1",
                "--logtostderr=1",
                f"--minloglevel={glog_level}",
            ],
            name=["metavision_recorder"],
            namespace="sensors",
            output="screen",
        ),
    ]
