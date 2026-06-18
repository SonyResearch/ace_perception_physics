# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""This file will launch multiple APS cameras."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ace_loggers.glog_helper import roslog_to_glog_level


def generate_launch_description():
    """Resolve parameter paths and launch multiple APS cameras."""

    return LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info", description="Set the log level of the launchfile"),
            DeclareLaunchArgument("config", description="Lab specific configuration, e.g. zrh00, zrh01, tyo02."),
            DeclareLaunchArgument(
                "parameters",
                default_value="",
                description="Parameters file name, if empty then will use config instead",
            ),
            DeclareLaunchArgument(
                "cluster", default_value="", description="Cluster configuration if any, e.g. _a, _b."
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )


def launch_setup(context):
    """
    Calls all relevant ROS nodes
    """
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))

    share_path = get_package_share_directory("aps")

    parameters_path = LaunchConfiguration("parameters").perform(context)
    if parameters_path != "":
        camera_params_path = share_path + "/parameters/camera_parameters/" + parameters_path + ".yaml"
    else:
        camera_params_path = [
            share_path + "/parameters/camera_parameters/",
            LaunchConfiguration("config"),
            ".yaml",
        ]
    params_path = [
        share_path + "/parameters/multi_camera_parameters/",
        LaunchConfiguration("config"),
        LaunchConfiguration("cluster"),
        ".yaml",
    ]
    return [
        Node(
            package="aps",
            executable="multi_camera_standalone",
            parameters=[
                {
                    "params_path": params_path,
                    "camera_params_path": camera_params_path,
                }
            ],
            arguments=[
                "--colorlogtostderr=1",
                "--logtostderr=1",
                f"--minloglevel={glog_level}",
            ],
            name=["multi_camera"],
            namespace="sensors",
            output="screen",
        ),
    ]
