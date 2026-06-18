# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""This file will launch a ROS node to detect balls in APS images."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ace_loggers.glog_helper import roslog_to_glog_level


def generate_launch_description():
    """Resolve parameter paths and launch ball detection aps node."""

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
            DeclareLaunchArgument(
                "debug",
                default_value="",
                description='Enables debug mode with additional output. Possible values are "detection" and "mask".',
            ),
            DeclareLaunchArgument(
                "ignore_gpu_checks", default_value="false", description="Ignore GPU checks (optional, default = false)."
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )


def launch_setup(context):
    """
    Calls all relevant ROS nodes
    """
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))

    share_path = get_package_share_directory("ball_detection_aps")
    calib_share_path = get_package_share_directory("calibration")
    parameters_path = LaunchConfiguration("parameters").perform(context)
    cluster = LaunchConfiguration("cluster").perform(context)
    if parameters_path != "":
        params_path = share_path + "/parameters/" + parameters_path + cluster + ".yaml"
    else:
        params_path = [
            share_path + "/parameters/",
            LaunchConfiguration("config"),
            LaunchConfiguration("cluster"),
            ".yaml",
        ]

    return [
        Node(
            package="ball_detection_aps",
            executable="ball_detection_aps_standalone",
            parameters=[
                {
                    "params_path": params_path,
                    "camera_calib_params_path": [
                        calib_share_path + "/parameters/camera_calibration/",
                        LaunchConfiguration("config"),
                        ".yaml",
                    ],
                    "debug": ParameterValue(LaunchConfiguration("debug"), value_type=str),
                    "ignore_gpu_checks": LaunchConfiguration("ignore_gpu_checks"),
                }
            ],
            arguments=[
                "--colorlogtostderr=1",
                "--logtostderr=1",
                f"--minloglevel={glog_level}",
            ],
            name=["ball_detection_aps"],
            namespace="sensors",
            output="screen",
        ),
    ]
