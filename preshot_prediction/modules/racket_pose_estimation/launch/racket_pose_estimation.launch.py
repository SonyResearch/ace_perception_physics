# Confidential, Copyright 2026, Sony AI, All rights reserved.
"""This file will launch a ROS node to estimate racket pose."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ace_loggers.glog_helper import roslog_to_glog_level


def generate_launch_description():
    """Resolve parameter paths and launch ball pose-estimation node."""

    launch_description = LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info", description="Set the log level of the launchfile"),
            DeclareLaunchArgument("config", description="lab specific configuration, e.g. tyo, zrch."),
            DeclareLaunchArgument(
                "parameters",
                default_value="",
                description="Parameters file name, if empty then will use config instead",
            ),
            DeclareLaunchArgument("cluster", default_value="", description="camera cluster suffix"),
            DeclareLaunchArgument("version", default_value="v11n", description="network_version"),
            DeclareLaunchArgument("tune", default_value="false", description="Tune racket pose estimator parameters"),
            OpaqueFunction(function=launch_setup),
        ]
    )

    return launch_description


def launch_setup(context):
    """
    Calls all relevant ROS nodes
    """
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))
    racket_share_path = get_package_share_directory("racket_pose_estimation")
    camera_share_path = get_package_share_directory("calibration")

    config = LaunchConfiguration("config").perform(context)
    cluster = LaunchConfiguration("cluster").perform(context)
    version = LaunchConfiguration("version").perform(context)
    tune = LaunchConfiguration("tune").perform(context)
    parameters = LaunchConfiguration("parameters").perform(context)
    if parameters == "":
        parameters = config

    racket_detector = f"{racket_share_path}/models/racket_{version}.onnx"

    return [
        Node(
            package="racket_pose_estimation",
            executable="racket_pose_estimation_standalone",
            parameters=[
                {
                    "camera_config_path": camera_share_path + "/parameters/camera_calibration/" + config + ".yaml",
                    "racket_config_path": racket_share_path + "/parameters/" + parameters + cluster + ".yaml",
                    "racket_detector": racket_detector,
                    "tune": tune,
                }
            ],
            arguments=[
                ["--colorlogtostderr=1"],
                ["--logtostderr=1"],
                f"--minloglevel={glog_level}",
            ],
            name=["racket_pose_estimation"],
            namespace="sensors",
            output="screen",
        ),
        Node(
            package="racket_pose_estimation",
            executable="racket_pose_filter",
            name="racket_pose_filter",
            output="screen",
            parameters=[{"config": "config_mekf"}],
        ),
    ]
