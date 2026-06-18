# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""This file will launch a ROS node to estimate player pose."""

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
            DeclareLaunchArgument(
                "fast",
                default_value="True",
                description="Use fast models for keypoints detection (yolo nano) or not (yolo medium)",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )

    return launch_description


def launch_setup(context):
    """
    Calls all relevant ROS nodes
    """
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))
    player_share_path = get_package_share_directory("player_pose")
    camera_share_path = get_package_share_directory("calibration")
    config = LaunchConfiguration("config").perform(context)
    cluster = LaunchConfiguration("cluster").perform(context)
    use_fast = LaunchConfiguration("fast").perform(context).lower() in ["true", "y", "yes", "1"]

    parameters = LaunchConfiguration("parameters").perform(context)
    if parameters == "":
        parameters = config

    if use_fast:
        model_name = "yolo11n-pose-batch.onnx"
    else:
        model_name = "yolo11m-pose-batch.onnx"

    return [
        Node(
            package="player_pose",
            executable="player_pose_standalone",
            parameters=[
                {
                    "camera_config_path": camera_share_path + "/parameters/camera_calibration/" + config + ".yaml",
                    "player_config_path": player_share_path + "/parameters/" + parameters + cluster + ".yaml",
                    "keypoints_detector": player_share_path + "/models/" + model_name,
                }
            ],
            arguments=[
                ["--colorlogtostderr=1"],
                ["--logtostderr=1"],
                f"--minloglevel={glog_level}",
            ],
            name=["player_pose"],
            namespace="sensors",
            output="screen",
        ),
        Node(
            package="player_pose",
            executable="player_pose_filter",
            name="player_pose_filter",
            output="screen",
            parameters=[{"config": "defaults"}],
        ),
    ]
