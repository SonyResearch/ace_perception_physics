# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""This file will launch a ROS node."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ace_loggers.glog_helper import roslog_to_glog_level


def generate_launch_description():
    """Resolve parameters for the node."""

    launch_description = LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info", description="Set the log level of the launchfile"),
            DeclareLaunchArgument("config", description="configuration name"),
            OpaqueFunction(function=launch_setup),
        ]
    )

    return launch_description


def launch_setup(context):
    """
    Calls all relevant ROS nodes
    """
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))
    # Module path
    share_path = get_package_share_directory("ball_detection_trt")

    return [
        Node(
            package="ball_detection_trt",
            executable="ball_detection_trt_standalone",
            parameters=[
                {
                "params_path": [share_path, "/parameters/", LaunchConfiguration("config"), ".yaml"],
                "camera_calib_params_path": [
                    "/shared/camera_calibration/",
                    LaunchConfiguration("config"),
                    ".yaml",
                    ]
                }
            ],
            arguments=[
                ["--colorlogtostderr=1"],
                ["--logtostderr=1"],
                f"--minloglevel={glog_level}",
            ],
            name=["ball_detection_trt"],
            namespace="sensors",
            output="screen",
        ),
    ]
