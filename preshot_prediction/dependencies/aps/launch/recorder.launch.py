# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""This file will launch an APS camera recorder."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ace_loggers.glog_helper import roslog_to_glog_level


def generate_launch_description():
    """Launch an APS camera recorder."""

    return LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info", description="Set the log level of the launchfile"),
            DeclareLaunchArgument("config", description="Lab specific configuration, e.g. zrh00, zrh01, tyo02."),
            DeclareLaunchArgument(
                "type",
                default_value="jpg",
                description="Type of recording, either raw, png or jpg (optional, default is jpg).",
            ),
            DeclareLaunchArgument("name", default_value="", description="Name of recording (optional)."),
            DeclareLaunchArgument(
                "topics",
                default_value="all",
                description="Comma-separated list of topics (optional, by default all image topics will be recorded).",
            ),
            DeclareLaunchArgument(
                "topic_filter",
                default_value="",
                description="A string for partial matching with topic names (used with topics:=all).",
            ),
            DeclareLaunchArgument(
                "ignore_gpu_checks", default_value="false", description="Ignore GPU checks (optional, default = false)."
            ),
            DeclareLaunchArgument("record_path", default_value="", description="Record path"),
            DeclareLaunchArgument("sampling", default_value="1", description="Sampling modulator (default = 1)"),
            DeclareLaunchArgument("quality", default_value="70", description="Quality factor 0~100 (default = 70)"),
            DeclareLaunchArgument(
                "publish_events",
                default_value="false",
                description="Publish image file path events when recorded (default = false)",
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

    return [
        Node(
            package="aps",
            executable="recorder_standalone",
            parameters=[
                {
                    "params_path": [
                        share_path + "/parameters/recorder_parameters/",
                        LaunchConfiguration("config"),
                        ".yaml",
                    ],
                    "recording_type": LaunchConfiguration("type"),
                    "name": ParameterValue(LaunchConfiguration("name"), value_type=str),
                    "topics": LaunchConfiguration("topics"),
                    "topic_filter": ParameterValue(LaunchConfiguration("topic_filter"), value_type=str),
                    "ignore_gpu_checks": LaunchConfiguration("ignore_gpu_checks"),
                    "record_path": LaunchConfiguration("record_path"),
                    "sampling": LaunchConfiguration("sampling"),
                    "quality": LaunchConfiguration("quality"),
                    "publish_events": LaunchConfiguration("publish_events"),
                }
            ],
            arguments=[
                "--colorlogtostderr=1",
                "--logtostderr=1",
                f"--minloglevel={glog_level}",
            ],
            name=["recorder"],
            namespace="sensors",
            output="screen",
        ),
    ]
