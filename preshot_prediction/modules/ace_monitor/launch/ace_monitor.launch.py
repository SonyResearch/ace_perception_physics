# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""This file will launch a ROS node to estimate player pose."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ace_loggers.glog_helper import roslog_to_glog_level


def generate_launch_description():
    """Resolve parameter paths and launch ace_monitor node."""
    launch_description = LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info", description="Set the log level of the launchfile"),
            DeclareLaunchArgument(
                "config", default_value="", description="lab specific configuration, e.g. tyo, zrch."
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
    config = LaunchConfiguration("config").perform(context)

    return [
        Node(
            package="ace_monitor",
            executable="ace_monitor_standalone",
            parameters=[],
            arguments=[f"--config={config}", "--colorlogtostderr=1", "--logtostderr=1", f"--minloglevel={glog_level}"],
            name="ace_monitor",
            namespace="utils",
            output="screen",
        ),
    ]
