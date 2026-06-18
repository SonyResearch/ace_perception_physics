# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""This file will launch a ROS node to triangulate balls from detections in multiple calibrated cameras."""

from ace_loggers.glog_helper import roslog_to_glog_level
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Resolve parameter paths and launch ball triangulation node."""

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
                "detection_parameters",
                default_value="",
                description="Ball Detection Parameters file name, if empty then will use config instead",
            ),
            DeclareLaunchArgument(
                "clusters", default_value="", description="Comma-separated cluster configurations if any, e.g. '_a,_b'."
            ),
            DeclareLaunchArgument("name", default_value="ball_triangulation", description="Node name use in publisher"),
            DeclareLaunchArgument("use_aps", default_value="True", description="Use aps cameras."),
            DeclareLaunchArgument("use_evs", default_value="False", description="Use evs cameras."),
            OpaqueFunction(function=launch_setup),
        ]
    )


def launch_setup(context):
    """
    Calls all relevant ROS nodes
    """
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))
    config = LaunchConfiguration("config").perform(context)
    clusters = LaunchConfiguration("clusters").perform(context).replace(" ", "").split(",")
    name = LaunchConfiguration("name").perform(context)

    parameters = LaunchConfiguration("parameters").perform(context)
    detection_parameters = LaunchConfiguration("detection_parameters").perform(context)
    if parameters == "":
        parameters = config
    if detection_parameters == "":
        detection_parameters = config
    camera_types = []
    if LaunchConfiguration("use_aps").perform(context).lower() in ["true", "y", "yes", "1"]:
        camera_types.append("aps")
    if LaunchConfiguration("use_evs").perform(context).lower() in ["true", "y", "yes", "1"]:
        camera_types.append("evs")

    share_path = get_package_share_directory("multi_ball_triangulation")
    params_path = share_path + "/parameters/" + parameters + ".yaml"

    ball_detection_params_paths = []
    for camera_type in camera_types:
        ball_detection_share_path = get_package_share_directory("ball_detection_" + camera_type)
        for cluster in clusters:
            ball_detection_params_paths.append(
                ball_detection_share_path + "/parameters/" + detection_parameters + cluster + ".yaml"
            )

    calib_share_path = get_package_share_directory("calibration")
    camera_calib_params_path = calib_share_path + "/parameters/camera_calibration/" + config + ".yaml"

    return [
        Node(
            package="multi_ball_triangulation",
            executable="multi_ball_triangulation_standalone",
            parameters=[
                {
                    "params_path": params_path,
                    "ball_detection_params_paths": ball_detection_params_paths,
                    "camera_calib_params_path": camera_calib_params_path,
                }
            ],
            arguments=[
                "--colorlogtostderr=1",
                "--logtostderr=1",
                f"--minloglevel={glog_level}",
            ],
            name=name,
            namespace="sensors",
            output="screen",
        ),
    ]
