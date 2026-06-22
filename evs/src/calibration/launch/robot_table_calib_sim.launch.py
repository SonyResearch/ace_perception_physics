# pylint: skip-file
# SPDX-License-Identifier: MIT
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
def roslog_to_glog_level(level: str) -> int:
    return {"debug": 0, "info": 0, "warn": 1, "warning": 1, "error": 2, "fatal": 3}.get(level.lower(), 0)
from robot_models.robot_parameters import RobotScenario

from ace_launch.helper_robot_common_nodes import ball_shooter_node
from ace_launch.helper_robot_simulator_nodes import bullet_simulator_node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info", description="Set the log level of the launchfile"),
            DeclareLaunchArgument("control_frequency", description="Frequency of robot control input in Hz"),
            DeclareLaunchArgument(
                "robot_name", default_value="staubli_60l", description="Short name of the robot model to use."
            ),
            DeclareLaunchArgument(
                "ball_prediction_config",
                default_value="parameters_default.yaml",
                description="The EKF config file path",
            ),
            LogInfo(msg="Log level:"),
            LogInfo(msg=LaunchConfiguration("log_level")),
            LogInfo(msg="Control frequency:"),
            LogInfo(msg=LaunchConfiguration("control_frequency")),
            LogInfo(msg="Robot name:"),
            LogInfo(msg=LaunchConfiguration("robot_name")),
            LogInfo(msg="Ball prediction config:"),
            LogInfo(msg=LaunchConfiguration("ball_prediction_config")),
            OpaqueFunction(function=launch_setup),
        ]
    )


def launch_setup(context):
    glog_level = roslog_to_glog_level(LaunchConfiguration("log_level").perform(context))
    robot_name = LaunchConfiguration("robot_name").perform(context)
    assert robot_name in RobotScenario.valid_robot_names(), "Unknown robot name %s! Valid names are %s." % (
        robot_name,
        RobotScenario.valid_robot_names(),
    )
    robot_technical_name_with_tool = RobotScenario.technical_robot_names_with_tool()[robot_name]

    config_file = LaunchConfiguration("ball_prediction_config").perform(context)
    ekf_config = os.path.join(get_package_share_directory("trajectory_estimation"), "config", config_file)

    control_frequency = int(LaunchConfiguration("control_frequency").perform(context))

    # The fanuc robot is positioned on the +x side of the table, so the serves should be done by player 1.
    # Opposite holds for the staubli robots
    if robot_name == "fanuc":
        serving_player = 1
    else:
        serving_player = 2

    launched_nodes = []
    ball_freq = control_frequency
    shooter_profile = "default"

    launched_nodes += bullet_simulator_node(
        use_gui=True,
        robot_name_left=robot_name if robot_name == "fanuc" else "",
        robot_name_right=robot_name if robot_name != "fanuc" else "",
        initial_ros_broadcast=False,
        use_hardware_robot_model=True,
        glog_level=glog_level,
        simulator_step_size=1 / ball_freq,
        physics_config=ekf_config,
    )
    launched_nodes += ball_shooter_node(
        robot_technical_name_with_tool=robot_technical_name_with_tool,
        serving_player=serving_player,
        glog_level=glog_level,
        cannon_setup=shooter_profile,
    )

    ball_topic_remapping = (
        "/sensors/ball_pose_estimation/poses",
        f"/sensors/ball_pose_estimation/{robot_technical_name_with_tool}/poses",
    )
    launched_nodes += [
        Node(
            package="calibration",
            executable="stepped_robot_table_calibration.py",
            name="calibrator",
            output="screen",
            arguments=["--colorlogtostderr=1", "--logtostderr=1", "--minloglevel=" + str(glog_level)],
            remappings=[ball_topic_remapping],
            emulate_tty=True,
        )
    ]

    return launched_nodes
