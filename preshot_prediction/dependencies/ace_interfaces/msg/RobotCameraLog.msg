# This ROS message is used log the clock of the robot + the camera in one ros topic
# while storing the current state of the robot and the camera.

std_msgs/Header header_camera
std_msgs/Header header_robot
sensor_msgs/JointState joint_state
geometry_msgs/Pose ball_pose
