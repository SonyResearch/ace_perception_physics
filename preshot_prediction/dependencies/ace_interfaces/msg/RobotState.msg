# This ROS message is used to publish the current state of the robot. `zero_jacobian` indicates the jacobian matrix from the tool frame to the base.
# `header_robot` and `header_camera` are needed to time synchronization.
# `zero_jacobian`, `mass` and `gravity` are row major matrices.

std_msgs/Header header_robot
std_msgs/Header header_camera
geometry_msgs/Pose end_effector_pose
geometry_msgs/Twist end_effector_velocity
geometry_msgs/Accel end_effector_acceleration
geometry_msgs/Wrench end_effector_force
sensor_msgs/JointState joint_state
std_msgs/Float64MultiArray zero_jacobian
std_msgs/Float64MultiArray mass
std_msgs/Float64MultiArray gravity
