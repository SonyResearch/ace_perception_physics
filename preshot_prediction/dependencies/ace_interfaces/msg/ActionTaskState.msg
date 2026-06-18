# This ROS message is used to interfacing with the polynomial motion planner at the task-level. `target_` values indicate the time when the
# racket should be at the specified position/velocity/acceleration.

std_msgs/Header header
ace_interfaces/RigidBodyState target_state
std_msgs/Float64 target_time
