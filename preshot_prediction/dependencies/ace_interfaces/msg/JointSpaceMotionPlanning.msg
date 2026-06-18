# This ROS message is used to interfacing with the ruckig motion planner at the task-level. `current_` indicates the current state of the joints
# `final_` indicates the target state of the joints. `inference_time` is the time where we inquiry the state of ruckig.

ace_interfaces/JointFullState current_joint_state
ace_interfaces/JointFullState final_joint_state
RigidBodyState final_corresponding_end_state
RigidBodyState current_corresponding_end_state

std_msgs/Float64 final_time
std_msgs/Float64 current_time
std_msgs/Float64 inference_time
std_msgs/Float64 motion_duration

float64 ball_time
float64 robot_time
