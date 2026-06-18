# This ROS message is used to interfacing with the polynomial motion planner at the task-level. `current_` indicates the current state of the end-effector.
# `final_` indicates the final values of the target state of the end-effector. `inference_time` is the time where we inquiry the state of the polynomial planner.
# If `hitting_over_span_of_time` is set to True and `dt_span_of_time` is a positive small value, the polynomial motion planner moves with the orientation equal to the final
# end-effector's state from final_time-hitting_over_span_of_time till final_time+hitting_over_span_of_time with a velocity equal to final_task_vel

std_msgs/Header header
geometry_msgs/Pose  current_task_pose
geometry_msgs/Pose  final_task_pose
geometry_msgs/Twist current_task_vel
geometry_msgs/Twist final_task_vel
geometry_msgs/Accel current_task_acc
geometry_msgs/Accel final_task_acc
geometry_msgs/Twist current_task_jerk
geometry_msgs/Twist final_task_jerk
std_msgs/Float64 final_time
std_msgs/Float64 current_time
std_msgs/Float64 inference_time
std_msgs/Bool hitting_over_span_of_time
std_msgs/Float64 dt_span_of_time

float64[] current_vel
float64[] target_vel

float64 ball_time
float64 robot_time
