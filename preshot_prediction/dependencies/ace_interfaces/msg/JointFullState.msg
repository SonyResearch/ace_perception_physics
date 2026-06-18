# This ROS message is used to publish the position/velocity/acceleration and jerk of the robot joints in one topic.

std_msgs/Header header

string[] name
float64[] position
float64[] velocity
float64[] acceleration
float64[] jerk
float64[] effort
