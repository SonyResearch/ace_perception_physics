# This ROS message is used to log the  goal state for the ball on the opponent side of the table.
# The `header` should be filled with the time of the ball.
# Maybe one, or multiple of the following topics are specific.

std_msgs/Header header
float64[] position_goal
float64[] velocity_goal
float64[] spin_goal
float64 flight_time_goal
float64 ball_time
float64 robot_time
