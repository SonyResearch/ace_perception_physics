# This ROS message is used  log the action and the state of an RL agent for the goal conditioned RL.
# The `action_` variables indicate the out put of the RL agent and the `state_` + `desired_goal_`  variables indicate the input.
# The `header` should be filled with the time of the ball.

std_msgs/Header header
float64[] action
float64[] state
float64 ball_time
float64 robot_time
float64 computation_inference_time
