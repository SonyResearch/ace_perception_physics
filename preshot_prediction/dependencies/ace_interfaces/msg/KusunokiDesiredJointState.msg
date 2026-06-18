# This ROS message is used to send target joint states to the Kusunoki interface while synchronizing to the Fanuc clock.
# While operating in this configuration, the Kusunoki is controlled at a higher multiple of the Fanuc control frequency (125Hz), but the state
# updates are broadcasted at the rhythm of the Fanuc UDP cycle, i.e. every 125 Hz.
# The idea is that the external listener (i.e. controller/planner) will subscribe to the Kusunoki state topic and send
# back a desired joint position for every current state message received.
# This requires that the desired joint message contains multiple values for the stage for every one for the Fanuc, as the
# control frequency of the former is n times faster.
# The field `stage_state` contains the n required positions for the stage, where each of these has 3 axes (X, Y, R).
# The field `fanuc_state` contains the single message for Fanuc, which will be composed of 6 axes (J1, J2, J3, J4, J5, J6).

sensor_msgs/JointState[] stage_state
sensor_msgs/JointState fanuc_state
