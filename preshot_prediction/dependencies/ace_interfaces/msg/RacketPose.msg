# This ROS message is used to publish the position, velocity and orientation of the racket.

PerceptionHeader header

bool tracked                      # is the racket visible?
string serial                     # racket ID
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
geometry_msgs/Quaternion orientation
