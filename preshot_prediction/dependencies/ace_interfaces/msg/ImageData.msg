# This contains time-stamped image data and additional information such as frame rate.

# Currently, `sensor_msgs/Image` and `std_msgs/Header` messages can't be loaned, preventing zero-copy usage
# In order to support zero copy, the following modifications are applied:
# - string -> char[<MAX_SIZE>]
# - uint8[] -> uint8[<MAX_SIZE>]

PerceptionHeader header

float64 frame_rate
char[20] encoding

# Image dimensions [in pixels]
uint16 height
uint16 width

# Image ROI top-left corner [in pixels]
uint16 offset_x
uint16 offset_y

# stride of the image [in bytes]
uint16 step

# Image data
uint8[1555200] data
