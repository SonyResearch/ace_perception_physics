# Ball Triangulation - Parameters #

The ball is triangulated from synchronized ball detections from different APS/EVS cameras. The triangulation is implemented using the Direct Linear Transform (DLT), see [this Wikipedia article](https://en.wikipedia.org/wiki/Triangulation_(computer_vision)) for more information. All parameters are defined in [multi_ball_triangulation_parameters.hpp](../include/multi_ball_triangulation/multi_ball_triangulation_parameters.hpp).

The parameters are stored under [parameters/](../parameters/) in yaml files.

## Parameters ##
* timeout_ms [ms]: Time between first ball detection of a certain frame id arrives until the ball detection times out. If not all ball detections from all cameras for the given frame id arrive within this time interval, the node times out and tries to triangulate the ball position for the next frame.
* use_variable_response_time [bool]: If enabled, triangulation is performed immediately after receiving the last missing frame, otherwise will be waiting for buffer to get full before performing triangulation.
* camera_buffer_length [int]: The size of the buffer that stores incoming ball-detection frames per camera. The larger the value, the longer the lag between ball-detection and ball-triangulation, and the less likely frames will get overwritten due to synchronization and transmission issues. A reasonable value is between 1 (real-time) and 3 (unlikely to overwrite frames).
* worker_queue_length [int]: The size of the queue for the grouped-frames pending triangulation. Most likely you want to keep it as low as 3 unless the processing PC experiences variable CPU loads.
* max_triangulated_balls [int]: The maximum number of expect balls in the scene. If more balls than that are detected, no detection will be reported.

## VOI filter ##
* voi_filter_enable [bool]: Enable/disable filtering based on VOI (volume-of-interest), defined as an AABB (axis-aligned bounding-box) in the world's origin basis.
* voi_filter_min_corner [-]: Minimum 3D corner defining the VOI's bounding-box in the form of a triplet, float vector.
* voi_filter_max_corner [-]: Maximum 3D corner defining the VOI's bounding-box in the form of a triplet, float vector.

# TODO(cv3d): document parameters
