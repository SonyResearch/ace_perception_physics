# Ball Detection from APS images - Parameters #

Balls are detected according to their color, motion and appearance (e.g., circularity). All parameters are defined in [ball_detection_aps_parameters.hpp](../include/ball_detection_aps/ball_detection_aps_parameters.hpp).

The parameters are stored under [parameters/](../parameters/) in yaml files.

## CUDA device ##
* cuda_device_uuid [str]: Universially unique identifier (UUID) of CUDA device which will be used for ball detection. Run `nvidia-smi -L` to list all available GPUs and their UUID.

## Color filter ##
* blur_kernel_size [px]:
* hsv_lower_boundary [-]: Lower boundary of HSV bandpass filter. Minimum values are [0, 0, 0].
* hsv_upper_boundary [-]: Upper boundary of HSV bandpass filter. Maximum values are [255, 255, 179].

## Motion filter ##
* motion_filter_enable [bool]: Enable/disable filtering based on motion.
* motion_filter_delay [s]: Delay for which difference between images is computed. This image is then * passed through a bandpass filter.
* motion_filter_lower_boundary: Lower boundary of motion bandpass filter. Minimum value is 0.
* motion_filter_upper_boundary: Upper boundary of motion bandpass filter. Maximum value is 255.

## Appearance filter ##
* min_circularity_ratio: Ratio of (blob_circumference)\*\*2/(4\*pi\*blob_area). The ratio will increase the more a blob looks like a circle. Example values: circle = 1.0, square = 0.785 (pi/4).
* min_radius [px]: Minimum required ball radius in camera image.

You can use `aps_show_camera_images` with `ros2 launch ball_detection_aps ball_detection_aps.launch.py debug:=mask` to visualize the resulting mask.  If both border and VOI filters are enabled, then the resulting mask is their intersection.

## border filter ##
* border_filter_enable [bool]: Enable/disable filtering based on border margin.
* border_filter_plane_depth [float]: defines the metric distance from camera center to the plane at which the margin is applied.
* border_filter_margin_size [float]: defines the metric distance from image left/right/top/bottom borders in which detections are excluded.

Similar to VOI filter, you can use `aps_show_camera_images` with `ros2 launch ball_detection_aps ball_detection_aps.launch.py debug:=mask` to visualize the resulting mask. If both border and VOI filters are enabled, then the resulting mask is their intersection.

## Cameras ##
* cameras [-]: List of cameras for which ball detection (and later triangualation) is done.
