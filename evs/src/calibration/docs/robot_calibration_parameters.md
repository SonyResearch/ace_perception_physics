# Robot Calibration Parameters #

The robot's pose and closed-loop delay is calibrated with respect to the perception system. All parameters are defined in [robot_calibration_parameters.hpp](../include/calibration/robot_calibration_parameters.hpp).

The parameters are stored under [parameters/robot_calibration/]([OUTDATED]../parameters/robot_calibration) in yaml files.

## Transformation ##
* T_RO [-]: Transformation matrix from *R*obot coordinate frame in the *O*rigin coordinate frame. More information on the coordinate frames can be found in [Coordinate Frames](coordinate_frames.md).

## Others ##
* delay [s]: Closed-loop delay from capturing images until commands are being received by the robot.
