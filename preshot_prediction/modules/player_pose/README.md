# Player Pose
Native implementation for human pose estimation, utilizing tensorrt for accelerated inference over movenet and yolov8 networks. This module supports the following features:

- [x] ROS based publisher for human bbox detection, 2d keypoints detections, 3d keypoints estimations
- [x] BBox detection per camera
- [x] 2D keypoints detection per camera w/confidence
- [x] 3D keypoints triangulation w/covariance
- [x] Temporal filtering for bbox and 2d keypoints detection
- [x] Python interface
- [x] ROS launcher
- [x] Automatic ONNX --> TRT model conversion (GPU specific build)


After building, you can run it via ros launcher:

`ros2 launch player_pose player_pose_estimation.launch.py config:=<lab> fast:=no`

`fast` flag allows to use lightning models for keypoint inference (vs thunder models). Set to false for more accurate inferences.

When launched, the following topics will be published:

`/sensors/[CameraName]/player_detection`: Contains the 2D bbox detection of the player per camera, along its confidence
`/sensors/[CameraName]/player_pose_2d`: Contains the 2D keypoints detection per camera, along their confidence
`/sensors/[PlayerName]/pose_3d`: Contains the triangulated keypoints along their projection error and inference confidence

## Videos of player pose in action
https://github.com/SonyResearch/project_ace/assets/117256524/8e3e5ae3-d0e0-4f81-8fef-994632c7300d


https://github.com/SonyResearch/project_ace/assets/117256524/6083dd9f-eeb5-47c5-886a-445946ad1352

Side-by-side:
https://github.com/SonyResearch/project_ace/assets/117256524/7b7de4e8-66ec-41a2-b6eb-556b95ab94ea

Keypoints error for the right wrist, elbow and hip (using Vive Tracker mounted on the body as a ground truth):

![Errors](https://github.com/SonyResearch/project_ace/assets/117256524/31c60a15-93c6-4ef9-a237-e3fd38922cdc)
