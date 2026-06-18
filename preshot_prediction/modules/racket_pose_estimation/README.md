This module provides the ability to estimate racket pose (6DoF) from multiple observations.

A pre-trained model is used to extract racket position and main keypoints which can be used to triangulate and fit the orientation axis is deployed.
Racket pose is published via ros messages, so it can be subscribed into by end users. The module also provide pybind interfaces if needed to be used directly from python side without ROS in the middle. Check sample code for that.

Player pose information are also used to improve the detections, by localizing the ROI to be around player's hands. To use this feature, player pose estimator should be running in the background. If not, then by default it will use the top left 640x640 part of the image and run the inference on.

To run racket pose estimator:
launch racket_pose_estimation racket_pose_estimation.launch.py config:=tyo01 cluster:=_a version:=v8n

`cluster` is optional, used if you have multi-clusters cameras
`version` represents the version of the model to be used. By default it will use racket_`v8n`.onnx model located under `models` folder . This might get deprecated in future.
