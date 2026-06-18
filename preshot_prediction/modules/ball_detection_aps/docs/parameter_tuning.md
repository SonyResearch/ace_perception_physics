# Parameter Tuning #

To tune the color filter in real time, launch the `ball_detection_aps` node in debug mode, i.e.
```
ros2 launch ball_detection_aps ball_detection_aps.launch.py config:=<config> debug:=true
```

In debug node, `ball_detection_aps` outputs additional information, allows for changing the HSV color filter values and inspect the filtered images in real time. To inspect the images, run the `aps_show_camera_images` tool.
```
aps_show_camera_images
```

Note that the debug mode increases the workload on the computer and should not be used for any real-time experiments.
