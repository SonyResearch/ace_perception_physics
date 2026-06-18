# ace_monitor (standalone fork)

A self-contained ROS 2 Humble package that opens an ImGui/ImPlot/OpenGL window
with several monitoring pages for perception topics. This is a de-coupled fork
of the original `ace_monitor`: every proprietary internal dependency
(`project_ace_ui`, `rclzmq`, `ace_loggers`, `ace_yaml`, `arena_native_ros`,
`ace_rt_profiles`, `vision_common`, `physics_layer`, `calibration`,
`triangulation`, `dot_projector`, `multi_ball_triangulation`,
`state_estimation`, `trajectory_estimation`, `cuda_common`,
`ace_cmake_utils`, `physics_models`) has been replaced by an in-tree shim or
removed.

The package builds against vanilla ROS 2 Humble plus standard Ubuntu/system
libraries. Dear ImGui and ImPlot are fetched at configure-time via
`FetchContent`.

## Workspace layout

```
ace_monitor/
├── ace_interfaces/       # ROS 2 .msg package (trimmed, self-contained)
├── data/                 # configs, shaders, models (camera/racket meshes)
├── external/shims/       # header-only shims for proprietary deps
├── include/              # public headers
├── launch/               # ROS 2 launch
├── src/                  # library + standalone executable
└── CMakeLists.txt
```

## System dependencies

```bash
sudo apt update
sudo apt install \
    build-essential cmake git \
    libglfw3-dev libglew-dev libglm-dev libstb-dev \
    libfreetype-dev libassimp-dev \
    libopencv-dev libeigen3-dev libyaml-cpp-dev
```

A working ROS 2 Humble installation is required (`source
/opt/ros/humble/setup.bash`).

## Build

```bash
mkdir -p ~/ros2_ws/src
cp -r ace_monitor ~/ros2_ws/src/
cd ~/ros2_ws

source /opt/ros/humble/setup.bash
colcon build --packages-select ace_interfaces ace_monitor
```

The first build downloads Dear ImGui (`v1.90.4`) and ImPlot (`v0.16`) into the
build tree via CMake `FetchContent`; this requires internet access.

## Run

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ace_monitor ace_monitor.launch.py
# or directly:
ros2 run ace_monitor ace_monitor_standalone
```

## Available pages

| Category    | Page                  | Notes                                                       |
|-------------|-----------------------|-------------------------------------------------------------|
| Perception  | Camera Monitor        | Per-camera image-stream FPS                                 |
| Perception  | Ball Detector         | 2D ball-detection frame rate                                |
| Perception  | Ball Triangulation    | 3D ball position FPS / counts                               |
| Perception  | Ball Pose Estimation  | Position/velocity/spin estimation                           |
| Perception  | Player Pose           | 3D player keypoints                                         |
| Perception  | Racket Pose           | Racket pose & estimate                                      |
| Perception  | Perception Delay      | End-to-end perception delay                                 |
| Visualizer  | Perception Visualizer | 3D scene with cameras / table / volume of interest          |
| Prediction  | Estimator Topics      | Auto-discovered estimator topic FPS                         |
| Prediction  | Ball State Est.       | Predicted ball trajectories                                 |
| GCS         | GCS                   | Spin-rate panel                                             |
| Alarms      | Alarms                | Aggregate alarm display                                     |

Every page that previously required proprietary perception code now subscribes
to the corresponding `ace_interfaces` topic — actual processing performed in
the original pipelines is replaced by a no-op shim, so plots stay empty unless
matching messages are published on the wire.

## Compatibility caveats

* The `Perception Visualizer` only loads the table/floor/ball meshes if the
  `physics_models` package is present; otherwise it shows just the grid + axis.
* Camera/calibration loading requires the `calibration` package; without it the
  visualizer page is functional but contains no calibrated cameras.
* The Vision/Physics/Calibration shims expose an API-compatible no-op
  implementation so the GUI runs end-to-end without those upstream packages.

## License

Apache-2.0 for the standalone fork. The original code base is © Sony AI; this
fork only retains the structural code that was rewritten or wrapped to make
the package buildable in isolation.
