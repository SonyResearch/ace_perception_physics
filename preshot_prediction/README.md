<h1> Fine-Grained Pre-Shot Anticipation for Robot Table Tennis: Framework, Markerless Racket Pose Tracking, and an Elite-Level Game-Play Dataset </h1>

This repositry contains the code for the paper "Data-Driven Event-Based Vision and Dataset for Robotic Table Tennis Rallies".



# ACE Perception (rclcpp port)

ROS 2 perception stack for table-tennis ball detection, triangulation and
player/racket pose estimation. This tree was originally written against an
internal `rclzmq` middleware and a number of proprietary helper packages.
It has been ported to upstream **ROS 2 Humble (`rclcpp`)** and standard
`ament_cmake` so that it can be built with `colcon`.

## Workspace layout

```
src/ace_perception/
├── dependencies/        # Foundation libraries (built first)
│   ├── ace_interfaces           # ROS 2 .msg / .srv definitions
│   ├── ace_yaml                 # YAML utility library + pybind11 bindings
│   ├── ace_rt_profiles          # Real-time profile manager + bindings
│   ├── ace_loggers              # Binary data logger (C++/Python)
│   ├── base_parameters          # Parameter base classes
│   ├── ball_detector            # 2D ball detection (OpenCV)
│   ├── dot_detector             # 2D marker point detection
│   ├── dot_projector            # Dot projection for triangulation
│   ├── trt_ball_detector        # TensorRT-based ball detector + pybind11
│   ├── vision_common            # Spin/velocity estimators
│   ├── calibration              # Camera/robot calibration parameters
│   ├── triangulation            # Multi-view 3D triangulation
│   ├── aps                      # APS camera driver (Spinnaker/FLIR)
│   └── cuda_common              # CUDA helpers + TensorRT wrapper
│
└── modules/             # Pipeline stages (build last)
    ├── ball_detection_aps       # Ball detection node (OpenCV-based)
    ├── ball_detection_trt       # Ball detection node (TensorRT-based)
    ├── multi_ball_triangulation # Multi-camera 3D ball localization
    ├── player_pose              # Human pose estimation (TensorRT)
    ├── racket_pose_estimation   # 6D racket pose estimation
    └── ace_monitor              # ImGui/OpenGL perception dashboard (optional, needs glfw3/GLEW)
```

## Build status

All 19 packages build successfully from a clean `colcon build` (~2 min).

| Package | Layer | Status | Notes |
|---|---|---|---|
| ace_interfaces | dependency | ✅ builds | All `.msg` and `.srv` files included via `file(GLOB)` |
| ace_yaml | dependency | ✅ builds | |
| ace_rt_profiles | dependency | ✅ builds | |
| ace_loggers | dependency | ✅ builds | `ace_helpers::SignalHandler` vendored as no-op stub; `boost_time.hpp` made C++17-compatible |
| base_parameters | dependency | ✅ builds | |
| ball_detector | dependency | ✅ builds | |
| dot_detector | dependency | ✅ builds | |
| dot_projector | dependency | ✅ builds | |
| trt_ball_detector | dependency | ✅ builds | TensorRT ball detector with pybind11 bindings |
| vision_common | dependency | ✅ builds | |
| calibration | dependency | ✅ builds | `ace_containers::IndexedMap` vendored (supports key and integer-index access) |
| triangulation | dependency | ✅ builds | |
| aps | dependency | ✅ builds | Requires Spinnaker SDK at `/opt/spinnaker/` |
| cuda_common | dependency | ✅ builds | Forces `g++-10` as CUDA host compiler; `CMAKE_CUDA_STANDARD=14`, `CMAKE_CXX_STANDARD=17` |
| ball_detection_aps | module | ✅ builds | OpenCV-based ball detection pipeline |
| ball_detection_trt | module | ✅ builds | TensorRT-based ball detection pipeline |
| multi_ball_triangulation | module | ✅ builds | |
| player_pose | module | ✅ builds | |
| racket_pose_estimation | module | ✅ builds | |
| ace_monitor | module | ⚠️ optional | Requires `libglfw3-dev`, `libglew-dev`; already uses `rclcpp` — no porting needed |

All 19 packages (14 dependencies + 5 modules) build out of the box with
the commands below.

## Prerequisites

| Software | Tested version | Notes |
|---|---|---|
| Ubuntu | 22.04 | |
| ROS 2 | Humble | `sudo apt install ros-humble-desktop` |
| Python | 3.10 | system default on 22.04 |
| CMake | ≥ 3.18 | required by `cuda_common` |
| GCC | 10 and 11 | both needed: gcc-11 for the C++ packages, g++-10 as the CUDA host compiler |
| CUDA Toolkit | 11.5 | needed for `cuda_common`; newer versions (12.x) work too but the host-compiler logic in `cuda_common/CMakeLists.txt` may need tweaking |
| TensorRT | 8.x | needed for `cuda_common`, `trt_ball_detector`, `ball_detection_trt`, `player_pose`, `racket_pose_estimation` |
| Eigen3 | ≥ 3.4 | |
| OpenCV | ≥ 4.5 (4.8 for `cuda_common`) | with CUDA support for `cuda_common` |
| yaml-cpp | ≥ 0.6 | |
| Boost | system | date_time, etc. |
| glog / gflags | system | |
| nlohmann-json | system | `apt install nlohmann-json3-dev` |
| pybind11 | system | `apt install pybind11-dev` |
| Spinnaker SDK | ≥ 2.x | FLIR camera driver; needed for `aps` package. Install to `/opt/spinnaker/` |

Apt one-liner for the typical system packages:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop ros-humble-ament-cmake \
  build-essential cmake git \
  g++-10 g++-11 \
  python3-colcon-common-extensions python3-pip \
  pybind11-dev python3-pybind11 \
  libeigen3-dev libopencv-dev libyaml-cpp-dev \
  libgoogle-glog-dev libgflags-dev libboost-all-dev \
  nlohmann-json3-dev
```

CUDA / TensorRT have to be installed separately following NVIDIA's
instructions (Debian repository or runfile installer).

## Building with colcon

This repo is intended to live under a colcon workspace, e.g.:

```
~/ros2_ws/
└── src/
    └── ace_perception/        ← this repository
```

From the workspace root:

```bash
# 1. Source ROS 2
source /opt/ros/humble/setup.bash

# 2. Build all 19 packages
colcon build \
  --base-paths src/ace_perception \
  --cmake-args -DBUILD_TESTING=OFF

# 3. Source the resulting overlay
source install/setup.bash
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--packages-select <pkg>` | Build a single package |
| `--packages-up-to <pkg>` | Build the package and all its deps |
| `--packages-skip cuda_common trt_ball_detector ball_detection_trt player_pose racket_pose_estimation` | Skip CUDA / TensorRT packages on a CPU-only machine |
| `--cmake-args -DBUILD_TESTING=ON` | Re-enable tests (some require additional deps such as `gflags` linked into `gtest`) |
| `--cmake-args -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11` | Override the CUDA host compiler |
| `--event-handlers console_direct+` | Stream all build output live |

### Building only the dependency layer

```bash
colcon build --base-paths src/ace_perception \
  --packages-up-to ace_interfaces ace_yaml ace_rt_profiles ace_loggers \
                   base_parameters ball_detector dot_detector dot_projector \
                   trt_ball_detector vision_common calibration triangulation \
                   cuda_common aps \
  --cmake-args -DBUILD_TESTING=OFF
```

### Building a single module

```bash
colcon build --base-paths src/ace_perception \
  --packages-select ball_detection_trt \
  --cmake-args -DBUILD_TESTING=OFF
```

## Running the modules

Each module exposes a standalone executable that is installed to
`install/<package>/lib/<package>/<package>_standalone`. After sourcing
the overlay you can run them either directly or via `ros2 run`:

```bash
# Ball detection — OpenCV-based (APS cameras)
ros2 run ball_detection_aps ball_detection_aps_standalone \
  --ros-args --params-file install/ball_detection_aps/share/ball_detection_aps/parameters/ball_detection_aps.yaml

# Ball detection — TensorRT-based (GPU required)
ros2 run ball_detection_trt ball_detection_trt_standalone \
  --ros-args --params-file install/ball_detection_trt/share/ball_detection_trt/parameters/ball_detection_trt.yaml

# 3D triangulation
ros2 run multi_ball_triangulation multi_ball_triangulation_standalone \
  --ros-args --params-file install/multi_ball_triangulation/share/multi_ball_triangulation/parameters/multi_ball_triangulation.yaml

# Player pose estimation (TensorRT, GPU required)
ros2 run player_pose player_pose_standalone \
  --ros-args --params-file install/player_pose/share/player_pose/parameters/player_pose.yaml

# Racket pose estimation (TensorRT, GPU required)
ros2 run racket_pose_estimation racket_pose_estimation_standalone \
  --ros-args --params-file install/racket_pose_estimation/share/racket_pose_estimation/parameters/racket_pose_estimation.yaml
```

Each `modules/<pkg>/launch/` directory also contains a `*.launch.py`
that wires up the standalone with its parameter file:

```bash
ros2 launch ball_detection_aps ball_detection_aps.launch.py
```

The standalones expect input messages on the topics that the `aps` camera
driver publishes (synchronised camera images per camera name); without a
running camera driver they will start up but stay idle.

## What changed during the port

For full transparency, every change made during the rclzmq → rclcpp
port and the build-fix work:

1. **`rclzmq` → `rclcpp`** — bulk renamed in every CMakeLists, package.xml,
   header and source file (≈ 25 files).
2. **`ace_find_package` → `find_package`**, **`ace_add_library` → `add_library`**,
   **`ace_pybind11_add_module` → `pybind11_add_module`** — the proprietary
   `ace_cmake_utils` helpers were inlined to standard CMake commands.
3. **`find_package(ace_cmake_utils …)`** lines were dropped from every
   CMakeLists.
4. **C++ standard** — every `dependencies/*` and `modules/*` CMakeLists got an
   explicit `set(CMAKE_CXX_STANDARD 20)` (or 17 for `cuda_common`).
5. **`ace_loggers/boost_time.hpp`** — replaced `std::ranges::find` /
   `std::ranges::all_of` with C++17-compatible loops so the header does not
   force C++20 onto downstream packages.
6. **`ace_helpers::SignalHandler`** — vendored a no-op stub at
   `dependencies/ace_loggers/include/ace_helpers/SignalHandler.hpp` and
   removed the `find_package(ace_helpers …)` calls and the
   `<depend>ace_helpers</depend>` declarations.
7. **`ace_containers::IndexedMap`** — vendored a minimal implementation at
   `dependencies/calibration/include/ace_containers/indexed_map.hpp`
   supporting `operator[]` (by key and by integer index), `at(key)`,
   `at(index)`, `GetIndex()`, `contains()` and standard iteration.
   `find_package(ace_containers ...)` and the `<depend>ace_containers</depend>`
   lines were removed.
8. **`cuda_common`** — pinned `CMAKE_CUDA_HOST_COMPILER` to `g++-10` (CUDA
   11.5 cannot consume libstdc++ 11 headers in `bayer_resize.cu`).
9. **`cuda_common::TRTEngine`** — updated header and source to the latest
   version with async stream-based inference API (`RunInferenceOnStream`,
   `CopyOutputsOnStream`, `UnpackPinnedOutputs`), CUDA graph capture
   (`CaptureGraph`, `RunGraphOnStream`, `IsGraphCaptured`), and external
   GPU buffer support (`SetExternalInputGpu`, `SetExternalOutputGpu`).
10. **`ace_rt_profiles/CMakeLists.txt`** — added a missing
    `find_package(ament_cmake REQUIRED)` so that `ament_target_dependencies`
    is available.
11. **`aps`** — ported from `ace_cmake_utils`; added `FindSpinnaker.cmake`
    under `dependencies/aps/cmake/` for the Spinnaker SDK.
12. **`dot_detector`, `dot_projector`** — ported from `ace_cmake_utils` to
    standard CMake.
13. **`ace_interfaces`** — all `.msg` and `.srv` files consolidated;
    CMakeLists uses `file(GLOB)` for automatic discovery.
14. **`trt_ball_detector`** — ported from `ace_cmake_utils` to standard CMake
    (`ace_find_package` → `find_package`, `ace_add_library` → `add_library`,
    `ace_pybind11_add_module` → `pybind11_add_module`).
15. **`ball_detection_trt`** — ported from `rclzmq` → `rclcpp` in both
    CMakeLists/package.xml and all C++ source files; removed `ace_cmake_utils`.

`setup.py` files still import the proprietary `ace_setuptools`, but they are
**not** invoked by the `ament_cmake` build (Python is installed via
`ament_python_install_package(...)` from CMake). They only matter if you run
`pip install .` directly inside one of the package directories — in that
case replace `from ace_setuptools import setup` with
`from setuptools import setup` and supply the standard kwargs.


## Citation
For details about model architecture, training and evaluation, please check our paper
available on [arxiv.org](https://arxiv.org/abs/).

```bibtex
@misc{takahashi2026,
      title={Fine-Grained Pre-Shot Anticipation for Robot Table Tennis: Framework, Markerless Racket Pose Tracking, and an Elite-Level Game-Play Dataset},
      author={Naoya Takahashi, Yamen Saraiji, Yin Bi, Christian Fujiwara, Hamdi Salhloul, Etienne, Peter Duerr},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/},
}
```