# Ace Real Time Profiles

A centralized place for controlling real time profiles  
Each profile contains settings such as cpu affinity, scheduler policies, etc

The configuration file is automatically deduced from the `ACE_LAB` and `ACE_PC` environment variables, `ace_rt_profiles/config/<ACE_LAB>/<ACE_PC>.yaml`  
If `DART_RUN_ID` or `CI` env vars are set, then `ACE_LAB` and `ACE_PC` will be ignored in favor of loading `dart.yaml` or `ci.yaml` respectively

## Syntax

```yaml
profile:    # str, name of the profile
  affinity: # optional list[int], list of cpu cores
  policy:   # optional str, one of the following: SCHED_OTHER | SCHED_BATCH | SCHED_IDLE | SCHED_FIFO | SCHED_RR
  priority: # optional int, required if policy is set to SCHED_FIFO or SCHED_RR
```

## Example

See the files inside `ace_rt_profiles/config/` folder

## Utilities

The `cat_ace_rt_profiles` executable allows to check which profiles are available in the current environment  
It will load the configuration file as per the environment variable resolution rules explained above  
And print the contents of the resulting yaml file to the terminal

```
cat_ace_rt_profiles [--help]
```

## API

See [profile_manager.hpp](include/ace_rt_profiles/profile_manager.hpp) and [profile_guard.hpp](include/ace_rt_profiles/profile_guard.hpp) for the full documentation

### C++

```cpp
#include <ace_rt_profiles/profile_guard.hpp>
#include <ace_rt_profiles/profile_manager.hpp>

{
  // save the current profile
  ace_rt_profiles::ProfileGuard guard;

  // apply the profile foo
  const auto& manager = ace_rt_profiles::ProfileManager::GetInstance();
  manager.Apply("foo");

  // threads spawned internally by ROS will run under the foo profile
  rclcpp::init(...);
}

// guard went out of scope, previous profile is restored for the main thread
// but threads spawned by ROS are still running under profile foo
```

### Python

```python
from ace_rt_profiles import ProfileGuard, ProfileManager

with ProfileGuard():
  manager = ProfileManager.get_instance()
  manager.apply("foo")
```

## Profiles

The following is a list of common profiles used during deployment (i.e. games)  
They **must** be defined in the respective PC configuration file

### Control

```yaml
arena_native_ros
arena_realworld_musashi
arena_tactics
collision_detection_robot_interface
mpc_solvers_and_collision_detection
multi_ball_triangulation
multistep_action_process
musashi_interface_logger
perception_default_ros
primitives_child
robot_interface
robot_ros_wrapper
trajectory_estimator
trajectory_estimator_mpc
```

### Perception

```yaml
ace_monitor
aps
arena_native_ros
ball_detection_aps
multi_ball_triangulation
perception_default_ros
player_pose_extractor
racket_pose_extractor
trajectory_estimator
trajectory_estimator_ace_monitor
```

### Scoreboard

```yaml
arena_native_ros
perception_default_ros
trajectory_estimator
```
