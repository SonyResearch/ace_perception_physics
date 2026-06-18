# Ball Detection APS #

This package detects balls based on color from APS images.

## Quick Start ##
```
ros2 launch ball_detection_aps ball_detection_aps.launch.py config:=<config>
```

Launch arguments
* `config` (required): Configuration for which parameters are loaded, e.g., `config:=zrh00`.
* `debug` (optional): Argument to start node in debug mode to output additional information and for tuning of the parameters. Use either `debug:=false` (default) or `debug:=true`.

## Further information ##
* [Ball detection parameters](docs/ball_detection_aps_parameters.md)
* [Parameter tuning](docs/parameter_tuning.md)
