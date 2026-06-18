# ACE Loggers

Common logging tools available for both C++ and Python.

## ACE Data Logger

The ACE Data Logger is a high-performance binary logging library engineered for capturing real-time data efficiently. Designed with speed and data integrity in mind, it is ideal for applications where performance is critical.

### Features

- **High-Speed Logging**: Optimized for fast data capture with minimal overhead.
- **Robust File Management**: Supports exclusive access to prevent accidental overwrites, safeguarding your log files.
- **Structured Metadata Support**: Each log file can include structured metadata in JSON format. Required keys include `module_name` and `log_version`, with the option to define additional keys for better data management.
- **Thread Safety**: Ensures safe logging across multiple threads without data corruption or race conditions.
- **Customizable Buffering**: Tailor buffer and flush sizes to meet specific performance needs.

### Installation

To install the ACE Data Logger, set up your development environment with the necessary dependencies using the `tools/installation/install_*` scripts. This will install essential libraries such as `nlohmann/json` and `glog`.

### Usage

1. **Basic Setup**: Create an instance of the `DataWriter` class, specifying the log file name and metadata.

2. **Logging Data**: Use the `operator<<` overload to write various data types to your log file. For example:
   ```cpp
   datalogger::DataWriter writer("logfile.ace", metadata);
   writer << yourData;  // yourData can be a primitive type, struct, etc.
   ```

3. **Reading Logs**: To read logged data, instantiate the `DataReader` class with the log file name and use the `operator>>` overload to extract data back into your variables:
   ```cpp
   datalogger::DataReader reader("logfile.ace");
   reader >> yourData;  // Retrieve previously logged data.
   ```

### Configuration

- **Disabling the Logger**: The data logger is enabled by default. To disable it globally, set the environment variable `ACE_DATALOGGER_DISABLE` to `1`.

- **Changing the Log File Path**: By default, log files are saved to the `ace_logs` directory in the current working directory. Change the path globally by setting the environment variable `ACE_DATALOGGER_PATH`, e.g., `/var/tmp/ace_logs` or `/home/sony/ace_logs`.

- **Timezone Configuration**: The default timezone is UTC. Change it globally by setting the environment variable `ACE_DATALOGGER_TIMEZONE` to your preferred timezone, e.g., `Asia/Tokyo` or `Europe/Zurich`.

### Example Implementations

Refer to the demo logger examples for [Python](include/ace_loggers/datalogger_demo.hpp) and [C++](src/datalogger_demo_pybind.cpp) in this package. These examples demonstrate how to implement the `DataWriter` and `DataReader` functionalities, including best practices for logger initialization, data writing, and safe shutdown of logging sessions.

### Metadata Requirements

The data logger necessitates specific metadata for proper categorization of logs:
- **`module_name`**: Identifies the logging module or component.
- **`log_version`**: Indicates the version of the log format.

### Metadata Example

```cpp
auto metadata = {
    {"module_name", "example_module"},
    {"log_version", "1.0.0"}
};
```

### Important Notes

- Ensure all required metadata keys are present before starting the logging process, as they cannot be modified afterward.
- It is advisable to include experiment parameters in the metadata for reproducibility purposes.
- Implement effective file buffering to enhance logging performance, especially in scenarios with high-frequency data capture.
- Ensure setting real time profile to minimize the impact of logging on your main process.
- Consider creating a `datalogger_utils.py` file in your package with a `read_log_file()` function and a `RosLoader` class to support yielding from Python and sending the logs as ROS2 messages.

### Logs Control

You can start or stop log recording for the entire network using a remote log controller, such as the [log_control](tools/log_control) utility or the [scoreboard](../realtime_game_state/tools/scores_tracker.py) app.

### Automatic Logs Upload

For details on automatic log uploads, please refer to [this documentation](https://sonyresearch.atlassian.net/wiki/spaces/SON/pages/4037836810/Automatic+Logs+Upload).
