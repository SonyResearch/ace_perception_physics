// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "evs/multi_camera_parameters.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace evs {

MultiCameraParameters::MultiCameraParameters() : BaseParameters("multi_camera_parameters") {}

bool MultiCameraParameters::UpdateParametersFromYaml() {
  try {
    master_serial_number = yaml_node_["master_serial_number"].as<std::string>();
    slave_serial_numbers = yaml_node_["slave_serial_numbers"].as<std::vector<std::string>>();
  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool MultiCameraParameters::UpdateYamlFromParameters() {
  try {
    yaml_node_["master_serial_number"] = master_serial_number;
    yaml_node_["slave_serial_numbers"] = slave_serial_numbers;
  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace evs
