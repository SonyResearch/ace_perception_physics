// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "aps/multi_camera_parameters.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace aps {

MultiCameraParameters::MultiCameraParameters() : BaseParameters("multi_camera_parameters") {}

bool MultiCameraParameters::UpdateParametersFromYaml() {
  try {
    software_trigger = yaml_node_["software_trigger"].as<bool>();
    first_as_master = yaml_node_["first_as_master"].as<bool>();
    serial_numbers = yaml_node_["serial_numbers"].as<std::vector<std::string>>();
    rt_profile = yaml_node_["rt_profile"].as<std::string>();
    sampling = yaml_node_["sampling"].as<float>();
  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool MultiCameraParameters::UpdateYamlFromParameters() {
  try {
    yaml_node_["software_trigger"] = software_trigger;
    yaml_node_["first_as_master"] = first_as_master;
    yaml_node_["serial_numbers"] = serial_numbers;
    yaml_node_["rt_profile"] = rt_profile;
  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace aps
