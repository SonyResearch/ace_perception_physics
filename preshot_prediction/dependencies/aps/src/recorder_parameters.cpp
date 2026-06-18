// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "aps/recorder_parameters.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace aps {

RecorderParameters::RecorderParameters() : BaseParameters("recorder_parameters") {}

bool RecorderParameters::UpdateParametersFromYaml() {
  try {
    // cuda device UUID
    cuda_device_uuids = yaml_node_["cuda_device_uuids"].as<std::vector<std::string>>();

  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool RecorderParameters::UpdateYamlFromParameters() {
  try {
    // cuda device UUID
    if (cuda_device_uuids.empty()) {
      LOG(ERROR) << "No CUDA device uuid set.\n";
      return false;
    }
    yaml_node_["cuda_device_uuids"] = cuda_device_uuids;
  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace aps
