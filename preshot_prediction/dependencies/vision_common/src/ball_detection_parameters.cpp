// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "vision_common/ball_detection_parameters.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace vision_common {

BallDetectionParameters::BallDetectionParameters() : BaseParameters("ball_detection_parameters") {}

bool BallDetectionParameters::UpdateParametersFromYaml() {
  try {
    // camera names
    camera_names = yaml_node_["camera_names"].as<std::vector<std::string>>();
  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool BallDetectionParameters::UpdateYamlFromParameters() {
  try {
    // camera names
    yaml_node_["camera_names"] = camera_names;
  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace vision_common
