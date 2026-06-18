#include "ball_detection_trt/ball_detection_parameters.hpp"

#include <iostream>

#include "ace_loggers/logging.hpp"

namespace ball_detection_trt {
BallDetectionParameters::BallDetectionParameters() : BaseParameters("ball_detection_parameters_trt") {}
bool BallDetectionParameters::UpdateParametersFromYaml() {
  try {
    // cuda device UUID
    model_engine_path = yaml_node_["model_engine_path"].as<std::string>();
    device_ids = yaml_node_["device_ids"].as<std::vector<int>>();
    sampling = yaml_node_["sampling"].as<float>();
    batch_size = yaml_node_["batch_size"].as<int>();
    camera_names = yaml_node_["camera_names"].as<std::vector<std::string>>();
    minimum_confidence = yaml_node_["minimum_confidence"].as<float>();

    // cpu affinity
    rt_profile = yaml_node_["rt_profile"].as<std::string>();

  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool BallDetectionParameters::UpdateYamlFromParameters() { return true; }

}  // namespace ball_detection_trt
