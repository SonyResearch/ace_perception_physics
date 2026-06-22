// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "evs/camera_parameters.hpp"

// #include "ace_loggers/ace_loggers.hpp"

namespace evs {

CameraParameters::CameraParameters() : BaseParameters("camera_parameters") {}

bool CameraParameters::UpdateParametersFromYaml() {
  try {
    // Biases.
    bias_diff = yaml_node_["bias_diff"].as<int>();
    bias_diff_off = yaml_node_["bias_diff_off"].as<int>();
    bias_diff_on = yaml_node_["bias_diff_on"].as<int>();
    bias_fo_p = yaml_node_["bias_fo_p"].as<int>();
    bias_hpf = yaml_node_["bias_hpf"].as<int>();
    bias_refr = yaml_node_["bias_refr"].as<int>();

  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool CameraParameters::UpdateYamlFromParameters() {
  try {
    // Biases.
    yaml_node_["bias_diff"] = bias_diff;
    yaml_node_["bias_diff_off"] = bias_diff_off;
    yaml_node_["bias_diff_on"] = bias_diff_on;
    yaml_node_["bias_fo_p"] = bias_fo_p;
    yaml_node_["bias_hpf"] = bias_hpf;
    yaml_node_["bias_refr"] = bias_refr;

  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace evs
