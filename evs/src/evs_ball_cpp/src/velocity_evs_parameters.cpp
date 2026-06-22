// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "evs_ball_cpp/velocity_evs_parameters.hpp"

#include <glog/logging.h>

namespace velocity_prediction_evs {

VelocityEVSParameters::VelocityEVSParameters() : BaseParameters("velocity_evs_parameters") {}

bool VelocityEVSParameters::UpdateParametersFromYaml() {
  try {
    // camera bias parameters
    bias_diff = yaml_node_["biases"]["bias_diff"].as<int>();
    bias_diff_off = yaml_node_["biases"]["bias_diff_off"].as<int>();
    bias_diff_on = yaml_node_["biases"]["bias_diff_on"].as<int>();
    bias_hpf = yaml_node_["biases"]["bias_hpf"].as<int>();
    bias_refr = yaml_node_["biases"]["bias_refr"].as<int>();

    erc_event_rate = yaml_node_["erc_event_rate"].as<int>();

    // camera serial number
    master_serial_number = yaml_node_["cameras"]["master_serial_number"].as<std::string>();
    slave_serial_number_0 = yaml_node_["cameras"]["slave_serial_number_0"].as<std::string>();
    slave_serial_number_1 = yaml_node_["cameras"]["slave_serial_number_1"].as<std::string>();
    slave_serial_number_2 = yaml_node_["cameras"]["slave_serial_number_2"].as<std::string>();
    number_cameras = yaml_node_["cameras"]["number_cameras"].as<int>();

    // detection algorithms
    batch_size = yaml_node_["algorithm"]["batch_size"].as<int>();
    model_path = yaml_node_["algorithm"]["model_path"].as<std::string>();
    dt_accumulate_us = yaml_node_["algorithm"]["dt_accumulate_us"].as<int>();
    number_streams = yaml_node_["algorithm"]["number_streams"].as<int>();

    // camera names
    camera_names = yaml_node_["camera_names"].as<std::vector<std::string>>();

    // crop parameters
    height = yaml_node_["crop_resolution"]["height"].as<int>();
    width = yaml_node_["crop_resolution"]["width"].as<int>();
    parts = yaml_node_["crop_resolution"]["parts"].as<int>();
    channels = yaml_node_["crop_resolution"]["channels"].as<int>();
    num_bbox = yaml_node_["crop_resolution"]["num_bbox"].as<int>();

    timeout_ms = yaml_node_["timeout_ms"].as<int>();
    use_variable_response_time = yaml_node_["use_variable_response_time"].as<bool>();
    camera_buffer_length = yaml_node_["camera_buffer_length"].as<int>();
    worker_queue_length = yaml_node_["worker_queue_length"].as<int>();

    // Threshold for velocity detection
    threshold = yaml_node_["threshold"].as<float>();
    velocity_constant = yaml_node_["velocity_constant"].as<float>();

    // Sync. frequency (APS or Ethercat)
    trigger_freq = yaml_node_["trigger_freq"].as<float>();
    trigger_exposure_time = yaml_node_["trigger_exposure_time"].as<int>();

  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool VelocityEVSParameters::UpdateYamlFromParameters() {
  try {
    // camera bias parameters
    yaml_node_["biases"]["bias_diff"] = bias_diff;
    yaml_node_["biases"]["bias_diff_off"] = bias_diff_off;
    yaml_node_["biases"]["bias_diff_on"] = bias_diff_on;
    yaml_node_["biases"]["bias_hpf"] = bias_hpf;
    yaml_node_["biases"]["bias_refr"] = bias_refr;

    yaml_node_["erc_event_rate"] = erc_event_rate;

    // camera serial number
    yaml_node_["cameras"]["master_serial_number"] = master_serial_number;
    yaml_node_["cameras"]["slave_serial_number_0"] = slave_serial_number_0;
    yaml_node_["cameras"]["slave_serial_number_1"] = slave_serial_number_1;
    yaml_node_["cameras"]["slave_serial_number_2"] = slave_serial_number_2;
    yaml_node_["cameras"]["number_cameras"] = number_cameras;

    // detection algorithms
    yaml_node_["algorithm"]["batch_size"] = batch_size;
    yaml_node_["algorithm"]["model_path"] = model_path;
    yaml_node_["algorithm"]["dt_accumulate_us"] = dt_accumulate_us;

    // camera names
    yaml_node_["camera_names"] = camera_names;

    // crop parameters
    yaml_node_["crop_resolution"]["height"] = height;
    yaml_node_["crop_resolution"]["width"] = width;
    yaml_node_["crop_resolution"]["parts"] = parts;

  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}


}  // namespace velocity_prediction_evs
