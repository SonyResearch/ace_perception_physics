// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "base_parameters/base_parameters.hpp"

namespace velocity_prediction_evs {

class VelocityEVSParameters final : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<VelocityEVSParameters>;
  using ConstSharedPtr = std::shared_ptr<const VelocityEVSParameters>;

  explicit VelocityEVSParameters();

  // camera bias parameters
  int bias_diff;
  int bias_diff_off;
  int bias_diff_on;
  int bias_hpf;
  int bias_refr;

  int erc_event_rate;

  // camera serial number
  std::string master_serial_number;
  std::string slave_serial_number_0;
  std::string slave_serial_number_1;
  std::string slave_serial_number_2;
  int number_cameras;

  // detection algorithms
  int batch_size;
  std::string model_path;
  int dt_accumulate_us;
  int number_streams;

  // camera names for which ball detection (and triangulation) is done
  std::vector<std::string> camera_names;

  bool triangulation_available;
  int timeout_ms;
  bool use_variable_response_time;
  int camera_buffer_length;
  int worker_queue_length;

  // crop parameters
  int height;
  int width;
  int parts;
  int channels;
  int num_bbox;

  // Threshold for velocity detection
  float threshold;
  float velocity_constant;

  float trigger_freq;
  int trigger_exposure_time;

 private:
  // auxiliary functions
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};
 

}  // namespace velocity_prediction_evs
