// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "base_parameters/base_parameters.hpp"

namespace ball_detection_trt {

class BallDetectionParameters : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<BallDetectionParameters>;
  using ConstSharedPtr = std::shared_ptr<const BallDetectionParameters>;

  explicit BallDetectionParameters();
  virtual ~BallDetectionParameters() = default;

  std::string model_engine_path;

  float sampling{1};  // sequenceID sampling (to reduce frequency if there is a bottle neck on inference)
  std::vector<int> device_ids;
  int batch_size{1};
  std::vector<std::string> camera_names;
  float minimum_confidence{0.3F};

  std::string rt_profile;

  static bool CheckSampling(size_t sample_id, float modular) {
    return static_cast<int>(static_cast<float>(sample_id) - modular * floor(static_cast<float>(sample_id) / modular)) ==
           0;
  }

 protected:
  // auxiliary functions
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace ball_detection_trt
