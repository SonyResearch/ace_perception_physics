// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "base_parameters/base_parameters.hpp"

namespace vision_common {

class BallDetectionParameters : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<BallDetectionParameters>;
  using ConstSharedPtr = std::shared_ptr<const BallDetectionParameters>;

  explicit BallDetectionParameters();

  // camera names for which ball detection (and triangulation) is done
  std::vector<std::string> camera_names;

 protected:
  // auxiliary functions
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace vision_common
