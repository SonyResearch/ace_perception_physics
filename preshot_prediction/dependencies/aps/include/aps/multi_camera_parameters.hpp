// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "base_parameters/base_parameters.hpp"

namespace aps {

class MultiCameraParameters final : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<MultiCameraParameters>;
  using ConstSharedPtr = std::shared_ptr<const MultiCameraParameters>;

  explicit MultiCameraParameters();

  bool software_trigger;
  bool first_as_master;
  std::vector<std::string> serial_numbers;
  float sampling = 1;

  std::string rt_profile;

 private:
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace aps
