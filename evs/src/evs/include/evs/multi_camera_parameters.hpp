// SPDX-License-Identifier: MIT
#pragma once

#include "base_parameters/base_parameters.hpp"

namespace evs {

class MultiCameraParameters final : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<MultiCameraParameters>;
  using ConstSharedPtr = std::shared_ptr<const MultiCameraParameters>;

  explicit MultiCameraParameters();

  std::string master_serial_number;
  std::vector<std::string> slave_serial_numbers;

 private:
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace evs
