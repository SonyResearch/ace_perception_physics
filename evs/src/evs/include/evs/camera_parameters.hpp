// SPDX-License-Identifier: MIT
#pragma once

#include "base_parameters/base_parameters.hpp"

namespace evs {

class CameraParameters final : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<CameraParameters>;
  using ConstSharedPtr = std::shared_ptr<const CameraParameters>;

  explicit CameraParameters();

  // Biases.
  // The exact meaning of the biases can be found in Prophesee's documentation and depend on the camera
  // that is used.
  // https://docs.prophesee.ai/stable/hw/manuals/biases.html?highlight=biases
  int bias_diff;
  int bias_diff_off;
  int bias_diff_on;
  int bias_fo_p;
  int bias_hpf;
  int bias_refr;

 private:
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace evs
