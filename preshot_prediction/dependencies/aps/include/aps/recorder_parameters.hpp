// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "base_parameters/base_parameters.hpp"

namespace aps {

class RecorderParameters final : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<RecorderParameters>;
  using ConstSharedPtr = std::shared_ptr<const RecorderParameters>;

  explicit RecorderParameters();

  // cuda device UUID
  std::vector<std::string> cuda_device_uuids;

  int sampling{1};
  int quality{70};
  bool publish_events{false};

 private:
  // auxiliary functions
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace aps
