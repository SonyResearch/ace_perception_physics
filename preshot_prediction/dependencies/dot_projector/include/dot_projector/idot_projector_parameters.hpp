// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "eigen3/Eigen/Eigen"
#include "opencv2/opencv.hpp"

namespace dot_projector {

class IDotProjectorParameters {
 public:
  using SharedPtr = std::shared_ptr<IDotProjectorParameters>;
  using ConstSharedPtr = std::shared_ptr<const IDotProjectorParameters>;

  virtual ~IDotProjectorParameters() = default;

  bool projector_enable;
  float projector_ball_radius;
};

}  // namespace dot_projector
