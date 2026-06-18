// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <vector>

#include "calibration/camera_calibration_parameters.hpp"
#include "dot_projector/idot_projector_parameters.hpp"
#include "eigen3/Eigen/Eigen"

namespace dot_projector {

class IDotProjector {
 public:
  using SharedPtr = std::shared_ptr<IDotProjector>;
  using ConstSharedPtr = std::shared_ptr<const IDotProjector>;

  virtual ~IDotProjector() = default;

  virtual bool SetParameters(const IDotProjectorParameters::SharedPtr& params_ptr) = 0;
  virtual void ProjectMarkers(const Eigen::Ref<const Eigen::Matrix2Xf>& distorted_points,
                              const calibration::Camera& camera, const Eigen::Vector3f& triangulated_point,
                              std::vector<Eigen::Vector3f>& projected_points) = 0;
};

}  // namespace dot_projector
