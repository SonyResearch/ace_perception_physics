// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "dot_projector/idot_projector.hpp"

namespace dot_projector {

class DotProjector : public IDotProjector {
 public:
  using SharedPtr = std::shared_ptr<DotProjector>;
  using ConstSharedPtr = std::shared_ptr<const DotProjector>;

  explicit DotProjector();

  bool SetParameters(const IDotProjectorParameters::SharedPtr& params_ptr) override;
  void ProjectMarkers(const Eigen::Ref<const Eigen::Matrix2Xf>& distorted_points, const calibration::Camera& camera,
                      const Eigen::Vector3f& triangulated_point,
                      std::vector<Eigen::Vector3f>& projected_points) override;

 private:
  IDotProjectorParameters::SharedPtr params_ptr_;
};

}  // namespace dot_projector
