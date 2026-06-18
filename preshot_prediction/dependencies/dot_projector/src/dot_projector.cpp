// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "dot_projector/dot_projector.hpp"

#include "vision_common/vision_common.hpp"

namespace dot_projector {

DotProjector::DotProjector() = default;

bool DotProjector::SetParameters(const IDotProjectorParameters::SharedPtr& params_ptr) {
  params_ptr_ = params_ptr;

  return true;
}

void DotProjector::ProjectMarkers(const Eigen::Ref<const Eigen::Matrix2Xf>& distorted_points,
                                  const calibration::Camera& camera, const Eigen::Vector3f& triangulated_point,
                                  std::vector<Eigen::Vector3f>& projected_points) {
  if (!params_ptr_->projector_enable || distorted_points.cols() == 0) {
    return;
  }
  const Eigen::Matrix2Xf undistorted_points = camera.UndistortPointsVectorized(distorted_points);
  const Eigen::Matrix3Xf rays_dir =
    (camera.projection_matrix_inv.block<3, 3>(0, 0) * undistorted_points.colwise().homogeneous())
      .colwise()
      .normalized();
  Eigen::Matrix3Xf points_3d;
  vision_common::ComputeNearestIntersections(camera.projection_matrix_inv.block<3, 1>(0, 3) - triangulated_point,
                                             rays_dir, Eigen::Vector3f::Zero(), params_ptr_->projector_ball_radius,
                                             points_3d);

  // Filter out outlier points that did not land on the ball
  Eigen::Matrix<bool, Eigen::Dynamic, 1> mask =
    (points_3d.colwise().norm().array() - params_ptr_->projector_ball_radius).abs() < 1e-3;
  for (Eigen::Index col_i = 0; col_i < points_3d.cols(); ++col_i) {
    if (mask[col_i]) {
      projected_points.emplace_back(points_3d.col(col_i));
    }
  }

  // Merge detections that are very close
  const float pattern_min_distance = 5e-3;  // TODO(cv3d): expose to parameters
  vision_common::FilterUniquePointsInplace(projected_points, pattern_min_distance);
}

}  // namespace dot_projector
