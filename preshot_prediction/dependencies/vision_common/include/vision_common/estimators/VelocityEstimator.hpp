/**
 * @author Yamen Saraiji
 * @copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <list>
#include <memory>

#include "ace_yaml/ace_yaml.hpp"

namespace vision_common {
class VelocityEstimator {
 public:
  using SharedPtr = std::shared_ptr<VelocityEstimator>;
  using ConstSharedPtr = std::shared_ptr<const VelocityEstimator>;

 private:
  double poses_timeout_{9999};
  size_t min_vel_history_length_{3};
  size_t max_vel_history_length_{9};

  double last_timestamp_{-1};

  Eigen::Vector3d last_velocity_;
  Eigen::Vector3d last_cov_;

  Eigen::Vector3d prev_position_;
  Eigen::Matrix3d prev_cov_matrix_;

  std::list<Eigen::Vector3d> velocity_history_;

 public:
  VelocityEstimator(/* args */);
  ~VelocityEstimator() = default;

  [[nodiscard]] bool InitializeYaml(YAML::Node params);
  [[nodiscard]] bool Initialize(int poses_frequency = 200, size_t min_velocity_history_length = 6,
                                size_t max_velocity_history_length = 20);

  [[nodiscard]] bool EstimateVelocity(double timestamp, const Eigen::Vector3d& position);

  [[nodiscard]] const Eigen::Vector3d& GetLastVelocity() const { return last_velocity_; }
  [[nodiscard]] const Eigen::Matrix3d& GetLastCovarianceMatrix() const { return prev_cov_matrix_; }
  [[nodiscard]] const Eigen::Vector3d& GetLastCovariance() const { return last_cov_; }

  void Reset();
};

}  // namespace vision_common
