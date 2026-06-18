/**
 * @author Etienne Walther, Yamen Saraiji
 * @copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <list>
#include <memory>

#include "ace_yaml/ace_yaml.hpp"

namespace vision_common {
class SpinEstimator {
 public:
  using SharedPtr = std::shared_ptr<SpinEstimator>;
  using ConstSharedPtr = std::shared_ptr<const SpinEstimator>;

 private:
  double poses_timeout_{9999};
  size_t min_spin_history_length_{3};
  size_t max_spin_history_length_{7};
  double spin_obs_cov_scaling_{0};

  double last_timestamp_{-1};

  Eigen::Vector3d last_spin_;
  Eigen::Vector3d last_cov_;

  Eigen::Quaterniond prev_rotation_;
  Eigen::Matrix3d prev_cov_matrix_;
  Eigen::Vector3d median_axis_;

  double inliers_cos_distance_{0.95};

  int min_inliers_{3};

  std::list<Eigen::AngleAxisd> spin_history_;
  [[nodiscard]] bool CalculateHistoryMedian();

 public:
  SpinEstimator(/* args */);
  ~SpinEstimator() = default;

  [[nodiscard]] bool InitializeYaml(YAML::Node params);
  [[nodiscard]] bool Initialize(int poses_frequency = 200, size_t min_spin_history_length = 6,
                                size_t max_spin_history_length = 20, double spin_obs_cov_scaling = 1e-3,
                                double inliers_cos_distance = 0.95);

  [[nodiscard]] bool EstimateSpin(double timestamp, const Eigen::Quaterniond& orientation);
  [[nodiscard]] bool FilterSpin(double timestamp, const Eigen::Vector3d& spin);

  [[nodiscard]] const Eigen::Vector3d& GetLastSpin() const { return last_spin_; }
  [[nodiscard]] const Eigen::Matrix3d& GetLastCovarianceMatrix() const { return prev_cov_matrix_; }
  [[nodiscard]] const Eigen::Vector3d& GetLastCovariance() const { return last_cov_; }

  void Reset();
};

}  // namespace vision_common
