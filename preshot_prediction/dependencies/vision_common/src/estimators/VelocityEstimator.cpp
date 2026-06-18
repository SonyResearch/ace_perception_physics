// Confidential, Copyright 2024, Sony AI, All rights reserved

#include "vision_common/estimators/VelocityEstimator.hpp"

#include <iostream>

namespace vision_common {
VelocityEstimator::VelocityEstimator(/* args */) : last_velocity_(0, 0, 0), prev_position_(0, 0, 0) {}

bool VelocityEstimator::InitializeYaml(YAML::Node node) {
  int poses_frequency = ace_yaml::SafeGetValue<int>(node["poses_frequency"], -1);
  auto min_history_length = ace_yaml::SafeGetValue<size_t>(node["min_history_length"], 6);
  auto max_history_length = ace_yaml::SafeGetValue<size_t>(node["max_history_length"], 20);

  return Initialize(poses_frequency, min_history_length, max_history_length);
}
bool VelocityEstimator::Initialize(int poses_frequency, size_t min_velocity_history_length,
                                   size_t max_velocity_history_length) {
  if (poses_frequency > 0) {
    poses_timeout_ = 1.0F / static_cast<double>(poses_frequency);
  } else {
    poses_timeout_ = 9999;
  }
  min_vel_history_length_ = min_velocity_history_length;
  max_vel_history_length_ = max_velocity_history_length;
  last_timestamp_ = -1;
  last_velocity_ = Eigen::Vector3d(0, 0, 0);
  last_cov_ = Eigen::Vector3d(0, 0, 0);
  return true;
}

bool VelocityEstimator::EstimateVelocity(double timestamp, const Eigen::Vector3d& position) {
  bool success = false;
  if (last_timestamp_ < 0) {
    last_timestamp_ = timestamp;
  }
  if (last_timestamp_ == -1) {
    prev_position_ = position;
  }
  auto time_diff = timestamp - last_timestamp_;
  if (time_diff > 0 && time_diff <= (poses_timeout_ + 1e-4)) {
    auto current_velocity = (position - prev_position_) / time_diff;
    if (last_timestamp_ != -1) {
      velocity_history_.push_front(current_velocity);
    }
    while (velocity_history_.size() > max_vel_history_length_) {
      velocity_history_.pop_back();
    }

    if (velocity_history_.size() >= min_vel_history_length_) {
      Eigen::Vector3d est_velocity(0, 0, 0);
      Eigen::Matrix<double, -1, 3> data_points;
      data_points.conservativeResize(static_cast<Eigen::Index>(velocity_history_.size()), data_points.cols());
      int count = 0;
      for (const auto& velocity : velocity_history_) {
        data_points.row(count) = velocity;
        est_velocity += velocity;
        ++count;
      }
      last_velocity_ = est_velocity / count;
      Eigen::MatrixXd centered = data_points.rowwise() - data_points.colwise().mean();
      prev_cov_matrix_ = (centered.adjoint() * centered) / static_cast<double>(data_points.rows() - 1);
      last_cov_ = prev_cov_matrix_.diagonal();
      success = true;
    }
  }
  prev_position_ = position;
  last_timestamp_ = timestamp;
  return success;
}

void VelocityEstimator::Reset() {
  velocity_history_.clear();
  last_velocity_ = Eigen::Vector3d(0, 0, 0);
  last_cov_ = Eigen::Vector3d(0, 0, 0);
  prev_cov_matrix_.setIdentity();
  last_timestamp_ = -1;
}

}  // namespace vision_common
