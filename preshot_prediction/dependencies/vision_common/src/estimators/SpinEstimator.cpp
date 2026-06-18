// Confidential, Copyright 2024, Sony AI, All rights reserved

#include "vision_common/estimators/SpinEstimator.hpp"

#include <iostream>

#include "ace_loggers/ace_loggers.hpp"

namespace vision_common {
SpinEstimator::SpinEstimator(/* args */) = default;

bool SpinEstimator::InitializeYaml(YAML::Node node) {
  int poses_frequency = ace_yaml::SafeGetValue<int>(node["poses_frequency"], -1);
  auto min_history_length = ace_yaml::SafeGetValue<size_t>(node["min_history_length"], 6);
  auto max_history_length = ace_yaml::SafeGetValue<size_t>(node["max_history_length"], 20);
  auto obs_cov_scaling = ace_yaml::SafeGetValue<double>(node["obs_cov_scaling"], 1e-3);
  auto inliers_cos_distance = ace_yaml::SafeGetValue<double>(node["inliers_cos_distance"], 0.95);
  min_inliers_ = ace_yaml::SafeGetValue<int>(node["min_inliers"], 3);

  return Initialize(poses_frequency, min_history_length, max_history_length, obs_cov_scaling, inliers_cos_distance);
}
bool SpinEstimator::Initialize(int poses_frequency, size_t min_spin_history_length, size_t max_spin_history_length,
                               double spin_obs_cov_scaling, double inliers_cos_distance) {
  if (poses_frequency > 0) {
    poses_timeout_ = 1.0F / static_cast<double>(poses_frequency);
  } else {
    poses_timeout_ = 9999;
  }
  min_spin_history_length_ = min_spin_history_length;
  max_spin_history_length_ = max_spin_history_length;
  spin_obs_cov_scaling_ = spin_obs_cov_scaling;
  inliers_cos_distance_ = inliers_cos_distance;
  last_timestamp_ = -1;
  last_spin_ = Eigen::Vector3d(0, 0, 0);
  last_cov_ = Eigen::Vector3d(0, 0, 0);
  LOG(INFO) << "SpinEstimator::Initialize() - Min Samples= " << min_spin_history_length_
            << ", Max Samples= " << max_spin_history_length_ << ", Samples timeout= " << poses_timeout_
            << ", Quality= " << inliers_cos_distance_;
  return true;
}

bool SpinEstimator::EstimateSpin(double timestamp, const Eigen::Quaterniond& orientation) {
  if (orientation.x() > 1e2) {
    // LOG(WARNING) << "SpinEstimator::EstimateSpin() - Invalid orientation for sample: " << timestamp;
    // invalid orientation
    return false;
  }
  auto curr_rotation = orientation.normalized();
  bool success = false;
  if (last_timestamp_ < 0) {
    last_timestamp_ = timestamp;
  }

  auto time_diff = timestamp - last_timestamp_;
  if (time_diff > 0 && time_diff <= (poses_timeout_ + 1e-4)) {
    auto rot_diff = curr_rotation * prev_rotation_.inverse();
    auto current_spin = Eigen::AngleAxisd(rot_diff);
    current_spin = Eigen::AngleAxisd(current_spin.angle() / time_diff, current_spin.axis());
    success = FilterSpin(timestamp, current_spin.axis() * current_spin.angle());
  }

  prev_rotation_ = std::move(curr_rotation);
  return success;
}

bool SpinEstimator::FilterSpin(double timestamp, const Eigen::Vector3d& spin) {
  if (fabs(spin.x()) > 1e3 || fabs(spin.y()) > 1e3 || fabs(spin.z()) > 1e3) {  // outlier spin
    // LOG(WARNING) << "SpinEstimator::EstimateSpin() - Invalid orientation for sample: " << timestamp;
    // invalid orientation
    return false;
  }
  bool success = false;
  if (last_timestamp_ < 0) {
    last_timestamp_ = timestamp;
  }
  auto time_diff = timestamp - last_timestamp_;
  if (time_diff > 0 && time_diff <= (poses_timeout_ + 1e-4)) {
    auto current_spin = Eigen::AngleAxisd(spin.norm(), spin / spin.norm());

    spin_history_.push_front(current_spin);
    if (spin_history_.size() > max_spin_history_length_) {
      spin_history_.pop_back();
    }

    if (spin_history_.size() >= min_spin_history_length_ && CalculateHistoryMedian()) {
      // calculate inliers (remove outliers)
      Eigen::Vector3d est_spin(0, 0, 0);
      Eigen::Matrix<double, -1, 3> data_points;
      data_points.conservativeResize(static_cast<Eigen::Index>(spin_history_.size()), data_points.cols());
      int count = 0;
      for (const auto& spin : spin_history_) {
        const auto& vec = spin.axis();
        if (vec.dot(median_axis_) >= inliers_cos_distance_) {
          auto spin_vector = spin.axis() * spin.angle();
          data_points.row(count) = spin_vector;
          est_spin += spin_vector;
          ++count;
        }
      }
      if (count >= min_inliers_) {
        data_points.conservativeResize(count, data_points.cols());
        last_spin_ = est_spin / count;
        Eigen::MatrixXd centered = data_points.rowwise() - data_points.colwise().mean();
        prev_cov_matrix_ = (centered.adjoint() * centered) / static_cast<double>(data_points.rows() - 1);
        prev_cov_matrix_ = prev_cov_matrix_ * spin_obs_cov_scaling_;
        last_cov_ = prev_cov_matrix_.diagonal();
        success = true;
      } else {
        // LOG(WARNING) << "SpinEstimator::EstimateSpin() - Not enought inliers were detected";
      }
    }
  } else {
    if (time_diff > 0) {
      LOG(WARNING) << "SpinEstimator::EstimateSpin() - Skip in time detected for sample: " << timestamp
                   << "ms --> Late by " << time_diff << "ms";
    }
  }
  last_timestamp_ = timestamp;
  return success;
}

void SpinEstimator::Reset() {
  spin_history_.clear();
  prev_rotation_.setIdentity();
  last_timestamp_ = -1;
}

bool SpinEstimator::CalculateHistoryMedian() {
  // get median value of the history
  static auto median_calc = [](const std::list<Eigen::AngleAxisd>& spin_history) {
    auto size = spin_history.size();
    Eigen::Vector3d median(0, 0, 0);
    if (size == 0) {
      return median;
    }
    using DataType = std::pair<int, const Eigen::AngleAxisd*>;
    std::vector<DataType> sorted(size);
    int index = 0;
    for (const auto& spin : spin_history) {
      sorted[index] = std::pair(index, &spin);
      ++index;
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const DataType& a, const DataType& b) { return a.second->angle() < b.second->angle(); });
    if (size % 2 == 0) {
      const auto& a = sorted[size / 2 - 1];
      const auto& b = sorted[size / 2];
      median = (a.second->axis() + b.second->axis()) / 2;
    } else {
      median = sorted[size / 2].second->axis();
    }
    return median;
  };

  median_axis_ = median_calc(spin_history_);
  return median_axis_.squaredNorm() != 0;
}

}  // namespace vision_common
