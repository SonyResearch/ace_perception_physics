// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "rclcpp/rclcpp.hpp"

namespace perception {

class RacketPoseROSNode;
class RacketPoseExtractor;

class RacketPoseEstimator : public rclcpp::Node {
 public:
  using SharedPtr = std::shared_ptr<RacketPoseEstimator>;
  using ConstSharedPtr = std::shared_ptr<const RacketPoseEstimator>;

 private:
  std::shared_ptr<RacketPoseROSNode> node_;
  std::shared_ptr<RacketPoseExtractor> extractor_;

  bool init_;

  bool tune_parameters_{false};
  rclcpp::TimerBase::SharedPtr tune_timer_;
  const std::string tune_window_name_ = "Racket Estimator Parameters";
  void TunerCallback();

 public:
  explicit RacketPoseEstimator(const rclcpp::NodeOptions& options);
  ~RacketPoseEstimator() override;
  void Initialize();
  void Destroy();
};
}  // namespace perception
