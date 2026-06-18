// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include "rclcpp/rclcpp.hpp"

namespace perception {

class PlayerPoseROSNode;
class PlayerPoseExtractor;

class PlayerPoseEstimator : public rclcpp::Node {
 public:
  using SharedPtr = std::shared_ptr<PlayerPoseEstimator>;
  using ConstSharedPtr = std::shared_ptr<const PlayerPoseEstimator>;

 private:
  std::shared_ptr<PlayerPoseROSNode> node_;
  std::shared_ptr<PlayerPoseExtractor> extractor_;

  bool init_;

 public:
  explicit PlayerPoseEstimator(const rclcpp::NodeOptions& options);
  ~PlayerPoseEstimator() override;
  void Initialize();
  void Destroy();
};
}  // namespace perception
