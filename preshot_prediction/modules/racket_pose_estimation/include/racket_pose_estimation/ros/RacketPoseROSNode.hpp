// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "racket_pose_estimation/RacketPoseExtractor.hpp"
#include "rclcpp/rclcpp.hpp"

namespace perception {
class RacketPoseROSNode {
 public:
  using SharedPtr = std::shared_ptr<RacketPoseROSNode>;
  using ConstSharedPtr = std::shared_ptr<const RacketPoseROSNode>;

 private:
  class RacketPoseROSNodeImpl;
  std::shared_ptr<RacketPoseROSNodeImpl> impl_;

 public:
  RacketPoseROSNode();

  void StartWithNode(std::shared_ptr<rclcpp::Node> node, RacketPoseExtractor::SharedPtr extractor);
  void Start(const std::string& node_name, RacketPoseExtractor::SharedPtr extractor);
  void Stop();
};
}  // namespace perception
