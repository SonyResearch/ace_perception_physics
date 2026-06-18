// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include "player_pose/PlayerPoseExtractor.hpp"
#include "rclcpp/rclcpp.hpp"

namespace perception {
class PlayerPoseROSNode {
 public:
  using SharedPtr = std::shared_ptr<PlayerPoseROSNode>;
  using ConstSharedPtr = std::shared_ptr<const PlayerPoseROSNode>;

 private:
  class PlayerPoseROSNodeImpl;
  std::shared_ptr<PlayerPoseROSNodeImpl> impl_;

 public:
  PlayerPoseROSNode();

  void StartWithNode(std::shared_ptr<rclcpp::Node> node, PlayerPoseExtractor::SharedPtr extractor, bool enable_bbox_pub,
                     bool enable_pose2d_pub, bool enable_pose3d_pub);
  void Start(const std::string& node_name, PlayerPoseExtractor::SharedPtr extractor, bool enable_bbox_pub,
             bool enable_pose2d_pub, bool enable_pose3d_pub);
  void Stop();
};
}  // namespace perception
