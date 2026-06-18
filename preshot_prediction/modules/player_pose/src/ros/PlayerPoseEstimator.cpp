// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "player_pose/ros/PlayerPoseEstimator.hpp"

#include <filesystem>
#include <memory>

#include "ace_loggers/ace_loggers.hpp"
#include "player_pose/PlayerPoseExtractor.hpp"
#include "player_pose/ros/PlayerPoseROSNode.hpp"

namespace perception {

PlayerPoseEstimator::PlayerPoseEstimator(const rclcpp::NodeOptions& options)
  : rclcpp::Node("player_pose_estimation", options) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }
  init_ = false;
}

PlayerPoseEstimator::~PlayerPoseEstimator() { Destroy(); }

void PlayerPoseEstimator::Initialize() {
  auto camera_config_path = declare_parameter<std::string>("camera_config_path");
  auto player_config_path = declare_parameter<std::string>("player_config_path");
  auto keypoints_detector = declare_parameter<std::string>("keypoints_detector");

  node_ = std::make_shared<perception::PlayerPoseROSNode>();
  extractor_ = std::make_shared<perception::PlayerPoseExtractor>(
    ::datalogger::ConstructFullLogName(std::filesystem::path(player_config_path).stem()));

  extractor_->Start(camera_config_path, player_config_path, keypoints_detector);

  node_->StartWithNode(shared_from_this(), extractor_, true, true, true);
  init_ = true;
}
void PlayerPoseEstimator::Destroy() { node_->Stop(); }
}  // namespace perception
