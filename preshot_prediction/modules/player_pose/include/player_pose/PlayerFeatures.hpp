// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once
#include <Eigen/Dense>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "ace_loggers/ace_loggers.hpp"

namespace perception {
class PlayerBBoxDetection {
 public:
  using SharedPtr = std::shared_ptr<PlayerBBoxDetection>;
  using ConstSharedPtr = std::shared_ptr<const PlayerBBoxDetection>;

  using Boundingbox = Eigen::Vector4i;
  using Keypoint = Eigen::Vector3f;

  size_t sequence_number{0};
  size_t player_index{0};
  size_t camera_index{0};
  float confidence{0};
  Boundingbox bbox;
  std::vector<Keypoint> keypoints;
};

class PlayerPoseDetection {
 public:
  using SharedPtr = std::shared_ptr<PlayerPoseDetection>;
  using ConstSharedPtr = std::shared_ptr<const PlayerPoseDetection>;

  using Keypoint = Eigen::Vector3f;

  size_t sequence_number{0};
  size_t player_index{0};
  size_t camera_index{0};
  PlayerBBoxDetection::Boundingbox bbox;
  std::vector<Keypoint> keypoints;
};

class PlayerPoseEstimate {
 public:
  using SharedPtr = std::shared_ptr<PlayerPoseEstimate>;
  using ConstSharedPtr = std::shared_ptr<const PlayerPoseEstimate>;

  int player_id;

  std::vector<Eigen::Vector3f> keypoints;
  std::vector<float> projection_error;
  std::vector<float> confidences;
};
class PlayerFrameFeatures {
 public:
  using SharedPtr = std::shared_ptr<PlayerFrameFeatures>;
  using ConstSharedPtr = std::shared_ptr<const PlayerFrameFeatures>;

  size_t sequence_number;
  std::vector<PlayerPoseDetection::SharedPtr> features;
  std::vector<PlayerPoseEstimate::SharedPtr> estimated_players;
};

}  // namespace perception
