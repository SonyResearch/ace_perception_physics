// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace perception {

class RacketPoseFeatures {
 public:
  using SharedPtr = std::shared_ptr<RacketPoseFeatures>;
  using ConstSharedPtr = std::shared_ptr<const RacketPoseFeatures>;

  using Boundingbox = Eigen::Vector4i;
  using Keypoint = Eigen::Vector3f;
  RacketPoseFeatures() { keypoints.resize(4); }

  size_t sequence_number{0};
  size_t player_index;
  size_t camera_index;
  float confidence{0};
  Boundingbox bbox;
  Eigen::Vector3i center;  // x,y,size
  std::vector<Keypoint> keypoints;

  void OffsetBy(int x, int y) {
    bbox.x() += x;
    bbox.y() += y;
    center.x() += x;
    center.y() += y;
    for (auto& kp : keypoints) {
      kp.x() += static_cast<float>(x);
      kp.y() += static_cast<float>(y);
    }
  }
};
class RacketEstimatedPose {
 public:
  using SharedPtr = std::shared_ptr<RacketEstimatedPose>;
  using ConstSharedPtr = std::shared_ptr<const RacketEstimatedPose>;

  void SetOrientation(const Eigen::Matrix<float, 4, 1>& quat) {
    // W coefficient must be passed first!
    orientation = Eigen::Quaternionf(quat(3), quat(0), quat(1), quat(2));
  }

  [[nodiscard]] Eigen::Matrix<float, 4, 1> GetOrientation() const {
    return Eigen::Matrix<float, 4, 1>(orientation.x(), orientation.y(), orientation.z(), orientation.w());
  }

  size_t racket_id;
  Eigen::Vector3f position;
  Eigen::Quaternionf orientation;
  int iterations_count{-1};                                   // number of iterations it took to find a solution
  float reprojection_err{std::numeric_limits<float>::max()};  // reprojection error in pixel
  float orientation_error{1};                                 // quality of orientation axis fitting [0-1]
  float orientation_confidence{
    0};  // confidence of orientation fitting [0-1], represents ratio of cameras used vs overall cameras
};
class RacketFrameFeatures {
 public:
  using SharedPtr = std::shared_ptr<RacketFrameFeatures>;
  using ConstSharedPtr = std::shared_ptr<const RacketFrameFeatures>;

  size_t sequence_number;
  std::vector<RacketPoseFeatures::SharedPtr> features;
  std::vector<RacketEstimatedPose::SharedPtr> estimated_rackets;
};

}  // namespace perception
