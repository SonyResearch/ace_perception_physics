// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include "player_pose/PlayerFeatures.hpp"
#include "player_pose/PlayerPoseParameters.hpp"

namespace perception {

class PersonDetector {
 public:
  using SharedPtr = std::shared_ptr<PersonDetector>;
  using ConstSharedPtr = std::shared_ptr<const PersonDetector>;

  using PlayerDetectedCallback = std::function<void(PlayerBBoxDetection::SharedPtr)>;

 private:
  class PersonDetectorImpl;
  std::shared_ptr<PersonDetectorImpl> impl_;

 public:
  PersonDetector();

  bool Initialize(PlayerPoseParameters::SharedPtr player_params, const std::string& model_path);
  void AddBBoxDetectionRequest(size_t seq_id, size_t player_index, size_t cam_index, cv::cuda::GpuMat& image, bool correct_rotation = false);
  void AddBBoxDetection(PlayerBBoxDetection::SharedPtr detection);
  bool GetLastBBoxDetection(size_t player_index, size_t camera_index, PlayerBBoxDetection::SharedPtr& person);

  void SetPlayerDetectedCallback(PlayerDetectedCallback callback);

  bool ExtractPerson(const std::vector<cv::cuda::GpuMat>& input, std::vector<cv::cuda::GpuMat*>* extracted,
                     std::vector<PlayerBBoxDetection::SharedPtr>& detection, bool correct_rotation = false);
};
}  // namespace perception
