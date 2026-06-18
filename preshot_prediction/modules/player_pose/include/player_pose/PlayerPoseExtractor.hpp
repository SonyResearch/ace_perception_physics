// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <thread>

#include "calibration/camera_calibration_parameters.hpp"
#include "player_pose/PlayerFeatures.hpp"
#include "player_pose/PlayerPoseParameters.hpp"

namespace perception {

class PlayerPoseExtractor {
 public:
  using SharedPtr = std::shared_ptr<PlayerPoseExtractor>;
  using ConstSharedPtr = std::shared_ptr<const PlayerPoseExtractor>;

  class ImageData {
   public:
    using SharedPtr = std::shared_ptr<ImageData>;
    explicit ImageData() = default;
    ~ImageData() = default;

    int camera_index{-1};
    cv::Mat image;
    cv::cuda::GpuMat uploaded_image;
    bool cropped{false};
    bool processed{false};
  };

  using PlayerDetectedCallback = std::function<void(size_t, size_t, size_t, const Eigen::Vector4i&,
                                                    float)>;  // seq_id,player_idx,cam_idx,bbox, confidence
  using PlayerPose2DCallback = std::function<void(size_t, const PlayerPoseDetection::SharedPtr&)>;  // seq_id,features
  using PlayerPose3DCallback =
    std::function<void(size_t, int, const PlayerPoseEstimate::SharedPtr&)>;  // seq_id,player_id,pose

 protected:
  class PlayerPoseExtractorImpl;
  std::shared_ptr<PlayerPoseExtractorImpl> impl_;

 public:
  explicit PlayerPoseExtractor(const std::string& log_name);
  ~PlayerPoseExtractor();

  void Start(const std::string& camera_calib_params_path, const std::string& Player_params_path,
             const std::string& model_path);
  void Stop();

  bool OnImages(size_t seq_id, std::vector<ImageData::SharedPtr>& images);

  void SetPlayerDetectedCallback(PlayerDetectedCallback callback);
  void SetPlayerPose2DCallback(PlayerPose2DCallback callback);
  void SetPlayerPose3DCallback(PlayerPose3DCallback callback);

  const std::vector<PlayerFrameFeatures>& GetLastDetections();
  PlayerPoseDetection::SharedPtr GetLastDetectionForCamera(int index);
  PlayerBBoxDetection::SharedPtr GetLastBBoxDetectionForCamera(int index);
  ::datalogger::DataWriter::SharedPtr GetDataLogger();
  void SetDataLoggerFileName(const std::string& name);

  calibration::CameraCalibrationParameters::SharedPtr GetCameraCalibrationParameters();
  PlayerPoseParameters::SharedPtr GetPlayerParameters();
};

}  // namespace perception
