// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <thread>

#include "calibration/camera_calibration_parameters.hpp"
#include "racket_pose_estimation/RacketFeatures.hpp"
#include "racket_pose_estimation/RacketParameters.hpp"

namespace perception {

class RacketPoseFeatures;
class RacketPoseExtractor {
 public:
  using SharedPtr = std::shared_ptr<RacketPoseExtractor>;
  using ConstSharedPtr = std::shared_ptr<const RacketPoseExtractor>;

  using ImageData = std::pair<size_t, cv::Mat>;
  using RacketPose3DCallback =
    std::function<void(size_t, size_t, RacketEstimatedPose::SharedPtr)>;  // seq_id, racket_id, pose
  using RacketDetection2DCallback =
    std::function<void(size_t, size_t, RacketPoseFeatures::SharedPtr)>;  // racket_id,camera_id,detection

 protected:
  class RacketPoseExtractorImpl;
  std::shared_ptr<RacketPoseExtractorImpl> impl_;

 public:
  explicit RacketPoseExtractor(const std::string& log_name);
  ~RacketPoseExtractor();

  bool Start(const std::string& camera_calib_params_path, const std::string& racket_params_path,
             const std::string& model_path, bool perform_estimation);
  void Stop();

  void SetROI(size_t racket_idx, size_t camera_idx, const Eigen::Vector4i& roi);
  bool GetROI(size_t racket_idx, size_t camera_idx, Eigen::Vector4i& roi);
  bool OnImages(size_t seq_id, const std::vector<ImageData>& images, bool async = true);

  void SetRacketPose3DCallback(RacketPose3DCallback callback);
  void SetRacketDetection2DCallback(RacketDetection2DCallback callback);
  const std::vector<RacketFrameFeatures>& GetLastDetections();
  RacketPoseFeatures::SharedPtr GetLastDetectionForCamera(size_t camera_idx);

  calibration::CameraCalibrationParameters::SharedPtr GetCameraCalibrationParameters();
  RacketParameters::SharedPtr GetRacketParameters();
  ::datalogger::DataWriter::SharedPtr GetDataLogger();
  void SetDataLoggerFileName(const std::string& name);
};

}  // namespace perception
