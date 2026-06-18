// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <triangulation/triangulation.hpp>

#include "base_parameters/base_parameters.hpp"
#include "calibration/camera_calibration_parameters.hpp"
#include "eigen3/Eigen/Eigen"
#include "player_pose/PlayerFeatures.hpp"
#include "player_pose/YoloObjectDetector.hpp"

namespace perception {

class PlayerPoseParameters final : public BaseParameters {
 public:
  using PlayerCameraPair = std::pair<size_t, size_t>;

  class CameraParameters {
   public:
    enum class RotationType { kNone, kCW, kCCW, kK180, kAuto };
    RotationType rotation_type{RotationType::kNone};
    std::string name;
    std::vector<uint16_t> resolution;
    int index;
    double rotation{0};
  };
  class CameraROI {
   public:
    std::string camera_name;
    size_t camera_calib_index;   // camera index in the calibration file
    size_t camera_params_index;  // camera index in this parameter file (params.cameras)
    cv::Rect2i roi;
  };
  class PlayerParameters {
   public:
    int player_id;
    std::string name;
    std::map<size_t, CameraROI> cameras;
  };
  class DetectorParameters : public YoloConfig {
   public:
    using SharedPtr = std::shared_ptr<DetectorParameters>;
    using ConstSharedPtr = std::shared_ptr<const DetectorParameters>;
    DetectorParameters() = default;
    int device_index{0};

    Eigen::Vector2f bbox_inflation{0.4, 0.2};  // inflation in %
    float latest_weight{0.8};
    int history_difference{10};
    int batch_size{4};
  };
  class ExtractorParameters {
   public:
    ExtractorParameters() = default;

    float sampling{1};
    size_t workers_count{2};
    int outdated_diff{16};  // diff in timestamp before reporting outdated
    bool apply_rotation_correction{true};
    float latest_weight{0.8};
    int history_difference{10};
    float min_keypoint_confidence{0.4};
    bool undistort_keypoints{true};

    std::string rt_profile;
  };
  class CameraCaptureParameters {
   public:
    CameraCaptureParameters() = default;
    int camera_buffer_length{2};
    int worker_queue_length{3};
    bool use_variable_response_time{false};
  };

  using SharedPtr = std::shared_ptr<PlayerPoseParameters>;
  using ConstSharedPtr = std::shared_ptr<const PlayerPoseParameters>;

  explicit PlayerPoseParameters();

  std::map<int, PlayerParameters> players;
  std::unordered_map<size_t, size_t> cameras_map;
  CameraCaptureParameters capture;
  std::vector<CameraParameters> cameras;
  ExtractorParameters extractor;
  triangulation::TriangulationParameters::SharedPtr triangulator;
  DetectorParameters::SharedPtr person_detector;
  std::set<std::string> cameras_names;

  // calculated from cameras calibration parameters
  std::unordered_map<size_t, cv::Mat> camera_rotation_mat;
  std::unordered_map<size_t, cv::Mat> camera_rotation_inv_mat;

  // Update internal parameters
  void UpdateInternalParameters();
  void ParseCameras(YAML::Node node);

  void InitializeCameras(calibration::CameraCalibrationParameters::SharedPtr camera_calib_params);
  cv::cuda::GpuMat RotateImage(size_t camera_index, cv::cuda::GpuMat frame);
  void InvRotateKeypoints(size_t camera_index, std::vector<cv::Point2f>& points);
  void InvRotateDetections(PlayerFrameFeatures& features);

  void InvRotateBBox(size_t camera_index, PlayerBBoxDetection::Boundingbox& bbox);

  template <typename T>
  T GetYamlValue(const std::string& name, const T& default_value) const {
    try {
      return yaml_node_[name].as<T>();
    } catch (YAML::Exception& e) {
    }
    return default_value;
  }

  static bool CheckSampling(size_t sample_id, float modular) {
    return static_cast<int>(static_cast<float>(sample_id) - modular * floor(static_cast<float>(sample_id) / modular)) ==
           0;
  }

 private:
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace perception
