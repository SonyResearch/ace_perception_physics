// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <triangulation/triangulation.hpp>

#include "base_parameters/base_parameters.hpp"
#include "eigen3/Eigen/Eigen"

namespace perception {

class RacketParameters final : public BaseParameters {
 public:
  using PlayerCameraPair = std::pair<size_t, size_t>;
  class CameraCaptureParameters {
   public:
    CameraCaptureParameters() = default;
    int camera_buffer_length{2};
    int worker_queue_length{3};
    bool use_variable_response_time{false};

    float left_hand{0.0};
    float right_hand{1.0};
  };
  class CameraParameters {
   public:
    std::string name;
    std::vector<uint16_t> resolution;
    int index;
  };
  class CameraROI {
   public:
    std::string camera_name;
    size_t camera_calib_index;   // camera index in the calibration file
    size_t camera_params_index;  // camera index in this parameter file (params.cameras)
    cv::Rect2i roi;

    Eigen::Vector4i intersect_roi(const Eigen::Vector4i& other_roi) const {
      // 1. Calculate the coordinates of the intersection boundaries
      int x1 = std::max(other_roi[0], roi.x);
      int y1 = std::max(other_roi[1], roi.y);
      int x2 = std::min(other_roi[0] + other_roi[2], roi.x + roi.width);
      int y2 = std::min(other_roi[1] + other_roi[3], roi.y + roi.height);

      // 2. Calculate width and height (clamping to 0 if no overlap)
      int width = std::max(0, x2 - x1);
      int height = std::max(0, y2 - y1);

      return Eigen::Vector4i(x1, y1, width, height);
    }
  };
  class PlayerParameters {
   public:
    int racket_id;
    std::string name;
    std::map<size_t, CameraROI> cameras;
  };
  class KeypointFitterParameters {
   public:
    KeypointFitterParameters() = default;
    int max_skips_reset{10};
    float max_delta_err_reset{1e-3};
    float max_error{1e-2};
    int max_iterations{200};
    float initial_rate{1e-1};

    float min_axis_length{10};  // length in pixels
  };
  class ExtractorParameters {
   public:
    ExtractorParameters() = default;

    int device_index{0};
    int opt_batch_size{3};
    int max_batch_size{6};

    size_t workers_count{2};
    size_t outdated_diff{16};  // diff in timestamp before reporting outdated

    float sampling{1};
    float confidence_threshold{0.3};
    float nms_threshold{0.4};
    float latest_weight{0.8};
    int history_difference{10};
    std::string rt_profile;
    Eigen::Vector2i roi{320, 320};  // ROI in pixels
    bool debug_output{false};
  };

  using SharedPtr = std::shared_ptr<RacketParameters>;
  using ConstSharedPtr = std::shared_ptr<const RacketParameters>;

  explicit RacketParameters();

  std::map<size_t, PlayerParameters> rackets;
  CameraCaptureParameters capture;
  KeypointFitterParameters fitter;
  ExtractorParameters extractor;
  triangulation::TriangulationParameters::SharedPtr triangulator;
  std::vector<CameraParameters> cameras;
  std::set<std::string> cameras_names;

  // Update internal parameters
  void UpdateInternalParameters();

  void ParseCameras(YAML::Node node);

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
