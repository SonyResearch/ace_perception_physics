// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include <random>
#include <vector>

#include "calibration/camera_calibration_parameters.hpp"
#include "eigen3/Eigen/Eigen"
#include "racket_pose_estimation/RacketParameters.hpp"

namespace cal = ::calibration;

namespace perception {
class RacketPoseFitter {
 public:
  using SharedPtr = std::shared_ptr<RacketPoseFitter>;
  using ConstSharedPtr = std::shared_ptr<const RacketPoseFitter>;

  RacketPoseFitter();

  void Initialize(RacketParameters::SharedPtr params) { params_ = params; }
  void SetCalibration(cal::CameraCalibrationParameters::SharedPtr calibration) { calibration_ = calibration; }
  void SetCameras(const std::vector<int>& cameras);
  const std::vector<int>& GetCameras() { return cameras_; }
  void ResetInitialRotation(float w, float x, float y, float z) {
    last_angles_ = Eigen::Quaternionf(w, x, y, z);
    last_error_ = 1e3F;
  }

  // keypoints are (x,y,confidence)
  Eigen::Quaternionf FitPoints(const Eigen::Vector3f& position,
                               const std::vector<std::vector<Eigen::Vector3f>>& keypoints, int& out_iterations,
                               float& out_error);

 protected:
  Eigen::Quaternionf last_angles_;
  float last_error_{9999};
  cal::CameraCalibrationParameters::SharedPtr calibration_;
  std::vector<int> cameras_;

  Eigen::Vector3f model_points_[2];

  std::vector<std::vector<Eigen::Vector2f>> projected_points_;
  std::vector<Eigen::Vector2f> zero_points_;
  std::vector<Eigen::Vector2f> key_points_center_;
  std::vector<float> projection_errors_;
  std::vector<float> view_weight_;
  std::mt19937 random_device_;

  std::vector<Eigen::Quaternionf> axis_flip_;

  RacketParameters::SharedPtr params_;

  float EvalAt(const Eigen::Vector3f& position, const Eigen::Quaternionf& q,
               const std::vector<std::vector<Eigen::Vector3f>>& target_axis, bool use_axis_order);

  void GetGradients(float curr_err, const Eigen::Vector3f& position, const Eigen::Quaternionf& q, float* gradients,
                    const std::vector<std::vector<Eigen::Vector3f>>& target_axis);
};

}  // namespace perception
