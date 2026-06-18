// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <unordered_set>

#include "ace_loggers/ace_loggers.hpp"
#include "calibration/camera_calibration_parameters.hpp"
#include "eigen3/Eigen/Eigen"

namespace triangulation {

// Covariance models.
enum class CovarianceModel { kInvalid = -1, kMetric, kPixel };

inline std::string CovarianceModelToString(const CovarianceModel& covariance_model) {
  switch (covariance_model) {
    case CovarianceModel::kMetric:
      return "metric";

    default:
      return "invalid";
  }
}

inline CovarianceModel StringToCovarianceModel(const std::string& s) {
  if (s == "metric") {
    return CovarianceModel::kMetric;
  }

  return CovarianceModel::kInvalid;
}

// Triangulation
struct TriangulatedPoint {
  using SharedPtr = std::shared_ptr<TriangulatedPoint>;
  using ConstSharedPtr = std::shared_ptr<const TriangulatedPoint>;

  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();

  std::vector<std::pair<int, int>> observation_indices;
  float reprojection_error = std::numeric_limits<float>::infinity();
};

struct TriangulationParameters {
  using SharedPtr = std::shared_ptr<TriangulationParameters>;
  using ConstSharedPtr = std::shared_ptr<const TriangulationParameters>;

  virtual ~TriangulationParameters() = default;

  // geometric error for correspondence matching.
  float geometric_error_threshold;

  // ball-radius error for correspondence matching.
  float radius_error_threshold;

  // Covariance.
  CovarianceModel covariance_model;
  float observation_variance;

  // Reprojection error minimization (nonlinear refinement).
  bool use_nonlinear_refinement;
  float tol_position;
  float tol_reprojection_error;
  int max_iterations;

  enum class GhostBallsPolicy { kNone, kDrop, kMerge };

  float similarity_threshold{0.4F};
  GhostBallsPolicy ghost_balls_policy{GhostBallsPolicy::kNone};
};

template <class DetectionPoint>
class Triangulation {
 public:
  using SharedPtr = std::shared_ptr<Triangulation>;
  using ConstSharedPtr = std::shared_ptr<const Triangulation>;
  using FilterTriangulationFunction = std::function<bool(const TriangulatedPoint&)>;

  explicit Triangulation(TriangulationParameters::ConstSharedPtr settings_ptr,
                         calibration::CameraCalibrationParameters::ConstSharedPtr camera_calibration_params_ptr,
                         ::datalogger::DataWriter::SharedPtr datawriter_ptr = nullptr);

  const TriangulationParameters::ConstSharedPtr& GetTriangulationParameters() { return settings_ptr_; };
  const calibration::CameraCalibrationParameters::ConstSharedPtr& GetCameraCalibrationParameters() {
    return camera_calib_params_ptr_;
  };

  bool SetTriangulationParameters(const TriangulationParameters::ConstSharedPtr& settings_ptr) {
    settings_ptr_ = settings_ptr;
    return true;
  };
  bool SetCameraCalibrationParameters(
    const calibration::CameraCalibrationParameters::ConstSharedPtr& camera_calibration_params_ptr) {
    camera_calib_params_ptr_ = camera_calibration_params_ptr;
    return true;
  };

  bool SetCameras(const std::vector<std::string>& camera_names);

  std::vector<TriangulatedPoint> TriangulatePoints(std::vector<std::vector<DetectionPoint>>& distorted_points);
  std::vector<TriangulatedPoint> TriangulateUncertainPoints(std::vector<std::vector<DetectionPoint>>& distorted_points,
                                                            std::vector<std::vector<Eigen::Vector2f>>& certainties);
  std::vector<TriangulatedPoint> TriangulateSortedPoints(
    std::vector<std::vector<DetectionPoint>>& distorted_points,
    std::vector<std::vector<std::pair<int, int>>>& point_correspondences);

  TriangulatedPoint TriangulatePoint(const std::vector<DetectionPoint>& observations,
                                     const std::unordered_set<int>& obs, const std::vector<int>& obs_to_cam_idx,
                                     bool undistort);
  [[nodiscard]] const ::datalogger::DataWriter::SharedPtr& GetDataLogger() const;

  void SetTriangulationFilter(FilterTriangulationFunction func) { triangulation_filter_ = func; }

 private:
  using Camera = calibration::Camera;

  bool init_ = false;

  // Triangulation parameters.
  TriangulationParameters::ConstSharedPtr settings_ptr_;
  calibration::CameraCalibrationParameters::ConstSharedPtr camera_calib_params_ptr_;

  FilterTriangulationFunction triangulation_filter_{nullptr};

  // Cameras.
  int num_cameras_;
  std::vector<size_t> camera_to_calib_indices_;

  static std::pair<float, float> ErrorCriteria(const DetectionPoint& pt_i, const DetectionPoint& pt_j,
                                               const Eigen::Matrix3f& F_ij, const Eigen::Vector2f& epipole_i,
                                               const Eigen::Vector2f& epipole_j);
  TriangulatedPoint TriangulatePointWithRefinement(std::vector<std::vector<DetectionPoint>>& distorted_points,
                                                   const std::vector<DetectionPoint>& undistorted_pts,
                                                   const std::unordered_set<int>& obs,
                                                   const std::vector<int>& obs_to_cam_idx,
                                                   const std::vector<int>& obs_to_pt_idx);
  TriangulatedPoint TriangulatePointDLT(const std::vector<DetectionPoint>& undistorted_pts,
                                        const std::unordered_set<int>& obs, const std::vector<int>& obs_to_cam_idx);
  TriangulatedPoint MinimizeReprojectionErrorGradDescent(
    const Eigen::Vector3f& x_guess, const std::vector<std::vector<DetectionPoint>>& distorted_points,
    const std::unordered_set<int>& obs, const std::vector<int>& obs_to_cam_idx, const std::vector<int>& obs_to_pt_idx);
  TriangulatedPoint MinimizeReprojectionErrorFixPoint(const Eigen::Vector3f& x_guess,
                                                      const std::vector<std::vector<DetectionPoint>>& distorted_points,
                                                      const std::unordered_set<int>& obs,
                                                      const std::vector<int>& obs_to_cam_idx,
                                                      const std::vector<int>& obs_to_pt_idx);
  // Covariance.
  Eigen::MatrixXf observation_cov_;
  Eigen::Matrix3f ComputeCovariance(const Eigen::MatrixX3f& jac, const std::unordered_set<int>& obs);

  ::datalogger::DataWriter::SharedPtr datawriter_ptr_;
};

}  // namespace triangulation
