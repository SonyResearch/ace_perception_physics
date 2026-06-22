// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <unordered_map>

#include "ace_containers/indexed_map.hpp"
#include "base_parameters/base_parameters.hpp"
#include "eigen3/Eigen/Eigen"
#include "opencv2/opencv.hpp"

namespace calibration {

enum class CameraModel { kInvalid = -1, kPinhole, kNumCameraModels };

inline std::string CameraModelToString(const CameraModel& camera_model) {
  switch (camera_model) {
    case CameraModel::kPinhole:
      return "pinhole";

    default:
      return "invalid";
  }
}

inline CameraModel StringToCameraModel(const std::string& s) {
  if (s == "pinhole") {
    return CameraModel::kPinhole;
  }

  return CameraModel::kInvalid;
}

enum class DistortionModel { kInvalid = -1, kEquidistant, kRadialTangential, kNumDistortionModels };

inline std::string DistortionModelToString(const DistortionModel& distortion_model) {
  switch (distortion_model) {
    case DistortionModel::kEquidistant:
      return "equidistant";
    case DistortionModel::kRadialTangential:
      return "radtan";

    default:
      return "invalid";
  }
}

inline DistortionModel StringToDistortionModel(const std::string& s) {
  if (s == "equidistant") {
    return DistortionModel::kEquidistant;
  }
  if (s == "radtan") {
    return DistortionModel::kRadialTangential;
  }

  return DistortionModel::kInvalid;
}

struct MultiViewData {
  using SharedPtr = std::shared_ptr<MultiViewData>;
  using ConstSharedPtr = std::shared_ptr<const MultiViewData>;

  Eigen::Matrix3f fundamental_matrix;  // The epipolar-constraint map between left and right views
  Eigen::Vector2f epipole_i;           // The eipole in the left view
  Eigen::Vector2f epipole_j;           // The eipole in the right view
};

class Camera {
 public:
  using SharedPtr = std::shared_ptr<Camera>;
  using ConstSharedPtr = std::shared_ptr<const Camera>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::vector<uint16_t> resolution;  // [width, height]

  CameraModel camera_model;
  Eigen::Matrix<float, 3, 3> camera_matrix;
  Eigen::Matrix<float, 3, 3> camera_matrix_inv;
  cv::Mat camera_matrix_cv;

  bool distortion_enable;
  DistortionModel distortion_model;
  uint n_radial_coeffs;
  uint n_tangential_coeffs;
  std::vector<float> distortion_coeffs;
  Eigen::VectorXf radial_coeffs;
  Eigen::VectorXf tangential_coeffs;

  Eigen::Matrix<float, 4, 4> T_camera_world;   // transformation from (W)orld to (C)amera frame
  Eigen::Matrix<float, 4, 4> T_world_camera;   // transformation from (C)amera to (W)orld frame
  Eigen::Matrix<float, 4, 4> T_camera_origin;  // transformation from (O)rigin to (C)amera frame
  Eigen::Matrix<float, 4, 4> T_origin_camera;  // transformation from (C)amera to (O)rigin frame

  float rotation;  // rotation of camera w.r.t. xy-plane of (O)rigin frame

  // (Un)distortion
  [[nodiscard]] Eigen::Vector2f UndistortPoint(const Eigen::Vector2f& p_d) const;
  [[nodiscard]] std::vector<Eigen::Vector2f> UndistortPoints(const std::vector<Eigen::Vector2f>& p_ds) const;
  [[nodiscard]] Eigen::Matrix2Xf UndistortPointsVectorized(const Eigen::Matrix2Xf& p_ds) const;
  void UndistortImage(const cv::Mat& img_distorted, cv::Mat& img_undistorted) const;
  [[nodiscard]] Eigen::Vector2f DistortPoint(const Eigen::Vector2f& p_u,
                                             Eigen::Matrix<float, 2, 3>* dprojection_dx = nullptr) const;
  [[nodiscard]] std::vector<Eigen::Vector2f> DistortPoints(const std::vector<Eigen::Vector2f>& p_us) const;
  [[nodiscard]] Eigen::Matrix2Xf DistortPointsVectorized(const Eigen::Matrix2Xf& p_us) const;
  void DistortImage(const cv::Mat& img_undistorted, cv::Mat& img_distorted) const;

  // Reprojection
  Eigen::Matrix<float, 3, 4> projection_matrix;  // projection matrix
  Eigen::Matrix4f projection_matrix_inv;
  [[nodiscard]] bool ProjectPointPybind(const Eigen::Vector3f& x, Eigen::Ref<Eigen::Vector2f> p_d) const {
    Eigen::Vector2f p_d_copy = p_d;
    bool ret = this->ProjectPoint(x, &p_d_copy);

    p_d = p_d_copy;
    return ret;
  }
  [[nodiscard]] std::tuple<Eigen::MatrixX2f, Eigen::Matrix<bool, Eigen::Dynamic, 1>> ProjectPointsPybind(
    const Eigen::MatrixX3f& xs) const {
    Eigen::MatrixX2f p_ds(xs.rows(), 2);
    Eigen::Matrix<bool, Eigen::Dynamic, 1> inliers(xs.rows(), 1);
    for (Eigen::Index i = 0; i < xs.rows(); ++i) {
      Eigen::Vector2f p_d = Eigen::Vector2f::Constant(2, 1, std::numeric_limits<float>::quiet_NaN());
      inliers(i, 0) = this->ProjectPoint(xs.row(i), &p_d);
      p_ds.row(i) = p_d;
    }

    return std::make_tuple(p_ds, inliers);
  }
  [[nodiscard]] bool ProjectPointExtendedPybind(const Eigen::Vector3f& X, Eigen::Ref<Eigen::Vector2f> p_d,
                                                Eigen::Ref<Eigen::Matrix<float, 2, 3>> dprojection_dx) const {
    Eigen::Vector2f p_d_copy = p_d;
    Eigen::Matrix<float, 2, 3> dprojection_dx_copy = dprojection_dx;
    bool ret = this->ProjectPoint(X, &p_d_copy, &dprojection_dx_copy);

    p_d = p_d_copy;
    dprojection_dx = dprojection_dx_copy;
    return ret;
  }
  bool ProjectPoint(const Eigen::Vector3f& x, Eigen::Vector2f* p_d,
                    Eigen::Matrix<float, 2, 3>* dprojection_dx = nullptr, bool apply_distortion = true) const;

  // Update internal parameters
  void UpdateInternalParameters(const Eigen::Matrix4f& T_world_origin = Eigen::Matrix4f::Identity());

 private:
  Eigen::Matrix<float, 3, 3> W_d_;
  static constexpr int kNd = 1000;
  static constexpr float kEps = 1e-8F;
  float r_d_max_;               // maximum distorted radius
  float s_aux_r_d_[kNd];        // scale from distorted radius
  float r_u_max_;               // maximum undistorted radius
  float s_aux_r_u_[kNd];        // scale from undistorted radius
  float ds_dr_u_aux_r_u_[kNd];  // derivative of scale with respect to undistorted radius
  cv::Mat undistortion_map_x_, undistortion_map_y_;
  cv::Mat distortion_map_x_, distortion_map_y_;

  void InitializeUndistortionLookupMap();
  void InitializeDistortionLookupMap();
  void InitializeOpenCVDistortionMaps();

  [[nodiscard]] float ComputeCacheFromDistortedRadius(const float& r_d) const;
  [[nodiscard]] Eigen::Vector2f UndistortNormalizedPoint(const Eigen::Vector2f& p_d) const;
  [[nodiscard]] Eigen::Matrix2Xf UndistortNormalizedPointsVectorized(const Eigen::Matrix2Xf& p_ds) const;

  [[nodiscard]] std::tuple<float, float> ComputeCacheFromUndistortedRadius(const float& r_u) const;
  [[nodiscard]] Eigen::Vector2f DistortNormalizedPoint(const Eigen::Vector2f& p_u,
                                                       Eigen::Matrix<float, 2, 3>* dprojection_dx = nullptr) const;
  [[nodiscard]] Eigen::Matrix2Xf DistortNormalizedPointsVectorized(const Eigen::Matrix2Xf& p_us) const;
};

class CameraCalibrationParameters final : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<CameraCalibrationParameters>;
  using ConstSharedPtr = std::shared_ptr<const CameraCalibrationParameters>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit CameraCalibrationParameters();

  std::vector<std::string> camera_names;
  // Note that maps instead of unordered map are used such that the order of cameras and fundamental matrices is
  // consistent.
  ace_containers::IndexedMap<std::string, Camera> cameras;
  ace_containers::IndexedMap<std::string, ace_containers::IndexedMap<std::string, MultiViewData>> multiview_data;
  Eigen::Matrix<float, 4, 4> T_world_origin;  // transformation from (O)rigin to (W)orld frame
  Eigen::Matrix<float, 4, 4> T_origin_world;  // transformation from (W)orld to (O)rigin frame

  // Update internal parameters
  void UpdateInternalParameters();

 private:
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace calibration
