// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "calibration/camera_calibration_parameters.hpp"

#include <cmath>

#include <glog/logging.h>

namespace calibration {

float Camera::ComputeCacheFromDistortedRadius(const float& r_d) const {
  // This function is solving for `s(r_d) = r_d / r_u(r_d)`
  switch (distortion_model) {
    case DistortionModel::kRadialTangential: {
      float r_u = r_d;

      while (true) {
        Eigen::VectorXf sqr_r_us(n_radial_coeffs);
        Eigen::VectorXf delta_coeffs(n_radial_coeffs);
        sqr_r_us[0] = r_u * r_u;
        delta_coeffs[0] = 3.F * sqr_r_us[0];
        for (uint i = 1; i < n_radial_coeffs; ++i) {
          sqr_r_us[i] = sqr_r_us[i - 1] * sqr_r_us[0];
          delta_coeffs[i] = (3.F + 2.F * static_cast<float>(i)) * sqr_r_us[i];
        }
        const Eigen::VectorXf dcoeffs_sqr_r_us = sqr_r_us.cwiseProduct(radial_coeffs);
        const float r_d_hat = r_u * (1.F + dcoeffs_sqr_r_us.sum());
        const float dr_d_hat_dr_u = 1.F + delta_coeffs.cwiseProduct(radial_coeffs).sum();
        const float delta_r_u = -(r_d_hat - r_d) / dr_d_hat_dr_u;
        r_u += delta_r_u;

        if (delta_r_u <= kEps) {
          break;
        }
      }

      return r_d / r_u;
    }
    case DistortionModel::kEquidistant: {
      float theta = r_d;

      while (true) {
        Eigen::VectorXf sqr_thetas(n_radial_coeffs);
        Eigen::VectorXf delta_coeffs(n_radial_coeffs);
        sqr_thetas[0] = theta * theta;
        delta_coeffs[0] = 3.F * sqr_thetas[0];
        for (uint i = 1; i < n_radial_coeffs; ++i) {
          sqr_thetas[i] = sqr_thetas[i - 1] * sqr_thetas[0];
          delta_coeffs[i] = (3.F + 2.F * static_cast<float>(i)) * sqr_thetas[i];
        }
        const Eigen::VectorXf dcoeffs_sqr_thetas = sqr_thetas.cwiseProduct(radial_coeffs);
        const float r_d_hat = theta * (1.F + dcoeffs_sqr_thetas.sum());
        const float dr_d_hat_dr_u = 1.F + delta_coeffs.cwiseProduct(radial_coeffs).sum();
        const float delta_theta = -(r_d_hat - r_d) / dr_d_hat_dr_u;
        theta += delta_theta;

        if (delta_theta <= kEps) {
          break;
        }
      }

      return r_d / std::tan(theta);
    }
    default: {
      throw std::runtime_error("Distortion model is not supported!");
    }
  }
}

Eigen::Vector2f Camera::UndistortNormalizedPoint(const Eigen::Vector2f& p_d) const {
  Eigen::Vector2f p_u;
  Eigen::Vector2f p_d_offset = p_d;
  for (int iter = 0; iter < 3; ++iter) {
    const float r_d = p_d_offset.norm();
    if (r_d >= r_d_max_) {
      // LOG(ERROR) << "r_d is outside of look-up table.\n" << r_d << std::endl;
    }

    const float n = std::min(r_d, r_d_max_) / r_d_max_ * (static_cast<float>(kNd) - 1.F);
    const int n_low = static_cast<int>(n);
    const float w = n - static_cast<float>(n_low);

    // Interpolate s(r_d) using look-up table.
    const float s = (1.F - w) * s_aux_r_d_[n_low] + w * s_aux_r_d_[n_low + 1];
    const float inv_s = 1.F / s;
    p_u = p_d_offset * inv_s;

    // Compensate the tangential components
    if (distortion_model != DistortionModel::kRadialTangential) {
      break;
    }
    const float two_x_y = 2 * p_u[0] * p_u[1];
    const float sqr_r_u2 = p_u.squaredNorm();
    const Eigen::Vector2f p_u2 = p_u.cwiseProduct(p_u);
    p_d_offset[0] = p_d[0] - (tangential_coeffs[0] * two_x_y + tangential_coeffs[1] * (sqr_r_u2 + 2 * p_u2[0]));
    p_d_offset[1] = p_d[1] - (tangential_coeffs[1] * two_x_y + tangential_coeffs[0] * (sqr_r_u2 + 2 * p_u2[1]));
  }
  return p_u;
}

Eigen::Matrix2Xf Camera::UndistortNormalizedPointsVectorized(const Eigen::Matrix2Xf& p_ds) const {
  Eigen::Matrix2Xf p_us;
  Eigen::Matrix2Xf p_ds_offset = p_ds;
  for (int iter = 0; iter < 3; ++iter) {
    const Eigen::Array<float, 1, -1> r_ds = p_ds_offset.colwise().norm();
    if (r_ds.maxCoeff() >= r_d_max_) {
      // LOG(ERROR) << "Coefficient of r_ds is outside of look-up table.\n";
    }

    const int64_t N = r_ds.cols();
    const Eigen::Array<float, 1, -1> ns = r_ds.min(r_d_max_) / r_d_max_ * (static_cast<float>(kNd) - 1.F);
    const Eigen::Array<int, 1, -1> n_lows = ns.cast<int>();
    const Eigen::Array<float, 1, -1> ws = ns - n_lows.cast<float>();

    // Interpolate s(r_ds) using look-up table.
    Eigen::Array<float, 1, -1> ss(N);
    for (uint i = 0; i < N; ++i) {
      ss(i) = (1.F - ws(i)) * s_aux_r_d_[n_lows(i)] + ws(i) * s_aux_r_d_[n_lows(i) + 1];
    }
    const Eigen::Array<float, 1, -1> inv_ss = ss.inverse();
    p_us = p_ds_offset.array().rowwise() * inv_ss;

    // Compensate the tangential components
    if (distortion_model != DistortionModel::kRadialTangential) {
      break;
    }
    const Eigen::Array<float, 1, -1> two_x_y = 2 * p_us.row(0).cwiseProduct(p_us.row(1));
    const Eigen::Array<float, 1, -1> sqr_r_u2 = p_us.colwise().squaredNorm();
    const Eigen::Matrix2Xf p_us2 = p_us.cwiseProduct(p_us);
    p_ds_offset.row(0) = p_ds.row(0).array() - (tangential_coeffs[0] * two_x_y +
                                                tangential_coeffs[1] * (sqr_r_u2.rowwise() + 2 * p_us2.row(0).array()));
    p_ds_offset.row(1) = p_ds.row(1).array() - (tangential_coeffs[1] * two_x_y +
                                                tangential_coeffs[0] * (sqr_r_u2.rowwise() + 2 * p_us2.row(1).array()));
  }
  return p_us;
}

std::tuple<float, float> Camera::ComputeCacheFromUndistortedRadius(const float& r_u) const {
  // This function is solving for `s(r_d(r_u)) = r_d(r_u) / r_u`
  switch (distortion_model) {
    case DistortionModel::kRadialTangential: {
      Eigen::VectorXf sqr_r_us(n_radial_coeffs);
      Eigen::VectorXf delta_coeffs(n_radial_coeffs);
      sqr_r_us[0] = r_u * r_u;
      delta_coeffs[0] = 3.F * sqr_r_us[0];
      for (uint i = 1; i < n_radial_coeffs; ++i) {
        sqr_r_us[i] = sqr_r_us[i - 1] * sqr_r_us[0];
        delta_coeffs[i] = (3.F + 2.F * static_cast<float>(i)) * sqr_r_us[i];
      }
      const Eigen::VectorXf dcoeffs_sqr_r_us = sqr_r_us.cwiseProduct(radial_coeffs);
      const float r_d = r_u * (1.F + dcoeffs_sqr_r_us.sum());
      const float dr_d_dr_u = 1.F + delta_coeffs.cwiseProduct(radial_coeffs).sum();

      const float s = r_d / r_u;
      const float ds_dr_u = 1.F / sqr_r_us[0] * (dr_d_dr_u * r_u - r_d);
      return std::make_tuple(s, ds_dr_u);
    }
    case DistortionModel::kEquidistant: {
      const float theta = std::atan(r_u);

      Eigen::VectorXf sqr_thetas(n_radial_coeffs);
      Eigen::VectorXf delta_coeffs(n_radial_coeffs);
      sqr_thetas[0] = theta * theta;
      delta_coeffs[0] = 3.F * sqr_thetas[0];
      for (uint i = 1; i < n_radial_coeffs; ++i) {
        sqr_thetas[i] = sqr_thetas[i - 1] * sqr_thetas[0];
        delta_coeffs[i] = (3.F + 2.F * static_cast<float>(i)) * sqr_thetas[i];
      }
      const Eigen::VectorXf dcoeffs_sqr_thetas = sqr_thetas.cwiseProduct(radial_coeffs);
      const float r_d = theta * (1.F + dcoeffs_sqr_thetas.sum());
      const float dr_d_dtheta = 1.F + delta_coeffs.cwiseProduct(radial_coeffs).sum();

      const float s = r_d / r_u;
      const float sqr_r_u2 = r_u * r_u;
      const float sqr_r_u3 = sqr_r_u2 * r_u;
      const float ds_dr_u = 1.F / sqr_r_u3 * (dr_d_dtheta * r_u / (1.F + sqr_r_u2) - r_d);
      return std::make_tuple(s, ds_dr_u);
    }
    default: {
      throw std::runtime_error("Distortion model is not supported!");
    }
  }
}

Eigen::Vector2f Camera::DistortNormalizedPoint(const Eigen::Vector2f& p_u,
                                               Eigen::Matrix<float, 2, 3>* dprojection_dx) const {
  const float r_u = p_u.norm();
  if (std::isnan(r_u)) {
    return p_u;
  }
  if (r_u >= r_u_max_) {
    // LOG(ERROR) << "r_u is outside of look-up table.\n" << r_u << std::endl;
  }
  if (r_u <= kEps) {
    return p_u;
  }

  const float n = (std::min(r_u, r_u_max_) / r_u_max_) * (static_cast<float>(kNd) - 1.F);
  const int n_low = static_cast<int>(n);
  const float w = n - static_cast<float>(n_low);
  // Interpolate s(r_u) using look-up table.
  const float s = (1.F - w) * s_aux_r_u_[n_low] + w * s_aux_r_u_[n_low + 1];

  // Compute gradient of distortion model (radial components)
  if (dprojection_dx != nullptr) {
    const float ds_dr_u = (1.F - w) * ds_dr_u_aux_r_u_[n_low] + w * ds_dr_u_aux_r_u_[n_low + 1];
    const Eigen::Matrix<float, 1, 2> ds_dp_u =
      ds_dr_u * p_u.colwise().homogeneous().transpose() * W_d_.block<3, 2>(0, 0);

    const Eigen::Matrix2f dp_d_dp_u =
      s * Eigen::Matrix2f::Identity() + (p_u - camera_matrix.block<2, 1>(0, 2)) * ds_dp_u;

    *dprojection_dx = dp_d_dp_u * *dprojection_dx;
  }

  Eigen::Vector2f p_d = s * p_u;
  if (distortion_model == DistortionModel::kRadialTangential) {
    // Compensate the tangential components
    const float two_x_y = 2 * p_u[0] * p_u[1];
    const float sqr_r_u2 = r_u * r_u;
    const Eigen::Vector2f p_u2 = p_u.cwiseProduct(p_u);
    p_d[0] += tangential_coeffs[0] * two_x_y + tangential_coeffs[1] * (sqr_r_u2 + 2 * p_u2[0]);
    p_d[1] += tangential_coeffs[1] * two_x_y + tangential_coeffs[0] * (sqr_r_u2 + 2 * p_u2[1]);

    // Compute gradient of distortion model (tangential components)
    if (dprojection_dx != nullptr) {
      // TODO(cv3d): Implement me!
    }
  }
  return p_d;
}

Eigen::Matrix2Xf Camera::DistortNormalizedPointsVectorized(const Eigen::Matrix2Xf& p_us) const {
  const Eigen::Array<float, 1, -1> r_us = p_us.colwise().norm();
  if (r_us.maxCoeff() >= r_u_max_) {
    // LOG(ERROR) << "Coefficient of r_us is outside of look-up table.\n";
  }

  const int64_t N = r_us.cols();
  const Eigen::Array<float, 1, -1> ns = r_us.min(r_u_max_) / r_u_max_ * (static_cast<float>(kNd) - 1.F);
  const Eigen::Array<int, 1, -1> n_lows = ns.cast<int>();
  const Eigen::Array<float, 1, -1> ws = ns - n_lows.cast<float>();

  // Interpolate s(r_us) using look-up table.
  Eigen::Array<float, 1, -1> ss(N);
  for (uint i = 0; i < N; ++i) {
    ss(i) = (1.F - ws(i)) * s_aux_r_u_[n_lows(i)] + ws(i) * s_aux_r_u_[n_lows(i) + 1];
  }
  Eigen::Matrix2Xf p_ds = p_us.array().rowwise() * ss;
  if (distortion_model == DistortionModel::kRadialTangential) {
    // Compensate the tangential components
    const Eigen::Array<float, 1, -1> two_x_y = 2 * p_us.row(0).cwiseProduct(p_us.row(1));
    const Eigen::Array<float, 1, -1> sqr_r_u2 = r_us * r_us;
    const Eigen::Matrix2Xf p_us2 = p_us.cwiseProduct(p_us);
    p_ds.row(0).array() +=
      tangential_coeffs[0] * two_x_y + tangential_coeffs[1] * (sqr_r_u2.rowwise() + 2 * p_us2.row(0).array());
    p_ds.row(1).array() +=
      tangential_coeffs[1] * two_x_y + tangential_coeffs[0] * (sqr_r_u2.rowwise() + 2 * p_us2.row(1).array());
  }
  return p_ds;
}

Eigen::Vector2f Camera::UndistortPoint(const Eigen::Vector2f& p_d) const {
  if (!distortion_enable) {
    return p_d;
  }

  const Eigen::Vector2f p_d_norm = (camera_matrix_inv * p_d.colwise().homogeneous()).head(2);
  const Eigen::Vector2f p_u_norm = UndistortNormalizedPoint(p_d_norm);
  return (camera_matrix * p_u_norm.colwise().homogeneous()).colwise().hnormalized();
}

std::vector<Eigen::Vector2f> Camera::UndistortPoints(const std::vector<Eigen::Vector2f>& p_ds) const {
  if (!distortion_enable || p_ds.empty()) {
    return p_ds;
  }

  std::vector<Eigen::Vector2f> p_us;
  p_us.reserve(p_ds.size());
  std::transform(p_ds.begin(), p_ds.end(), std::back_inserter(p_us),
                 std::bind(&Camera::UndistortPoint, this, std::placeholders::_1));
  return p_us;
}

Eigen::Matrix2Xf Camera::UndistortPointsVectorized(const Eigen::Matrix2Xf& p_ds) const {
  const int64_t N = p_ds.cols();
  if (!distortion_enable || N == 0) {
    return p_ds;
  }

  const Eigen::Matrix2Xf p_ds_norm = (camera_matrix_inv * p_ds.colwise().homogeneous()).topRows(2);
  const Eigen::Matrix2Xf p_us_norm = UndistortNormalizedPointsVectorized(p_ds_norm);
  return (camera_matrix * p_us_norm.colwise().homogeneous()).colwise().hnormalized();
}

void Camera::UndistortImage(const cv::Mat& img_distorted, cv::Mat& img_undistorted) const {
  if (!distortion_enable) {
    img_distorted.copyTo(img_undistorted);
    return;
  }

  cv::remap(img_distorted, img_undistorted, undistortion_map_x_, undistortion_map_y_, cv::INTER_LINEAR);
}

Eigen::Vector2f Camera::DistortPoint(const Eigen::Vector2f& p_u, Eigen::Matrix<float, 2, 3>* dprojection_dx) const {
  if (!distortion_enable) {
    return p_u;
  }

  const Eigen::Vector2f p_u_norm = (camera_matrix_inv * p_u.colwise().homogeneous()).head(2);
  const Eigen::Vector2f p_d_norm = DistortNormalizedPoint(p_u_norm, dprojection_dx);
  return (camera_matrix * p_d_norm.colwise().homogeneous()).colwise().hnormalized();
}

std::vector<Eigen::Vector2f> Camera::DistortPoints(const std::vector<Eigen::Vector2f>& p_us) const {
  if (!distortion_enable || p_us.empty()) {
    return p_us;
  }

  std::vector<Eigen::Vector2f> p_ds;
  p_ds.reserve(p_us.size());
  std::transform(p_us.begin(), p_us.end(), std::back_inserter(p_ds),
                 std::bind(&Camera::DistortPoint, this, std::placeholders::_1, nullptr));
  return p_ds;
}

Eigen::Matrix2Xf Camera::DistortPointsVectorized(const Eigen::Matrix2Xf& p_us) const {
  const int64_t N = p_us.cols();
  if (!distortion_enable || N == 0) {
    return p_us;
  }

  const Eigen::Matrix2Xf p_us_norm = (camera_matrix_inv * p_us.colwise().homogeneous()).topRows(2);
  const Eigen::Matrix2Xf p_ds_norm = DistortNormalizedPointsVectorized(p_us_norm);
  return (camera_matrix * p_ds_norm.colwise().homogeneous()).colwise().hnormalized();
}

bool Camera::ProjectPoint(const Eigen::Vector3f& x, Eigen::Vector2f* p_d, Eigen::Matrix<float, 2, 3>* dprojection_dx,
                          bool apply_distortion) const {
  if (p_d == nullptr) {
    LOG(ERROR) << "Invalid argument p_d (nullptr).\n";
    return false;
  }

  // Back-project point.
  const Eigen::Vector3f p_uh_ns = projection_matrix.block<3, 3>(0, 0) * x + projection_matrix.block<3, 1>(0, 3);
  if (p_uh_ns(2) < 0.F) {
    // LOG(ERROR) << "Point is behind camera (z: " << p_uh_ns(2) << ").\n";
    return false;
  }
  const Eigen::Vector3f p_uh = p_uh_ns / p_uh_ns(2);
  const Eigen::Vector2f p_u = p_uh.segment<2>(0);

  // Compute gradient of pinhole model
  if (dprojection_dx != nullptr) {
    Eigen::Matrix<float, 2, 3> dp_u_dx;
    const float inv_p_uh_2_2 = 1.F / (p_uh_ns(2) * p_uh_ns(2));
    dp_u_dx.row(0) = inv_p_uh_2_2 * (p_uh_ns(2) * projection_matrix.block<1, 3>(0, 0) -
                                     p_uh_ns(0) * projection_matrix.block<1, 3>(2, 0));
    dp_u_dx.row(1) = inv_p_uh_2_2 * (p_uh_ns(2) * projection_matrix.block<1, 3>(1, 0) -
                                     p_uh_ns(1) * projection_matrix.block<1, 3>(2, 0));

    *dprojection_dx = dp_u_dx;
  }
  if (apply_distortion) {
    *p_d = DistortPoint(p_u, dprojection_dx);
  } else {
    *p_d = p_u;
  }
  return true;
}

void Camera::DistortImage(const cv::Mat& img_undistorted, cv::Mat& img_distorted) const {
  if (!distortion_enable) {
    img_undistorted.copyTo(img_distorted);
    return;
  }

  cv::remap(img_undistorted, img_distorted, distortion_map_x_, distortion_map_y_, cv::INTER_LINEAR);
}

void Camera::UpdateInternalParameters(const Eigen::Matrix4f& T_world_origin) {
  // (Inverse) camera matrix
  camera_matrix_inv = camera_matrix.inverse();
  camera_matrix_cv = cv::Mat(3, 3, CV_32FC1);
  for (int m = 0; m < 3; ++m) {
    for (int n = 0; n < 3; ++n) {
      camera_matrix_cv.at<float>(m, n) = camera_matrix(m, n);
    }
  }

  // Transformation
  T_world_camera = T_camera_world.inverse();
  T_camera_origin = T_camera_world * T_world_origin;
  T_origin_camera = T_camera_origin.inverse();

  // (Un)Projection matrix
  projection_matrix = camera_matrix * T_camera_origin.block<3, 4>(0, 0);

  projection_matrix_inv.block<3, 4>(0, 0) = projection_matrix;
  projection_matrix_inv.row(3) << 0, 0, 0, 1;
  projection_matrix_inv = projection_matrix_inv.inverse();

  // (Un)distortion
  InitializeUndistortionLookupMap();
  InitializeDistortionLookupMap();
  InitializeOpenCVDistortionMaps();
}

void Camera::InitializeUndistortionLookupMap() {
  if (!distortion_enable) {
    return;
  }

  // find maximum distortion
  // (distortion function is monotonically increasing or decreasing, so checking corners is sufficient)
  Eigen::Vector3f p_d;
  p_d(2) = 1.F;
  Eigen::VectorXf r_ds(4);

  p_d(0) = 0.F;
  p_d(1) = 0.F;
  r_ds[0] = (camera_matrix_inv * p_d).head(2).norm();

  p_d(0) = static_cast<float>(resolution[0]);
  p_d(1) = 0.F;
  r_ds[1] = (camera_matrix_inv * p_d).head(2).norm();

  p_d(0) = static_cast<float>(resolution[0]);
  p_d(1) = static_cast<float>(resolution[1]);
  r_ds[2] = (camera_matrix_inv * p_d).head(2).norm();

  p_d(0) = 0.F;
  p_d(1) = static_cast<float>(resolution[1]);
  r_ds[3] = (camera_matrix_inv * p_d).head(2).norm();

  r_d_max_ = 0.F;
  for (float r_d : r_ds) {
    if (r_d > r_d_max_) {
      r_d_max_ = r_d;
    }
  }
  r_d_max_ *= 1.01F;  // add some margin

  // compute undistortion look-up table
  s_aux_r_d_[0] = 1.F;
  for (int n = 1; n < kNd; ++n) {
    const float r_d = r_d_max_ * (static_cast<float>(n) / (static_cast<float>(kNd) - 1.F));
    s_aux_r_d_[n] = ComputeCacheFromDistortedRadius(r_d);
  }
}

void Camera::InitializeDistortionLookupMap() {
  if (!distortion_enable) {
    return;
  }

  // Some cache to be used later with gradient of distortion model
  Eigen::Matrix3f D = Eigen::Matrix3f::Identity();
  D(2, 2) = 0.F;
  W_d_ = camera_matrix_inv.transpose() * D * camera_matrix_inv;

  // Find maximum radius.
  Eigen::Vector2f p_d;
  Eigen::Vector3f p_uh;
  p_uh(2) = 1.F;
  Eigen::VectorXf rs(4);

  p_d(0) = 0.F;
  p_d(1) = 0.F;
  p_uh.segment<2>(0) = UndistortPoint(p_d);
  rs[0] = (camera_matrix_inv * p_uh).head(2).norm();

  p_d(0) = static_cast<float>(resolution[0]);
  p_d(1) = 0.F;
  p_uh.segment<2>(0) = UndistortPoint(p_d);
  rs[1] = (camera_matrix_inv * p_uh).head(2).norm();

  p_d(0) = static_cast<float>(resolution[0]);
  p_d(1) = static_cast<float>(resolution[1]);
  p_uh.segment<2>(0) = UndistortPoint(p_d);
  rs[2] = (camera_matrix_inv * p_uh).head(2).norm();

  p_d(0) = 0.F;
  p_d(1) = static_cast<float>(resolution[1]);
  p_uh.segment<2>(0) = UndistortPoint(p_d);
  rs[3] = (camera_matrix_inv * p_uh).head(2).norm();

  r_u_max_ = 0.F;
  for (float r_u : rs) {
    if (r_u > r_u_max_) {
      r_u_max_ = r_u;
    }
  }
  r_u_max_ *= 1.01F;  // Add some margin.

  // Compute distortion look-up table.
  s_aux_r_u_[0] = 1.F;
  for (int n = 1; n < kNd; ++n) {
    const float r_u = r_u_max_ * (static_cast<float>(n) / (static_cast<float>(kNd) - 1.F));
    std::tie(s_aux_r_u_[n], ds_dr_u_aux_r_u_[n]) = ComputeCacheFromUndistortedRadius(r_u);
  }
}

void Camera::InitializeOpenCVDistortionMaps() {
  // Create (un)distortion maps.
  const cv::Size img_size(resolution[0], resolution[1]);
  distortion_map_x_.create(img_size, CV_32F);
  distortion_map_y_.create(img_size, CV_32F);
  undistortion_map_x_.create(img_size, CV_32F);
  undistortion_map_y_.create(img_size, CV_32F);

  // Initialize (un)distortion maps.
  for (int x = 0; x < img_size.width; ++x) {
    for (int y = 0; y < img_size.height; ++y) {
      const Eigen::Vector2f& p = Eigen::Vector2f(static_cast<float>(x), static_cast<float>(y));
      Eigen::Vector2f p_u = UndistortPoint(p);
      Eigen::Vector2f p_s = DistortPoint(p);

      distortion_map_x_.at<float>(y, x) = p_u.x();
      distortion_map_y_.at<float>(y, x) = p_u.y();
      undistortion_map_x_.at<float>(y, x) = p_s.x();
      undistortion_map_y_.at<float>(y, x) = p_s.y();
    }
  }
}

CameraCalibrationParameters::CameraCalibrationParameters() : BaseParameters("camera_calibration_parameters") {}

void CameraCalibrationParameters::UpdateInternalParameters() {
  T_origin_world = T_world_origin.inverse();

  // List of all cameras (names)
  camera_names.clear();
  for (const auto& [camera_name, camera] : cameras) {
    camera_names.emplace_back(camera_name);
  }

  // Compute fundamental matrices
  for (const auto& camera_name_i : camera_names) {
    const auto& camera_i = cameras[camera_name_i];
    const Eigen::Matrix3f& K_i = camera_i.camera_matrix;
    const Eigen::Matrix4f& T_camera_i_world = camera_i.T_camera_world;

    for (const auto& camera_name_j : camera_names) {
      auto& curr_multiview_data = multiview_data[camera_name_i][camera_name_j];
      if (camera_name_i == camera_name_j) {
        curr_multiview_data.fundamental_matrix = Eigen::Matrix3f::Zero();
        curr_multiview_data.epipole_i = Eigen::Vector2f::Zero();
        curr_multiview_data.epipole_j = Eigen::Vector2f::Zero();
      } else {
        const auto& camera_j = cameras[camera_name_j];
        const Eigen::Matrix3f& K_j = camera_j.camera_matrix;
        const Eigen::Matrix4f& T_world_camera_j = camera_j.T_world_camera;

        Eigen::Matrix4f T_camera_i_camera_j = T_camera_i_world * T_world_camera_j;
        Eigen::Matrix3f T_skew;
        T_skew << 0.F, -T_camera_i_camera_j(2, 3), T_camera_i_camera_j(1, 3), T_camera_i_camera_j(2, 3), 0.F,
          -T_camera_i_camera_j(0, 3), -T_camera_i_camera_j(1, 3), T_camera_i_camera_j(0, 3), 0.F;

        curr_multiview_data.fundamental_matrix =
          K_i.inverse().transpose() * T_skew * T_camera_i_camera_j.block<3, 3>(0, 0) * K_j.inverse();

        const Eigen::JacobiSVD<Eigen::Matrix3f> svd_i(curr_multiview_data.fundamental_matrix.transpose(),
                                                      Eigen::ComputeFullV);
        curr_multiview_data.epipole_i = svd_i.matrixV().col(2).colwise().hnormalized();

        const Eigen::JacobiSVD<Eigen::Matrix3f> svd_j(curr_multiview_data.fundamental_matrix, Eigen::ComputeFullV);
        curr_multiview_data.epipole_j = svd_j.matrixV().col(2).colwise().hnormalized();
      }
    }
  }
}

bool CameraCalibrationParameters::UpdateParametersFromYaml() {
  static const std::string KTermResetDefaults = "\033[0m";

  try {
    cameras.clear();
    if (yaml_node_.Type() == YAML::NodeType::Map) {
      // Process (O)rigin coordinate frame first.
      auto T_world_origin_yaml = yaml_node_["T_WO"].as<std::vector<std::vector<float>>>();
      if (T_world_origin_yaml.size() != 4) {
        LOG(ERROR) << "Invalid size of T_WO in " << yaml_path_ << "\n";
        return false;
      }
      for (uint i = 0; i < T_world_origin_yaml.size(); ++i) {
        if (T_world_origin_yaml[i].size() != 4) {
          LOG(ERROR) << "Invalid size of T_WO in " << yaml_path_ << "\n";
          return false;
        }

        std::memcpy(T_world_origin.col(i).data(), T_world_origin_yaml[i].data(),
                    T_world_origin_yaml[i].size() * sizeof(float));
      }
      T_world_origin.transposeInPlace();

      // Process (C)amera coordinate frames second due to dependencies on
      // (O)rigin.
      for (const auto& node : yaml_node_) {
        auto camera_name = node.first.as<std::string>();

        if (node.second.Type() != YAML::NodeType::Map) {
          LOG(WARNING) << "Skipping non-map node: " << camera_name << KTermResetDefaults << "\n";
          continue;
        }

        cameras[camera_name] = Camera();
        auto& camera = cameras[camera_name];

        // Resolution
        camera.resolution = node.second["resolution"].as<std::vector<uint16_t>>();
        if (camera.resolution.size() != 2) {
          LOG(ERROR) << "Invalid size of resolution in " << yaml_path_ << "\n";
          return false;
        }

        // Camera model
        camera.camera_model = StringToCameraModel(node.second["camera_model"].as<std::string>());
        if (camera.camera_model == CameraModel::kInvalid) {
          LOG(ERROR) << "Unsupported camera model in " << yaml_path_ << "\n";
          return false;
        }

        // Camera matrix
        auto camera_matrix = node.second["camera_matrix"].as<std::vector<std::vector<float>>>();
        if (camera_matrix.size() != 3) {
          LOG(ERROR) << "Invalid size of camera_matrix in " << yaml_path_ << "\n";
          return false;
        }
        for (uint i = 0; i < camera_matrix.size(); ++i) {
          if (camera_matrix[i].size() != 3) {
            LOG(ERROR) << "Invalid size of camera_matrix in " << yaml_path_ << "\n";
            return false;
          }

          std::memcpy(camera.camera_matrix.col(i).data(), camera_matrix[i].data(),
                      camera_matrix[i].size() * sizeof(float));
        }
        camera.camera_matrix.transposeInPlace();

        // Is distortion enabled?
        if (node.second["distortion_enable"].IsDefined()) {
          camera.distortion_enable = node.second["distortion_enable"].as<bool>();
        } else {
          camera.distortion_enable = true;  // enabled by default
        }

        // Distortion model
        camera.distortion_model = StringToDistortionModel(node.second["distortion_model"].as<std::string>());
        if (camera.distortion_model == DistortionModel::kInvalid) {
          LOG(ERROR) << "Unsupported distortion model in " << yaml_path_ << "\n";
          return false;
        }
        camera.n_radial_coeffs = (camera.distortion_model == DistortionModel::kEquidistant) ? 4 : 2;
        camera.n_tangential_coeffs = (camera.distortion_model == DistortionModel::kEquidistant) ? 0 : 2;

        // Distortion coefficients
        camera.distortion_coeffs = node.second["distortion_coeffs"].as<std::vector<float>>();
        if (camera.distortion_coeffs.size() != camera.n_radial_coeffs + camera.n_tangential_coeffs) {
          LOG(ERROR) << "Invalid size of distortion_coeffs in " << yaml_path_ << "\n";
          return false;
        }
        camera.radial_coeffs =
          Eigen::Map<const Eigen::VectorXf>(camera.distortion_coeffs.data(), camera.n_radial_coeffs);
        camera.tangential_coeffs = Eigen::Map<const Eigen::VectorXf>(
          camera.distortion_coeffs.data() + camera.n_radial_coeffs, camera.n_tangential_coeffs);

        // Transformation from (W)orld to (C)amera frame
        auto T_camera_world_yaml = node.second["T_CW"].as<std::vector<std::vector<float>>>();
        if (T_camera_world_yaml.size() != 4) {
          LOG(ERROR) << "Invalid size of T_CW in " << yaml_path_ << "\n";
          return false;
        }
        for (uint i = 0; i < T_camera_world_yaml.size(); ++i) {
          if (T_camera_world_yaml[i].size() != 4) {
            LOG(ERROR) << "Invalid size of T_CW in " << yaml_path_ << "\n";
            return false;
          }

          std::memcpy(camera.T_camera_world.col(i).data(), T_camera_world_yaml[i].data(),
                      T_camera_world_yaml[i].size() * sizeof(float));
        }
        camera.T_camera_world.transposeInPlace();

        // Rotation
        camera.rotation = node.second["rotation"].as<float>();

        // Initialize interal parameters
        camera.UpdateInternalParameters(T_world_origin);
      }

      // Update internal parameters
      UpdateInternalParameters();

    } else {
      LOG(ERROR) << "Invalid yaml file (" << yaml_path_ << "), expected a map of cameras.\n";
      return false;
    }
  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool CameraCalibrationParameters::UpdateYamlFromParameters() {
  try {
    for (const auto& [camera_name, camera] : cameras) {
      YAML::Node camera_node;

      camera_node["resolution"] = camera.resolution;
      camera_node["resolution"].SetStyle(YAML::EmitterStyle::Flow);

      camera_node["camera_model"] = CameraModelToString(camera.camera_model);
      std::vector<std::vector<float>> camera_matrix;
      for (uint i = 0; i < 3; ++i) {
        std::vector<float> row;
        for (uint j = 0; j < 3; ++j) {
          row.push_back(camera.camera_matrix(i, j));
        }
        camera_matrix.push_back(row);
      }
      camera_node["camera_matrix"] = camera_matrix;
      for (uint i = 0; i < camera_node["camera_matrix"].size(); ++i) {
        camera_node["camera_matrix"][i].SetStyle(YAML::EmitterStyle::Flow);
      }

      camera_node["distortion_enable"] = camera.distortion_enable;
      camera_node["distortion_model"] = DistortionModelToString(camera.distortion_model);
      camera_node["distortion_coeffs"] = camera.distortion_coeffs;
      camera_node["distortion_coeffs"].SetStyle(YAML::EmitterStyle::Flow);

      std::vector<std::vector<float>> T_camera_world_yaml;
      for (uint i = 0; i < 4; ++i) {
        std::vector<float> row;
        for (uint j = 0; j < 4; ++j) {
          row.push_back(camera.T_camera_world(i, j));
        }
        T_camera_world_yaml.push_back(row);
      }
      camera_node["T_CW"] = T_camera_world_yaml;
      for (uint i = 0; i < camera_node["T_CW"].size(); ++i) {
        camera_node["T_CW"][i].SetStyle(YAML::EmitterStyle::Flow);
      }

      camera_node["rotation"] = camera.rotation;

      yaml_node_[camera_name] = camera_node;
    }

    std::vector<std::vector<float>> T_world_origin_yaml;
    for (uint i = 0; i < 4; ++i) {
      std::vector<float> row;
      for (uint j = 0; j < 4; ++j) {
        row.push_back(T_world_origin(i, j));
      }
      T_world_origin_yaml.push_back(row);
    }
    yaml_node_["T_WO"] = T_world_origin_yaml;
    for (uint i = 0; i < yaml_node_["T_WO"].size(); ++i) {
      yaml_node_["T_WO"][i].SetStyle(YAML::EmitterStyle::Flow);
    }

  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace calibration
