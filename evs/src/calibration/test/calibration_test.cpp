// SPDX-License-Identifier: MIT
#include <algorithm>
#include <chrono>
#include <random>

#include "calibration/camera_calibration_parameters.hpp"
#include "gtest/gtest.h"
#include "test/root_dir.hpp"

namespace calibration {

calibration::CameraCalibrationParameters::SharedPtr camera_calibration_params_ptr =
  std::make_shared<calibration::CameraCalibrationParameters>();

class CalibrationTests : public ::testing::Test {
 public:
  explicit CalibrationTests() : random_number_generator_(random_device_()) {
    // Params setup
    camera_calibration_params_ptr->Initialize(kRootDir + "/test/parameters/camera_calibration.yaml");
    // Set cameras used for tests.
    camera_names_ = camera_calibration_params_ptr->camera_names;

    // Set up random number generator.
    random_number_generator_.seed(std::time(nullptr));
  }

 protected:
  // Auxiliary variables
  std::vector<std::string> camera_names_;

  std::random_device random_device_;
  std::mt19937 random_number_generator_;
  std::uniform_real_distribution<float> uniform_;
  float UniformSample(const float& low, const float& high) {
    return low + uniform_(random_number_generator_) * (high - low);
  }
  std::uniform_int_distribution<int> uniform_int_0_100_;
  int UniformIntSample0100() { return uniform_int_0_100_(random_number_generator_); }
  std::normal_distribution<float> normal_;
  float NormalSample(const float& mean, const float& stddev) {
    return mean + normal_(random_number_generator_) * stddev;
  }
};

TEST_F(CalibrationTests, TestUndistortionDistortion) {
  // Uniformly at random select (distorted) points in camera image, undistort then distort points sequentially
  // and test if final points coincide with inital ones.
  const int num_points = 5000;
  const float max_undistortion_error = 1e-3F;  // in pixels

  for (const auto& [camera_name, camera] : camera_calibration_params_ptr->cameras) {
    ///////////////////////
    // Generate test data.
    ///////////////////////
    std::vector<Eigen::Vector2f> distorted_points_in;
    distorted_points_in.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
      distorted_points_in.emplace_back(UniformSample(0.F, static_cast<float>(camera.resolution[0])),
                                       UniformSample(0.F, static_cast<float>(camera.resolution[1])));
    }

    ///////////////////////
    // Tests undistortion then distortion.
    ///////////////////////
    std::vector<Eigen::Vector2f> undistorted_points = camera.UndistortPoints(distorted_points_in);
    std::vector<Eigen::Vector2f> distorted_points_out = camera.DistortPoints(undistorted_points);

    for (int i = 0; i < num_points; ++i) {
      EXPECT_NEAR(distorted_points_in[i][0], distorted_points_out[i][0], max_undistortion_error);
      EXPECT_NEAR(distorted_points_in[i][1], distorted_points_out[i][1], max_undistortion_error);
    }
  }

  SUCCEED();
}

TEST_F(CalibrationTests, TestDistortionUndistortion) {
  // Uniformly at random select (undistorted) points in camera image, distort then undistort points sequentially
  // and test if final points coincide with inital ones.
  const int num_points = 5000;
  const float max_distortion_error = 1e-3F;  // in pixels

  for (const auto& [camera_name, camera] : camera_calibration_params_ptr->cameras) {
    ///////////////////////
    // Generate test data.
    ///////////////////////
    std::vector<Eigen::Vector2f> undistorted_points_in;
    undistorted_points_in.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
      undistorted_points_in.emplace_back(UniformSample(0.F, static_cast<float>(camera.resolution[0])),
                                         UniformSample(0.F, static_cast<float>(camera.resolution[1])));
    }

    ///////////////////////
    // Tests distortion then undistortion.
    ///////////////////////
    std::vector<Eigen::Vector2f> distorted_points = camera.DistortPoints(undistorted_points_in);
    std::vector<Eigen::Vector2f> undistorted_points_out = camera.UndistortPoints(distorted_points);

    for (int i = 0; i < num_points; ++i) {
      EXPECT_NEAR(undistorted_points_in[i][0], undistorted_points_out[i][0], max_distortion_error);
      EXPECT_NEAR(undistorted_points_in[i][1], undistorted_points_out[i][1], max_distortion_error);
    }
  }

  SUCCEED();
}

TEST_F(CalibrationTests, TestOpencvUndistortion) {
  // Uniformly at random select (distorted) points in camera image, undistort points sequentially
  // and test if undistorted points coincide with OpenCV.
  const int num_points = 5000;
  const float max_undistortion_error = 2e-2F;  // in pixels

  for (const auto& [camera_name, camera] : camera_calibration_params_ptr->cameras) {
    ///////////////////////
    // Generate test data.
    ///////////////////////
    std::vector<Eigen::Vector2f> distorted_points_vec;
    distorted_points_vec.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
      distorted_points_vec.emplace_back(UniformSample(0.F, static_cast<float>(camera.resolution[0])),
                                        UniformSample(0.F, static_cast<float>(camera.resolution[1])));
    }

    std::vector<Eigen::Vector2f> undistorted_points_cv;
    undistorted_points_cv.reserve(num_points);
    const cv::Mat distorted_points_cv(num_points, 1, CV_32FC2, distorted_points_vec.data());
    const cv::Mat undistorted_points_cv_map(num_points, 1, CV_32FC2, undistorted_points_cv.data());
    auto src = *const_cast<Eigen::Matrix3f*>(&camera.camera_matrix);
    const cv::Mat K_aux(static_cast<int>(src.cols()), static_cast<int>(src.rows()), CV_32F,
                        static_cast<void*>(src.data()), src.outerStride() * sizeof(float));
    const cv::Mat K = K_aux.t();
    const cv::Mat D(1, camera.n_radial_coeffs + camera.n_tangential_coeffs, CV_32F,
                    const_cast<float*>(camera.distortion_coeffs.data()));

    ///////////////////////
    // Tests undistortion.
    ///////////////////////
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector2f> undistorted_points = camera.UndistortPoints(distorted_points_vec);
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);

    start_time = std::chrono::high_resolution_clock::now();
    if (camera.distortion_model == DistortionModel::kEquidistant) {
      cv::fisheye::undistortPoints(distorted_points_cv, undistorted_points_cv_map, K, D, cv::noArray(), K);
    } else {
      cv::undistortPoints(distorted_points_cv, undistorted_points_cv_map, K, D, cv::noArray(), K);
    }
    stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time_opencv = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    std::printf("Camera: %s, Duration normal: %ldus, \tDuration opencv: %ldus\n", camera_name.c_str(),
                elapsed_time.count(), elapsed_time_opencv.count());

    for (int i = 0; i < num_points; ++i) {
      EXPECT_NEAR(undistorted_points[i][0], undistorted_points_cv[i][0], max_undistortion_error);
      EXPECT_NEAR(undistorted_points[i][1], undistorted_points_cv[i][1], max_undistortion_error);
    }
  }

  SUCCEED();
}

TEST_F(CalibrationTests, TestVectorizedUndistortion) {
  // Uniformly at random select (distorted) points in camera image, undistort points sequentially
  // and vectorized and test if undistorted points coincide.
  const int num_points = 5000;
  const float max_undistortion_error = 1e-9F;  // in pixels

  for (const auto& [camera_name, camera] : camera_calibration_params_ptr->cameras) {
    ///////////////////////
    // Generate test data.
    ///////////////////////
    std::vector<Eigen::Vector2f> distorted_points_vec;
    distorted_points_vec.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
      distorted_points_vec.emplace_back(UniformSample(0.F, static_cast<float>(camera.resolution[0])),
                                        UniformSample(0.F, static_cast<float>(camera.resolution[1])));
    }

    ///////////////////////
    // Tests undistortion.
    ///////////////////////
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector2f> undistorted_points_vec = camera.UndistortPoints(distorted_points_vec);
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);

    start_time = std::chrono::high_resolution_clock::now();
    Eigen::Matrix2Xf undistorted_points = camera.UndistortPointsVectorized(
      Eigen::Map<Eigen::Matrix2Xf>(reinterpret_cast<float*>(distorted_points_vec.data()), 2, num_points));
    stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time_vectorized = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    std::printf("Camera: %s, Duration normal: %ldus, \tDuration vectorized: %ldus\n", camera_name.c_str(),
                elapsed_time.count(), elapsed_time_vectorized.count());

    for (int i = 0; i < num_points; ++i) {
      EXPECT_NEAR(undistorted_points_vec[i][0], undistorted_points.col(i)[0], max_undistortion_error);
      EXPECT_NEAR(undistorted_points_vec[i][1], undistorted_points.col(i)[1], max_undistortion_error);
    }
  }

  SUCCEED();
}

TEST_F(CalibrationTests, TestVectorizedDistortion) {
  // Uniformly at random select (distorted) points in camera image, undistort points to obtain undistorted points in
  // valid range, and then distort sequentially and vectorized and test if distorted points coincide.
  const int num_points = 5000;
  const float max_distortion_error = 1e-9F;  // in pixels

  for (const auto& [camera_name, camera] : camera_calibration_params_ptr->cameras) {
    ///////////////////////
    // Generate test data.
    ///////////////////////
    std::vector<Eigen::Vector2f> undistorted_points_vec;
    undistorted_points_vec.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
      undistorted_points_vec.emplace_back(UniformSample(0.F, static_cast<float>(camera.resolution[0])),
                                          UniformSample(0.F, static_cast<float>(camera.resolution[1])));
    }

    ///////////////////////
    // Tests distortion.
    ///////////////////////
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector2f> distorted_points_vec = camera.DistortPoints(undistorted_points_vec);
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);

    start_time = std::chrono::high_resolution_clock::now();
    Eigen::Matrix2Xf distorted_points = camera.DistortPointsVectorized(
      Eigen::Map<Eigen::Matrix2Xf>(reinterpret_cast<float*>(undistorted_points_vec.data()), 2, num_points));
    stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time_vectorized = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    std::printf("Camera: %s, Duration normal: %ldus, \tDuration vectorized: %ldus\n", camera_name.c_str(),
                elapsed_time.count(), elapsed_time_vectorized.count());

    for (int i = 0; i < num_points; ++i) {
      EXPECT_NEAR(distorted_points_vec[i][0], distorted_points.col(i)[0], max_distortion_error);
      EXPECT_NEAR(distorted_points_vec[i][1], distorted_points.col(i)[1], max_distortion_error);
    }
  }

  SUCCEED();
}

TEST_F(CalibrationTests, TestGradient) {
  // Uniformly at random select a 3D-space point around the camera center axis, project
  // and distort it, then test if its gradient can lead to convergence.
  const int num_points = 500;
  const float distance_from_cam = 1.F;  // in meters
  const float max_rand_radius = .1F;    // in meters
  const int max_grad_iter = 15;
  const float max_grad_error = 1e-3F;  // in pixels

  for (const auto& [camera_name, camera] : camera_calibration_params_ptr->cameras) {
    const Eigen::Vector3f x_center =
      camera.T_origin_camera.block<3, 1>(0, 3) + distance_from_cam * camera.T_origin_camera.block<3, 1>(0, 2);
    const Eigen::Vector2f p_dst = camera.camera_matrix.block<2, 1>(0, 2);
    const float gamma = 1.F / (camera.camera_matrix(0, 0) * camera.camera_matrix(1, 1));

    Eigen::Vector3f x;
    Eigen::Vector2f p_d;
    Eigen::Matrix<float, 2, 3> dp_d_dx;
    Eigen::Vector2f error;
    for (int i = 0; i < num_points; ++i) {
      x = x_center + Eigen::Vector3f(UniformSample(-max_rand_radius, max_rand_radius),
                                     UniformSample(-max_rand_radius, max_rand_radius),
                                     UniformSample(-max_rand_radius, max_rand_radius));
      for (int iter_idx = 0; iter_idx < max_grad_iter; ++iter_idx) {
        camera.ProjectPoint(x, &p_d, &dp_d_dx);
        error = p_d - p_dst;
        x -= gamma * dp_d_dx.transpose() * error;
      }
      EXPECT_LT(error.norm(), max_grad_error);
    }
  }

  SUCCEED();
}

TEST_F(CalibrationTests, TestEpipoles) {
  // Confirms that all epipoles are computed correctly
  for (const auto& camera_name_i : camera_names_) {
    for (const auto& camera_name_j : camera_names_) {
      const auto& curr_multiview_data = camera_calibration_params_ptr->multiview_data[camera_name_i][camera_name_j];
      EXPECT_LT(
        (curr_multiview_data.epipole_i.colwise().homogeneous().transpose() * curr_multiview_data.fundamental_matrix)
          .norm(),
        1e-3F);
      EXPECT_LT((curr_multiview_data.fundamental_matrix * curr_multiview_data.epipole_j.colwise().homogeneous()).norm(),
                1e-3F);
    }
  }

  SUCCEED();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}  // namespace calibration
