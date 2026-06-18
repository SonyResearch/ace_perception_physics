// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "triangulation/triangulation.hpp"

#include <algorithm>
#include <chrono>
#include <random>

#include "ace_loggers/ace_loggers.hpp"
#include "gtest/gtest.h"
#include "test/root_dir.hpp"
#include "triangulation/datalogger.hpp"

namespace triangulation {
namespace {
const TriangulationParameters::SharedPtr kTriangulationSettingsPtr = std::make_shared<TriangulationParameters>();
const calibration::CameraCalibrationParameters::SharedPtr kCameraCalibrationParamsPtr =
  std::make_shared<calibration::CameraCalibrationParameters>();

class TriangulationTests : public ::testing::Test {
 public:
  explicit TriangulationTests()
    : triangulator_(kTriangulationSettingsPtr, kCameraCalibrationParamsPtr),
      random_number_generator_(random_device_()) {
    // Params setup
    kTriangulationSettingsPtr->geometric_error_threshold = 0.75;
    kTriangulationSettingsPtr->radius_error_threshold = 0.15;
    kTriangulationSettingsPtr->covariance_model = CovarianceModel::kMetric;
    kTriangulationSettingsPtr->use_nonlinear_refinement = false;
    kTriangulationSettingsPtr->tol_position = 1e-10;
    kTriangulationSettingsPtr->tol_reprojection_error = 1e-8;
    kTriangulationSettingsPtr->max_iterations = 100;
    kCameraCalibrationParamsPtr->Initialize(kRootDir + "/test/parameters/camera_calibration.yaml");
    // Set cameras used for tests.
    camera_names_ = kCameraCalibrationParamsPtr->camera_names;
    triangulator_.SetCameras(camera_names_);

    // Set up random number generator.
    random_number_generator_.seed(std::time(nullptr));
  }

 protected:
  Triangulation<Eigen::Vector2f> triangulator_;

  // Auxiliary variables
  std::vector<std::string> camera_names_;

  std::random_device random_device_;
  std::mt19937 random_number_generator_;
  float UniformSample(float low, float high) {
    std::uniform_real_distribution<float> uniform(low, high);
    return uniform(random_number_generator_);
  }

  int UniformIntSample0100() {
    std::uniform_int_distribution<int> uniform;
    return uniform(random_number_generator_);
  }

  float NormalSample(float mean, float stddev) {
    std::normal_distribution<float> normal{mean, stddev};
    return normal(random_number_generator_);
  }
};

TEST_F(TriangulationTests, TestTriangulatePoints) {
  // Uniformly at random select points in a box, back-project the points to the
  // cameras, add (zero-mean) noise the the back-projected points, shuffle the points, and
  // test if the initial 3D points can be triangulated.

  const int num_points_3d = 10;

  const float box_x_low = -3.F;
  const float box_x_high = 3.F;
  const float box_y_low = -2.F;
  const float box_y_high = 2.F;
  const float box_z_low = 0.F;
  const float box_z_high = 2.F;

  const float noise_stddev = 0.1F;  // in pixels; standard deviation of noise

  const float max_triangulation_error = 5e-3F;  // in meters;

  ///////////////////////
  // Generate test data.
  ///////////////////////
  const int num_cameras = static_cast<int>(camera_names_.size());
  std::vector<Eigen::Vector3f> points_3d;
  std::vector<std::vector<Eigen::Vector2f>> distorted_points(num_cameras);
  points_3d.reserve(num_points_3d);
  while (true) {
    // Sample 3D point.
    Eigen::Vector3f candidate_point;
    candidate_point << UniformSample(box_x_low, box_x_high), UniformSample(box_y_low, box_y_high),
      UniformSample(box_z_low, box_z_high);

    // Back-project point onto cameras.
    std::vector<std::pair<int, Eigen::Vector2f>> candidate_distorted_points;
    for (int camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const auto& camera = kCameraCalibrationParamsPtr->cameras.at(camera_names_[camera_idx]);

      Eigen::Vector2f distorted_point;
      if (camera.ProjectPoint(candidate_point, &distorted_point)) {
        // Add noise to back-projected point.
        distorted_point.x() += NormalSample(0.F, noise_stddev);
        distorted_point.y() += NormalSample(0.F, noise_stddev);

        // Check if distorted point include noise is still in cameras field of view.
        if (0.F <= distorted_point.x() && distorted_point.x() < static_cast<float>(camera.resolution[0] + 1) &&
            0.F <= distorted_point.y() && distorted_point.y() < static_cast<float>(camera.resolution[1] + 1)) {
          candidate_distorted_points.emplace_back(camera_idx, distorted_point);
        }
      }
    }

    // Check if there are sufficient distorted points for triangulation, i.e., two or more distorted points.
    if (candidate_distorted_points.size() >= 2) {
      for (auto& [camera_idx, distorted_point] : candidate_distorted_points) {
        distorted_points[camera_idx].emplace_back(distorted_point);
      }

      points_3d.emplace_back(candidate_point);
    }

    // Check if sufficient valid 3D points were found.
    if (static_cast<int>(points_3d.size()) == num_points_3d) {
      break;
    }
  }

  // Shuffle distorted points.
  for (int camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
    std::shuffle(distorted_points[camera_idx].begin(), distorted_points[camera_idx].end(),
                 std::mt19937(std::random_device()()));
  }

  /////////////////////////////
  // Tests triangulate points.
  /////////////////////////////
  // Triangulate points (and find point correspondences in the process).
  std::vector<TriangulatedPoint> points_3d_triangulated = triangulator_.TriangulatePoints(distorted_points);

  // Test if the same number of 3D points was recovered.
  EXPECT_EQ(points_3d.size(), points_3d_triangulated.size());

  // Compute error to original 3D points
  std::vector<std::vector<float>> error(num_points_3d);
  for (int i = 0; i < num_points_3d; ++i) {
    error[i].resize(num_points_3d, std::numeric_limits<float>::infinity());
    for (int j = 0; j < num_points_3d; ++j) {
      error[i][j] = (points_3d_triangulated[i].position - points_3d[j]).norm();
    }
  }

  // Greedily assign triangulated points to the original 3D point with the least error,
  // and test that triangulation error is below threshold.
  for (int i = 0; i < num_points_3d; ++i) {
    const auto min_it = std::ranges::min_element(error[i]);

    if (*min_it < max_triangulation_error) {
      const int idx = static_cast<int>(std::distance(error[i].begin(), min_it));

      // Mark found original 3D point as used by setting error from triangulated point
      // to this point to infinity;
      for (int j = 0; j < num_points_3d; ++j) {
        error[j][idx] = std::numeric_limits<float>::infinity();
      }
    } else {
      std::printf("error %f\n", *min_it);
      FAIL();
    }
  }

  // Done. All triangulated points were assigned to a unique original 3D point and their
  // triangulation error was below the given threshold.
  SUCCEED();
}

TEST_F(TriangulationTests, TestTriangulateSortedPoints) {
  // Uniformly at random select points in a box, back-project the points to the
  // cameras, add (zero-mean) noise the the back-projected points, shuffle the points, and
  // test if the initial 3D points can be triangulated.

  const int num_points_3d = 10;

  const float box_x_low = -3.F;
  const float box_x_high = 3.F;
  const float box_y_low = -2.F;
  const float box_y_high = 2.F;
  const float box_z_low = 0.F;
  const float box_z_high = 2.F;

  const float noise_stddev = 0.1F;  // in pixels; standard deviation of noise

  const float max_triangulation_error = 5e-3F;  // in meters;

  ///////////////////////
  // Generate test data.
  ///////////////////////
  const int num_cameras = static_cast<int>(camera_names_.size());
  std::vector<Eigen::Vector3f> points_3d;
  std::vector<std::vector<Eigen::Vector2f>> distorted_points(num_cameras);
  std::vector<std::vector<std::pair<int, int>>> point_correspondences;
  points_3d.reserve(num_points_3d);
  point_correspondences.reserve(num_points_3d);
  while (true) {
    // Sample 3D point.
    Eigen::Vector3f candidate_point;
    candidate_point << UniformSample(box_x_low, box_x_high), UniformSample(box_y_low, box_y_high),
      UniformSample(box_z_low, box_z_high);

    // Back-project point onto cameras.
    std::vector<std::pair<int, Eigen::Vector2f>> candidate_distorted_points;
    for (int camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const auto& camera = kCameraCalibrationParamsPtr->cameras.at(camera_names_[camera_idx]);

      Eigen::Vector2f distorted_point;
      if (camera.ProjectPoint(candidate_point, &distorted_point)) {
        // Add noise to back-projected point.
        distorted_point.x() += NormalSample(0.F, noise_stddev);
        distorted_point.y() += NormalSample(0.F, noise_stddev);

        // Check if distorted point include noise is still in cameras field of view.
        if (0.F <= distorted_point.x() && distorted_point.x() < static_cast<float>(camera.resolution[0] + 1) &&
            0.F <= distorted_point.y() && distorted_point.y() < static_cast<float>(camera.resolution[1] + 1)) {
          candidate_distorted_points.emplace_back(camera_idx, distorted_point);
        }
      }
    }

    // Check if there are sufficient distorted points for triangulation, i.e., two or more distorted points.
    if (candidate_distorted_points.size() >= 2) {
      point_correspondences.emplace_back();
      for (auto& [camera_idx, distorted_point] : candidate_distorted_points) {
        point_correspondences.back().emplace_back(camera_idx, distorted_points[camera_idx].size());
        distorted_points[camera_idx].emplace_back(distorted_point);
      }

      points_3d.emplace_back(candidate_point);
    }

    // Check if sufficient valid 3D points were found.
    if (static_cast<int>(points_3d.size()) == num_points_3d) {
      break;
    }
  }

  /////////////////////////////
  // Tests triangulate points.
  /////////////////////////////
  // Triangulate points (and find point correspondences in the process).
  std::vector<TriangulatedPoint> points_3d_triangulated =
    triangulator_.TriangulateSortedPoints(distorted_points, point_correspondences);

  // Test if the same number of 3D points was recovered.
  EXPECT_TRUE(points_3d.size() == points_3d_triangulated.size());

  // Test if triangulation error for all points is below given threshold.
  for (int i = 0; i < num_points_3d; ++i) {
    EXPECT_LT((points_3d_triangulated[i].position - points_3d[i]).norm(), max_triangulation_error);
  }

  // Done. The triangulation error of all points was below the given threshold.
  SUCCEED();
}

TEST_F(TriangulationTests, TestNonlinearRefinement) {
  // Uniformly at random select points in a box, back-project the points to the
  // cameras, add (zero-mean) noise the the back-projected points, triangulate with
  // and without nonlinear refinement and compare the error to the selected points
  const int num_points_3d = 500;

  const float box_x_low = -3.F;
  const float box_x_high = 3.F;
  const float box_y_low = -2.F;
  const float box_y_high = 2.F;
  const float box_z_low = 0.F;
  const float box_z_high = 2.F;

  const float noise_stddev = 0.05F;      // in pixels; standard deviation of noise
  const float max_error = 5e-3F;         // in meters
  const float diff_reprojection = 0.5F;  // in pixels

  ///////////////////////
  // Generate test data.
  ///////////////////////
  const int num_cameras = static_cast<int>(camera_names_.size());
  std::vector<Eigen::Vector3f> points_3d;
  std::vector<std::vector<Eigen::Vector2f>> distorted_points(num_cameras);
  std::vector<std::vector<std::pair<int, int>>> point_correspondences;
  point_correspondences.reserve(num_points_3d);
  Eigen::Matrix<float, 2, 3> dp_d_dx;
  points_3d.reserve(num_points_3d);

  while (true) {
    // Sample 3D point.
    Eigen::Vector3f candidate_point;
    candidate_point << UniformSample(box_x_low, box_x_high), UniformSample(box_y_low, box_y_high),
      UniformSample(box_z_low, box_z_high);

    // Back-project point onto cameras.
    std::vector<std::pair<int, Eigen::Vector2f>> candidate_distorted_points;
    for (int camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const auto& camera = kCameraCalibrationParamsPtr->cameras.at(camera_names_[camera_idx]);

      Eigen::Vector2f distorted_point;

      if (camera.ProjectPoint(candidate_point, &distorted_point, &dp_d_dx)) {
        // Add noise to back-projected point.
        distorted_point.x() += NormalSample(0.F, noise_stddev);
        distorted_point.y() += NormalSample(0.F, noise_stddev);
        // Check if distorted point include noise is still in cameras field of view.
        if (0.F <= distorted_point.x() && distorted_point.x() < static_cast<float>(camera.resolution[0] + 1) &&
            0.F <= distorted_point.y() && distorted_point.y() < static_cast<float>(camera.resolution[1] + 1)) {
          candidate_distorted_points.emplace_back(camera_idx, distorted_point);
        }
        // Check if distorted point include noise is still in cameras field of view.
      }
    }
    // Store point if in field of view of >= 2 cameras
    if (candidate_distorted_points.size() >= 2) {
      point_correspondences.emplace_back();
      for (auto& [camera_idx, distorted_point] : candidate_distorted_points) {
        point_correspondences.back().emplace_back(camera_idx, distorted_points[camera_idx].size());
        distorted_points[camera_idx].emplace_back(distorted_point);
      }
      // Store the corresponing 3D point
      points_3d.emplace_back(candidate_point);
    }
    // Check if sufficient valid 3D points were found.
    if (static_cast<int>(points_3d.size()) == num_points_3d) {
      break;
    }
  }

  // Triangulate without nonlinear refinement
  kTriangulationSettingsPtr->use_nonlinear_refinement = false;
  std::vector<TriangulatedPoint> points_3d_triangulated =
    triangulator_.TriangulateSortedPoints(distorted_points, point_correspondences);

  // Triangulate with nonlinear refinement
  kTriangulationSettingsPtr->use_nonlinear_refinement = true;
  std::vector<TriangulatedPoint> points_3d_triangulated_refined =
    triangulator_.TriangulateSortedPoints(distorted_points, point_correspondences);

  EXPECT_EQ(points_3d.size(), points_3d_triangulated_refined.size());
  // Project back once again
  for (int point_idx = 0; point_idx < num_points_3d; ++point_idx) {
    EXPECT_LT((points_3d_triangulated_refined[point_idx].position - points_3d[point_idx]).norm(), max_error);
    float dist_projected_point = 0;
    float dist_projected_point_refined = 0;
    for (int camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const auto& camera = kCameraCalibrationParamsPtr->cameras.at(camera_names_[camera_idx]);
      Eigen::Vector2f projected_point;
      Eigen::Vector2f projected_point_refined;
      if (camera.ProjectPoint(points_3d_triangulated[point_idx].position, &projected_point) &&
          camera.ProjectPoint(points_3d_triangulated_refined[point_idx].position, &projected_point_refined) &&
          std::cmp_less(point_idx, distorted_points[camera_idx].size())) {
        if (distorted_points[camera_idx][point_idx].norm() > 0) {
          if (0.F <= projected_point.x() && projected_point.x() < static_cast<float>(camera.resolution[0] + 1) &&
              0.F <= projected_point.y() && projected_point.y() < static_cast<float>(camera.resolution[1] + 1)) {
            dist_projected_point += (projected_point - distorted_points[camera_idx][point_idx]).norm();
            dist_projected_point_refined += (projected_point_refined - distorted_points[camera_idx][point_idx]).norm();
          }
        }
      }
    }
    EXPECT_LT(abs(dist_projected_point_refined - dist_projected_point), diff_reprojection);
  }

  SUCCEED();
}
}  // namespace
}  // namespace triangulation

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
