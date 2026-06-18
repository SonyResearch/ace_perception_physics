// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "triangulation/triangulation.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

#include "ace_loggers/ace_loggers.hpp"
#include "triangulation/datalogger.hpp"

namespace triangulation {

template <class DetectionPoint>
Triangulation<DetectionPoint>::Triangulation(
  const TriangulationParameters::ConstSharedPtr settings_ptr,
  const calibration::CameraCalibrationParameters::ConstSharedPtr camera_calibration_params_ptr,
  const ::datalogger::DataWriter::SharedPtr datawriter_ptr)
  : settings_ptr_(std::move(settings_ptr)),
    camera_calib_params_ptr_(std::move(camera_calibration_params_ptr)),
    datawriter_ptr_(std::move(datawriter_ptr)) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(std::string("triangulation").c_str());
  }
}

template <class DetectionPoint>
bool Triangulation<DetectionPoint>::SetCameras(const std::vector<std::string>& camera_names) {
  init_ = false;
  num_cameras_ = static_cast<int>(camera_names.size());

  // Store indices to camera parameters for easy access
  camera_to_calib_indices_.reserve(num_cameras_);
  for (const auto& camera_name : camera_names) {
    if (!camera_calib_params_ptr_->cameras.contains(camera_name)) {
      LOG(ERROR) << "Cannot find camera \"" << camera_name << "\" in provided camera calibration parameters.\n";
      return false;
    }

    camera_to_calib_indices_.emplace_back(camera_calib_params_ptr_->cameras.GetIndex(camera_name));
  }
  // Store observation covariance
  observation_cov_.resize(2 * num_cameras_, 2 * num_cameras_);
  observation_cov_.setIdentity();
  observation_cov_ *= settings_ptr_->observation_variance;

  init_ = true;

  return true;
}

template <class DetectionPoint>
std::vector<TriangulatedPoint> Triangulation<DetectionPoint>::TriangulateUncertainPoints(
  std::vector<std::vector<DetectionPoint>>& distorted_points, std::vector<std::vector<Eigen::Vector2f>>& certainties) {
  observation_cov_.setIdentity();
  int count_sigmas = 0;
  for (const auto& certainty : certainties) {
    for (const auto& model_sigma : certainty) {
      observation_cov_.block<1, 1>(2 * count_sigmas, 2 * count_sigmas) *= model_sigma(0);
      observation_cov_.block<1, 1>(2 * count_sigmas + 1, 2 * count_sigmas + 1) *= model_sigma(1);
      count_sigmas++;
    }
  }
  std::vector<TriangulatedPoint> triangulated_points = TriangulatePoints(distorted_points);
  return triangulated_points;
}

template <class DetectionPoint>
std::vector<TriangulatedPoint> Triangulation<DetectionPoint>::TriangulatePoints(
  std::vector<std::vector<DetectionPoint>>& distorted_points) {
  std::vector<TriangulatedPoint> triangulated_points;
  auto logger_hook = ::datalogger::defer([&] {
    datawriter_ptr_ << std::chrono::high_resolution_clock::now() << static_cast<uint32_t>(0) << distorted_points
                    << triangulated_points;
  });

  // Sanity checks
  if (!init_) {
    LOG(ERROR) << "Triangulation is not initialized.\n";
    return triangulated_points;
  }

  if (static_cast<int>(distorted_points.size()) != num_cameras_) {
    LOG(ERROR) << "Size of distorted_points does not match number of cameras.\n";
    return triangulated_points;
  }

  // Check sufficient number of observations
  int num_obs = 0;
  std::vector<int> obs_to_cam_idx;
  std::vector<int> obs_to_pt_idx;
  for (int cam_idx = 0; cam_idx < num_cameras_; ++cam_idx) {
    for (int pt_idx = 0; pt_idx < static_cast<int>(distorted_points[cam_idx].size()); ++pt_idx) {
      ++num_obs;
      obs_to_cam_idx.emplace_back(cam_idx);
      obs_to_pt_idx.emplace_back(pt_idx);
    }
  }
  if (num_obs < 2) {
    return triangulated_points;
  }

  // Undistort points (for each camera)
  std::vector<DetectionPoint> undistorted_points(num_obs);
  for (int i = 0; i < num_obs; ++i) {
    const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[i]]);
    undistorted_points[i].head(2) =
      camera.UndistortPoint(distorted_points[obs_to_cam_idx[i]][obs_to_pt_idx[i]].head(2));
    // Store the radius information (if available as third element) in the undistorted point
    if (std::is_same_v<DetectionPoint, Eigen::Vector3f>) {
      undistorted_points[i][2] = distorted_points[obs_to_cam_idx[i]][obs_to_pt_idx[i]][2];
    }
  }

  // Compute geometric error for all observations
  Eigen::MatrixXf geometric_error(num_obs, num_obs);
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> match(num_obs, num_obs);
  for (int i = 0; i < num_obs; ++i) {
    for (int j = i; j < num_obs; ++j) {
      if (i == j) {
        geometric_error(i, j) = 0.F;
        match(i, j) = false;
        continue;
      }

      // Off-diagonal elements
      if (obs_to_cam_idx[i] == obs_to_cam_idx[j]) {
        geometric_error(i, j) = 1e9F;
        match(i, j) = false;
      } else {
        const auto& multiview_data_ij =
          camera_calib_params_ptr_->multiview_data.at(camera_to_calib_indices_[obs_to_cam_idx[i]])
            .at(camera_to_calib_indices_[obs_to_cam_idx[j]]);
        float radius_error;
        std::tie(geometric_error(i, j), radius_error) =
          ErrorCriteria(undistorted_points[i], undistorted_points[j], multiview_data_ij.fundamental_matrix,
                        multiview_data_ij.epipole_i, multiview_data_ij.epipole_j);
        match(i, j) = geometric_error(i, j) <= settings_ptr_->geometric_error_threshold &&
                      radius_error <= settings_ptr_->radius_error_threshold;
      }

      // Symmetric matrix
      geometric_error(j, i) = geometric_error(i, j);
      match(j, i) = match(i, j);
    }
  }

  // Find correspondences between observations
  std::vector<std::unordered_set<int>> pt_to_obs;
  pt_to_obs.reserve(num_obs / 2);

  std::vector<std::vector<int>> obs_to_pt(num_obs);
  for (int i = 0; i < num_obs; ++i) {
    obs_to_pt.reserve(num_obs / 2);
  }

  std::vector<float> pt_to_score;
  pt_to_score.reserve(num_obs / 2);
  // A score is assigned to each (3D) pt. The score is the average geometric error of the observations and the number
  // of observations used. The more observations the better, but if two (3D) pts have the same number of observations,
  // the one with lower average geometric error is preferred.

  const float score_separator = 10.F * settings_ptr_->geometric_error_threshold;

  for (int i = 0; i < num_obs - 1; ++i) {
    for (int j = i + 1; j < num_obs; ++j) {
      if (match(i, j)) {
        // Two observations match, i.e., their geometric error is below a treshold

        // 1. Find (3D) pts where observations are already being used
        std::unordered_set<int> pts(obs_to_pt[i].begin(), obs_to_pt[i].end());
        pts.insert(obs_to_pt[j].begin(), obs_to_pt[j].end());

        // 2. Check if observations match with existing (3D) pt
        bool pt_added = false;
        for (const auto& pt : pts) {
          const int check_point = pt_to_obs[pt].count(i) == 1 ? j : i;
          const int obs_0 = *pt_to_obs[pt].begin();
          const int obs_1 = *(++pt_to_obs[pt].begin());
          if (match(check_point, obs_0) && match(check_point, obs_1)) {
            // Remove matches between observations already assigned to (3D) pt and current observation
            for (const auto& obs : pt_to_obs[pt]) {
              match(check_point, obs) = false;
              match(obs, check_point) = false;
            }

            // Add current observation to (3D) pt
            pt_to_obs[pt].emplace(check_point);
            obs_to_pt[check_point].push_back(pt);
            pt_to_score[pt] +=
              geometric_error(i, j) -
              score_separator * static_cast<float>(2 * pt_to_obs[pt].size() - 1);  // (N**2 - (N-1)**2) = 2*N - 1
            // Note: For the sake of effiency only one geometric error per observation is used.

            pt_added = true;
            // break;
            // Note: The line above could be uncommented for additional speed up. In theory, only a single point should
            // match, hence after one was found, there is no need to check further points.
          }
        }

        // 3. Add new (3D) pt if there does not yet exist a (3D) pt that also matches with current observation pair
        if (!pt_added) {
          pt_to_obs.emplace_back(std::unordered_set<int>({i, j}));

          const int pt = static_cast<int>(pt_to_obs.size()) - 1;
          obs_to_pt[i].push_back(pt);
          obs_to_pt[j].push_back(pt);

          pt_to_score.emplace_back(geometric_error(i, j) - score_separator * 2 * 2);
        }
      }
    }
  }

  // Select (3D) pts according to 1) number of observations and 2) geometric error. Note that each observation can only
  // be used once.
  for (auto i = 0u; i < pt_to_score.size(); ++i) {
    pt_to_score[i] /= static_cast<float>(pt_to_obs[i].size());
    // Note: this step is only cosmetics.
  }
  std::vector<int> sorted_idx(pt_to_score.size());
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(),
            [&pt_to_score](const int& i, const int& j) { return pt_to_score[i] < pt_to_score[j]; });
  // Note that we could also have sort according to the two sorting criterias mentioned above directly in the comparison
  // function, but it is found to be computationally more efficient to have a single sorting criteria.
  std::vector<bool> pt_validity(pt_to_score.size(), true);
  std::vector<int> trig_indicies;
  for (const auto& pt : sorted_idx) {
    if (pt_validity[pt]) {
      // Triangulate (3D) pt
      trig_indicies.push_back(pt);
      const TriangulatedPoint triangulated_point = TriangulatePointWithRefinement(
        distorted_points, undistorted_points, pt_to_obs[pt], obs_to_cam_idx, obs_to_pt_idx);
      // Append to vector of triangulated points.
      triangulated_points.emplace_back(triangulated_point);

      // Mark (3D) pts that use the same observations as invalid.
      // Note: One should only remove the observations from other points and recompute their score until there are no
      // more valid (3D) pts left, i.e., with two and more observations.
      for (const auto& obs : pt_to_obs[pt]) {
        for (const auto& pt_invalid : obs_to_pt[obs]) {
          pt_validity[pt_invalid] = false;
        }
      }
    }
  }

  // ghost balls
  if (settings_ptr_->ghost_balls_policy != TriangulationParameters::GhostBallsPolicy::kNone) {
    std::vector<std::vector<size_t>> similarity(triangulated_points.size());
    std::vector<bool> marked(triangulated_points.size(), false);
    for (size_t i = 0; i < triangulated_points.size(); ++i) {
      if (marked[i]) {
        continue;  // skip this point, already processed as a similarity
      }
      similarity[i].push_back(i);
      const auto& p1 = triangulated_points[i].position;
      for (size_t j = i + 1; j < triangulated_points.size(); ++j) {
        const auto& p2 = triangulated_points[j].position;
        const float distance = (p2 - p1).squaredNorm();
        if (distance > settings_ptr_->similarity_threshold * settings_ptr_->similarity_threshold) {
          continue;
        }
        marked[j] = true;
        similarity[i].push_back(j);
      }
    }
    std::vector<TriangulatedPoint> ghost_free_points;
    for (const auto& sim : similarity) {
      if (sim.empty()) {
        continue;
      }
      if (sim.size() == 1 || settings_ptr_->ghost_balls_policy == TriangulationParameters::GhostBallsPolicy::kDrop) {
        ghost_free_points.emplace_back(triangulated_points[sim.front()]);
        continue;
      }
      // process identified similarities

      // collect 2d detections from all ghost balls
      std::unordered_set<int> pt_to_ob;
      using PointWeight = std::tuple<int, float, int>;
      std::vector<PointWeight> pts_weight;
      for (auto pt_idx : sim) {
        auto points = pt_to_obs[trig_indicies[pt_idx]];  // get the points list using trig_indicies
        pt_to_ob.insert(points.begin(), points.end());

        for (auto pt2d_idx : points) {
          const auto& pt_2d = undistorted_points[pt2d_idx];
          const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[pt2d_idx]]);
          const Eigen::Vector3f camera_position = camera.T_origin_camera.block<3, 1>(0, 3);

          float pt_weight{0};
          if (pt_2d.size() > 2) {
            pt_weight = pt_2d[2];  // use the radius as the weight
          } else {
            // otherwise use the distance to camera
            pt_weight = (triangulated_points[pt_idx].position - camera_position).squaredNorm();
          }
          pts_weight.emplace_back(pt2d_idx, pt_weight, pt_idx);
        }
      }

      // sort by weight
      std::sort(pts_weight.begin(), pts_weight.end(),
                [](const PointWeight& a, const PointWeight& b) { return std::get<1>(a) > std::get<1>(b); });
      int selected_point = -1;
      {
        static constexpr float kRejectionThreshold =
          0.3F;  // reject 30% of weak detections (i.e. distance from camera, or small radius)
        int rejection_count = static_cast<int>(std::round(static_cast<float>(pts_weight.size()) * kRejectionThreshold));
        for (auto it = pts_weight.begin(); it != pts_weight.end(); ++it, --rejection_count) {
          if (rejection_count > 0) {
            pt_to_ob.erase(std::get<0>(*it));
          } else {
            // for valid detections, check if they all belong to the same triangulation
            if (selected_point == -1) {
              selected_point = std::get<2>(*it);
            } else if (selected_point != std::get<2>(*it)) {
              selected_point = -1;
              break;
            }
          }
        }
      }

      if (selected_point == -1) {
        TriangulatedPoint triangulated_point =
          TriangulatePointWithRefinement(distorted_points, undistorted_points, pt_to_ob, obs_to_cam_idx, obs_to_pt_idx);
        ghost_free_points.emplace_back(std::move(triangulated_point));
      } else {
        // don't retriangulate
        ghost_free_points.emplace_back(triangulated_points[selected_point]);
      }
    }
    triangulated_points = std::move(ghost_free_points);
  }
  if (triangulation_filter_) {
    // perform filtering
    std::vector<TriangulatedPoint> filtered;
    for (const auto& triangulated_point : triangulated_points) {
      if (triangulation_filter_(triangulated_point)) {
        filtered.emplace_back(triangulated_point);
      }
    }
    triangulated_points = std::move(filtered);
  }

  return triangulated_points;
}

template <class DetectionPoint>
std::vector<TriangulatedPoint> Triangulation<DetectionPoint>::TriangulateSortedPoints(
  std::vector<std::vector<DetectionPoint>>& distorted_points,
  std::vector<std::vector<std::pair<int, int>>>& point_correspondences) {
  std::vector<TriangulatedPoint> triangulated_points;
  auto logger_hook = ::datalogger::defer([&] {
    datawriter_ptr_ << std::chrono::high_resolution_clock::now() << static_cast<uint32_t>(1) << distorted_points
                    << point_correspondences << triangulated_points;
  });

  // Sanity checks
  if (!init_) {
    LOG(ERROR) << "Triangulation is not initialized.\n";
    return triangulated_points;
  }

  if (static_cast<int>(distorted_points.size()) != num_cameras_) {
    LOG(ERROR) << "Size of distorted_points does not match number of cameras.\n";
    return triangulated_points;
  }

  // Check sufficient number of observations
  int num_obs = 0;
  std::vector<int> obs_to_cam_idx;
  std::vector<int> obs_to_pt_idx;
  std::vector<std::vector<int>> cam_pt_to_obs(num_cameras_);
  for (int cam_idx = 0; cam_idx < num_cameras_; ++cam_idx) {
    for (int pt_idx = 0; pt_idx < static_cast<int>(distorted_points[cam_idx].size()); ++pt_idx) {
      cam_pt_to_obs[cam_idx].emplace_back(num_obs);
      obs_to_cam_idx.emplace_back(cam_idx);
      obs_to_pt_idx.emplace_back(pt_idx);
      ++num_obs;
    }
  }
  if (num_obs < 2) {
    return triangulated_points;
  }

  // Re-format point correspondences
  std::vector<std::unordered_set<int>> pt_to_obs;
  pt_to_obs.reserve(point_correspondences.size());
  for (auto& pt : point_correspondences) {
    if (pt.size() < 2) {
      continue;
    }

    pt_to_obs.emplace_back();
    for (const auto& point_correspondence : pt) {
      pt_to_obs.back().insert(cam_pt_to_obs[point_correspondence.first][point_correspondence.second]);
    }
  }

  // Undistort points (for each camera)
  std::vector<DetectionPoint> undistorted_points(num_obs);
  for (int i = 0; i < num_obs; ++i) {
    const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[i]]);
    undistorted_points[i].head(2) =
      camera.UndistortPoint(distorted_points[obs_to_cam_idx[i]][obs_to_pt_idx[i]].head(2));
    // Store the radius information (if available as third element) in the undistorted point
    if (std::is_same_v<DetectionPoint, Eigen::Vector3f>) {
      undistorted_points[i][2] = distorted_points[obs_to_cam_idx[i]][obs_to_pt_idx[i]][2];
    }
  }

  // Triangulate points
  for (auto& pt_to_ob : pt_to_obs) {
    // Triangulate (3D) pt
    TriangulatedPoint triangulated_point =
      TriangulatePointWithRefinement(distorted_points, undistorted_points, pt_to_ob, obs_to_cam_idx, obs_to_pt_idx);
    // Append to vector of triangulated points.
    triangulated_points.emplace_back(std::move(triangulated_point));
  }

  return triangulated_points;
}

template <class DetectionPoint>
TriangulatedPoint Triangulation<DetectionPoint>::TriangulatePoint(const std::vector<DetectionPoint>& observations,
                                                                  const std::unordered_set<int>& obs,
                                                                  const std::vector<int>& obs_to_cam_idx,
                                                                  bool undistort) {
  TriangulatedPoint triangulated_point;
  if (obs.empty()) {
    return triangulated_point;
  }
  if (undistort) {
    std::vector<DetectionPoint> undistorted(observations.size());
    for (size_t i = 0; i < observations.size(); ++i) {
      const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[i]]);
      undistorted[i].head(2) = camera.UndistortPoint(observations[i].head(2));
    }
    triangulated_point = TriangulatePointDLT(undistorted, obs, obs_to_cam_idx);
  } else {
    triangulated_point = TriangulatePointDLT(observations, obs, obs_to_cam_idx);
  }
  // calculate reprojection error
  const Eigen::Vector3f& pos = triangulated_point.position;
  float reprojection_error = 0;
  float total_triangulations = 0;
  for (size_t i = 0; i < observations.size(); ++i) {
    Eigen::Vector2f p_d(0, 0);
    const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[i]]);
    if (camera.ProjectPoint(pos, &p_d, nullptr)) {
      const Eigen::Vector2f& distance_error = observations[i].head(2) - p_d;
      reprojection_error += distance_error.squaredNorm();
      total_triangulations += 1;
    }
  }
  if (total_triangulations > 0) {
    triangulated_point.reprojection_error = sqrtf(reprojection_error) / total_triangulations;
  } else {
    triangulated_point.reprojection_error = std::numeric_limits<double>::quiet_NaN();
  }
  return triangulated_point;
}
template <class DetectionPoint>
std::pair<float, float> Triangulation<DetectionPoint>::ErrorCriteria(const DetectionPoint& pt_i,
                                                                     const DetectionPoint& pt_j,
                                                                     const Eigen::Matrix3f& F_ij,
                                                                     const Eigen::Vector2f& epipole_i,
                                                                     const Eigen::Vector2f& epipole_j) {
  // Computes (1) Sampson distance, (2) radius error (if radius is passed)
  //
  // Sampson distance is a first-order approximation of the geometric error,
  // i.e., distance of point j to epipolar line corresponding to pt_i in camera j.
  // See "Multi View Geometry in Computer Vision", Second Edition, R. Hartley and A. Zisserman, Section 11.4.3

  const Eigen::Vector3f& pt_i_h = pt_i.head(2).colwise().homogeneous();
  const Eigen::Vector3f& pt_j_h = pt_j.head(2).colwise().homogeneous();
  const Eigen::Vector3f& epiline_i_in_j = pt_i_h.transpose() * F_ij;
  const Eigen::Vector3f& epiline_j_in_i = F_ij * pt_j_h;
  const float& v = pt_i_h.dot(epiline_j_in_i);

  const float& sampson_distance =
    (v * v) / (epiline_i_in_j(0) * epiline_i_in_j(0) + epiline_i_in_j(1) * epiline_i_in_j(1) +
               epiline_j_in_i(0) * epiline_j_in_i(0) + epiline_j_in_i(1) * epiline_j_in_i(1));

  // If radius information is provided, compute the radius error
  // We basically start with the radius from left view, map it to the right view,
  // and finally compare the mapped and actual radii.
  // For robustness, radius is sampled in the orthognal direction to the epipole
  if (std::is_same_v<DetectionPoint, Eigen::Vector3f>) {
    // First, compute the direction from the epipole for the center of left view
    Eigen::Vector2f r_dir_i = (pt_i.head(2) - epipole_i).normalized();
    // Find the orthognal the direction to the previous direction
    // [x, y] -> [-y, x] is a 2D cross product
    std::swap(r_dir_i[0], r_dir_i[1]);
    r_dir_i[0] *= -1.F;

    // Compute a point along the left circumference in the orthognal direction & map it to the right view
    // The left point image becomes a line on the right view
    const Eigen::Vector3f& r_epiline = (pt_i.head(2) + pt_i[2] * r_dir_i).colwise().homogeneous().transpose() * F_ij;
    // The first two components of the line are the line direction (orthognal to the right epipole)
    Eigen::Vector2f r_dir_j = r_epiline.head(2).normalized();
    // And we need to find the direction passing through the right epipole
    // [x, y] -> [-y, x] is a 2D cross product
    std::swap(r_dir_j[0], r_dir_j[1]);
    r_dir_j[0] *= -1.F;

    // Using the new right-epipole direction, compute a point on the right observation's expected circumference
    const Eigen::Vector2f& p_tan_j = epipole_j + r_dir_j.dot(pt_j.head(2) - epipole_j) * r_dir_j;
    const float& r_j_hat = (p_tan_j - pt_j.head(2)).norm();

    // Measure the error between expected radius and actual one on the right view
    // Clip the difference at maximum to the actual radius
    const float r_diff = std::min(pt_j[2], std::abs(pt_j[2] - r_j_hat));
    // Slightly bias the reprojection error by the radius error as well
    return std::make_pair(sampson_distance + 1e-2F * r_diff, r_diff / pt_j[2]);
  }
  return std::make_pair(sampson_distance, 0.F);
}

template <class DetectionPoint>
TriangulatedPoint Triangulation<DetectionPoint>::TriangulatePointWithRefinement(
  std::vector<std::vector<DetectionPoint>>& distorted_points, const std::vector<DetectionPoint>& undistorted_points,
  const std::unordered_set<int>& obs, const std::vector<int>& obs_to_cam_idx, const std::vector<int>& obs_to_pt_idx) {
  TriangulatedPoint triangulated_point = TriangulatePointDLT(undistorted_points, obs, obs_to_cam_idx);

  if (settings_ptr_->use_nonlinear_refinement) {
    triangulated_point = MinimizeReprojectionErrorFixPoint(triangulated_point.position, distorted_points, obs,
                                                           obs_to_cam_idx, obs_to_pt_idx);
  }

  // Add camera and point indices info
  triangulated_point.observation_indices.reserve(obs.size());
  std::transform(obs.begin(), obs.end(), std::back_inserter(triangulated_point.observation_indices),
                 [&](const int& obs_i) { return std::make_pair(obs_to_cam_idx[obs_i], obs_to_pt_idx[obs_i]); });

  return triangulated_point;
}
template <class DetectionPoint>
TriangulatedPoint Triangulation<DetectionPoint>::TriangulatePointDLT(const std::vector<DetectionPoint>& undistorted_pts,
                                                                     const std::unordered_set<int>& obs,
                                                                     const std::vector<int>& obs_to_cam_idx) {
  // This function implements a triangulation using DLT (direct linear transform) from N observations (N >= 2).
  // See "Multi View Geometry in Computer Vision", Second Edition, R. Hartley and A. Zisserman, Section 12.2

  TriangulatedPoint triangulated_point;

  Eigen::MatrixX4f A(2 * obs.size(), 4);
  int counter = 0;
  for (const auto& obs_i : obs) {
    const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[obs_i]]);
    A.row(2 * counter) = undistorted_pts[obs_i].x() * camera.projection_matrix.row(2) - camera.projection_matrix.row(0);
    A.row(2 * counter + 1) =
      undistorted_pts[obs_i].y() * camera.projection_matrix.row(2) - camera.projection_matrix.row(1);

    if (settings_ptr_->covariance_model == CovarianceModel::kMetric) {
      A.row(2 * counter) /= static_cast<Eigen::Vector4f>(A.row(2 * counter)).head(3).norm();
      A.row(2 * counter + 1) /= static_cast<Eigen::Vector4f>(A.row(2 * counter + 1)).head(3).norm();
    }

    ++counter;
  }
  const Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinV);
  Eigen::MatrixXf V = svd.matrixV();

  if (settings_ptr_->covariance_model == CovarianceModel::kMetric) {
    triangulated_point.position = V.block<3, 1>(0, 3) / V(3, 3);
    triangulated_point.covariance(0, 0) = static_cast<float>(2 * std::pow(svd.singularValues()(3), 2) / counter);
    return triangulated_point;
  }

  triangulated_point.position = V.block<3, 1>(0, 3) / V(3, 3);
  return triangulated_point;
}

template <class DetectionPoint>
TriangulatedPoint Triangulation<DetectionPoint>::MinimizeReprojectionErrorGradDescent(
  const Eigen::Vector3f& x_guess, const std::vector<std::vector<DetectionPoint>>& distorted_points,
  const std::unordered_set<int>& obs, const std::vector<int>& obs_to_cam_idx, const std::vector<int>& obs_to_pt_idx) {
  // Set up triangulated point.
  TriangulatedPoint triangulated_point;
  // triangulated_point.num_cameras = static_cast<int>(obs.size());
  triangulated_point.position = x_guess;

  // Gradient descent
  Eigen::MatrixX3f jac(2 * obs.size(), 3);
  Eigen::Vector3f prev_grad;
  Eigen::Vector3f grad;
  float prev_reprojection_error;
  float& reprojection_error = triangulated_point.reprojection_error;
  Eigen::Vector3f prev_x;
  Eigen::Vector3f& x = triangulated_point.position;
  float gamma = 1e-8F;  // Learning rate for gradient descent

  for (int i = 0; i < settings_ptr_->max_iterations; ++i) {
    prev_grad = grad;
    grad = Eigen::Vector3f::Zero();
    prev_reprojection_error = reprojection_error;
    reprojection_error = 0.F;

    int obs_i_counter = 0;
    for (const auto& obs_i : obs) {
      Eigen::Vector2f p_d;
      Eigen::Matrix<float, 2, 3> dp_d_dx;
      const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[obs_i]]);
      if (camera.ProjectPoint(x, &p_d, &dp_d_dx)) {
        if (settings_ptr_->use_nonlinear_refinement) {
          jac.middleRows<2>(2 * obs_i_counter++) = dp_d_dx;
        }
        const Eigen::Vector2f& distance_error =
          distorted_points[obs_to_cam_idx[obs_i]][obs_to_pt_idx[obs_i]].head(2) - p_d;
        grad += -2.F * dp_d_dx.transpose() * distance_error;
        reprojection_error += distance_error.squaredNorm();
      } else {
        LOG(WARNING) << "Projection failed; Returning triangulated point from DLT.\n";
        triangulated_point.position = x_guess;
        return triangulated_point;
      }
    }
    reprojection_error = sqrtf(reprojection_error) / static_cast<float>(obs.size());

    // Check improvement of reprojection error
    if (i > 1 && std::fabs(reprojection_error - prev_reprojection_error) < settings_ptr_->tol_reprojection_error) {
      triangulated_point.covariance = ComputeCovariance(jac, obs);
      return triangulated_point;
    }

    if (i > 0) {
      // Check for vanishing gradient, i.e., solution is close to local optimum.
      const float dgrad_sqnorm = (grad - prev_grad).squaredNorm();
      if (dgrad_sqnorm < 1e-5F) {
        gamma = 1e-8F;
      } else {
        // Update learning rate using Wolfe conditions.
        gamma = std::fabs((x - prev_x).transpose() * (grad - prev_grad)) / (grad - prev_grad).squaredNorm();
      }
    }

    prev_x = x;
    x -= gamma * grad;

    // Check change in position
    if (i > 0 && (x - prev_x).norm() < settings_ptr_->tol_position) {
      break;
    }
  }

  // Compute reprojection error and (if required) covariance
  reprojection_error = 0.F;
  int obs_i_counter = 0;
  for (const auto& obs_i : obs) {
    Eigen::Vector2f p_d;

    bool proj_point_success;
    const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[obs_i]]);
    if (settings_ptr_->use_nonlinear_refinement) {
      Eigen::Matrix<float, 2, 3> dp_d_dx;
      proj_point_success = camera.ProjectPoint(x, &p_d, &dp_d_dx);
      if (proj_point_success) {
        jac.middleRows<2>(2 * obs_i_counter++) = dp_d_dx;
      }
    } else {
      proj_point_success = camera.ProjectPoint(x, &p_d);
    }

    if (proj_point_success) {
      const Eigen::Vector2f& distance_error =
        distorted_points[obs_to_cam_idx[obs_i]][obs_to_pt_idx[obs_i]].head(2) - p_d;
      reprojection_error += distance_error.squaredNorm();
    } else {
      LOG(WARNING) << "Projection failed; Returning triangulated point from DLT.\n";
      return triangulated_point;
    }
  }
  reprojection_error = sqrtf(reprojection_error) / static_cast<float>(obs.size());

  if (settings_ptr_->use_nonlinear_refinement) {
    triangulated_point.covariance = ComputeCovariance(jac, obs);
  }
  return triangulated_point;
}

template <class DetectionPoint>
TriangulatedPoint Triangulation<DetectionPoint>::MinimizeReprojectionErrorFixPoint(
  const Eigen::Vector3f& x_guess, const std::vector<std::vector<DetectionPoint>>& distorted_points,
  const std::unordered_set<int>& obs, const std::vector<int>& obs_to_cam_idx, const std::vector<int>& obs_to_pt_idx) {
  // Set up triangulated point.
  TriangulatedPoint triangulated_point;
  triangulated_point.position = x_guess;

  // Fix-point method.
  Eigen::MatrixX3f jac(2 * obs.size(), 3);
  Eigen::VectorXf error(2 * obs.size());
  float prev_reprojection_error;
  float& reprojection_error = triangulated_point.reprojection_error;
  Eigen::Vector3f prev_x;
  Eigen::Vector3f& x = triangulated_point.position;

  for (int i = 0; i < settings_ptr_->max_iterations; ++i) {
    prev_reprojection_error = reprojection_error;
    reprojection_error = 0.F;

    int obs_i_counter = 0;
    for (const auto& obs_i : obs) {
      Eigen::Vector2f p_d;
      Eigen::Matrix<float, 2, 3> dp_d_dx;
      const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[obs_i]]);
      if (camera.ProjectPoint(x, &p_d, &dp_d_dx)) {
        jac.middleRows<2>(2 * obs_i_counter) = dp_d_dx;
        const Eigen::Vector2f& distance_error =
          distorted_points[obs_to_cam_idx[obs_i]][obs_to_pt_idx[obs_i]].head(2) - p_d;
        error.segment<2>(2 * obs_i_counter++) = distance_error;
        reprojection_error += distance_error.squaredNorm();
      } else {
        LOG(WARNING) << "Projection failed; Returning triangulated point from DLT.\n";
        triangulated_point.position = x_guess;
        return triangulated_point;
      }
    }
    reprojection_error = sqrtf(reprojection_error) / static_cast<float>(obs.size());

    // Check improvement of reprojection error
    if (i > 1 && std::fabs(reprojection_error - prev_reprojection_error) < settings_ptr_->tol_reprojection_error) {
      triangulated_point.covariance = ComputeCovariance(jac, obs);
      return triangulated_point;
    }

    prev_x = x;
    x += (jac.transpose() * jac).inverse() * (jac.transpose() * error);

    // Check change in position
    if (i > 0 && (x - prev_x).norm() < settings_ptr_->tol_position) {
      break;
    }
  }

  // Compute reprojection error
  reprojection_error = 0.F;
  int obs_i_counter = 0;
  for (const auto& obs_i : obs) {
    const auto& camera = camera_calib_params_ptr_->cameras.at(camera_to_calib_indices_[obs_to_cam_idx[obs_i]]);

    Eigen::Vector2f p_d;
    bool proj_point_success;
    if (settings_ptr_->use_nonlinear_refinement) {
      Eigen::Matrix<float, 2, 3> dp_d_dx;
      proj_point_success = camera.ProjectPoint(x, &p_d, &dp_d_dx);
      if (proj_point_success) {
        jac.middleRows<2>(2 * obs_i_counter++) = dp_d_dx;
      }
    } else {
      proj_point_success = camera.ProjectPoint(x, &p_d);
    }
    if (proj_point_success) {
      const Eigen::Vector2f& distance_error =
        distorted_points[obs_to_cam_idx[obs_i]][obs_to_pt_idx[obs_i]].head(2) - p_d;
      reprojection_error += distance_error.squaredNorm();
    } else {
      LOG(WARNING) << "Projection failed; Returning triangulated point from DLT.\n";
      triangulated_point.position = x_guess;
      return triangulated_point;
    }
  }
  reprojection_error = sqrtf(reprojection_error) / static_cast<float>(obs.size());

  triangulated_point.covariance = ComputeCovariance(jac, obs);

  return triangulated_point;
}

template <class DetectionPoint>
Eigen::Matrix3f Triangulation<DetectionPoint>::ComputeCovariance(const Eigen::MatrixX3f& jac,
                                                                 const std::unordered_set<int>& obs) {
  const Eigen::Matrix3f inv = (jac.transpose() * jac).inverse();
  Eigen::Matrix3f cov =
    inv * (jac.transpose() * observation_cov_.block(0, 0, 2 * obs.size(), 2 * obs.size()) * jac) * inv;

  return cov;
}

template <class DetectionPoint>
const ::datalogger::DataWriter::SharedPtr& Triangulation<DetectionPoint>::GetDataLogger() const {
  return datawriter_ptr_;
}

// Explicit template instantiation
template class Triangulation<Eigen::Vector2f>;  // points only
template class Triangulation<Eigen::Vector3f>;  // points with radii

}  // namespace triangulation
