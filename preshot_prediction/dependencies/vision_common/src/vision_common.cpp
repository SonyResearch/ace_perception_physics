// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "vision_common/vision_common.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace vision_common {

size_t GetFactorial(const size_t& n, const size_t& k) { return (n <= k) ? 1 : n * GetFactorial(n - 1, k); }

void FitCone(const Eigen::Ref<const Eigen::Matrix3Xf>& /*unit_sphere_points*/, Eigen::Vector3f& /*cone_axis*/,
             float& cone_angle) {
  // TODO(cv3d): Implement me
  cone_angle = 0;
}

/*
 * Finds the nearest ray-sphere intersection point by computing ray traveled distances from the equation:
 *  ```
 *  || ray_start + distances * rays_dir - sphere_center, axis=1 || == sphere_radius
 *  ```
 * and then returns the intersection point, which is `ray_start + distances * rays_dir`
 */
void ComputeNearestIntersections(const Eigen::Vector3f& ray_start, const Eigen::Matrix3Xf& rays_dir,
                                 const Eigen::Vector3f& sphere_center, const float& sphere_radius,
                                 Eigen::Matrix3Xf& points_3d) {
  const Eigen::Vector3f center_ray = sphere_center - ray_start;
  const float distance_max = center_ray.norm();
  const Eigen::VectorXf ray_cos = rays_dir.adjoint() * center_ray / distance_max;
  const Eigen::VectorXf ray_tan = ((1.F + ray_cos.array()) * (1.F - ray_cos.array())).pow(0.5) / ray_cos.array();

  // An initial estimation
  Eigen::VectorXf distances = Eigen::VectorXf::Constant(rays_dir.cols(), distance_max - sphere_radius);
  Eigen::VectorXf y_dists;
  Eigen::VectorXf sphere_sin;
  Eigen::VectorXf sphere_cos;
  for (size_t i = 0; i < 3; ++i) {  // It converges very fast
    y_dists = ray_tan.cwiseProduct(distances);
    sphere_sin = (y_dists / sphere_radius).array().max(0).min(1);
    sphere_cos = ((1.F + sphere_sin.array()) * (1.F - sphere_sin.array())).pow(0.5);
    distances = distance_max - sphere_cos.array() * sphere_radius;
  }

  points_3d = (rays_dir.array().rowwise() * distances.adjoint().array()).matrix().colwise() + ray_start;
}

void FilterUniquePointsInplace(std::vector<Eigen::Vector3f>& points, float eps) {
  // TODO(cv3d): Consider replacing with DBSCAN(eps, minPts=2)
  std::vector<Eigen::Vector3f> points_tmp(points);
  std::vector<int> indices_to_remove;

  // this loop can be changed to a recursive function
  while (true) {
    points_tmp = points;
    const int npoints = static_cast<int>(points.size());
    const int k_nearest_neighbor = 2;  // we exclude the point itself, so 1 less than the reference python code
    Eigen::MatrixXf distances_within_eps(npoints, k_nearest_neighbor);
    Eigen::MatrixXi indices_within_eps(npoints, k_nearest_neighbor);

    // initialize
    for (int i = 0; i < npoints; i++) {
      for (int j = 0; j < k_nearest_neighbor; j++) {
        distances_within_eps(i, j) = -1;
        indices_within_eps(i, j) = npoints;
      }
    }

    // create k_nearest_neighbor matrix
    // can be improved by exploiting symmetries, vectorization (does Eigen take care of this?) (and parallelization? -
    // depends on the threading structure) use -1 if not within eps distance
    for (int i = 0; i < npoints; i++) {
      for (int j = 0; j < npoints; j++) {
        // exclude self
        if (i != j) {
          // compute distance
          float d = (points[i] - points[j]).norm();
          d = d < eps ? d : -1;
          int idx = j;

          // insert in NN structures
          if (d >= 0) {
            for (int k = 0; k < k_nearest_neighbor; k++) {
              if (distances_within_eps(i, k) >= 0 && d < distances_within_eps(i, k)) {
                // shift values to make space for the new one
                for (int kr = k_nearest_neighbor - 1; kr >= k + 1; kr--) {
                  distances_within_eps(i, kr) = distances_within_eps(i, kr - 1);
                  indices_within_eps(i, kr) = indices_within_eps(i, kr - 1);
                }
                distances_within_eps(i, k) = d;
                indices_within_eps(i, k) = idx;
                break;
              }
              if (distances_within_eps(i, k) < 0) {
                distances_within_eps(i, k) = d;
                indices_within_eps(i, k) = idx;
                break;
              }
            }
          }
        }
      }
    }

    // clustering
    for (int i = 0; i < npoints; i++) {
      int number_of_neighbors = 1;
      for (int k = 0; k < k_nearest_neighbor; k++) {
        if (indices_within_eps(i, k) < npoints) {
          points_tmp[i] += points[indices_within_eps(i, k)];
          number_of_neighbors++;
        }
      }
      points_tmp[i] /= static_cast<float>(number_of_neighbors);
    }

    indices_to_remove.clear();
    for (int i = 0; i < npoints - 1; i++) {
      if (std::binary_search(indices_to_remove.begin(), indices_to_remove.end(), i)) {
        continue;
      }
      for (int j = i + 1; j < npoints; j++) {
        if ((points_tmp[i] - points_tmp[j]).norm() < 1e-5) {
          indices_to_remove.insert(std::upper_bound(indices_to_remove.begin(), indices_to_remove.end(), j), j);
        }
      }
    }

    // remove duplicate points
    for (auto i = indices_to_remove.rbegin(); i != indices_to_remove.rend(); i++) {  // NOLINT
      points_tmp.erase(points_tmp.begin() + *i);
    }

    // need to check that points and pointsTmp are the same
    // if (distances_within_eps.maxCoeff() < 0 || points.size() <= 1 || indices_to_remove.size() == 0) {
    if (points == points_tmp) {
      break;
    }
    points.swap(points_tmp);
  }
  points = points_tmp;
}

}  // namespace vision_common
