// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include <cstddef>

#include "eigen3/Eigen/Core"
#include "opencv2/opencv.hpp"

namespace vision_common {

// Compute AABB corners from min and max ones
// Works for both 2D & 3D spaces
template <typename T, int Dim>
void ComputeAABBFromMinMaxCorners(Eigen::Matrix<T, Dim, 1 << Dim>& corners, const Eigen::Matrix<T, Dim, 1>& min_corner,
                                  const Eigen::Matrix<T, Dim, 1>& max_corner) {
  for (uint32_t corner_index = 0; corner_index < (1 << Dim); corner_index++) {
    const uint32_t& corner_gray_code = corner_index ^ (corner_index >> 1);
    for (uint32_t axis_index = 0; axis_index < Dim; axis_index++) {
      const uint32_t& axis_bit_code = 1 << axis_index;
      corners(axis_index, corner_index) =
        ((corner_gray_code & axis_bit_code) ? max_corner[axis_index] : min_corner[axis_index]);
    }
  }
}

template <typename T>
struct opencv_matrix;

template <>
struct opencv_matrix<double> {
  static const int kType = CV_64F;
};
template <>
struct opencv_matrix<float> {
  static const int kType = CV_32F;
};
template <>
struct opencv_matrix<uint8_t> {
  static const int kType = CV_8U;
};

template <typename T, int Rows, int Cols>
cv::Mat EigenToCV(Eigen::Matrix<T, Rows, Cols>& src) {
  // Assuming src is ColMajor and its data is also column-wise
  cv::Mat dst(static_cast<int>(src.cols()), static_cast<int>(src.rows()), opencv_matrix<T>::kType,
              static_cast<void*>(src.data()), src.outerStride() * sizeof(T));
  return dst;
}

size_t GetFactorial(const size_t& n, const size_t& k = 1);

template <typename T>
void ComputePermutations(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& elements, const size_t& k,
                         std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>>& perms) {
  const size_t n = elements.rows();
  if (k > n) {
    throw std::runtime_error("k cannot be greater than n");
  }

  std::vector<size_t> divisors(k);
  divisors[k - 1] = 1;
  for (size_t row_idx = 1; row_idx < k; ++row_idx) {
    divisors[k - row_idx - 1] = divisors[k - row_idx] * n;
  }

  const size_t n_perms = GetFactorial(n, n - k);
  perms.reserve(n_perms);

  const size_t n_iters = divisors[0] * n;
  for (size_t iter_idx = 0; iter_idx < n_iters; ++iter_idx) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> perm(k, 1);
    bool is_valid = true;
    for (size_t row_idx = 0; row_idx < k; ++row_idx) {
      perm[row_idx] = elements[static_cast<Eigen::Index>(iter_idx / divisors[row_idx]) % n];
      const auto& perm_curr_end = perm.begin() + row_idx;
      if (std::find(perm.begin(), perm_curr_end, perm[row_idx]) != perm_curr_end) {
        iter_idx += divisors[row_idx] - 1 - iter_idx % divisors[row_idx];
        is_valid = false;
        break;
      }
    }
    if (!is_valid) {
      continue;
    }
    perms.emplace_back(perm);
  }
  if (perms.size() != n_perms) {
    throw std::runtime_error("ComputePermutations did not result in the expected permutations!");
  }
}

template <typename T>
void ComputeCombinations(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& elements, const size_t& k,
                         std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>>& combs,
                         const size_t& combinations_threshold = std::numeric_limits<size_t>::max()) {
  const size_t n = elements.rows();
  if (k > n) {
    throw std::runtime_error("k cannot be greater than n");
  }

  const size_t n_combs = GetFactorial(n, std::max(k, n - k)) / GetFactorial(std::min(k, n - k));
  combs.reserve(n_combs);

  // Source: https://rosettacode.org/wiki/Combinations#C.2B.2B
  std::string bitmask(k, 1);  // k leading 1's
  bitmask.resize(n, 0);       // n-k trailing 0's

  // generate combinations from indices and permute bitmask
  do {
    Eigen::Matrix<T, Eigen::Dynamic, 1> comb(k, 1);
    Eigen::Index comb_idx = 0;
    for (size_t idx = 0; idx < n; ++idx) {
      if (!bitmask[idx]) {
        continue;
      }
      comb[comb_idx++] = elements[idx];
    }
    combs.emplace_back(comb);
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()) && combs.size() < combinations_threshold);
  if (combs.size() != std::min(n_combs, combinations_threshold)) {
    throw std::runtime_error("ComputeCombinations did not result in the expected combinations!");
  }
}

template <typename T, size_t Repeat>
void ComputeCartesianProduct(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& elements,
                             std::vector<Eigen::Matrix<T, Repeat, 1>>& products) {
  const size_t n = elements.rows();

  std::vector<size_t> divisors(Repeat);
  divisors[Repeat - 1] = 1;
  for (size_t row_idx = 1; row_idx < Repeat; ++row_idx) {
    divisors[Repeat - row_idx - 1] = divisors[Repeat - row_idx] * n;
  }

  const size_t n_products = divisors[0] * n;
  products.reserve(n_products);
  for (size_t col_idx = 0; col_idx < n_products; ++col_idx) {
    Eigen::Matrix<T, Repeat, 1> product;
    for (size_t row_idx = 0; row_idx < Repeat; ++row_idx) {
      product[row_idx] = elements[static_cast<Eigen::Index>(col_idx / divisors[row_idx]) % n];
    }
    products.emplace_back(product);
  }
  if (products.size() != n_products) {
    throw std::runtime_error("ComputeCartesianProduct did not result in the expected products!");
  }
}

void FitCone(const Eigen::Ref<const Eigen::Matrix3Xf>& unit_sphere_points, Eigen::Vector3f& cone_axis,
             float& cone_angle);

/*
 * Finds the nearest ray-sphere intersection point by computing ray traveled distances from the equation:
 *  ```
 *  || ray_start + distances * rays_dir - sphere_center, axis=1 || == sphere_radius
 *  ```
 * and then returns the intersection point, which is `ray_start + distances * rays_dir`
 */
void ComputeNearestIntersections(const Eigen::Vector3f& ray_start, const Eigen::Matrix3Xf& rays_dir,
                                 const Eigen::Vector3f& sphere_center, const float& sphere_radius,
                                 Eigen::Matrix3Xf& points_3d);

void FilterUniquePointsInplace(std::vector<Eigen::Vector3f>& points, float eps);

}  // namespace vision_common
