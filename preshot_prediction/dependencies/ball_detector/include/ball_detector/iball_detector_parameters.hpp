// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "eigen3/Eigen/Eigen"
#include "opencv2/opencv.hpp"

namespace ball_detector {

class IBallDetectorParameters {
 public:
  using SharedPtr = std::shared_ptr<IBallDetectorParameters>;
  using ConstSharedPtr = std::shared_ptr<const IBallDetectorParameters>;

  virtual ~IBallDetectorParameters() = default;

  // color filter
  int blur_kernel_size;
  cv::Scalar hsv_lower_boundary;
  cv::Scalar hsv_upper_boundary;

  // motion filter
  bool motion_filter_enable;
  double motion_filter_delay;
  int motion_filter_lower_boundary;
  int motion_filter_upper_boundary;

  // appearance filter
  double min_circularity_ratio;
  double min_radius;
};

}  // namespace ball_detector
