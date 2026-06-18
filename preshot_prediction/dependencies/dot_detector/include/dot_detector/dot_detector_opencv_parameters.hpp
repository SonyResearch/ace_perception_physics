// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <memory>

namespace dot_detector {

class DotDetectorOpenCVParameters {
 public:
  using SharedPtr = std::shared_ptr<DotDetectorOpenCVParameters>;
  using ConstSharedPtr = std::shared_ptr<const DotDetectorOpenCVParameters>;

  virtual ~DotDetectorOpenCVParameters() = default;

  // adaptive-thresholding parameters
  float markers_opencv_adaptive_threshold_power;
  float markers_opencv_adaptive_threshold_ratio;
};

}  // namespace dot_detector
