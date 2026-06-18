// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <memory>

namespace dot_detector {

class IDotDetectorParameters {
 public:
  using SharedPtr = std::shared_ptr<IDotDetectorParameters>;
  using ConstSharedPtr = std::shared_ptr<const IDotDetectorParameters>;

  virtual ~IDotDetectorParameters() = default;

  // Is detector enabled
  bool markers_enable;

  // Minimum circularity of acceptable markers
  float markers_min_circularity_ratio;

  // Minimum and maximum marker radius, relative to detected ball
  float markers_min_radius_ratio;
  float markers_max_radius_ratio;

  // Ball-edge visibility constraint
  float markers_cone_angle_threshold;
};

}  // namespace dot_detector
