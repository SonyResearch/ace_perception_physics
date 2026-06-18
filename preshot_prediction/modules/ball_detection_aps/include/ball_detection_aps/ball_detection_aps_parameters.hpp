// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "ball_detector/iball_detector_parameters.hpp"
#include "dot_detector/dot_detector_opencv_parameters.hpp"
#include "dot_detector/idot_detector_parameters.hpp"
#include "eigen3/Eigen/Eigen"
#include "opencv2/opencv.hpp"
#include "vision_common/ball_detection_parameters.hpp"

namespace ball_detection_aps {

class BallDetectionAPSParameters final : public vision_common::BallDetectionParameters,
                                         public ball_detector::IBallDetectorParameters,
                                         public dot_detector::IDotDetectorParameters,
                                         public dot_detector::DotDetectorOpenCVParameters {
 public:
  using SharedPtr = std::shared_ptr<BallDetectionAPSParameters>;
  using ConstSharedPtr = std::shared_ptr<const BallDetectionAPSParameters>;

  // cuda device UUID
  std::vector<std::string> cuda_device_uuids;

  // border filter
  bool border_filter_enable;
  double border_filter_plane_depth;
  double border_filter_margin_size;

  std::string rt_profile;

 private:
  // auxiliary functions
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace ball_detection_aps
