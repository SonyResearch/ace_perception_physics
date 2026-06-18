// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "eigen3/Eigen/Eigen"
#include "opencv2/opencv.hpp"

namespace trt_ball_detector {

class BallDetectorParameters {
 public:
  using SharedPtr = std::shared_ptr<BallDetectorParameters>;
  using ConstSharedPtr = std::shared_ptr<const BallDetectorParameters>;

  virtual ~BallDetectorParameters() = default;

  std::string onnx_engine_path;

  int device_id{0};
  int batch_size{1};

  bool copy_heatmap{false};

};

}  // namespace trt_ball_detector
