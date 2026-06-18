// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once
#include <opencv2/opencv.hpp>

#include "racket_pose_estimation/RacketFeatures.hpp"
#include "racket_pose_estimation/RacketParameters.hpp"

namespace perception {

class RacketInference {
 public:
  using SharedPtr = std::shared_ptr<RacketInference>;
  using ConstSharedPtr = std::shared_ptr<const RacketInference>;

 private:
  class RacketInferenceImpl;
  std::shared_ptr<RacketInferenceImpl> impl_;
  bool ParseResults(const float* output, RacketPoseFeatures::SharedPtr& results);

 public:
  RacketInference();
  bool Initialize(const std::string& model_path, RacketParameters::SharedPtr racket_params);

  bool DetectRacketKeypoints(const std::vector<cv::cuda::GpuMat>& inputs, RacketFrameFeatures& results);
  virtual cv::cuda::GpuMat PreProcessImage(const cv::Mat& bayer);
  const Eigen::Vector3i& GetInputSize();
};

}  // namespace perception
