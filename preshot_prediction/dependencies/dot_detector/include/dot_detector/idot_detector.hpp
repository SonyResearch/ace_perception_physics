// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <memory>

#include "dot_detector/idot_detector_parameters.hpp"
#include "opencv2/opencv.hpp"

namespace dot_detector {

class IDotDetector {
 public:
  using SharedPtr = std::shared_ptr<IDotDetector>;
  using ConstSharedPtr = std::shared_ptr<const IDotDetector>;

  virtual ~IDotDetector() = default;

  [[nodiscard]] virtual bool SetParameters(const IDotDetectorParameters::SharedPtr& params_ptr) = 0;
  virtual bool SetCudaDeviceID(const int& cuda_device_id) = 0;
  virtual bool SetBayerImage(cv::InputArray& bayer_img) = 0;
  virtual bool SetBgrImage(cv::InputArray& bgr_img) = 0;
  virtual bool SetGrayImage(cv::InputArray& gray_img) = 0;
  virtual void DetectMarkers(const cv::Point2f& ball_center, const float& ball_radius,
                             std::vector<std::pair<cv::RotatedRect, float>>& marker_detections) const = 0;
  virtual void GetDetectionMask(cv::OutputArray& detection_mask) const = 0;
};

}  // namespace dot_detector
