// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <memory>

#include "ace_loggers/ace_loggers.hpp"
#include "ball_detector/iball_detector_parameters.hpp"
#include "opencv2/opencv.hpp"

namespace ball_detector {

class IBallDetector {
 public:
  using SharedPtr = std::shared_ptr<IBallDetector>;
  using ConstSharedPtr = std::shared_ptr<const IBallDetector>;

  virtual ~IBallDetector() = default;

  virtual bool SetDataWriter(const ::datalogger::DataWriter::SharedPtr& datawriter_ptr) = 0;
  virtual bool SetParameters(const IBallDetectorParameters::SharedPtr& params_ptr) = 0;
  virtual bool SetFrameRate(const double& frame_rate) = 0;
  virtual bool SetCudaDeviceID(const int& cuda_device_id) = 0;
  virtual bool SetBayerImage(cv::InputArray& bayer_img) = 0;
  virtual bool SetBgrImage(cv::InputArray& bgr_img) = 0;
  virtual void GetBgrImage(cv::OutputArray& bgr_img) const = 0;
  virtual void DetectBalls(std::vector<std::pair<cv::Point2f, float>>& ball_detections,
                           cv::InputArray& valid_mask) const = 0;
  virtual void GetDetectionMask(cv::OutputArray& detection_mask) const = 0;
  [[nodiscard]] virtual const ::datalogger::DataWriter::SharedPtr& GetDataLogger() const = 0;
};

}  // namespace ball_detector
