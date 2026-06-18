// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "dot_detector/dot_detector_opencv_parameters.hpp"
#include "dot_detector/idot_detector.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"

namespace dot_detector {

class DotDetectorOpenCV : public IDotDetector {
 public:
  using SharedPtr = std::shared_ptr<DotDetectorOpenCV>;
  using ConstSharedPtr = std::shared_ptr<const DotDetectorOpenCV>;

  explicit DotDetectorOpenCV(DotDetectorOpenCVParameters::SharedPtr opencv_params_ptr);
  ~DotDetectorOpenCV() override;

  [[nodiscard]] bool SetParameters(const IDotDetectorParameters::SharedPtr& params_ptr) override;
  bool SetCudaDeviceID(const int& cuda_device_id) override;
  bool SetBayerImage(cv::InputArray& bayer_img) override;
  bool SetBgrImage(cv::InputArray& bgr_img) override;
  bool SetGrayImage(cv::InputArray& gray_img) override;
  void DetectMarkers(const cv::Point2f& ball_center, const float& ball_radius,
                     std::vector<std::pair<cv::RotatedRect, float>>& marker_detections) const override;
  void GetDetectionMask(cv::OutputArray& detection_mask) const override;

 private:
  void EnsureInit(const cv::Size& img_size);
  static cv::RotatedRect FitEllipse(const std::vector<cv::Point2i>& points, const size_t& n_points_fit = 5);

  IDotDetectorParameters::SharedPtr params_ptr_;
  const DotDetectorOpenCVParameters::SharedPtr opencv_params_ptr_;

  float avg_radius_ratio_;
  float visibility_sin_;

  cv::Size img_size_;
  std::unique_ptr<cv::cuda::Stream> stream_ptr_;
  cv::cuda::GpuMat bayer_img_;
  cv::cuda::GpuMat bgr_img_;
  cv::cuda::GpuMat gray_img_;
  cv::Mat detection_mask_;
};

}  // namespace dot_detector
