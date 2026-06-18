// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "ball_detector/iball_detector.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"

namespace ball_detector {

class BallDetectorOpenCV : public IBallDetector {
 public:
  using SharedPtr = std::shared_ptr<BallDetectorOpenCV>;
  using ConstSharedPtr = std::shared_ptr<const BallDetectorOpenCV>;

  explicit BallDetectorOpenCV();
  ~BallDetectorOpenCV() override = default;

  bool SetDataWriter(const ::datalogger::DataWriter::SharedPtr& datawriter_ptr) override;
  bool SetParameters(const IBallDetectorParameters::SharedPtr& params_ptr) override;
  bool SetFrameRate(const double& frame_rate) override;
  bool SetCudaDeviceID(const int& cuda_device_id) override;
  bool SetBayerImage(cv::InputArray& bayer_img) override;
  bool SetBgrImage(cv::InputArray& bgr_img) override;
  void GetBgrImage(cv::OutputArray& bgr_img) const override;
  void DetectBalls(std::vector<std::pair<cv::Point2f, float>>& ball_detections,
                   cv::InputArray& valid_mask) const override;

  void ExtractBallsFromMask(std::vector<std::pair<cv::Point2f, float>>& ball_detections,
                            cv::InputArray& valid_mask) const;

  void SetDetectionMask(const cv::Mat& mask_img);
  void GetDetectionMask(cv::OutputArray& detection_mask) const override;
  [[nodiscard]] const ::datalogger::DataWriter::SharedPtr& GetDataLogger() const override;

 private:
  void EnsureInit(const cv::Size& img_size);

  IBallDetectorParameters::SharedPtr params_ptr_;

  cv::Size img_size_;
  double frame_rate_{0};
  std::unique_ptr<cv::cuda::Stream> stream_ptr_;
  cv::cuda::GpuMat bayer_img_;
  cv::cuda::GpuMat bgr_img_;
  cv::cuda::GpuMat blur_img_;
  cv::cuda::GpuMat hsv_img_;
  cv::cuda::GpuMat hsv_channel_[3];
  cv::cuda::GpuMat gray_img_;
  cv::cuda::GpuMat tmp_img_;
  cv::cuda::GpuMat color_mask_;
  cv::cuda::GpuMat motion_mask_;
  cv::cuda::GpuMat combined_mask_;
  cv::Mat detection_mask_;
  int motion_filter_length_{0};
  std::unique_ptr<cv::cuda::GpuMat[]> circular_buffer_;
  int circular_buffer_head_{0};
  int circular_buffer_tail_{0};
  cv::Ptr<cv::cuda::Filter> gaussian_ptr_;

  ::datalogger::DataWriter::SharedPtr datawriter_ptr_ = nullptr;
};

}  // namespace ball_detector
