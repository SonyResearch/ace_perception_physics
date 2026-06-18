// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ball_detector/ball_detector_opencv.hpp"

#include <memory>
#include <stdexcept>

namespace ball_detector {

BallDetectorOpenCV::BallDetectorOpenCV() : img_size_(cv::Size(0, 0)) {}

bool BallDetectorOpenCV::SetDataWriter(const ::datalogger::DataWriter::SharedPtr& datawriter_ptr) {
  datawriter_ptr_ = datawriter_ptr;

  return true;
}

bool BallDetectorOpenCV::SetParameters(const IBallDetectorParameters::SharedPtr& params_ptr) {
  params_ptr_ = params_ptr;

  return true;
}

bool BallDetectorOpenCV::SetFrameRate(const double& frame_rate) {
  frame_rate_ = frame_rate;

  return true;
}

bool BallDetectorOpenCV::SetCudaDeviceID(const int& cuda_device_id) {
  // Set CUDA device id
  cv::cuda::setDevice(cuda_device_id);
  stream_ptr_ = std::make_unique<cv::cuda::Stream>();

  return true;
}

bool BallDetectorOpenCV::SetBayerImage(cv::InputArray& bayer_img) {
  EnsureInit(bayer_img.size());
  if (bayer_img.isGpuMat()) {
    bayer_img_ = bayer_img.getGpuMat();
  } else {
    bayer_img_.upload(bayer_img, *stream_ptr_);
  }
  cv::cuda::cvtColor(bayer_img_, bgr_img_, cv::COLOR_BayerBG2BGR, 0, *stream_ptr_);

  return true;
}

bool BallDetectorOpenCV::SetBgrImage(cv::InputArray& bgr_img) {
  EnsureInit(bgr_img.size());
  if (bgr_img.isGpuMat()) {
    bgr_img_ = bgr_img.getGpuMat();
  } else {
    bgr_img_.upload(bgr_img, *stream_ptr_);
  }

  return true;
}

void BallDetectorOpenCV::GetBgrImage(cv::OutputArray& bgr_img) const { bgr_img.getGpuMatRef() = bgr_img_; }

void BallDetectorOpenCV::DetectBalls(std::vector<std::pair<cv::Point2f, float>>& ball_detections,
                                     cv::InputArray& valid_mask) const {
  auto logger_hook = ::datalogger::defer([&] {
    datawriter_ptr_ << std::chrono::high_resolution_clock::now() << ball_detections.size();
    const Eigen::Vector2f img_size(bayer_img_.cols, bayer_img_.rows);
    for (const auto& [ball_center, ball_radius] : ball_detections) {
      const Eigen::Vector2f center(ball_center.x, ball_center.y);
      const float extended_radius = 1.1F * ball_radius;
      const auto min_corner = ((center.array() - extended_radius).cwiseMax(0).cast<int>() / 2) * 2;
      const auto max_corner = ((center.array() + extended_radius).cwiseMin(img_size.array()).cast<int>() / 2) * 2;
      const cv::Rect img_roi(min_corner[0], min_corner[1], max_corner[0] - min_corner[0],
                             max_corner[1] - min_corner[1]);
      datawriter_ptr_ << img_roi << static_cast<cv::Mat>(bayer_img_(img_roi)) << ball_center << ball_radius;
    }
  });

  if (bgr_img_.empty()) {
    throw std::runtime_error("Please call SetBayerImage() or SetBgrImage() first");
  }
  if (!valid_mask.empty() && valid_mask.size() != bgr_img_.size()) {
    throw std::runtime_error("Mask and image must have the same 2D size");
  }
  auto* this_unsafe = const_cast<BallDetectorOpenCV*>(this);

  gaussian_ptr_->apply(bgr_img_, blur_img_, *this_unsafe->stream_ptr_);

  // Compute color mask.
  cv::cuda::cvtColor(blur_img_, hsv_img_, cv::COLOR_BGR2HSV, 3, *this_unsafe->stream_ptr_);
  cv::cuda::split(hsv_img_, this_unsafe->hsv_channel_, *this_unsafe->stream_ptr_);

  for (int channel_idx = 0; channel_idx < 3; ++channel_idx) {
    cv::cuda::threshold(hsv_channel_[channel_idx], tmp_img_, params_ptr_->hsv_lower_boundary[channel_idx], 255,
                        cv::THRESH_BINARY, *this_unsafe->stream_ptr_);
    cv::cuda::threshold(hsv_channel_[channel_idx], hsv_channel_[channel_idx],
                        params_ptr_->hsv_upper_boundary[channel_idx], 255, cv::THRESH_BINARY_INV,
                        *this_unsafe->stream_ptr_);
    cv::cuda::bitwise_and(hsv_channel_[channel_idx], tmp_img_, hsv_channel_[channel_idx], cv::noArray(),
                          *this_unsafe->stream_ptr_);
  }
  // Combine masks
  cv::cuda::bitwise_and(hsv_channel_[1], hsv_channel_[2], hsv_channel_[1], cv::noArray(), *this_unsafe->stream_ptr_);
  cv::cuda::bitwise_and(hsv_channel_[0], hsv_channel_[1], color_mask_, cv::noArray(), *this_unsafe->stream_ptr_);

  // Compute motion mask and combine with color mask.
  if (params_ptr_->motion_filter_enable) {
    cv::cuda::cvtColor(blur_img_, circular_buffer_[circular_buffer_head_], cv::COLOR_BGR2GRAY, 1,
                       *this_unsafe->stream_ptr_);

    // Update head and tail of circular buffer.
    this_unsafe->circular_buffer_head_ = (circular_buffer_head_ + 1) % motion_filter_length_;
    if (circular_buffer_head_ == circular_buffer_tail_) {
      this_unsafe->circular_buffer_tail_ = (circular_buffer_tail_ + 1) % motion_filter_length_;
    }

    cv::cuda::absdiff(circular_buffer_[circular_buffer_tail_],
                      circular_buffer_[(circular_buffer_head_ + motion_filter_length_ - 1) % motion_filter_length_],
                      tmp_img_, *this_unsafe->stream_ptr_);
    cv::cuda::threshold(tmp_img_, motion_mask_, params_ptr_->motion_filter_lower_boundary, 255, cv::THRESH_BINARY,
                        *this_unsafe->stream_ptr_);

    cv::cuda::bitwise_and(color_mask_, motion_mask_, combined_mask_, cv::noArray(), *this_unsafe->stream_ptr_);
  }

  // Download mask from GPU.
  if (params_ptr_->motion_filter_enable) {
    combined_mask_.download(detection_mask_, *this_unsafe->stream_ptr_);
  } else {
    color_mask_.download(detection_mask_, *this_unsafe->stream_ptr_);
  }
  this_unsafe->stream_ptr_->waitForCompletion();
  ExtractBallsFromMask(ball_detections, valid_mask);
}
void BallDetectorOpenCV::ExtractBallsFromMask(std::vector<std::pair<cv::Point2f, float>>& ball_detections,
                                              cv::InputArray& valid_mask) const {
  // Extract ball positions.
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(detection_mask_, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  std::vector<std::pair<cv::RotatedRect, float>> markers;
  for (auto& contour : contours) {
    double U = cv::arcLength(contour, true);
    double A = cv::contourArea(contour, false);

    double circularity = 4.0 * M_PI * A / std::pow(U, 2);
    if (circularity >= params_ptr_->min_circularity_ratio) {
      cv::Point2f ball_center;
      float ball_radius;
      cv::minEnclosingCircle(contour, ball_center, ball_radius);

      if (ball_radius > params_ptr_->min_radius &&
          (valid_mask.empty() || valid_mask.getMat().at<uint8_t>(static_cast<cv::Point2i>(ball_center)) > 0)) {
        ball_detections.emplace_back(ball_center, ball_radius);
      }
    }
  }
}

void BallDetectorOpenCV::EnsureInit(const cv::Size& img_size) {
  // Allocate memory if current buffers are not compatible
  if (img_size_ == img_size) {
    return;
  }

  if (params_ptr_->motion_filter_enable && frame_rate_ == 0) {
    throw std::runtime_error("Please call SetFrameRate() first");
  }
  img_size_ = img_size;

  // Allocate GPU memory.
  bayer_img_.create(img_size_.height, img_size_.width, CV_8UC1);
  bgr_img_.create(img_size_.height, img_size_.width, CV_8UC3);
  blur_img_.create(img_size_.height, img_size_.width, CV_8UC3);
  hsv_img_.create(img_size_.height, img_size_.width, CV_8UC3);
  for (auto& channel : hsv_channel_) {
    channel.create(img_size_.height, img_size_.width, CV_8UC1);
  }
  gray_img_.create(img_size_.height, img_size_.width, CV_8UC1);
  tmp_img_.create(img_size_.height, img_size_.width, CV_8UC1);
  color_mask_.create(img_size_.height, img_size_.width, CV_8UC1);
  motion_mask_.create(img_size_.height, img_size_.width, CV_8UC3);
  combined_mask_.create(img_size_.height, img_size_.width, CV_8UC3);

  if (params_ptr_->motion_filter_enable) {
    motion_filter_length_ = static_cast<int>(params_ptr_->motion_filter_delay * frame_rate_ + 1.0);
    circular_buffer_ = std::make_unique<cv::cuda::GpuMat[]>(motion_filter_length_);
  }
  gaussian_ptr_ = cv::cuda::createGaussianFilter(
    CV_8UC3, CV_8UC3, cv::Size(params_ptr_->blur_kernel_size, params_ptr_->blur_kernel_size), 0.0);

  // Allocate CPU memory.
  detection_mask_.create(img_size_.height, img_size_.width, CV_8UC1);
}

void BallDetectorOpenCV::SetDetectionMask(const cv::Mat& mask_img) { detection_mask_ = mask_img; }
void BallDetectorOpenCV::GetDetectionMask(cv::OutputArray& detection_mask) const {
  detection_mask.getMatRef() = detection_mask_;
}

[[nodiscard]] const ::datalogger::DataWriter::SharedPtr& BallDetectorOpenCV::GetDataLogger() const {
  return datawriter_ptr_;
}

}  // namespace ball_detector
