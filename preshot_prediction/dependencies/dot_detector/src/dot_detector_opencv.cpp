// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "dot_detector/dot_detector_opencv.hpp"

#include <cstddef>
#include <numeric>
#include <utility>

namespace dot_detector {

DotDetectorOpenCV::DotDetectorOpenCV(DotDetectorOpenCVParameters::SharedPtr opencv_params_ptr)
  : params_ptr_(nullptr), opencv_params_ptr_(std::move(opencv_params_ptr)) {}

DotDetectorOpenCV::~DotDetectorOpenCV() = default;

bool DotDetectorOpenCV::SetParameters(const IDotDetectorParameters::SharedPtr& params_ptr) {
  params_ptr_ = params_ptr;

  avg_radius_ratio_ = 0.5F * (params_ptr_->markers_min_radius_ratio + params_ptr_->markers_max_radius_ratio);
  visibility_sin_ = std::sin(params_ptr_->markers_cone_angle_threshold);

  return true;
}

bool DotDetectorOpenCV::SetCudaDeviceID(const int& cuda_device_id) {
  // Set CUDA device id
  cv::cuda::setDevice(cuda_device_id);
  stream_ptr_ = std::make_unique<cv::cuda::Stream>();

  return true;
}

bool DotDetectorOpenCV::SetBayerImage(cv::InputArray& bayer_img) {
  if (!params_ptr_->markers_enable) {
    return false;
  }
  EnsureInit(bayer_img.size());
  if (bayer_img.isGpuMat()) {
    bayer_img_ = bayer_img.getGpuMat();
  } else {
    bayer_img_.upload(bayer_img, *stream_ptr_);
  }
  cv::cuda::cvtColor(bayer_img_, gray_img_, cv::COLOR_BayerBG2GRAY, 0, *stream_ptr_);

  return true;
}

bool DotDetectorOpenCV::SetBgrImage(cv::InputArray& bgr_img) {
  if (!params_ptr_->markers_enable) {
    return false;
  }
  EnsureInit(bgr_img.size());
  if (bgr_img.isGpuMat()) {
    bgr_img_ = bgr_img.getGpuMat();
  } else {
    bgr_img_.upload(bgr_img, *stream_ptr_);
  }
  cv::cuda::cvtColor(bgr_img_, gray_img_, cv::COLOR_BGR2GRAY, 0, *stream_ptr_);

  return true;
}

bool DotDetectorOpenCV::SetGrayImage(cv::InputArray& gray_img) {
  if (!params_ptr_->markers_enable) {
    return false;
  }
  EnsureInit(gray_img.size());
  if (gray_img.isGpuMat()) {
    gray_img_ = gray_img.getGpuMat();
  } else {
    gray_img_.upload(gray_img, *stream_ptr_);
  }

  return true;
}

void DotDetectorOpenCV::DetectMarkers(const cv::Point2f& ball_center, const float& ball_radius,
                                      std::vector<std::pair<cv::RotatedRect, float>>& marker_detections) const {
  if (!params_ptr_->markers_enable) {
    return;
  }
  if (gray_img_.empty()) {
    throw std::runtime_error("Please call SetBayerImage(), SetBgrImage() or SetGrayImage() first");
  }

  const cv::Point2i ball_center_int(static_cast<int>(std::round(ball_center.x)),
                                    static_cast<int>(std::round(ball_center.y)));
  const auto ball_radius_extended = static_cast<int>(std::round(ball_radius) + 1);
  cv::Rect roi(std::max(0, ball_center_int.x - ball_radius_extended),
               std::max(0, ball_center_int.y - ball_radius_extended), 2 * ball_radius_extended,
               2 * ball_radius_extended);
  roi.width = std::min(roi.width, gray_img_.size().width - roi.x);
  roi.height = std::min(roi.height, gray_img_.size().height - roi.y);

  auto* this_unsafe = const_cast<DotDetectorOpenCV*>(this);

  // Download roi_img from GPU
  const cv::cuda::GpuMat roi_img_gpu(gray_img_, roi);
  cv::Mat roi_img;
  roi_img_gpu.download(roi_img, *this_unsafe->stream_ptr_);
  this_unsafe->stream_ptr_->waitForCompletion();

  const int adaptive_ksize = std::max(3, 1 + 2 * static_cast<int>(std::round(2 * ball_radius * avg_radius_ratio_) / 2));
  const float adaptive_threshold =
    static_cast<float>(std::pow(adaptive_ksize, opencv_params_ptr_->markers_opencv_adaptive_threshold_power)) *
    opencv_params_ptr_->markers_opencv_adaptive_threshold_ratio;

  if (detection_mask_.size() != roi.size()) {
    this_unsafe->detection_mask_.create(roi.height, roi.width, CV_8UC1);
  }
  cv::adaptiveThreshold(roi_img, detection_mask_, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                        adaptive_ksize, adaptive_threshold);

  // Extract markers
  std::vector<std::vector<cv::Point2i>> circle_contours;
  cv::findContours(detection_mask_, circle_contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  for (auto& contour : circle_contours) {
    cv::RotatedRect marker_rect = FitEllipse(contour);
    marker_rect.center.x += static_cast<float>(roi.x);
    marker_rect.center.y += static_cast<float>(roi.y);
    if (std::pow(marker_rect.center.x - ball_center.x, 2) + std::pow(marker_rect.center.y - ball_center.y, 2) >=
        std::pow(std::min(visibility_sin_ * ball_radius, ball_radius - 1), 2)) {
      continue;
    }

    const double circularity_num = 0.5 * marker_rect.size.width * marker_rect.size.height;
    const double circularity_denum = 0.25 * std::pow(marker_rect.size.width, 2) + std::pow(marker_rect.size.height, 2);
    const double circularity = circularity_num / circularity_denum;
    const double marker_radius = std::pow(circularity_num + circularity_denum, 0.5);
    const double radius_ratio = marker_radius / ball_radius;

    if (circularity < params_ptr_->markers_min_circularity_ratio ||
        radius_ratio < params_ptr_->markers_min_radius_ratio || radius_ratio > params_ptr_->markers_max_radius_ratio) {
      continue;
    }

    marker_detections.emplace_back(marker_rect, marker_radius);
  }
}

cv::RotatedRect DotDetectorOpenCV::FitEllipse(const std::vector<cv::Point2i>& points, const size_t& n_points_fit) {
  const size_t n_points = points.size();
  if (n_points <= 2) {
    cv::Point2f center = std::accumulate(points.begin(), points.end(), cv::Point2i(0, 0));
    center.x /= static_cast<float>(n_points);
    center.y /= static_cast<float>(n_points);
    const auto radius = static_cast<float>(std::pow(n_points, 0.5F));
    return cv::RotatedRect(center, cv::Size2f(radius, radius), 0);
  }

  if (n_points < std::max(static_cast<size_t>(5), n_points_fit)) {
    const cv::Moments moments = cv::moments(points);
    if (moments.m00 == 0) {
      cv::Point2f center = std::accumulate(points.begin(), points.end(), cv::Point2i(0, 0));
      center.x /= static_cast<float>(n_points);
      center.y /= static_cast<float>(n_points);
      return cv::RotatedRect(center, cv::Size2f(static_cast<float>(n_points), 0), 0);
    }
    const cv::Point2f center(static_cast<float>(moments.m10 / moments.m00),
                             static_cast<float>(moments.m01 / moments.m00));
    const float radius = std::pow(static_cast<float>(moments.m00 / M_PI), 0.5F);
    return cv::RotatedRect(center, cv::Size2f(radius, radius), 0);
  }

  // return cv::fitEllipse(points);
  return cv::minAreaRect(points);
}

void DotDetectorOpenCV::EnsureInit(const cv::Size& img_size) {
  // Allocate memory if current buffers are not compatible
  if (img_size_ == img_size) {
    return;
  }
  img_size_ = img_size;

  // Allocate GPU memory.
  bayer_img_.create(img_size_.height, img_size_.width, CV_8UC1);
  bgr_img_.create(img_size_.height, img_size_.width, CV_8UC3);
  gray_img_.create(img_size_.height, img_size_.width, CV_8UC1);
}

void DotDetectorOpenCV::GetDetectionMask(cv::OutputArray& detection_mask) const {
  detection_mask.getMatRef() = detection_mask_;
}

}  // namespace dot_detector
