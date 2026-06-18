// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "trt_ball_detector/ball_detector_parameters.hpp"

namespace trt_ball_detector {

class DecodingResult {
  // detection per image
 public:
  using SharedPtr = std::shared_ptr<DecodingResult>;

  DecodingResult() = default;
  ~DecodingResult() = default;


  void reset() {
    sequence_id = 0;
    confidences.clear();
    ball_positions.clear();
    ball_velocities.clear();
    radius.clear();
    heatmap.release();
  }

  uint64_t sequence_id{0};
  std::vector<float> confidences;
  std::vector<Eigen::Vector2f> ball_positions;   // x, y
  std::vector<Eigen::Vector2f> ball_velocities;  // vx, vy
  std::vector<float> radius;  // radius
  cv::Mat heatmap;                               // H×W single-channel float32
};

class BallDetector {
 public:
  using SharedPtr = std::shared_ptr<BallDetector>;

  BallDetector();
  ~BallDetector() = default;

  bool Initialize(BallDetectorParameters::SharedPtr parameters);

  void ResetEncodedImages();

  // Encode a batch of images that belongs to the same timestamp
  bool EncodeImages(uint64_t sequence_id, const std::vector<cv::Mat>& images);

  // This will block until the GPU finishes processing the most recently enqueued batch (if any)
  // and return the decoding results for that batch.
  std::vector<DecodingResult::SharedPtr> GetDecodingResults();

 private:
  class BallDetectorImpl;
  std::shared_ptr<BallDetectorImpl> impl_;
};

}  // namespace trt_ball_detector
