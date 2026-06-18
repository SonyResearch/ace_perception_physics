// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once
#include <Eigen/Dense>
#include <list>
#include <mutex>
#include <string>

namespace ace_monitor::common_values {

struct CategoryNames {
  static const std::string kCamera;
  static const std::string kBallDetection;
  static const std::string kBallTriangulation;
  static const std::string kBallPoseEstimation;
  static const std::string kEstimatorTopic;
  static const std::string kEstimatorPredictor;

  static const std::string kRacket;
  static const std::string kPlayer;
};
class CameraDetection {
 public:
  int all_fps;
  int valid_fps;
  int markers_fps;
  bool ball_visible{false};
};
class CameraDetectionList {
 public:
  std::vector<std::string> names;
};
class BallDetection {
 public:
  double timestamp;
  size_t sequence_number;
  Eigen::Vector2f center;
  float radius;
};

class BallTriangulation {
 public:
  size_t timestamp;
  Eigen::Vector3f position;
  int num_cameras;
};
class BallPoseEstimation {
 public:
  class Estimation {
   public:
    Estimation() : position(0, 0, 0), velocity(0, 0, 0), spin(0, 0, 0), orientation(1, 0, 0, 0) {}
    double timestamp{0};
    size_t sequence_number{0};
    Eigen::Vector3f position;
    Eigen::Vector3f velocity;
    Eigen::Vector3f spin;
    Eigen::Quaternionf orientation;
    int num_cameras{0};
  };
  bool enabled{true};

  int all_fps{0};
  int valid_fps{0};
  std::list<Estimation> latest;
  std::list<Estimation> history;
  std::mutex history_mutex;
};

class NamesList {
 public:
  std::vector<std::string> names;
};

///
class PlayerDetection2D {
 public:
  double timestamp;
  size_t sequence_number;
  Eigen::Vector4i bbox;
  float confidence;
};
class PlayerPoseEstimation2D {
 public:
  double timestamp;
  size_t sequence_number;
  Eigen::Vector2i keypoints[17];
  float confidence[17];
};

class PlayerPoseEstimation3D {
 public:
  double timestamp;
  size_t sequence_number;
  Eigen::Vector3f keypoints[17];
  float projection_err[17];
  float confidence[17];
  bool valid[17];
};
class PlayerPoseEstimation3DList : public NamesList {
 public:
};

class RacketPoseEstimation3D {
 public:
  enum class RacketType { kVive, kEstimated, kOther };

  double timestamp;
  size_t sequence_number;
  Eigen::Vector3f position;
  Eigen::Quaternionf orientation;
  bool visible{true};
  bool enabled{true};
  float projection_error;
  RacketType type{RacketType::kOther};
};
class RacketPoseEstimation3DList : public NamesList {
 public:
};

class EstimatorPredictor {
 public:
  using PredictionVector = std::pair<double, std::vector<Eigen::Vector3f>>;
  PredictionVector predictions;
  bool enabled{true};
  std::mutex data_mutex;
};
class EstimatorTopic : public EstimatorPredictor {
 public:
  int all_fps;
};
class EstimatorPredictorList : public NamesList {
 public:
};
class EstimatorTopicsList : public EstimatorPredictorList {
 public:
};

}  // namespace ace_monitor::common_values
