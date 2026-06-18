// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "ace_interfaces/msg/balls_with_attributes.hpp"
#include "ace_interfaces/msg/logger_control.hpp"
#include "ace_interfaces/msg/points_with_covariance.hpp"
#include "calibration/camera_calibration_parameters.hpp"
#include "dot_projector/idot_projector.hpp"
#include "eigen3/Eigen/Eigen"
#include "multi_ball_triangulation/multi_ball_triangulation_parameters.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"
#include "triangulation/triangulation.hpp"
#include "vision_common/ball_detection_parameters.hpp"
#include "vision_common/concurrent_circular_buffer.hpp"
#include "vision_common/synced_circular_buffers.hpp"

namespace multi_ball_triangulation {

[[nodiscard]] inline uint64_t GetSequenceNumber(
  const ace_interfaces::msg::BallsWithAttributes::ConstSharedPtr& msg_ptr) {
  return msg_ptr->header.sequence_number;
}

class MultiBallTriangulation : public rclcpp::Node {
 public:
  explicit MultiBallTriangulation(const rclcpp::NodeOptions& options);
  ~MultiBallTriangulation() override;

 private:
  using BallsWithAttributes = ace_interfaces::msg::BallsWithAttributes;
  using CameraBuffers =
    utils::SyncedCircularBuffers<ace_interfaces::msg::BallsWithAttributes::ConstSharedPtr, uint64_t, GetSequenceNumber>;
  using WorkerQueue = utils::ConcurrentCircularBuffer<CameraBuffers::GroupsOfSyncedItems>;

  bool init_;
  bool Initialize(const std::string& params_path, const std::vector<std::string>& ball_detection_params_paths,
                  const std::string& camera_calib_params_path);

  bool IsPointInVOI(const triangulation::TriangulatedPoint& triangulated_point);

  const MultiBallTriangulationParameters::SharedPtr params_ptr_;
  calibration::CameraCalibrationParameters::SharedPtr camera_calib_params_ptr_;
  std::vector<std::string> camera_names_;

  // Triangulator.
  int num_cameras_;
  std::vector<size_t> camera_to_calib_indices_;
  std::unique_ptr<triangulation::Triangulation<Eigen::Vector3f>> triangulator_ptr_;
  void TriangulateBallPositions(const CameraBuffers::GroupsOfSyncedItems::mapped_type& detections,
                                ace_interfaces::msg::PointsWithCovariance& message);

  std::unique_ptr<dot_projector::IDotProjector> dot_projector_ptr_;

  // Producer-consumer queues
  std::unique_ptr<CameraBuffers> camera_buffers_ptr_;
  std::unique_ptr<WorkerQueue> worker_queue_ptr_;

  std::thread status_thread_;
  std::thread worker_thread_;
  std::atomic<bool> canceled_;
  void WorkerThread();

  mutable std::mutex reload_mutex_;
  mutable std::mutex mutex_;
  std::condition_variable reload_;
  std::thread reload_thread_;
  std::atomic<bool> reload_needed_;
  static constexpr std::chrono::milliseconds kReloadTimeout{100};
  void ReloadCalibrationThread();

  // Publisher and subscribers.
  rclcpp::Publisher<ace_interfaces::msg::PointsWithCovariance>::SharedPtr pub_;

  std::vector<rclcpp::Subscription<ace_interfaces::msg::BallsWithAttributes>::SharedPtr> subs_;
  void BallDetectionCallback(const int& camera_index,
                             const ace_interfaces::msg::BallsWithAttributes::ConstSharedPtr& msg_ptr);

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr calib_reload_sub_;
  void ReloadCalibrationCallback(const std_msgs::msg::Bool::ConstSharedPtr& msg_ptr);

  rclcpp::Subscription<ace_interfaces::msg::LoggerControl>::SharedPtr loggercontrol_sub_;
  void LoggerControlCallback(const ace_interfaces::msg::LoggerControl::ConstSharedPtr& msg_ptr);

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr execution_mode_sub_;

  // Statistics
  static constexpr std::chrono::seconds kStatusReportInterval{5};
  std::atomic<size_t> total_triangulations_counter_{0};
  std::atomic<size_t> complete_triangulations_counter_{0};
  std::atomic<size_t> limited_triangulations_counter_{0};
  std::atomic<size_t> zero_triangulations_counter_{0};
  std::atomic<size_t> outdated_detections_counter_{0};
  std::atomic<size_t> overwritten_detections_counter_{0};
  std::atomic<size_t> timeout_counter_{0};
  void ReportStatus();
};

}  // namespace multi_ball_triangulation
