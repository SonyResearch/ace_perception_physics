// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <ace_rt_profiles/rt_thread.hpp>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <thread>
#include <trt_ball_detector/ball_detector.hpp>

#include "ace_interfaces/msg/balls_with_attributes.hpp"
#include "ace_interfaces/msg/image_data.hpp"
#include "ace_interfaces/msg/logger_control.hpp"
#include "ball_detection_trt/ball_detection_parameters.hpp"
#include "calibration/camera_calibration_parameters.hpp"
#include "rclcpp/rclcpp.hpp"

namespace ball_detection_trt {
class BallDetectionTRT : public rclcpp::Node {
 public:
  explicit BallDetectionTRT(const rclcpp::NodeOptions& options);
  ~BallDetectionTRT() override;

 private:
  bool init_;
  std::shared_ptr<BallDetectionParameters> params_ptr_;
  std::vector<std::shared_ptr<trt_ball_detector::BallDetector>> ball_detectors_ptr_;
  const calibration::CameraCalibrationParameters::SharedPtr camera_calib_params_ptr_;
  std::vector<rclcpp::Subscription<ace_interfaces::msg::ImageData>::SharedPtr> subs_;
  rclcpp::Subscription<ace_interfaces::msg::LoggerControl>::SharedPtr loggercontrol_sub_;
  std::vector<rclcpp::Publisher<ace_interfaces::msg::BallsWithAttributes>::SharedPtr> pubs_;

  // per device
  class Message{
  public:
    size_t camera_index;
    ace_interfaces::msg::PerceptionHeader header;
  };
  struct QueueEntry {
    ace_interfaces::msg::ImageData::ConstSharedPtr message;
    unsigned int camera_index;
  };
  std::vector<std::atomic<bool>> detections_ready_;
  std::vector<std::vector<Message>> device_messages_;

  std::atomic<bool> canceled_;
  std::deque<QueueEntry> shared_queue_;
  std::mutex shared_queue_mutex_;
  std::condition_variable cv_shared_queue_;
  std::unique_ptr<std::mutex[]> publisher_mutexes_;
  std::unique_ptr<std::condition_variable[]> cv_publisher_;
  static constexpr std::chrono::milliseconds kMessagesWaitDuration{100};
  std::vector<ace_rt_profiles::RTThread> worker_threads_;
  std::vector<ace_rt_profiles::RTThread> publisher_threads_;


  void WorkerThread(unsigned int device_index);
  void PublishThread(unsigned int device_index);

  bool Initialize(const std::string& params_path, const std::string& camera_calib_params_path);
  void ProcessImage(const ace_interfaces::msg::ImageData::ConstSharedPtr& message, unsigned int camera_index);
  void LoggerControlCallback(const ace_interfaces::msg::LoggerControl::ConstSharedPtr& msg_ptr);
};
}  // namespace ball_detection_trt
