// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <ace_rt_profiles/rt_thread.hpp>
#include <atomic>
#include <condition_variable>
#include <thread>

#include "ace_interfaces/msg/balls_with_attributes.hpp"
#include "ace_interfaces/msg/image_data.hpp"
#include "ace_interfaces/msg/logger_control.hpp"
#include "ace_interfaces/srv/image_data.hpp"
#include "ball_detection_aps/ball_detection_aps_parameters.hpp"
#include "ball_detector/iball_detector.hpp"
#include "calibration/camera_calibration_parameters.hpp"
#include "dot_detector/idot_detector.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"

namespace ball_detection_aps {

enum class ImageType : uint8_t {
  kRaw,
  kDetection,
  kValidMask,
};

class BallDetectionAPS : public rclcpp::Node {
 public:
  explicit BallDetectionAPS(const rclcpp::NodeOptions& options);
  ~BallDetectionAPS() override;

  enum class DebugMode : uint8_t {
    kOff,
    kDetection,
    kValidMask,
  };

  enum class ParameterID : uint8_t {
    kHLow = 0,
    kHHigh,
    kSLow,
    kSHigh,
    kVLow,
    kVHigh,
  };
  bool UpdateParameterValue(ParameterID parameter_id, int value);

 private:
  bool init_;
  bool Initialize(const std::string& params_path, const std::string& camera_calib_params_path, const std::string& debug,
                  const bool& ignore_gpu_checks);

  std::vector<std::unique_ptr<ball_detector::IBallDetector>> ball_detector_ptrs_;
  std::vector<std::unique_ptr<dot_detector::IDotDetector>> dot_detector_ptrs_;
  const BallDetectionAPSParameters::SharedPtr params_ptr_;
  const calibration::CameraCalibrationParameters::SharedPtr camera_calib_params_ptr_;
  std::vector<cv::Mat> valid_masks_;

  Eigen::Matrix<float, 3, 8> voi_filter_corners_;

  std::vector<rclcpp::Subscription<ace_interfaces::msg::ImageData>::SharedPtr> subs_;
  void ProcessImage(const ace_interfaces::msg::ImageData::ConstSharedPtr& message, unsigned int camera_index);

  rclcpp::Subscription<ace_interfaces::msg::LoggerControl>::SharedPtr loggercontrol_sub_;
  void LoggerControlCallback(const ace_interfaces::msg::LoggerControl::ConstSharedPtr& msg_ptr);

  std::vector<rclcpp::Publisher<ace_interfaces::msg::BallsWithAttributes>::SharedPtr> pubs_;

  // Worker thread.
  std::atomic<bool> canceled_;
  std::unique_ptr<std::mutex[]> mutexes_;
  std::unique_ptr<std::condition_variable[]> condition_variable_messages_available_;
  static constexpr std::chrono::milliseconds kMessagesWaitDuration{100};
  std::vector<bool> messages_available_;
  std::vector<ace_interfaces::msg::ImageData::ConstSharedPtr> messages_;
  std::vector<ace_rt_profiles::RTThread> threads_;
  void WorkerThread(unsigned int camera_index, int cuda_device_id);

  // Debug output.
  DebugMode debug_;
  rclcpp::TimerBase::SharedPtr debug_timer_;
  const std::string debug_window_name_ = "Ball Detection Parameters";
  std::vector<rclcpp::Publisher<ace_interfaces::msg::ImageData>::SharedPtr> debug_pubs_;
  void DebugCallback();

  rclcpp::Service<ace_interfaces::srv::ImageData>::SharedPtr image_data_service_;
  void ImageDataService(ace_interfaces::srv::ImageData::Request::SharedPtr request,
                        ace_interfaces::srv::ImageData::Response::SharedPtr response);
};

}  // namespace ball_detection_aps
