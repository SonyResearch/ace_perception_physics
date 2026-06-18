// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <ace_rt_profiles/rt_sub_profile.hpp>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

#include "ace_interfaces/msg/image_data.hpp"
#include "ace_interfaces/msg/statistics.hpp"
#include "ace_interfaces/srv/trigger.hpp"
#include "aps/blackfly_s_camera.hpp"
#include "aps/blackfly_s_interface_event_handler.hpp"
#include "aps/camera_parameters.hpp"
#include "aps/multi_camera_parameters.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

namespace aps {

class MultiCamera : public rclcpp::Node {
 public:
  explicit MultiCamera(const rclcpp::NodeOptions& options);
  ~MultiCamera() override;

 private:
  using clock = std::chrono::high_resolution_clock;

  // general
  bool init_;
  static constexpr std::string_view kUSBFSMemorySizeFile = "/sys/module/usbcore/parameters/usbfs_memory_mb";
  static constexpr int kMinUSBFSMemorySize = 1000;  // [MB]
  bool Initialize();

  // camera
  Spinnaker::SystemPtr system_ptr_;
  BlackflySInterfaceEventHandler interface_event_handler_;
  MultiCameraParameters::SharedPtr params_;
  CameraParameters::SharedPtr camera_params_;
  std::unordered_map<std::string, BlackflySCamera::SharedPtr> cams_;
  double frame_rate_;
  double sequence_number_to_ns_;

  // camera timing check
  uint64_t frame_interval_ns_;
  std::vector<uint64_t> previous_seq_nums_;
  std::vector<uint64_t> previous_ts_;
  static constexpr double kTimingCheckStartPeriod = 0.5;    // [s]
  static constexpr uint64_t kTimingMaxDeviationNs = 10000;  // [ns]
  uint64_t timing_check_start_seq_num_;

  // frames skipping to reduce sync issues
  static constexpr double kCallbackTimeRatio = 0.4;
  uint64_t callback_max_time_us_;
  std::vector<uint64_t> frames_to_skip_;

  // statistics on messages sync (via sequence number)
  uint64_t stats_print_rate_;
  std::mutex counter_mutex_;
  std::vector<uint32_t> state_counters_;
  void UpdateAndPrintStatistics(const size_t& camera_idx, const uint64_t& curr_seq_num, const uint64_t& curr_timestamp);

  bool InitializeCamera(const std::string& serial_number, bool is_master,
                        const ace_rt_profiles::RTSubProfile& rt_subprofile);
  bool SetAcqusitionParameters(BlackflySCamera::SharedPtr& cam_ptr, bool is_master);
  bool StartAcquisition();
  bool StopAcquisition();
  bool ResetDevices();

  // publishers and services
  std::vector<rclcpp::Publisher<ace_interfaces::msg::ImageData>::SharedPtr> pubs_;
  std::vector<rclcpp::Publisher<ace_interfaces::msg::Statistics>::SharedPtr> pubs_stats_;
  std::vector<std::pair<ace_interfaces::msg::Statistics, clock::time_point>> stats_;
  rclcpp::Service<ace_interfaces::srv::Trigger>::SharedPtr trigger_service_;
  clock::time_point trigger_time_;

  // Sync state
  std_msgs::msg::Float32 sync_msg_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr sync_pub_;
  uint64_t stats_pub_rate_;

  // EtherCAT subscriber
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr ecat_sync_sub_;
  clock::time_point ecat_restart_time_;

  void DeviceArrivalCallback(const std::string& serial_number);
  void DeviceRemovalCallback(const std::string& serial_number);
  void DeviceCallback(const size_t& camera_idx, const std::string& event_name);
  void ImageCallback(const size_t& camera_idx, Spinnaker::ImagePtr& image_ptr);
  void TriggerService(ace_interfaces::srv::Trigger::Request::SharedPtr request,
                      ace_interfaces::srv::Trigger::Response::SharedPtr response);
};

}  // namespace aps
