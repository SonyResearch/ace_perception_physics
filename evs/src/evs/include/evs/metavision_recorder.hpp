// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "evs/camera_parameters.hpp"
#include "evs/multi_camera_parameters.hpp"
#include "metavision/hal/device/device.h"
#include "rclcpp/rclcpp.hpp"

namespace evs {

class MetavisionRecorder : public rclcpp::Node {
 public:
  explicit MetavisionRecorder(const rclcpp::NodeOptions& options);
  ~MetavisionRecorder() override;

 private:
  // General
  bool init_;
  bool Initialize(const std::string& multi_camera_params_path, const std::string& camera_params_path,
                  const std::string& name);
  std::string recording_path_;

  // EVS cameras (TODO: change this to EVS abstraction layer once ready)
  MultiCameraParameters::SharedPtr multi_camera_params_;
  CameraParameters::SharedPtr camera_params_;
  std::unordered_map<std::string, std::unique_ptr<Metavision::Device>> cams_;
  std::atomic<bool> canceled_;
  std::vector<std::thread> cam_threads_;

  bool InitializeCamera(const std::string& serial_number, bool is_master = false);
  void StartRecording();
  void StopRecording();
};

}  // namespace evs
