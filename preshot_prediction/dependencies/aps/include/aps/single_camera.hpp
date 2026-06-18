// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <memory>
#include <string>

#include "ace_interfaces/msg/image_data.hpp"
#include "aps/blackfly_s_camera.hpp"
#include "aps/blackfly_s_interface_event_handler.hpp"
#include "aps/camera_parameters.hpp"
#include "rclcpp/rclcpp.hpp"

namespace aps {

class SingleCamera : public rclcpp::Node {
 public:
  explicit SingleCamera(const rclcpp::NodeOptions& options);
  ~SingleCamera() override;

 private:
  // general
  bool init_;
  static constexpr std::string_view kUSBFSMemorySizeFile = "/sys/module/usbcore/parameters/usbfs_memory_mb";
  static constexpr int kMinUSBFSMemorySize = 1000;
  bool Initialize();

  // camera
  Spinnaker::SystemPtr system_ptr_;
  BlackflySInterfaceEventHandler interface_event_handler_;
  CameraParameters::SharedPtr camera_params_;
  const std::string serial_number_;
  BlackflySCamera::SharedPtr cam_;
  double frame_rate_;
  double sequence_number_to_ns_;

  bool StartAcquisition();
  bool StopAcquisition();

  // publisher
  rclcpp::Publisher<ace_interfaces::msg::ImageData>::SharedPtr pub_;
  void DeviceArrivalCallback(const std::string& serial_number);
  void DeviceRemovalCallback(const std::string& serial_number);
  void DeviceCallback(const std::string& event_name);
  void ImageCallback(Spinnaker::ImagePtr& image_ptr);
};

}  // namespace aps
