// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "aps/blackfly_s_interface_event_handler.hpp"

#include <cstdio>

#include "glog/logging.h"

namespace aps {

BlackflySInterfaceEventHandler::BlackflySInterfaceEventHandler() {
  if (!google::IsGoogleLoggingInitialized()) {
    const std::string logging_name = "interface_event_handler";
    google::InitGoogleLogging(logging_name.c_str());
  }
}

bool BlackflySInterfaceEventHandler::RegisterDeviceArrivalCallback(std::function<void(const std::string&)> callback) {
  if (arrival_callback_ != nullptr) {
    LOG(WARNING) << "Only a single arrival callback can be registered" << std::endl;
    return false;
  }
  arrival_callback_ = callback;
  return true;
}

bool BlackflySInterfaceEventHandler::RegisterDeviceRemovalCallback(std::function<void(const std::string&)> callback) {
  if (removal_callback_ != nullptr) {
    LOG(WARNING) << "Only a single removal callback can be registered" << std::endl;
    return false;
  }
  removal_callback_ = callback;
  return true;
}

void BlackflySInterfaceEventHandler::OnDeviceArrival(Spinnaker::CameraPtr cam_ptr) {
  Spinnaker::TransportLayerDevice device(&cam_ptr->GetTLDeviceNodeMap());
  const auto serial_number = static_cast<std::string>(device.DeviceSerialNumber.GetValue());
  arrival_callback_(serial_number);
}

void BlackflySInterfaceEventHandler::OnDeviceArrival(uint64_t serial_number) {
  arrival_callback_(std::to_string(serial_number));
}

void BlackflySInterfaceEventHandler::OnDeviceRemoval(Spinnaker::CameraPtr cam_ptr) {
  Spinnaker::TransportLayerDevice device(&cam_ptr->GetTLDeviceNodeMap());
  const auto serial_number = static_cast<std::string>(device.DeviceSerialNumber.GetValue());
  removal_callback_(serial_number);
}

void BlackflySInterfaceEventHandler::OnDeviceRemoval(uint64_t serial_number) {
  removal_callback_(std::to_string(serial_number));
}

}  // namespace aps
