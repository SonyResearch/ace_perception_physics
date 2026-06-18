// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "aps/blackfly_s_device_event_handler.hpp"

#include <cstdio>

#include "glog/logging.h"

namespace aps {

BlackflySDeviceEventHandler::BlackflySDeviceEventHandler(const std::string serial_number)
  : serial_number_(std::move(serial_number)) {
  if (!google::IsGoogleLoggingInitialized()) {
    const std::string logging_name = serial_number_ + "_device_event_handler";
    google::InitGoogleLogging(logging_name.c_str());
  }
}

bool BlackflySDeviceEventHandler::RegisterDeviceCallback(std::function<void(const std::string&)> callback) {
  if (callback_ != nullptr) {
    LOG(WARNING) << "Only a single callback can be registered" << std::endl;
    return false;
  }
  callback_ = callback;
  return true;
}

void BlackflySDeviceEventHandler::OnDeviceEvent(Spinnaker::GenICam::gcstring event_name) {
  callback_(static_cast<const char*>(event_name));
}

}  // namespace aps
