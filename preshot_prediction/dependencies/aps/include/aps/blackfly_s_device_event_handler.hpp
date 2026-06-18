// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <Spinnaker.h>

#include <functional>
#include <string>

namespace aps {

class BlackflySDeviceEventHandler : public Spinnaker::DeviceEventHandler {
 public:
  // constructor & destructor
  explicit BlackflySDeviceEventHandler(std::string serial_number);

  bool RegisterDeviceCallback(std::function<void(const std::string&)> callback);
  void OnDeviceEvent(Spinnaker::GenICam::gcstring event_name) override;

 private:
  const std::string serial_number_;
  std::function<void(const std::string&)> callback_;
};

}  // namespace aps
