// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <Spinnaker.h>

#include <functional>
#include <string>

namespace aps {

class BlackflySInterfaceEventHandler : public Spinnaker::InterfaceEventHandler {
 public:
  // constructor & destructor
  explicit BlackflySInterfaceEventHandler();

  bool RegisterDeviceArrivalCallback(std::function<void(const std::string&)> callback);
  bool RegisterDeviceRemovalCallback(std::function<void(const std::string&)> callback);

  void OnDeviceArrival(Spinnaker::CameraPtr cam_ptr) override;
  void OnDeviceRemoval(Spinnaker::CameraPtr cam_ptr) override;

  [[deprecated("Will be removed when all PCs are on Ubuntu 22.04 or more")]] void OnDeviceArrival(
    uint64_t serial_number);
  [[deprecated("Will be removed when all PCs are on Ubuntu 22.04 or more")]] void OnDeviceRemoval(
    uint64_t serial_number);

 private:
  std::function<void(const std::string&)> arrival_callback_;
  std::function<void(const std::string&)> removal_callback_;
};

}  // namespace aps
