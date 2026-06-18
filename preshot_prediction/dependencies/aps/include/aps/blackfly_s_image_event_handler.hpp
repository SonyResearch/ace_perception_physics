// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <Spinnaker.h>

#include <ace_rt_profiles/rt_sub_profile.hpp>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace aps {

class BlackflySImageEventHandler : public Spinnaker::ImageEventHandler {
 public:
  // constructor & destructor
  explicit BlackflySImageEventHandler(std::string serial_number);
  ~BlackflySImageEventHandler() override;

  bool RegisterImageCallback(std::function<void(Spinnaker::ImagePtr&)> callback);
  void OnImageEvent(Spinnaker::ImagePtr image_ptr) override;

  void SetRTSubProfile(const std::optional<ace_rt_profiles::RTSubProfile>& rt_subprofile) {
    rt_subprofile_ = rt_subprofile;
  }

 private:
  std::optional<ace_rt_profiles::RTSubProfile> rt_subprofile_ = std::nullopt;
  const std::string serial_number_;
  std::function<void(Spinnaker::ImagePtr&)> callback_ = nullptr;
};

}  // namespace aps
