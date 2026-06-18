// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "aps/blackfly_s_image_event_handler.hpp"

#include <ace_rt_profiles/profile_manager.hpp>
#include <cstdio>

#include "ace_loggers/ace_loggers.hpp"

namespace aps {

BlackflySImageEventHandler::BlackflySImageEventHandler(const std::string serial_number)
  : serial_number_(std::move(serial_number)) {
  if (!google::IsGoogleLoggingInitialized()) {
    const std::string logging_name = serial_number_ + "_image_event_handler";
    google::InitGoogleLogging(logging_name.c_str());
  }
}

bool BlackflySImageEventHandler::RegisterImageCallback(std::function<void(Spinnaker::ImagePtr&)> callback) {
  if (callback_ != nullptr) {
    LOG(WARNING) << "Only a single callback can be registered" << std::endl;
    return false;
  }
  callback_ = callback;
  return true;
}

void BlackflySImageEventHandler::OnImageEvent(Spinnaker::ImagePtr image_ptr) {
  if (rt_subprofile_) {
    ace_rt_profiles::ProfileManager::GetInstance().Apply(*rt_subprofile_);
    rt_subprofile_.reset();
  }
  if (image_ptr->IsIncomplete()) {
    LOG(WARNING) << "Received image event on incomplete image.\n";
    return;
  }
  callback_(image_ptr);
}

BlackflySImageEventHandler::~BlackflySImageEventHandler() = default;

}  // namespace aps
