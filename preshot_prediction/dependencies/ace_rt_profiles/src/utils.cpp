// Confidential, Copyright 2025, Sony AI, All rights reserved

#include "ace_rt_profiles/utils.hpp"

#include <glog/logging.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace ace_rt_profiles::utils {
std::string GetConfigFile() {
  const std::string prefix = ament_index_cpp::get_package_share_directory("ace_rt_profiles") + "/config/";

  if (std::getenv("DART_RUN_ID")) {
    LOG(WARNING) << "DART_RUN_ID environment variable is set, ignoring ACE_LAB and loading DART real time settings";
    return prefix + "dart.yaml";
  }

  if (std::getenv("CI")) {
    LOG(WARNING) << "CI environment variable is set, ignoring ACE_LAB and loading CI real time settings";
    return prefix + "ci.yaml";
  }

  const char* ace_lab = std::getenv("ACE_LAB");
  if (ace_lab == nullptr) {
    throw std::runtime_error("ACE_LAB environment variable is not set. Can't load real time profiles");
  };

  const char* ace_pc = std::getenv("ACE_PC");
  if (ace_pc == nullptr) {
    throw std::runtime_error("ACE_PC environment variable is not set. Can't load real time profiles");
  };

  return prefix + ace_lab + "/" + ace_pc + ".yaml";
}
}  // namespace ace_rt_profiles::utils
