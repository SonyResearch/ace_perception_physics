// Confidential, Copyright 2025, Sony AI, All rights reserved

#include "ace_rt_profiles/profile_guard.hpp"

#include <glog/logging.h>

#include <system_error>

#include "ace_rt_profiles/profile_manager.hpp"

namespace ace_rt_profiles {

ProfileGuard::ProfileGuard() : policy_(sched_getscheduler(kSelfPid)) {
  if (sched_getaffinity(kSelfPid, sizeof(affinity_), &affinity_) != EXIT_SUCCESS) {
    throw std::system_error(errno, std::generic_category(), "Failed to get affinity");
  }

  if (sched_getparam(kSelfPid, &param_) != EXIT_SUCCESS) {
    throw std::system_error(errno, std::generic_category(), "Failed to get scheduler param");
  }
}

ProfileGuard::~ProfileGuard() {
  // check if affinity and policy are the same (mostly just to avoid log spam)
  cpu_set_t cur_affinity;
  sched_param cur_param;
  if (sched_getaffinity(kSelfPid, sizeof(cur_affinity), &cur_affinity) == EXIT_SUCCESS &&
      CPU_EQUAL(&cur_affinity, &affinity_) && sched_getscheduler(kSelfPid) == policy_ &&
      sched_getparam(kSelfPid, &cur_param) == EXIT_SUCCESS && cur_param.sched_priority == param_.sched_priority) {
    return;
  }

  // can't throw exceptions on destructors, so just print error message below
  bool success = true;
  success &= sched_setaffinity(kSelfPid, sizeof(affinity_), &affinity_) == EXIT_SUCCESS;
  success &= sched_setscheduler(kSelfPid, policy_, &param_) == EXIT_SUCCESS;

  // create affinity string for logging
  std::ostringstream affinity_oss;
  affinity_oss << "[";
  if (CPU_COUNT(&affinity_) != static_cast<int>(std::thread::hardware_concurrency())) {
    bool insert_comma = false;
    for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
      if (CPU_ISSET(i, &affinity_)) {
        affinity_oss << (insert_comma ? ", " : "") << i;
        insert_comma = true;
      }
    }
  }
  affinity_oss << "]";

  // get policy name for logging
  std::string policy_name = "UNKNOWN";
  for (const auto& [key, val] : ProfileManager::kPolicies) {
    if (val == policy_) {
      policy_name = key;
      break;
    }
  }

  std::ostringstream log;
  log << " in thread id " << std::this_thread::get_id() << ". Affinity: " << affinity_oss.str()
      << ", policy: " << policy_name << ", priority: " << param_.sched_priority;

  if (success) {
    LOG(INFO) << "Reverted to previous profile" << log.str();
  } else {
    LOG(ERROR) << "ERROR: " << std::strerror(errno) << ". Failed to revert to previous profile" << log.str();
  }
}
}  // namespace ace_rt_profiles
