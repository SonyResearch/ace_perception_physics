// Confidential, Copyright 2025, Sony AI, All rights reserved

#include "ace_rt_profiles/profile_manager.hpp"

#include <glog/logging.h>

#include <optional>

#include "ace_rt_profiles/utils.hpp"

namespace ace_rt_profiles {

const ProfileManager& ProfileManager::GetInstance() {
  static const ProfileManager Instance;
  return Instance;
}

ProfileManager::ProfileManager() : config_file_(utils::GetConfigFile()), config_(ace_yaml::LoadFile(config_file_)) {}

void ProfileManager::Apply(const std::string& profile) const { Apply(kSelfPid, profile); }

void ProfileManager::Apply(int pid, const std::string& profile) const {
  if (!config_[profile]) {
    throw std::runtime_error("Missing profile " + profile + " in " + config_file_);
  }

  ApplyAffinity(GetAffinity(profile), pid, profile);
  ApplyPolicy(pid, profile);
}

void ProfileManager::Apply(const RTSubProfile& subprofile) const { Apply(kSelfPid, subprofile); }

void ProfileManager::Apply(int pid, const RTSubProfile& subprofile) const {
  if (!config_[subprofile.rt_profile]) {
    throw std::runtime_error("Missing profile " + subprofile.rt_profile + " in " + config_file_);
  }

  const std::vector<int> affinity = GetAffinity(subprofile.rt_profile);
  if (subprofile.affinity_index >= affinity.size()) {
    throw std::out_of_range("Affinity index " + std::to_string(subprofile.affinity_index) +
                            " out of range of profile " + subprofile.rt_profile + " in " + config_file_);
  }

  ApplyAffinity({affinity.at(subprofile.affinity_index)}, pid, subprofile.rt_profile);
  ApplyPolicy(pid, subprofile.rt_profile);
}

void ProfileManager::ApplyAffinity(const std::vector<int>& affinity, int pid, const std::string& profile) const {
  if (affinity.empty()) {
    return;
  }

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  for (const int core : affinity) {
    CPU_SET(core, &cpu_set);
  }

  // check if affinity is the same (mostly just to avoid log spam)
  if (cpu_set_t cur; sched_getaffinity(pid, sizeof(cur), &cur) == EXIT_SUCCESS && CPU_EQUAL(&cur, &cpu_set)) {
    return;
  }

  if (sched_setaffinity(pid, sizeof(cpu_set), &cpu_set) != EXIT_SUCCESS) {
    throw std::system_error(errno, std::generic_category(),
                            "Failed to apply affinity settings for profile " + profile + " in " + config_file_);
  }

  // create affinity string for logging
  std::ostringstream oss;
  oss << "[";
  bool insert_comma = false;
  for (const int core : affinity) {
    oss << (insert_comma ? ", " : "") << core;
    insert_comma = true;
  }
  oss << "]";

  LOG(INFO) << "Applied real time profile " << profile << " to thread id " << std::this_thread::get_id()
            << ". Affinity: " << oss.str();
}

void ProfileManager::ApplyPolicy(int pid, const std::string& profile) const {
  const auto policy_node = config_[profile]["policy"];
  if (!policy_node || policy_node.IsNull()) {
    return;
  }

  std::string policy_name;
  try {
    policy_name = policy_node.as<std::string>();
  } catch (const std::exception&) {
    throw std::runtime_error("policy must be a string, profile " + profile + " in " + config_file_);
  }

  if (policy_name.empty()) {
    return;
  }

  if (!kPolicies.contains(policy_name)) {
    throw std::runtime_error("Invalid policy " + policy_name + " for profile " + profile + " in " + config_file_);
  }
  const int policy = kPolicies.at(policy_name);

  const auto priority_node = config_[profile]["priority"];
  if (priority_node && !ace_yaml::CheckNodeType<int>(priority_node)) {
    throw std::runtime_error("priority must be an int, profile " + profile + " in " + config_file_);
  }

  if (!priority_node && (policy == SCHED_FIFO || policy == SCHED_RR)) {
    throw std::runtime_error("Real time policies require priority to be set, profile " + profile + " in " +
                             config_file_);
  }

  const int priority = ace_yaml::SafeGetValue<int>(priority_node, 0);
  const int min_priority = sched_get_priority_min(policy);
  const int max_priority = sched_get_priority_max(policy);
  const sched_param param = {.sched_priority = std::clamp(priority, min_priority, max_priority)};

  // check if policy and priority are the same (mostly just to avoid log spam)
  if (sched_param cur; sched_getscheduler(pid) == policy && sched_getparam(pid, &cur) == EXIT_SUCCESS &&
                       cur.sched_priority == param.sched_priority) {
    return;
  }

  if (param.sched_priority > kMaxSafePriority) {
    LOG(WARNING) << "Priorities higher than " << kMaxSafePriority
                 << " are reserved for critical kernel tasks. Applying priority " << param.sched_priority
                 << " is not recommended";
  }

  if (sched_setscheduler(pid, policy, &param) != EXIT_SUCCESS) {
    throw std::system_error(errno, std::generic_category(),
                            "Failed to apply policy settings for profile " + profile + " in " + config_file_);
  }

  LOG(INFO) << "Applied real time profile " << profile << " to thread id " << std::this_thread::get_id()
            << ". Policy: " << policy_name << ", priority: " << param.sched_priority;
}

std::vector<RTThread> ProfileManager::MakeThreads(const std::string& profile) const {
  return MakeThreads(profile, GetAffinity(profile).size(), AffinityMode::kSequential);
}

std::vector<RTThread> ProfileManager::MakeThreads(const std::string& profile, size_t amount, AffinityMode mode) const {
  std::vector<RTThread> threads;
  threads.reserve(amount);

  const std::vector<int> affinity = GetAffinity(profile);

  // translate hybrid into the other modes
  if (mode == AffinityMode::kHybrid) {
    mode = amount == affinity.size() ? AffinityMode::kSequential : AffinityMode::kHomogeneous;
  }

  // apply real time profile before executing user code
  switch (mode) {
    case AffinityMode::kHomogeneous: {
      for (size_t i = 0; i < amount; ++i) {
        threads.emplace_back([this, profile]() { Apply(profile); });
      }
      break;
    }

    case AffinityMode::kSequential: {
      for (size_t i = 0; i < amount; ++i) {
        // modulo operation ensures we loop from the start once we run out of cores
        const std::optional<int> core =
          affinity.empty() ? std::nullopt : std::make_optional<int>(affinity.at(i % affinity.size()));

        threads.emplace_back([this, profile, core]() {
          if (core) {
            ApplyAffinity({*core}, kSelfPid, profile);
          }

          ApplyPolicy(kSelfPid, profile);
        });
      }
      break;
    }

    default:
      throw std::runtime_error("Invalid affinity mode " + std::to_string(static_cast<int>(mode)));
  }

  return threads;
}

std::vector<int> ProfileManager::GetAffinity(const std::string& profile) const {
  if (!config_[profile]) {
    throw std::runtime_error("Missing profile " + profile + " in " + config_file_);
  }

  const auto affinity = config_[profile]["affinity"];
  if (!affinity || affinity.IsNull() || (affinity.IsSequence() && affinity.size() == 0)) {
    return {};
  }

  if (!affinity.IsSequence()) {
    throw std::runtime_error("affinity must be a sequence, profile " + profile + " in " + config_file_);
  }

  std::vector<int> cores;
  for (const auto& it : affinity) {
    const int core = it.as<int>();
    if (core < 0) {
      throw std::runtime_error("CPU id must be non-negative integer, profile " + profile + " in " + config_file_);
    }
    cores.push_back(core);
  }

  return cores;
}
}  // namespace ace_rt_profiles
