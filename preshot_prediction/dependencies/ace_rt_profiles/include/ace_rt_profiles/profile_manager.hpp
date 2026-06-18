// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <sched.h>

#include <ace_yaml/ace_yaml.hpp>
#include <string>
#include <vector>

#include "ace_rt_profiles/rt_sub_profile.hpp"
#include "ace_rt_profiles/rt_thread.hpp"

namespace ace_rt_profiles {

/**
 * Strategies for setting thread affinity.
 */
enum class AffinityMode {
  /** Assign each thread to the next core in the affinity list, looping from the start as needed. */
  kSequential,

  /** Assign all threads to all cores in the affinity list. */
  kHomogeneous,

  /** If same number of threads and cores performs 1:1 assignment, otherwise fallback to HOMOGENEOUS mode. */
  kHybrid
};

/**
 * Thread-safe singleton responsible for managing real time profiles.
 * Each profile contains settings such as cpu affinity, scheduler priorities, etc
 */
class ProfileManager final {
  // helper class for unit tests
  friend class ProfileManagerTester;

 public:
  /*
   * Loads real time profiles from a yaml file.
   * File is automatically deduced from ACE_LAB env var.
   * If CI env var is set, ACE_LAB is ignored and CI profiles will be loaded.
   *
   * Profiles are loaded into memory the first time this function is called.
   * Future calls re-use the in-memory config instead of reloading the file.
   *
   * @return The singleton instance.
   *
   * @throw std::runtime_error If config file is missing or env vars are not set.
   */
  static const ProfileManager& GetInstance();

  // no copies allowed for singletons
  ProfileManager(ProfileManager const&) = delete;
  void operator=(ProfileManager const&) = delete;

  /** Shorthand for ProfileManager::Apply(0, profile) */
  void Apply(const std::string& profile) const;

  /**
   * Applies the given profile.
   *
   * @param pid PID (or TID) to apply profile to. Applies to the calling thread if 0.
   * @param profile Key in the config file containing the settings to apply.
   *
   * @throw std::runtime_error If profile does not exist or config file is ill-formed.
   * @throw std::system_error If there was any error applying the profile.
   */
  void Apply(int pid, const std::string& profile) const;

  /** Shorthand for ProfileManager::Apply(0, subprofile) */
  void Apply(const RTSubProfile& subprofile) const;

  /**
   * Applies the given subprofile.
   * For using threads please see ProfileManager::MakeThreads instead.
   *
   * @param pid PID (or TID) to apply profile to. Applies to the calling thread if 0.
   * @param subprofile Subprofile to apply.
   *
   * @throw std::out_of_range If the index in the subprofile is out of range of the affinity list
   * @throw std::runtime_error If profile does not exist or config file is ill-formed.
   * @throw std::system_error If there was any error applying the profile.
   */
  void Apply(int pid, const RTSubProfile& subprofile) const;

  /**
   * Shorthand for ProfileManager::MakeThreads(profile, GetAffinity(profile).size(), AffinityMode::kSequential)
   * See the above overload for more details.
   *
   * The number of threads created matches the size of the affinity list.
   * Each of those threads is assigned to a different core.
   * If the affinity list is empty, no threads are created.
   *
   * Other settings are applied to all threads equally.
   *
   * ProfileManager::MakeThreads("example")
   *  example:
   *    affinity: [0, 1]    # create 2 threads A and B. A runs on core 0 and B on core 1
   *    policy: SCHED_FIFO  # A and B will both run with the SCHED_FIFO policy
   *    priority: 89        # A and B will both run with priority 89
   *
   * @param profile Key in the config file containing the settings to apply.
   *
   * @return The real time threads.
   */
  std::vector<RTThread> MakeThreads(const std::string& profile) const;

  /**
   * Creates the given amount of threads with a real time profile.
   * The profile settings will be applied once RTThread::Run is called, before user code runs.
   *
   * The actual OS threads will only be spawned once RTThread::Run is called.
   *
   * In HOMOGENEOUS mode, the profile is applied to all threads equally.
   *
   * In SEQUENTIAL mode, each thread will be assigned a single core.
   * The cores in the affinity list are looped in sequence as needed to match the number of threads.
   * Different threads will be assigned to the same core if the number of threads is higher than the number of cores.
   * Other settings are applied to all threads equally.
   *
   * In HYBRID mode, runs as SEQUENTIAL mode if the number of threads matches the number of cores.
   * Otherwise runs as HOMOGENEOUS mode.
   *
   * ProfileManager::MakeThreads("example", 3, AffinityMode::kSequential)
   *  example:
   *    affinity: [0, 1]    # create 3 threads A, B and C. A and C run on core 0, B on core 1.
   *    policy: SCHED_FIFO  # A, B and C will all run with the SCHED_FIFO policy
   *    priority: 89        # A, B and C will all run with priority 89
   *
   * @param profile Key in the config file containing the settings to apply.
   * @param amount Number of threads to create.
   * @param mode Strategy for setting the affinity of the threads.
   *
   * @return The real time threads.
   */
  std::vector<RTThread> MakeThreads(const std::string& profile, size_t amount, AffinityMode mode) const;

  /**
   * Get the cpu affinities for the given profile.
   *
   * @param profile The profile to search for affinity.
   *
   * @return List of cpu cores defined in the affinity setting of the given profile.
   *
   * @throw std::runtime_error If profile does not exist or config file is ill-formed.
   */
  std::vector<int> GetAffinity(const std::string& profile) const;

 private:
  // private as this is a singleton
  ProfileManager();
  ~ProfileManager() = default;

  /**
   * Applies the given cpu affinity.
   * Does nothing if affinity is empty.
   *
   * @param affinity Affinity to apply. Result from ProfileManager::GetAffinity(profile).
   * @param pid PID (or TID) to apply affinity to. Applies to the calling thread if 0.
   * @param profile Name of the respective profile, used for logging purposes only.
   *
   * @throw std::system_error If there was any error applying the affinity.
   */
  void ApplyAffinity(const std::vector<int>& affinity, int pid, const std::string& profile) const;

  /**
   * Applies scheduler policy settings for the given profile.
   * Does nothing if no policy settings are defined in the config.
   *
   * @param pid PID (or TID) to apply settings to. Applies to the calling thread if 0.
   * @param profile Key in the config file to read settings from (assumed to exist)
   *
   * @throw std::runtime_error If profile is missing or config file is ill-formed.
   * @throw std::system_error If there was any error applying the profile.
   */
  void ApplyPolicy(int pid, const std::string& profile) const;

  // --- ATTRS ---

  /** The absolute path of the config file. */
  const std::string config_file_;

  /** The yaml object representing the config file. */
  const YAML::Node config_;

  /** Priorities 90-99 are reserved for critical kernel tasks */
  static constexpr int kMaxSafePriority = 89;

  /** Represents the pid of the current thread. */
  static constexpr int kSelfPid = 0;

 public:
  /** Mapping of possible policies and their names. */
  static inline const std::unordered_map<std::string, int> kPolicies = {{"SCHED_OTHER", SCHED_OTHER},
                                                                        {"SCHED_BATCH", SCHED_BATCH},
                                                                        {"SCHED_IDLE", SCHED_IDLE},
                                                                        {"SCHED_FIFO", SCHED_FIFO},
                                                                        {"SCHED_RR", SCHED_RR}};
};

}  // namespace ace_rt_profiles
