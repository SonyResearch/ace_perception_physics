// Confidential, Copyright 2025, Sony AI, All rights reserved

#include <sched.h>

#include <string>

namespace ace_rt_profiles {

/**
 * RAII style mechanism for applying temporary profiles to the current thread.
 * The current profile is saved on creation and restored on destruction.
 * Useful when invoking some external API that spawns threads out of your control (e.g. ROS).
 *
 * Example:
 * {
 *   ProfileGuard guard;        // save the current profile
 *   manager.Apply("foo")       // apply the profile foo
 *   rclcpp::init(...);         // threads spawned internally by ROS will run under the foo profile
 * }                            // guard goes out of scope, previous profile is restored for the main thread
 *                              // but threads spawned by ROS are still running under profile foo
 */
class ProfileGuard {
 public:
  /**
   * Save the current profile and restore it when the object is destroyed.
   */
  ProfileGuard();

  /**
   * Restore the saved profile.
   */
  ~ProfileGuard();

 private:
  /** Saved affinity. */
  cpu_set_t affinity_;

  /** Saved param. */
  sched_param param_;

  /** Saved policy. */
  int policy_;

  /** Represents the pid of the current thread. */
  static constexpr int kSelfPid = 0;
};

}  // namespace ace_rt_profiles
