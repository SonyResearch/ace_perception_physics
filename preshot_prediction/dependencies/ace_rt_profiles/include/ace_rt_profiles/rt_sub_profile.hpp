// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <string>

namespace ace_rt_profiles {

/**
 * Helper struct that represents a subset of a real time profile with a single core of the original affinity list.
 * It's useful for distributing affinities via custom logic (e.g. assign one core to each event handler in a container).
 * Other settings (e.g. policy and priority) are applied as normal.
 *
 * For using threads please see ProfileManager::MakeThreads instead.
 */
struct RTSubProfile {
  /** Name of the original real time profile. */
  std::string rt_profile;

  /** Index of the cpu core in the affinity list to apply. */
  size_t affinity_index = 0;
};
}  // namespace ace_rt_profiles
