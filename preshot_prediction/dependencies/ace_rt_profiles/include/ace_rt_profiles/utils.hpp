// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <string>

namespace ace_rt_profiles::utils {

/**
 * The configuration file is automatically deduced from the ACE_LAB and ACE_PC environment variables
 * ace_rt_profiles/config/<ACE_LAB>/<ACE_PC>.yaml
 *
 * If DART_RUN_ID or CI env vars are set, then ACE_LAB and ACE_PC will be ignored
 * in favor of loading dart.yaml or ci.yaml respectively
 *
 * @return The config file
 */
std::string GetConfigFile();

}  // namespace ace_rt_profiles::utils
