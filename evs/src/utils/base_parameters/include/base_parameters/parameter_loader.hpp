// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

template <typename T>
int LoadInterfaceParameters(const YAML::Node& params, const std::vector<std::string>& param_list,
                            std::map<std::string, T>& values) {
  for (const std::string& s : param_list) {
    try {
      values[s] = params[s].as<T>();
    } catch (YAML::Exception& e) {
      PLOG(ERROR) << "[ERROR] Unable to extract parameter " << s;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
