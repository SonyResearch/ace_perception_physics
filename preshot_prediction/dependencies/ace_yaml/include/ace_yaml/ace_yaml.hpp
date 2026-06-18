// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <filesystem>

#include "yaml-cpp/yaml.h"

namespace ace_yaml {

constexpr char kInheritanceNodeName[] = "defaults";
constexpr char kSearchPathNodeName[] = "__search_paths__";
constexpr char kAttributesNodeName[] = "__attributes__";

YAML::Node LoadFile(const std::string& file_path);
YAML::Node LoadString(const std::string& yaml);
std::filesystem::path ResolvePath(const std::string& file_path);

// simple helper function to check if a node is of type T
template <typename T>
bool CheckNodeType(YAML::Node node) {
  try {
    node.as<T>();
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}
template <typename T>
T SafeGetValue(YAML::Node node, const T& def) {
  try {
    return node.as<T>();
  } catch (const std::exception& e) {
    return def;
  }
}
}  // namespace ace_yaml
