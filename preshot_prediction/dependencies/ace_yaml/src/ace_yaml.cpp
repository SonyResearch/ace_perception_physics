// Confidential, Copyright 2025, Sony AI, All rights reserved.

#include "ace_yaml/ace_yaml.hpp"

#include <glog/logging.h>
#include <sys/stat.h>

#include <filesystem>
#include <regex>

#include "ament_index_cpp/get_package_share_directory.hpp"

namespace ace_yaml {

enum class YamlAttributes : unsigned int { kFinal = 0x1 };

inline void MergeInplace(YAML::Node& dst_nodes, const YAML::Node& src_nodes);

std::string resolveValue(const std::string& directive_name, const std::string& value) {
  if (directive_name == "package") {
    // find package location
    return ament_index_cpp::get_package_share_directory(value);
  }
  throw std::invalid_argument("[ERROR] invalid directive provided to yaml file: " + directive_name);
}

std::filesystem::path ResolvePath(const std::string& file_path) {
  std::regex regex(R"((.*)\$\{(.+):(.+)\}/(.+))");
  std::smatch m;

  std::filesystem::path result_path;

  if (std::regex_match(file_path, m, regex)) {
    // match 0 -> whole string
    // match 1 -> pre-path
    // match 2 -> directive_name (e.g. package)
    // match 3 -> value
    // match 4 -> post path

    if (!m.str(1).empty()) {
      result_path = m.str(1);
    } else {
      result_path = ".";
    }
    result_path /= resolveValue(m.str(2), m.str(3));
    result_path /= m.str(4);

  } else {
    return file_path;
  }
  return result_path;
}
std::string resolve_path(const std::filesystem::path& file_path, const std::list<std::filesystem::path>& search_paths) {
  if (file_path.is_absolute()) {
    return file_path;
  }
  struct stat buffer;
  for (const auto& path : search_paths) {
    std::string resolved = path / file_path;
    if (stat(resolved.c_str(), &buffer) == 0) {
      return resolved;
    }
  }
  return file_path;
}

YAML::Node LoadFile(const std::string& file_path, std::map<std::string, unsigned int>& attributes_map) {
  // LOG(INFO) << "Reading nodes from \"" << file_path << "\"";
  YAML::Node nodes = YAML::LoadFile(file_path);
  std::list<std::string> file_paths;
  std::list<std::string> attributes;
  std::list<std::filesystem::path> search_paths;
  unsigned int attributes_bitwise = 0;

  const auto& base_path = std::filesystem::path(file_path).parent_path();
  search_paths.push_back(base_path);
  if (YAML::Node search_paths_nodes = nodes[kSearchPathNodeName]) {
    if (!search_paths_nodes.IsSequence()) {
      LOG(ERROR) << "file=" << file_path << ": " << kSearchPathNodeName << " is not a sequence!";
      return nodes;
    }
    for (auto it = search_paths_nodes.begin(); it != search_paths_nodes.end(); ++it) {
      auto search_path = ResolvePath(it->as<std::string>());
      search_paths.push_back(search_path);
    }
    nodes.remove(kSearchPathNodeName);
  }

  if (YAML::Node attributes_nodes = nodes[kAttributesNodeName]) {
    if (!attributes_nodes.IsSequence()) {
      LOG(ERROR) << "file=" << file_path << ": " << kSearchPathNodeName << " is not a sequence!";
      return nodes;
    }
    for (auto it = attributes_nodes.begin(); it != attributes_nodes.end(); ++it) {
      auto attribute = it->as<std::string>();
      if (attribute == "final") {
        attributes_bitwise |= static_cast<unsigned int>(YamlAttributes::kFinal);
      }
    }
    nodes.remove(kAttributesNodeName);
  }

  if (YAML::Node inheritance_nodes = nodes[kInheritanceNodeName]) {
    if (!inheritance_nodes.IsSequence()) {
      LOG(ERROR) << "file=" << file_path << ": " << kInheritanceNodeName << " is not a sequence!";
      return nodes;
    }

    for (auto it = inheritance_nodes.begin(); it != inheritance_nodes.end(); ++it) {
      auto file_path = std::filesystem::path(it->as<std::string>());
      if (file_path == "_self_") {
        // skip hydra's keywords
        continue;
      }
      file_paths.push_front(ResolvePath(file_path));
    }
    nodes.remove(kInheritanceNodeName);
  }

  std::list<YAML::Node> root_nodes;
  // Go over the paths in a reverse order from the YAML elements
  for (auto& file_path : file_paths) {
    root_nodes.push_back(LoadFile(resolve_path(file_path, search_paths), attributes_map));
  }
  // now mark all nodes here with the set attributes
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto node = it->first.as<std::string>();

    auto node_it = attributes_map.find(node);
    if (node_it == attributes_map.end()) {
      attributes_map[node] = attributes_bitwise;
    } else if ((node_it->second & static_cast<unsigned int>(YamlAttributes::kFinal)) != 0) {
      throw std::invalid_argument("[ERROR] file=" + file_path + ": Node[" + node_it->first +
                                  "] is final in root config, yet its overriden here!");
    }
  }

  // finally do the merge
  for (auto& root : root_nodes) {
    MergeInplace(nodes, root);
  }

  return nodes;
}

inline void MergeInplace(YAML::Node& dst_nodes, const YAML::Node& src_nodes) {
  if (dst_nodes.IsNull() || !src_nodes.IsMap()) {
    return;
  }
  for (auto it = src_nodes.begin(); it != src_nodes.end(); ++it) {
    const std::string& key = it->first.as<std::string>();
    if (YAML::Node dst_node = dst_nodes[key]) {
      MergeInplace(dst_node, it->second);
    } else {
      dst_nodes[key] = it->second;
    }
  }
}

YAML::Node LoadFile(const std::string& file_path) {
  std::map<std::string, unsigned int> attributes_map;
  return LoadFile(file_path, attributes_map);
}
YAML::Node LoadString(const std::string& yaml) { return YAML::Load(yaml); }

}  // namespace ace_yaml
