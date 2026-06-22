// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <yaml-cpp/yaml.h>

class BaseParameters {
 public:
  explicit BaseParameters(std::string parameters_name);

  bool Initialize(std::string yaml_path);
  bool IsInitialized() const { return init_; };
  bool PrintParameters() const;
  bool ReadParametersFromFile();
  bool ReadParametersFromString(const std::string& yaml_str);
  bool WriteParametersToFile();
  std::string ToString();
  std::string ToString() const;

 protected:
  bool init_;
  const std::string parameters_name_;

  YAML::Node yaml_node_;
  std::string yaml_path_;

  // Auxiliary functions to be implemented in derived class.
  virtual bool UpdateParametersFromYaml() = 0;
  virtual bool UpdateYamlFromParameters() = 0;
};
