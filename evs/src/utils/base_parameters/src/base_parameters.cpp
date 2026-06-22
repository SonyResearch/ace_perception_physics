// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "base_parameters/base_parameters.hpp"

#include <fstream>
#include <iostream>

#include "ace_loggers/ace_loggers.hpp"

BaseParameters::BaseParameters(const std::string parameters_name) : parameters_name_(parameters_name) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(parameters_name_.c_str());
  }

  init_ = false;
  yaml_path_ = "path not set";
}

bool BaseParameters::Initialize(const std::string yaml_path) {
  if (init_) {
    LOG(WARNING) << parameters_name_ << " is already initialized.\n";
    return false;
  }

  yaml_path_ = yaml_path;
  if (!ReadParametersFromFile()) {
    return false;
  }

  init_ = true;
  return init_;
}

bool BaseParameters::PrintParameters() const {
  if (!init_) {
    LOG(WARNING) << "Unable to print parameters (uninitialized).\n";
    return false;
  }

  // obtain file name
  std::string file_name;

  size_t pos_end = yaml_path_.find(".yaml");
  if (pos_end == std::string::npos) {
    LOG(WARNING) << "Parameter file not ending with \".yaml\".\n";
  }

  size_t pos_start = yaml_path_.find_last_of('/');
  if (pos_start == std::string::npos) {
    pos_start = 0;
  }
  file_name = yaml_path_.substr(pos_start + 1, pos_end - (pos_start + 1));

  // print parameters info
  YAML::Emitter out;
  out << yaml_node_;

  std::cout << "\n";
  std::cout << "-------------------------------------------------------\n";
  std::cout << parameters_name_ << ": \"" << file_name << "\"\n";
  std::cout << "(path: " << yaml_path_ << ")\n";
  std::cout << "-------------------------------------------------------\n";
  std::cout << out.c_str() << "\n";
  std::cout << "-------------------------------------------------------\n";
  std::cout << "\n";

  return true;
}

bool BaseParameters::ReadParametersFromFile() {
  try {
    yaml_node_ = ace_yaml::LoadFile(yaml_path_);
  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  if (!UpdateParametersFromYaml()) {
    LOG(WARNING) << "Unable to upadte parameters from yaml file \"" << yaml_path_ << "\".\n";
    return false;
  }

  return true;
}

bool BaseParameters::ReadParametersFromString(const std::string &yaml_str) {
  try {
    yaml_node_ = ace_yaml::LoadString(yaml_str);
  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  if (!UpdateParametersFromYaml()) {
    LOG(WARNING) << "Unable to upadte parameters from yaml string.\n";
    return false;
  }

  return true;
}

bool BaseParameters::WriteParametersToFile() {
  if (!init_) {
    LOG(WARNING) << "Unable to write parameters (uninitialized).\n";
    return false;
  }

  if (!UpdateYamlFromParameters()) {
    LOG(WARNING) << "Unable to update YAML from parameters.\n";
    return false;
  }

  std::ofstream output_file_stream(yaml_path_);
  output_file_stream << yaml_node_;
  return true;
}

std::string BaseParameters::ToString() {
  if (!init_ || !UpdateYamlFromParameters()) {
    return "";
  }

  std::ostringstream output_stream;
  output_stream << yaml_node_;
  return output_stream.str();
}

std::string BaseParameters::ToString() const {
  if (!init_) {
    return "";
  }

  std::ostringstream output_stream;
  output_stream << yaml_node_;
  return output_stream.str();
}
