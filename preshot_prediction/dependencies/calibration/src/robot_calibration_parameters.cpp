// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "calibration/robot_calibration_parameters.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace calibration {

RobotCalibrationParameters::RobotCalibrationParameters() : BaseParameters("robot_calibration_parameters") {}

void RobotCalibrationParameters::UpdateInternalParameters() { T_origin_robot = T_robot_origin.inverse(); }

bool RobotCalibrationParameters::UpdateParametersFromYaml() {
  try {
    // Transformation.
    auto T_robot_origin_yaml = yaml_node_["T_RO"].as<std::vector<std::vector<double>>>();
    if (T_robot_origin_yaml.size() != 4) {
      LOG(ERROR) << "Invalid size of T_robot_origin in " << yaml_path_ << ".\n";
      return false;
    }
    for (int i = 0; i < static_cast<int>(T_robot_origin_yaml.size()); ++i) {
      if (T_robot_origin_yaml[i].size() != 4) {
        LOG(ERROR) << "Invalid size of T_robot_origin in " << yaml_path_ << ".\n";
        return false;
      }

      std::memcpy(T_robot_origin.col(i).data(), T_robot_origin_yaml[i].data(),
                  static_cast<int>(T_robot_origin_yaml[i].size()) * sizeof(double));
    }
    T_robot_origin.transposeInPlace();

  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  // Update internal parameters
  UpdateInternalParameters();

  return true;
}

bool RobotCalibrationParameters::UpdateYamlFromParameters() {
  try {
    // Transformation.
    std::vector<std::vector<double>> T_robot_origin_yaml;
    for (int i = 0; i < 4; ++i) {
      std::vector<double> row;
      row.reserve(4);
      for (int j = 0; j < 4; ++j) {
        row.push_back(T_robot_origin(i, j));
      }
      T_robot_origin_yaml.push_back(row);
    }
    yaml_node_["T_RO"] = T_robot_origin_yaml;
    for (uint i = 0; i < yaml_node_["T_RO"].size(); ++i) {
      yaml_node_["T_RO"][i].SetStyle(YAML::EmitterStyle::Flow);
    }

  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace calibration
