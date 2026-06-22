// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "base_parameters/base_parameters.hpp"
#include "eigen3/Eigen/Eigen"

namespace calibration {

class RobotCalibrationParameters final : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<RobotCalibrationParameters>;
  using ConstSharedPtr = std::shared_ptr<const RobotCalibrationParameters>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit RobotCalibrationParameters();

  // Transformation.
  Eigen::Matrix<double, 4, 4> T_origin_robot;  // transformation from (O)rigin to (R)obot frame
  Eigen::Matrix<double, 4, 4> T_robot_origin;  // transformation from (R)obot to (O)rigin frame

  // Update internal parameters
  void UpdateInternalParameters();

 private:
  // auxiliary functions
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace calibration
