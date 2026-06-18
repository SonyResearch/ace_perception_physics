// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <chrono>

#include "base_parameters/base_parameters.hpp"
#include "dot_projector/idot_projector_parameters.hpp"
#include "triangulation/triangulation.hpp"

namespace multi_ball_triangulation {

class MultiBallTriangulationParameters final : public BaseParameters,
                                               public triangulation::TriangulationParameters,
                                               public dot_projector::IDotProjectorParameters {
 public:
  using SharedPtr = std::shared_ptr<MultiBallTriangulationParameters>;
  using ConstSharedPtr = std::shared_ptr<const MultiBallTriangulationParameters>;

  explicit MultiBallTriangulationParameters();

  std::chrono::milliseconds timeout_ms;
  bool use_variable_response_time;
  int camera_buffer_length;
  int worker_queue_length;
  int max_triangulated_balls;

  // VOI filter
  bool voi_filter_enable;
  Eigen::Vector3f voi_filter_min_corner;
  Eigen::Vector3f voi_filter_max_corner;

  GhostBallsPolicy ghost_balls_policy_settings;

  void SetExecutionMode(const std::string& mode);

 private:
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace multi_ball_triangulation
