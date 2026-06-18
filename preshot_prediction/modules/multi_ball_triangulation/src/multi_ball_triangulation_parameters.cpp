// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "multi_ball_triangulation/multi_ball_triangulation_parameters.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace multi_ball_triangulation {

MultiBallTriangulationParameters::MultiBallTriangulationParameters()
  : BaseParameters("multi_ball_triangulation_parameters") {}

bool MultiBallTriangulationParameters::UpdateParametersFromYaml() {
  try {
    timeout_ms = std::chrono::milliseconds(yaml_node_["timeout_ms"].as<int>());
    use_variable_response_time = yaml_node_["use_variable_response_time"].as<bool>();
    camera_buffer_length = yaml_node_["camera_buffer_length"].as<int>();
    worker_queue_length = yaml_node_["worker_queue_length"].as<int>();
    max_triangulated_balls = yaml_node_["max_triangulated_balls"].as<int>();

    // Triangulation
    covariance_model =
      triangulation::StringToCovarianceModel(yaml_node_["triangulation"]["covariance_model"].as<std::string>());
    geometric_error_threshold = yaml_node_["triangulation"]["geometric_error_threshold"].as<float>();
    radius_error_threshold = yaml_node_["triangulation"]["radius_error_threshold"].as<float>();
    observation_variance = yaml_node_["triangulation"]["observation_variance"].as<float>();
    use_nonlinear_refinement = yaml_node_["triangulation"]["nonlinear_refinement"]["use"].as<bool>();
    tol_position = yaml_node_["triangulation"]["nonlinear_refinement"]["tol_position"].as<float>();
    tol_reprojection_error = yaml_node_["triangulation"]["nonlinear_refinement"]["tol_reprojection_error"].as<float>();
    max_iterations = yaml_node_["triangulation"]["nonlinear_refinement"]["max_iterations"].as<int>();

    // Projector
    projector_enable = yaml_node_["projector"]["enable"].as<bool>();
    projector_ball_radius = yaml_node_["projector"]["ball_radius"].as<float>();

    // VOI filter
    voi_filter_enable = yaml_node_["voi_filter"]["enable"].as<bool>();
    auto voi_filter_min_corner_yaml = yaml_node_["voi_filter"]["min_corner"].as<std::vector<float>>();
    if (voi_filter_min_corner_yaml.size() != 3) {
      LOG(ERROR) << "Invalid size of voi_filter::min_corner in " << yaml_path_ << ".\n";
      return false;
    }
    auto voi_filter_max_corner_yaml = yaml_node_["voi_filter"]["max_corner"].as<std::vector<float>>();
    if (voi_filter_max_corner_yaml.size() != 3) {
      LOG(ERROR) << "Invalid size of voi_filter::max_corner in " << yaml_path_ << ".\n";
      return false;
    }
    for (int i = 0; i < 3; ++i) {
      voi_filter_min_corner[i] = voi_filter_min_corner_yaml[i];
      voi_filter_max_corner[i] = voi_filter_max_corner_yaml[i];
    }

    similarity_threshold = yaml_node_["ghost_balls_filter"]["similarity_threshold"].as<float>();

    auto ghost_balls_policy_str = yaml_node_["ghost_balls_filter"]["filter_policy"].as<std::string>();

    if (ghost_balls_policy_str == "none") {
      ghost_balls_policy = triangulation::TriangulationParameters::GhostBallsPolicy::kNone;
    } else if (ghost_balls_policy_str == "drop") {
      ghost_balls_policy = triangulation::TriangulationParameters::GhostBallsPolicy::kDrop;
    } else if (ghost_balls_policy_str == "merge") {
      ghost_balls_policy = ghost_balls_policy_settings =
        triangulation::TriangulationParameters::GhostBallsPolicy::kMerge;
    } else {
      throw std::invalid_argument("Ghost balls filter policy is unkown: " + ghost_balls_policy_str);
    }

    // model_points
  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool MultiBallTriangulationParameters::UpdateYamlFromParameters() {
  try {
    yaml_node_["timeout_ms"] = static_cast<int>(timeout_ms.count());
    yaml_node_["use_variable_response_time"] = use_variable_response_time;
    yaml_node_["camera_buffer_length"] = camera_buffer_length;
    yaml_node_["worker_queue_length"] = worker_queue_length;
    yaml_node_["max_triangulated_balls"] = max_triangulated_balls;

    // Triangulation
    yaml_node_["triangulation"]["covariance_model"] = triangulation::CovarianceModelToString(covariance_model);
    yaml_node_["triangulation"]["geometric_error_threshold"] = geometric_error_threshold;
    yaml_node_["triangulation"]["radius_error_threshold"] = radius_error_threshold;
    yaml_node_["triangulation"]["observation_variance"] = observation_variance;
    yaml_node_["triangulation"]["nonlinear_refinement"]["use"] = use_nonlinear_refinement;
    yaml_node_["triangulation"]["nonlinear_refinement"]["tol_position"] = tol_position;
    yaml_node_["triangulation"]["nonlinear_refinement"]["tol_reprojection_error"] = tol_reprojection_error;
    yaml_node_["triangulation"]["nonlinear_refinement"]["max_iterations"] = max_iterations;

    // Projector
    yaml_node_["projector"]["enable"] = projector_enable;
    yaml_node_["projector"]["ball_radius"] = projector_ball_radius;

    // VOI filter
    yaml_node_["voi_filter"]["enable"] = voi_filter_enable;
    std::vector<float> voi_filter_min_corner_yaml = {voi_filter_min_corner[0], voi_filter_min_corner[1],
                                                     voi_filter_min_corner[2]};
    yaml_node_["voi_filter"]["min_corner"] = voi_filter_min_corner_yaml;
    std::vector<float> voi_filter_max_corner_yaml = {voi_filter_max_corner[0], voi_filter_max_corner[1],
                                                     voi_filter_max_corner[2]};
    yaml_node_["voi_filter"]["max_corner"] = voi_filter_max_corner_yaml;

    yaml_node_["voi_filter"]["min_corner"].SetStyle(YAML::EmitterStyle::Flow);
    yaml_node_["voi_filter"]["max_corner"].SetStyle(YAML::EmitterStyle::Flow);
  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

void MultiBallTriangulationParameters::SetExecutionMode(const std::string& mode) {
  if (mode == "calibration" || mode == "healthcheck") {
    ghost_balls_policy = GhostBallsPolicy::kNone;
  } else {
    ghost_balls_policy = ghost_balls_policy_settings;
  }
}
}  // namespace multi_ball_triangulation
