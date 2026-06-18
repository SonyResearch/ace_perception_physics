// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "racket_pose_estimation/RacketParameters.hpp"

#include <cmath>
#include <iostream>

#include "ace_loggers/ace_loggers.hpp"

namespace perception {

RacketParameters::RacketParameters() : BaseParameters("racket_parameters") {
  triangulator = std::make_shared<triangulation::TriangulationParameters>();
}

void RacketParameters::UpdateInternalParameters() {}

void RacketParameters::ParseCameras(YAML::Node node) {
  for (auto it = node.begin(); it != node.end(); ++it) {
    try {
      CameraParameters cam;
      cam.name = it->first.as<std::string>();
      cameras.push_back(cam);
    } catch (YAML::Exception& e) {
      LOG(WARNING) << "Failed to parse camera: " << it->first << " : " << e.what();
      continue;
    }
  }
}
bool RacketParameters::UpdateParametersFromYaml() {
  static const std::string KTermResetDefaults = "\033[0m";

  try {
    ParseCameras(yaml_node_["cameras"]);
    auto rackets_node = yaml_node_["rackets"];
    for (auto it = rackets_node.begin(); it != rackets_node.end(); ++it) {
      try {
        PlayerParameters params;
        params.name = it->first.as<std::string>();
        params.racket_id = it->second["id"].as<int>();

        for (const auto& cam_yaml : it->second["cameras"]) {
          CameraROI cam;
          cam.camera_name = cam_yaml["name"].as<std::string>();

          bool cam_found = false;

          for (size_t i = 0; i < cameras.size(); ++i) {
            if (cameras[i].name == cam.camera_name) {
              cam.camera_params_index = i;
              cam_found = true;
              break;
            }
          }
          if (!cam_found) {
            LOG(ERROR) << "Failed to find camera named: " << cam.camera_name << " from Racket ID: " << params.racket_id
                       << " in the camera list!";
            throw std::invalid_argument("Camera name not defined");
          }

          auto roi = ace_yaml::SafeGetValue<std::vector<int>>(cam_yaml["ROI"], {0, 0, 1440, 1080});
          cam.roi = cv::Rect2i(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]);

          params.cameras[cam.camera_params_index] = std::move(cam);
        }
        rackets[params.racket_id] = params;
      } catch (YAML::Exception& e) {
        LOG(WARNING) << "Failed to parse racket: " << it->first.as<int>() << " : " << e.what();
        continue;
      }
    }

    for (auto& r : this->rackets) {
      auto& cams = r.second.cameras;
      for (auto& cam : cams) {
        cameras_names.insert(cam.second.camera_name);
      }
    }

    auto capture_node = yaml_node_["capture"];
    capture.camera_buffer_length = capture_node["camera_buffer_length"].as<int>();
    capture.worker_queue_length = capture_node["worker_queue_length"].as<int>();
    capture.use_variable_response_time = capture_node["use_variable_response_time"].as<bool>();
    capture.left_hand = ace_yaml::SafeGetValue<float>(capture_node["left_hand"], capture.left_hand);
    capture.right_hand = ace_yaml::SafeGetValue<float>(capture_node["right_hand"], capture.right_hand);
    {
      auto iweight_sum = 1.0F / (capture.left_hand + capture.right_hand);
      capture.left_hand = capture.left_hand * iweight_sum;
      capture.right_hand = capture.right_hand * iweight_sum;
    }

    auto fitter_node = yaml_node_["fitter"];
    fitter.max_skips_reset = fitter_node["max_skips_reset"].as<int>();
    fitter.max_delta_err_reset = fitter_node["max_delta_err_reset"].as<float>();
    fitter.max_error = fitter_node["max_error"].as<float>();
    fitter.max_iterations = fitter_node["max_iterations"].as<int>();
    fitter.initial_rate = fitter_node["initial_rate"].as<float>();
    fitter.min_axis_length = fitter_node["min_axis_length"].as<float>();

    auto extractor_node = yaml_node_["extractor"];
    extractor.outdated_diff = extractor_node["outdated_diff"].as<int>();
    extractor.workers_count = extractor_node["workers_count"].as<int>();
    extractor.confidence_threshold = extractor_node["confidence_threshold"].as<float>();
    extractor.nms_threshold = extractor_node["nms_threshold"].as<float>();
    extractor.sampling = extractor_node["sampling"].as<float>();
    extractor.latest_weight = extractor_node["latest_weight"].as<float>();
    extractor.history_difference = extractor_node["history_difference"].as<int>();
    extractor.max_batch_size = ace_yaml::SafeGetValue<int>(extractor_node["max_batch_size"], extractor.max_batch_size);
    extractor.device_index = ace_yaml::SafeGetValue<int>(extractor_node["device_index"], extractor.device_index);
    extractor.opt_batch_size = ace_yaml::SafeGetValue<int>(extractor_node["opt_batch_size"], extractor.opt_batch_size);
    extractor.rt_profile = extractor_node["rt_profile"].as<std::string>();
    extractor.roi = Eigen::Vector2i(extractor_node["roi"].as<std::vector<int>>().data());
    extractor.debug_output = ace_yaml::SafeGetValue<bool>(extractor_node["debug_output"], extractor.debug_output);

    auto triangulator_node = yaml_node_["triangulation"];
    // Triangulation
    triangulator->covariance_model =
      triangulation::StringToCovarianceModel(triangulator_node["covariance_model"].as<std::string>());
    triangulator->geometric_error_threshold = triangulator_node["geometric_error_threshold"].as<float>();
    triangulator->radius_error_threshold = triangulator_node["radius_error_threshold"].as<float>();
    triangulator->observation_variance = triangulator_node["observation_variance"].as<float>();
    triangulator->use_nonlinear_refinement = triangulator_node["nonlinear_refinement"]["use"].as<bool>();
    triangulator->tol_position = triangulator_node["nonlinear_refinement"]["tol_position"].as<float>();
    triangulator->tol_reprojection_error =
      triangulator_node["nonlinear_refinement"]["tol_reprojection_error"].as<float>();
    triangulator->max_iterations = triangulator_node["nonlinear_refinement"]["max_iterations"].as<int>();

  } catch (YAML::Exception& e) {
    LOG(WARNING) << e.what();
  }

  return true;
}

bool RacketParameters::UpdateYamlFromParameters() { return true; }
}  // namespace perception
