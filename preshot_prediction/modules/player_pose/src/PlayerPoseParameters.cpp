// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "player_pose/PlayerPoseParameters.hpp"

#include <cmath>
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "ace_loggers/ace_loggers.hpp"

namespace perception {

PlayerPoseParameters::PlayerPoseParameters() : BaseParameters("player_parameters") {
  triangulator = std::make_shared<triangulation::TriangulationParameters>();
  person_detector = std::make_shared<DetectorParameters>();

  person_detector->class_names.clear();
  person_detector->class_names.emplace_back("person");  // only person
  person_detector->type = YoloModelType::kKeypoints;
}

void PlayerPoseParameters::UpdateInternalParameters() {}

void PlayerPoseParameters::ParseCameras(YAML::Node node) {
  for (auto it = node.begin(); it != node.end(); ++it) {
    try {
      CameraParameters cam;
      cam.name = it->first.as<std::string>();
      // if (cameras_names.find(cam.name) == cameras_names.end()) {
      //   continue;  // skip cameras not in the mask
      // }
      auto rotation_type = it->second["rotate"].as<std::string>();
      if (rotation_type == "none") {
        cam.rotation_type = CameraParameters::RotationType::kNone;
        cam.rotation = 0;
      } else if (rotation_type == "CW") {
        cam.rotation_type = CameraParameters::RotationType::kCW;
        cam.rotation = -90;
      } else if (rotation_type == "CCW") {
        cam.rotation_type = CameraParameters::RotationType::kCCW;
        cam.rotation = 90;
      } else if (rotation_type == "180") {
        cam.rotation_type = CameraParameters::RotationType::kK180;
        cam.rotation = 180;
      } else if (rotation_type == "auto") {
        cam.rotation_type = CameraParameters::RotationType::kAuto;
        cam.rotation = 0;  // to be calculated later..
      }

      LOG(INFO) << "Adding camera with: " << rotation_type << ":" << static_cast<int>(cam.rotation_type);
      cameras.push_back(cam);
    } catch (YAML::Exception& e) {
      LOG(WARNING) << "Failed to parse camera: " << it->first << " : " << e.what();
      continue;
    }
  }
}

bool PlayerPoseParameters::UpdateParametersFromYaml() {
  static const std::string KTermResetDefaults = "\033[0m";

  try {
    ParseCameras(yaml_node_["cameras"]);
    auto players_node = yaml_node_["players"];
    for (auto it = players_node.begin(); it != players_node.end(); ++it) {
      try {
        PlayerParameters params;
        params.name = it->first.as<std::string>();
        params.player_id = it->second["id"].as<int>();

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
            LOG(ERROR) << "Failed to find camera named: " << cam.camera_name << " from Player ID: " << params.player_id
                       << " in the camera list!";
            throw std::invalid_argument("Camera name not defined");
          }

          auto roi = ace_yaml::SafeGetValue<std::vector<int>>(cam_yaml["ROI"], {0, 0, 1440, 1080});
          cam.roi = cv::Rect2i(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]);

          params.cameras[cam.camera_params_index] = std::move(cam);
        }
        players[params.player_id] = params;
      } catch (YAML::Exception& e) {
        LOG(WARNING) << "Failed to parse player: " << it->first.as<int>() << " : " << e.what();
        continue;
      }
    }
    for (auto& r : players) {
      auto& cams = r.second.cameras;
      for (auto& cam : cams) {
        cameras_names.insert(cam.second.camera_name);
      }
    }

    auto capture_node = yaml_node_["capture"];
    capture.camera_buffer_length = capture_node["camera_buffer_length"].as<int>();
    capture.worker_queue_length = capture_node["worker_queue_length"].as<int>();
    capture.use_variable_response_time = capture_node["use_variable_response_time"].as<bool>();

    auto extractor_node = yaml_node_["extractor"];
    extractor.outdated_diff = extractor_node["outdated_diff"].as<int>();
    extractor.workers_count = extractor_node["workers_count"].as<int>();
    extractor.apply_rotation_correction = extractor_node["apply_rotation_correction"].as<bool>();
    extractor.latest_weight = extractor_node["latest_weight"].as<float>();
    extractor.history_difference = extractor_node["history_difference"].as<int>();
    extractor.min_keypoint_confidence = extractor_node["min_keypoint_confidence"].as<float>();
    extractor.undistort_keypoints = extractor_node["undistort_keypoints"].as<bool>();
    extractor.sampling = extractor_node["sampling"].as<float>();
    extractor.rt_profile = extractor_node["rt_profile"].as<std::string>();


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

    auto person_detector_node = yaml_node_["person_detector"];
    person_detector->conf_threshold = person_detector_node["conf_threshold"].as<float>();
    person_detector->nms_threshold = person_detector_node["nms_threshold"].as<float>();
    person_detector->top_k = person_detector_node["topK"].as<int>();
    person_detector->bbox_inflation =
      Eigen::Vector2f(person_detector_node["bbox_inflation"].as<std::vector<float>>().data());
    person_detector->latest_weight = person_detector_node["latest_weight"].as<float>();
    person_detector->history_difference = person_detector_node["history_difference"].as<int>();
    person_detector->batch_size = person_detector_node["batch_size"].as<int>();
    person_detector->device_index =
      ace_yaml::SafeGetValue<int>(person_detector_node["device_index"], person_detector->device_index);

  } catch (YAML::Exception& e) {
    LOG(WARNING) << e.what();
  }

  return true;
}

bool PlayerPoseParameters::UpdateYamlFromParameters() { return true; }

void PlayerPoseParameters::InitializeCameras(calibration::CameraCalibrationParameters::SharedPtr camera_calib_params) {
  for (size_t index = 0; index < cameras.size(); ++index) {
    auto& pcam = cameras[index];
    int cam_index = -1;
    for (int i = 0; i < static_cast<int>(camera_calib_params->camera_names.size()); i++) {
      if (camera_calib_params->camera_names[i] == pcam.name) {
        cam_index = i;
        break;
      }
    }
    if (cam_index == -1) {
      continue;
    }

    std::cout << "Mapping camera: " << pcam.name << " with index: " << cam_index << " to index: " << index << std::endl;
    cameras_map[cam_index] = index;  // map camera index
    auto& cam = camera_calib_params->cameras[pcam.name];
    pcam.resolution = cam.resolution;
    // calculate camera rotation along x (forward) axis
    if (pcam.rotation_type == PlayerPoseParameters::CameraParameters::RotationType::kAuto) {
      Eigen::Vector2f p0;
      Eigen::Vector2f p1;
      Eigen::Vector2f axis;
      cam.ProjectPoint(Eigen::Vector3f(0, 0, 0), &p0);
      cam.ProjectPoint(Eigen::Vector3f(0, 0, 0.1), &p1);
      axis = (p1 - p0);
      axis.normalize();
      pcam.rotation = atan2(axis.x(), -axis.y()) * 180 / M_PI;
    }

    camera_rotation_mat[cam_index] = cv::getRotationMatrix2D(
      cv::Point2f(static_cast<float>(cam.resolution[0]) / 2.F, static_cast<float>(cam.resolution[1]) / 2.F),
      pcam.rotation, 1.0);
    camera_rotation_inv_mat[cam_index] = cv::getRotationMatrix2D(
      cv::Point2f(static_cast<float>(cam.resolution[0]) / 2.F, static_cast<float>(cam.resolution[1]) / 2.F),
      -pcam.rotation, 1.0);
  }
}

cv::cuda::GpuMat PlayerPoseParameters::RotateImage(size_t camera_index, cv::cuda::GpuMat frame) {
  auto index = cameras_map[camera_index];  // map camera index
  const auto& pcam = cameras[index];
  using RotationType = PlayerPoseParameters::CameraParameters::RotationType;
  cv::cuda::GpuMat warped;

  switch (pcam.rotation_type) {
    case RotationType::kCW:
      cv::cuda::rotate(frame, warped, cv::Size(frame.rows, frame.cols), pcam.rotation, frame.rows - 1, 0,
                       cv::INTER_LINEAR);
      break;
    case RotationType::kCCW:
      cv::cuda::rotate(frame, warped, cv::Size(frame.rows, frame.cols), pcam.rotation, 0, frame.cols, cv::INTER_LINEAR);
      break;
    case RotationType::kK180:
      cv::cuda::rotate(frame, warped, cv::Size(frame.cols, frame.rows), pcam.rotation, frame.cols - 1, frame.rows - 1,
                       cv::INTER_LINEAR);
      break;
    case RotationType::kAuto:
      cv::cuda::warpAffine(frame, warped, camera_rotation_mat[camera_index], frame.size());
      break;
    default:
      warped = frame;
  }
  return warped;
}

void PlayerPoseParameters::InvRotateKeypoints(size_t camera_index, std::vector<cv::Point2f>& points) {
  auto index = cameras_map[camera_index];  // map camera index
  const auto& pcam = cameras[index];
  // calculate camera rotation along x (forward) axis
  using RotationType = PlayerPoseParameters::CameraParameters::RotationType;
  switch (pcam.rotation_type) {
    case RotationType::kNone:
      break;
    case RotationType::kCCW:
      for (auto& kp : points) {
        auto k = kp;
        kp.x = static_cast<float>(pcam.resolution[0]) - k.y;
        kp.y = k.x;
      }
      break;
    case RotationType::kCW:
      for (auto& kp : points) {
        auto k = kp;
        kp.x = k.y;
        kp.y = static_cast<float>(pcam.resolution[1]) - k.x;
      }
      break;
    case RotationType::kK180:
      for (auto& kp : points) {
        kp.x = static_cast<float>(pcam.resolution[0]) - kp.x;
        kp.y = static_cast<float>(pcam.resolution[1]) - kp.y;
      }
      break;
    case RotationType::kAuto: {
      auto& rot = camera_rotation_inv_mat[camera_index];
      cv::transform(points, points, rot);
    } break;
  }
}
void PlayerPoseParameters::InvRotateDetections(PlayerFrameFeatures& features) {
  // inverse rotation to be back in image space
  for (auto& detection : features.features) {
    auto index = cameras_map[detection->camera_index];  // map camera index
    auto& pcam = cameras[index];
    // calculate camera rotation along x (forward) axis
    using RotationType = PlayerPoseParameters::CameraParameters::RotationType;
    switch (pcam.rotation_type) {
      case RotationType::kNone:
        break;
      case RotationType::kCCW:
        for (auto& kp : detection->keypoints) {
          auto k = kp;
          kp.x() = static_cast<float>(pcam.resolution[0]) - k.y();
          kp.y() = k.x();
        }
        break;
      case RotationType::kCW:
        for (auto& kp : detection->keypoints) {
          auto k = kp;
          kp.x() = k.y();
          kp.y() = static_cast<float>(pcam.resolution[1]) - k.x();
        }
        break;
      case RotationType::kK180:
        for (auto& kp : detection->keypoints) {
          kp.x() = static_cast<float>(pcam.resolution[0]) - kp.x();
          kp.y() = static_cast<float>(pcam.resolution[1]) - kp.y();
        }
        break;
      case RotationType::kAuto: {
        auto& rot = camera_rotation_inv_mat[detection->camera_index];
        std::vector<cv::Point2f> points;
        for (auto& kpt : detection->keypoints) {
          points.emplace_back(kpt.x(), kpt.y());
        }
        cv::transform(points, points, rot);
        for (size_t i = 0; i < points.size(); ++i) {
          detection->keypoints[i].x() = points[i].x;
          detection->keypoints[i].y() = points[i].y;
        }
      } break;
    }
  }
}
void PlayerPoseParameters::InvRotateBBox(size_t camera_index, PlayerBBoxDetection::Boundingbox& rect) {
  std::vector<cv::Point2f> rect_points = {
    {static_cast<float>(rect.x()), static_cast<float>(rect.y())},
    {static_cast<float>(rect.x() + rect.z()), static_cast<float>(rect.y())},
    {static_cast<float>(rect.x() + rect.z()), static_cast<float>(rect.y() + rect.w())},
    {static_cast<float>(rect.x()), static_cast<float>(rect.y() + rect.w())}};
  InvRotateKeypoints(camera_index, rect_points);
  auto bbox = cv::boundingRect(rect_points);
  rect.x() = bbox.x;
  rect.y() = bbox.y;
  rect.z() = bbox.width;
  rect.w() = bbox.height;
}

}  // namespace perception
