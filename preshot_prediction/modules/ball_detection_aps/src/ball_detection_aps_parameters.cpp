// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ball_detection_aps/ball_detection_aps_parameters.hpp"

#include <numbers>

#include "ace_loggers/logging.hpp"
#include "ace_rt_profiles/profile_manager.hpp"

namespace ball_detection_aps {

bool BallDetectionAPSParameters::UpdateParametersFromYaml() {
  vision_common::BallDetectionParameters::UpdateParametersFromYaml();
  try {
    // cuda device UUID
    cuda_device_uuids = yaml_node_["cuda_device_uuids"].as<std::vector<std::string>>();

    // color filter
    blur_kernel_size = yaml_node_["color_filter"]["blur_kernel_size"].as<int>();

    const auto hsv_lower_boundary_yaml = yaml_node_["color_filter"]["hsv_lower_boundary"].as<std::array<double, 3>>();
    const auto hsv_upper_boundary_yaml = yaml_node_["color_filter"]["hsv_upper_boundary"].as<std::array<double, 3>>();
    for (int i = 0; i < 3; ++i) {
      hsv_lower_boundary[i] = hsv_lower_boundary_yaml[i];
      hsv_upper_boundary[i] = hsv_upper_boundary_yaml[i];
    }

    // motion filter
    motion_filter_enable = yaml_node_["motion_filter"]["enable"].as<bool>();
    motion_filter_delay = yaml_node_["motion_filter"]["delay"].as<double>();
    motion_filter_lower_boundary = yaml_node_["motion_filter"]["lower_boundary"].as<int>();
    motion_filter_upper_boundary = yaml_node_["motion_filter"]["upper_boundary"].as<int>();

    // appearance filter
    min_circularity_ratio = yaml_node_["appearance_filter"]["min_circularity_ratio"].as<double>();
    min_radius = yaml_node_["appearance_filter"]["min_radius"].as<double>();

    // border filter
    border_filter_enable = yaml_node_["border_filter"]["enable"].as<bool>();
    border_filter_plane_depth = yaml_node_["border_filter"]["plane_depth"].as<double>();
    border_filter_margin_size = yaml_node_["border_filter"]["margin_size"].as<double>();

    // markers
    markers_enable = yaml_node_["markers"]["enable"].as<bool>();
    markers_min_circularity_ratio = yaml_node_["markers"]["min_circularity_ratio"].as<float>();
    markers_min_radius_ratio = yaml_node_["markers"]["min_radius_ratio"].as<float>();
    markers_max_radius_ratio = yaml_node_["markers"]["max_radius_ratio"].as<float>();
    markers_cone_angle_threshold =
      yaml_node_["markers"]["cone_angle_threshold"].as<float>() * std::numbers::pi_v<float> / 180.0F;

    // markers (opencv)
    markers_opencv_adaptive_threshold_power = yaml_node_["markers"]["opencv"]["adaptive_threshold_power"].as<float>();
    markers_opencv_adaptive_threshold_ratio = yaml_node_["markers"]["opencv"]["adaptive_threshold_ratio"].as<float>();

    // cpu affinity
    rt_profile = yaml_node_["rt_profile"].as<std::string>();
    const auto cpu_affinity = ace_rt_profiles::ProfileManager::GetInstance().GetAffinity(rt_profile);

    if (cpu_affinity.size() < camera_names.size()) {
      LOG(WARNING) << "The number of allocated cores (cpu_affinity=" << cpu_affinity.size()
                   << ") is less than cameras count=" << camera_names.size()
                   << ". This might lead to drop in performance!";
    }
  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool BallDetectionAPSParameters::UpdateYamlFromParameters() {
  vision_common::BallDetectionParameters::UpdateYamlFromParameters();
  try {
    // cuda device UUID
    yaml_node_["cuda_device_uuids"] = cuda_device_uuids;
    if (cuda_device_uuids.empty()) {
      LOG(ERROR) << "No CUDA device uuid set.\n";
      return false;
    }

    // color filter
    yaml_node_["color_filter"]["blur_kernel_size"] = blur_kernel_size;
    const std::array hsv_lower_boundary_yaml = {hsv_lower_boundary[0], hsv_lower_boundary[1], hsv_lower_boundary[2]};
    yaml_node_["color_filter"]["hsv_lower_boundary"] = hsv_lower_boundary_yaml;
    const std::array hsv_upper_boundary_yaml = {hsv_upper_boundary[0], hsv_upper_boundary[1], hsv_upper_boundary[2]};
    yaml_node_["color_filter"]["hsv_upper_boundary"] = hsv_upper_boundary_yaml;

    yaml_node_["color_filter"]["hsv_lower_boundary"].SetStyle(YAML::EmitterStyle::Flow);
    yaml_node_["color_filter"]["hsv_upper_boundary"].SetStyle(YAML::EmitterStyle::Flow);

    // motion filter
    yaml_node_["motion_filter"]["enable"] = motion_filter_enable;
    yaml_node_["motion_filter"]["delay"] = motion_filter_delay;
    yaml_node_["motion_filter"]["lower_boundary"] = motion_filter_lower_boundary;
    yaml_node_["motion_filter"]["upper_boundary"] = motion_filter_upper_boundary;

    // appearance filter
    yaml_node_["appearance_filter"]["min_circularity_ratio"] = min_circularity_ratio;
    yaml_node_["appearance_filter"]["min_radius"] = min_radius;

    // border filter
    yaml_node_["border_filter"]["enable"] = border_filter_enable;
    yaml_node_["border_filter"]["plane_depth"] = border_filter_plane_depth;
    yaml_node_["border_filter"]["margin_size"] = border_filter_margin_size;

    // markers
    yaml_node_["markers"]["enable"] = markers_enable;
    yaml_node_["markers"]["min_circularity_ratio"] = markers_min_circularity_ratio;
    yaml_node_["markers"]["min_radius_ratio"] = markers_min_radius_ratio;
    yaml_node_["markers"]["max_radius_ratio"] = markers_max_radius_ratio;
    yaml_node_["markers"]["cone_angle_threshold"] =
      markers_cone_angle_threshold * std::numbers::inv_pi_v<float> * 180.0F;

    // markers (opencv)
    yaml_node_["markers"]["opencv"]["adaptive_threshold_power"] = markers_opencv_adaptive_threshold_power;
    yaml_node_["markers"]["opencv"]["adaptive_threshold_ratio"] = markers_opencv_adaptive_threshold_ratio;
  } catch (YAML::Exception& e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace ball_detection_aps
