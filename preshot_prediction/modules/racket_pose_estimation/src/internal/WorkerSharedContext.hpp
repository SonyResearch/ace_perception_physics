// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <mutex>

#include "racket_pose_estimation/RacketInference.hpp"
#include "racket_pose_estimation/RacketPoseExtractor.hpp"
#include "racket_pose_estimation/datalogger.hpp"
#include "triangulation/datalogger.hpp"

namespace perception {

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  for (auto& x : v) {
    out << x << ", ";
  }
  out << "\b\b]";
  return out;
}
template <typename T, typename V>
std::ostream& operator<<(std::ostream& out, const std::unordered_map<T, V>& v) {
  out << "{";
  for (auto& x : v) {
    out << x.first << ":" << x.second << ", ";
  }
  out << "\b\b}";
  return out;
}
template <typename T, typename V>
std::ostream& operator<<(std::ostream& out, const std::pair<T, V>& v) {
  out << "(";
  out << v.first << ":" << v.second;
  out << ")";
  return out;
}

class WorkerSharedContext {
 public:
  using SharedPtr = std::shared_ptr<WorkerSharedContext>;
  using ConstSharedPtr = std::shared_ptr<const WorkerSharedContext>;

  explicit WorkerSharedContext(std::string log_file) : log_file_name(std::move(log_file)) {
    using std::chrono::high_resolution_clock;
    last_time = high_resolution_clock::now();
  }
  ~WorkerSharedContext() { datawriter = nullptr; }

  void Initialize(calibration::CameraCalibrationParameters::SharedPtr camera_params,
                  RacketParameters::SharedPtr racket_params) {
    this->racket_params = racket_params;
    camera_calib_params = camera_params;
    int index = 0;
    std::vector<int> racket_ids;
    for (const auto& it : racket_params->rackets) {
      // Initialize triangulator.
      auto datawriter_ptr = std::make_shared<triangulation::datalogger::DataWriter>(
        ::datalogger::ConstructFullLogName("racket_pose_" + std::to_string(index) + "_triangulation"),
        racket_params->ToString(), camera_calib_params->ToString());
      auto triangulator =
        std::make_shared<TriangulatorType>(racket_params->triangulator, camera_calib_params, std::move(datawriter_ptr));
      triangulator->SetCameras(camera_calib_params->camera_names);

      triangulators[it.second.racket_id] = triangulator;

      racket_ids.push_back(it.second.racket_id);
      index++;
    }

    datawriter = std::make_shared<racket_pose_estimation::datalogger::DataWriter>(log_file_name, racket_ids);
  }

  void AverageDetection(RacketPoseFeatures::SharedPtr& detection) {
    std::lock_guard<std::mutex> lock(history_mutex);
    auto key = std::pair(detection->player_index, detection->camera_index);
    if (detections_history.find(key) == detections_history.end()) {
      detections_history[key] = std::vector<RacketPoseFeatures::SharedPtr>();
    }
    auto& history = detections_history[key];
    history.push_back(detection);

    static auto multiply_keypoints = [](const std::vector<RacketPoseFeatures::Keypoint>& keypoints, float weight) {
      std::vector<RacketPoseFeatures::Keypoint> result;
      result.reserve(keypoints.size());
      for (const auto& kp : keypoints) {
        result.emplace_back(kp * weight);
      }
      return result;
    };
    static auto sum_keypoints = [](std::vector<RacketPoseFeatures::Keypoint>& keypoints,
                                   const std::vector<RacketPoseFeatures::Keypoint>& other) {
      for (size_t i = 0; i < keypoints.size(); ++i) {
        keypoints[i] += other[i];
      }
    };
    float total_weight = racket_params->extractor.latest_weight;
    auto keypoints = multiply_keypoints(detection->keypoints, total_weight);
    float weight = 1 - racket_params->extractor.latest_weight;
    auto max_distance = static_cast<float>(racket_params->extractor.history_difference);
    // std::cout << detection->camera_index << " - Adding Detection: " << detection->sequence_number << std::endl;
    for (size_t i = 0; i < history.size() - 1; ++i) {
      auto dist = std::abs<float>(static_cast<float>(history[i]->sequence_number) -
                                  static_cast<float>(detection->sequence_number));
      if (history[i]->sequence_number == detection->sequence_number || dist > max_distance) {
        // std::cout << "\t Removing: " << history[i]->sequence_number << std::endl;
        auto it = history.begin();
        std::advance(it, i);
        history.erase(it);
        --i;
      } else {
        float w = weight * (1 - dist / max_distance);
        sum_keypoints(keypoints, multiply_keypoints(history[i]->keypoints, w));
        total_weight += w;
      }
    }
    // std::cout << "Averaging Keypoints: " << total_weight << "/" << history.size() << std::endl;
    detection->keypoints = multiply_keypoints(keypoints, 1.0F / total_weight);
  }
  void AddDetection(std::chrono::time_point<std::chrono::high_resolution_clock> start,
                    const RacketFrameFeatures& features) {
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    std::lock_guard<std::mutex> lock(data_mutex);
    {
      // write to log file
      datawriter << start << std::chrono::high_resolution_clock::now()
                 << ::datalogger::RacketPoseEstimationDataType::kRacketFrameFeatures;
      datawriter << features;
    }

    latest_features.push_back(features);
    last_frame_id = std::max<size_t>(last_frame_id, features.sequence_number);
    ++total_processed;
    auto now = high_resolution_clock::now();
    auto diff = duration_cast<milliseconds>(now - last_time).count();
    if (diff >= 1000) {
      LOG(INFO) << "Processed Detections FPS: "
                << static_cast<int>(static_cast<double>(total_processed) * 1e3 / static_cast<double>(diff));
      last_time = now;
      total_processed = 0;
    }

    // update last detections
    for (auto d : features.features) {
      if (d == nullptr) {
        continue;
      }
      auto key = RacketParameters::PlayerCameraPair(d->player_index, d->camera_index);
      if (last_detections_percamera.find(key) == last_detections_percamera.end()) {
        last_detections_percamera[key] = nullptr;
      }

      if (last_detections_percamera[key] == nullptr ||
          last_detections_percamera[key]->sequence_number < features.sequence_number ||
          features.sequence_number < (last_detections_percamera[key]->sequence_number - 30)) {
        AverageDetection(d);
        last_detections_percamera[key] = d;
        if (on_racket_detection2d_callback) {
          on_racket_detection2d_callback(d->player_index, d->camera_index, d);
        }
      }
    }
    if (on_racket_pose3d_callback) {
      if (features.estimated_rackets.empty()) {
        for (const auto& racket : racket_params->rackets) {
          on_racket_pose3d_callback(features.sequence_number, racket.first, nullptr);
        }
      } else {
        for (const auto& d : features.estimated_rackets) {
          on_racket_pose3d_callback(features.sequence_number, d->racket_id, d);
        }
      }
    }
  }
  const std::vector<RacketFrameFeatures>& GetAndClearDetections() {
    {
      std::lock_guard<std::mutex> lock(data_mutex);
      latest_features_fetched = std::move(latest_features);
    }

    std::sort(
      latest_features_fetched.begin(), latest_features_fetched.end(),
      [](const RacketFrameFeatures& a, RacketFrameFeatures& b) { return a.sequence_number < b.sequence_number; });
    return latest_features_fetched;
  }

  RacketPoseFeatures::SharedPtr GetLastDetections(size_t racket_idx, size_t camera_idx, int max_outdate_diff = -1) {
    if (max_outdate_diff == 0) {
      max_outdate_diff = static_cast<int>(racket_params->extractor.outdated_diff);
    }

    auto key = std::pair(racket_idx, camera_idx);
    std::lock_guard<std::mutex> lock(data_mutex);
    if (last_detections_percamera.find(key) == last_detections_percamera.end() ||
        last_detections_percamera[key] == nullptr) {
      return nullptr;
    }

    if (max_outdate_diff > 0 &&
        static_cast<int>(last_frame_id) - static_cast<int>(last_detections_percamera[key]->sequence_number) >
          max_outdate_diff) {
      // std::cout << "Out dated: " << camera_idx << ":" << last_frame_id << "/"
      //           << last_detections_percamera[camera_idx]->sequence_number << std::endl;
      return nullptr;
    }
    return last_detections_percamera[key];
  }

  void TriangulateDetections(RacketFrameFeatures& features) {
    //
    for (auto it : racket_params->rackets) {
      auto racket_id = it.first;
      // LOG(INFO) << "Triangulating for: " << racket_id;
      auto& player = it.second;
      auto& cams = player.cameras;
      // fill the missing camera indicies from last detections
      for (auto cam : cams) {
        // if (cam.second.camera_calib_index == -1) {
        //   continue;
        // }
        if (find_if(features.features.begin(), features.features.end(),
                    [&cam, racket_id](const RacketPoseFeatures::SharedPtr& f) {
                      if (f) {
                        return cam.second.camera_calib_index == static_cast<int>(f->camera_index) &&
                               racket_id == f->player_index;
                      }
                      return false;
                    }) == features.features.end()) {
          auto f = GetLastDetections(racket_id, cam.second.camera_calib_index, 0);
          if (f != nullptr) {
            // LOG(WARNING) << "Filling detection for camera: ";
            features.features.push_back(f);
          }
        }
      }

      if (triangulators.find(racket_id) == triangulators.end()) {
        continue;
      }

      std::vector<Eigen::Vector2f> detections;
      std::unordered_set<int> obs;
      std::vector<int> obs_to_cam_idx;

      std::vector<size_t> populated_camera_indicies;
      int total_observations = 0;
      for (auto& f : features.features) {
        if (!f || f->player_index != racket_id) {
          continue;
        }
        // std::cout << "Processing feature for racket/camera :" << racket_id << "/" << f->camera_index << std::endl;
        if (f->confidence > 0 && std::find_if(cams.begin(), cams.end(), [f](const auto& cam) {
                                   return f->camera_index == cam.second.camera_calib_index;
                                 }) != cams.end()) {
          ++total_observations;
          // std::cout << "Adding observations for camera: " << f->camera_index << std::endl;
          populated_camera_indicies.push_back(f->camera_index);
          // std::cout << "Adding detection: " << d->center.transpose() << std::endl;
          detections.emplace_back(f->center.head<2>().cast<float>());
          obs.insert(static_cast<int>(detections.size()) - 1);
          obs_to_cam_idx.push_back(static_cast<int>(f->camera_index));

          // LOG(INFO) << "\tCamera: " << f->camera_index << " --> " << f->center.head<2>().transpose();
        }
      }
      if (detections.size() < 2) {
        continue;
      }
      bool undistort_keypoints = false;
      triangulation::TriangulatedPoint pt;
      std::vector<std::vector<Eigen::Vector2f>> all_detections;
      all_detections.resize(camera_calib_params->cameras.size());
      for (size_t i = 0; i < populated_camera_indicies.size(); i++) {
        all_detections[populated_camera_indicies[i]].push_back(detections[i]);
      }
      auto triangulations = triangulators[racket_id]->TriangulatePoints(all_detections);
      if (triangulations.empty()) {
        pt = triangulators[racket_id]->TriangulatePoint(detections, obs, obs_to_cam_idx, undistort_keypoints);
      } else {
        pt = triangulations[0];
      }
      auto pose = std::make_shared<RacketEstimatedPose>();
      pose->racket_id = racket_id;

      // LOG(INFO) << "Triangulated position for " << racket_id << ": " << pt.position.transpose();
      pose->position = pt.position;
      pose->reprojection_err = pt.reprojection_error;
      features.estimated_rackets.push_back(pose);
    }
  }

  void SetROI(size_t player_idx, size_t camera_idx, const Eigen::Vector4i& roi) {
    std::lock_guard<std::mutex> lock(data_mutex);
    camera_roi[RacketParameters::PlayerCameraPair(player_idx, camera_idx)] = roi;
  }
  bool GetROI(size_t player_idx, size_t camera_idx, Eigen::Vector4i& roi) {
    std::lock_guard<std::mutex> lock(data_mutex);
    auto it = camera_roi.find(RacketParameters::PlayerCameraPair(player_idx, camera_idx));
    if (it == camera_roi.end()) {
      return false;
    }
    roi = it->second;
    return true;
  }

  //  private:
  std::mutex history_mutex;
  std::mutex data_mutex;
  size_t last_frame_id{0};
  std::map<RacketParameters::PlayerCameraPair, Eigen::Vector4i> camera_roi;
  std::map<RacketParameters::PlayerCameraPair, RacketPoseFeatures::SharedPtr> last_detections_percamera;
  std::vector<RacketFrameFeatures> latest_features;
  std::vector<RacketFrameFeatures> latest_features_fetched;
  std::map<RacketParameters::PlayerCameraPair, std::vector<RacketPoseFeatures::SharedPtr>> detections_history;

  size_t total_processed{0};
  std::chrono::high_resolution_clock::time_point last_time;

  using TriangulatorType = triangulation::Triangulation<Eigen::Vector2f>;

  std::map<size_t, std::shared_ptr<TriangulatorType>> triangulators;
  RacketParameters::SharedPtr racket_params;
  calibration::CameraCalibrationParameters::SharedPtr camera_calib_params;

  RacketPoseExtractor::RacketPose3DCallback on_racket_pose3d_callback;
  RacketPoseExtractor::RacketDetection2DCallback on_racket_detection2d_callback;

  std::string log_file_name;
  racket_pose_estimation::datalogger::DataWriter::SharedPtr datawriter;
};
}  // namespace perception
