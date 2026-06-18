// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <cuda_runtime.h>

#include <mutex>

#include "player_pose/PersonDetector.hpp"
#include "player_pose/PlayerPoseExtractor.hpp"
#include "player_pose/datalogger.hpp"
#include "triangulation/datalogger.hpp"
// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                                                                             \
  {                                                                                                                  \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE);                                                                                            \
    }                                                                                                                \
  }
namespace perception {

std::string keypoints_names[] = {
  "nose",  "r_eye", "l_eye", "r_ear", "l_ear",  "r_sho",  "l_sho", "r_elb", "l_elb",
  "r_wri", "l_wri", "r_hip", "l_hip", "r_knee", "l_knee", "r_ank", "l_ank",
};

class WorkerSharedContext {
 public:
  using SharedPtr = std::shared_ptr<WorkerSharedContext>;
  using ConstSharedPtr = std::shared_ptr<const WorkerSharedContext>;

  explicit WorkerSharedContext(std::string log_file) : log_file_name(std::move(log_file)) {
    using std::chrono::high_resolution_clock;
    last_time = high_resolution_clock::now();
  }
  virtual ~WorkerSharedContext() { image_processing_pool->Stop(); }

  void Initialize(calibration::CameraCalibrationParameters::SharedPtr camera_params,
                  PlayerPoseParameters::SharedPtr params) {
    this->player_params = params;
    camera_calib_params = camera_params;
    image_processing_pool = std::make_shared<ImageProcessingPool>(this);
    image_processing_pool->Start();

    std::vector<int> player_ids;
    for (const auto& it : player_params->players) {
      // Initialize triangulator.
      auto datawriter_ptr = std::make_shared<triangulation::datalogger::DataWriter>(
        ::datalogger::ConstructFullLogName("player_pose_" + std::to_string(it.second.player_id)),
        player_params->ToString(), camera_calib_params->ToString());
      auto triangulator =
        std::make_shared<TriangulatorType>(player_params->triangulator, camera_calib_params, std::move(datawriter_ptr));
      triangulator->SetCameras(camera_calib_params->camera_names);

      triangulators[it.first] = triangulator;
      player_ids.push_back(it.second.player_id);
    }

    last_bbox_detections.resize(camera_calib_params->cameras.size());
    datawriter = std::make_shared<player_pose::datalogger::DataWriter>(log_file_name, player_ids);
    std::cout << "Done initializing worker " << std::endl;
  }

  void AverageDetection(PlayerPoseDetection::SharedPtr detection) {
    std::scoped_lock<std::mutex> lock(history_mutex);
    detections_history.try_emplace(std::pair(detection->player_index, detection->camera_index),
                                   std::vector<PlayerPoseDetection::SharedPtr>());
    auto& history = detections_history[std::pair(detection->player_index, detection->camera_index)];
    history.push_back(detection);

    static auto multiply_keypoints = [](const std::vector<PlayerPoseDetection::Keypoint>& keypoints, float weight) {
      std::vector<PlayerPoseDetection::Keypoint> result;
      result.reserve(keypoints.size());
      for (const auto& kp : keypoints) {
        result.emplace_back(kp * weight);
      }
      return result;
    };
    static auto sum_keypoints = [](std::vector<PlayerPoseDetection::Keypoint>& keypoints,
                                   const std::vector<PlayerPoseDetection::Keypoint>& other) {
      for (size_t i = 0; i < keypoints.size(); ++i) {
        keypoints[i] += other[i];
      }
    };
    float total_weight = player_params->extractor.latest_weight;
    auto keypoints = multiply_keypoints(detection->keypoints, total_weight);
    float weight = 1 - player_params->extractor.latest_weight;
    auto max_distance = static_cast<float>(player_params->extractor.history_difference);
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
                    const PlayerFrameFeatures& features) {
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    std::scoped_lock<std::mutex> lock(data_mutex);
    {
      // data
      datawriter << start << std::chrono::high_resolution_clock::now()
                 << datalogger::PlayerPoseDataType::kPlayerFrameFeatures;
      datawriter << features;
    }
    latest_features.push_back(features);
    last_frame_id = std::max<size_t>(last_frame_id, features.sequence_number);
    ++total_processed;
    auto now = high_resolution_clock::now();
    auto diff = static_cast<double>(duration_cast<milliseconds>(now - last_time).count());
    if (diff >= 1000) {
      LOG(INFO) << "Processed Detections FPS: " << static_cast<int>(total_processed * 1e3 / diff);
      last_time = now;
      total_processed = 0;
    }
    // update last detections
    for (const auto& d : features.features) {
      auto key = std::pair(d->player_index, d->camera_index);
      if (last_detections_frameids.find(key) == last_detections_frameids.end()) {
        last_detections_frameids[key] = 0;
        last_detections_percamera[key] = nullptr;
      }

      if (last_detections_frameids[key] < features.sequence_number ||
          features.sequence_number < (last_detections_frameids[key] - 30)) {
        last_detections_frameids[key] = features.sequence_number;
        last_detections_percamera[key] = d;
      }
      if (on_player_pose2d_callback) {
        on_player_pose2d_callback(features.sequence_number, d);
      }
    }

    if (on_player_pose3d_callback) {
      if (features.estimated_players.empty()) {
        for (const auto& player : player_params->players) {
          on_player_pose3d_callback(features.sequence_number, player.first, nullptr);
        }
      } else {
        for (const auto& d : features.estimated_players) {
          on_player_pose3d_callback(features.sequence_number, d->player_id, d);
        }
      }
    }
  }
  const std::vector<PlayerFrameFeatures>& GetAndClearDetections() {
    {
      std::scoped_lock<std::mutex> lock(data_mutex);
      latest_features_fetched = std::move(latest_features);
    }

    std::sort(
      latest_features_fetched.begin(), latest_features_fetched.end(),
      [](const PlayerFrameFeatures& a, PlayerFrameFeatures& b) { return a.sequence_number < b.sequence_number; });
    return latest_features_fetched;
  }

  PlayerPoseDetection::SharedPtr GetLastDetections(size_t player_idx, size_t camera, int max_outdate_diff = -1) {
    if (max_outdate_diff == 0) {
      max_outdate_diff = outdated_diff;
    }
    std::scoped_lock<std::mutex> lock(data_mutex);
    auto key = std::pair(player_idx, camera);

    if (last_detections_frameids.find(key) == last_detections_frameids.end()) {
      return nullptr;
    }

    if (max_outdate_diff > 0 && static_cast<int>(last_frame_id - last_detections_frameids[key]) > max_outdate_diff) {
      // std::cout << "Out dated: " << camera << ":" << last_frame_id_ << "/" << last_detections_frameids_[key]
      //           << std::endl;
      return nullptr;
    }

    return last_detections_percamera[key];
  }

  void TriangulateDetections(PlayerFrameFeatures& features) {
    //
    for (auto it : player_params->players) {
      int player_id = it.first;
      auto& player = it.second;
      auto& cams = player.cameras;
      // fill the missing camera indicies from last detections
      for (auto cam : cams) {
        if (find_if(features.features.begin(), features.features.end(),
                    [&cam, player_id](const PlayerPoseDetection::SharedPtr& f) {
                      return cam.second.camera_calib_index == f->camera_index && f->player_index == player_id;
                    }) == features.features.end()) {
          auto f = GetLastDetections(player.player_id, cam.second.camera_calib_index, 0);
          if (f != nullptr) {
            // std::cout << "Filling detection for camera: " << cam << std::endl;
            features.features.push_back(f);
          }
        }
      }
      // std::cout << "Checking against camera indicies: " << cams << std::endl;
      std::vector<PlayerPoseDetection*> features_list;
      std::vector<size_t> populated_camera_indicies;
      int total_observations = 0;
      for (auto& f : features.features) {
        if (f->player_index != player_id) {
          continue;
        }
        // std::cout << "Processing feature for camera :" << f->camera_index << "/" << f->keypoints.size() << std::endl;
        if (std::find_if(cams.begin(), cams.end(), [f](const auto& cam) {
              return cam.second.camera_calib_index == f->camera_index;
            }) != cams.end()) {
          ++total_observations;
          // std::cout << "Adding observations for camera: " << f->camera_index << std::endl;
          populated_camera_indicies.push_back(f->camera_index);
          features_list.push_back(f.get());
        }
      }
      // std::cout << player_id << " - Attempting to triangulate for: " << total_observations << std::endl;
      if (total_observations < 2 || triangulators.find(player_id) == triangulators.end()) {
        continue;
      }

      auto pose = std::make_shared<PlayerPoseEstimate>();
      pose->player_id = player_id;
      pose->keypoints.resize(17);
      pose->projection_error.resize(17, std::numeric_limits<double>::quiet_NaN());
      pose->confidences.resize(17, 0);
      for (int i = 0; i < 17; ++i) {
        std::vector<Eigen::Vector2f> detections;
        std::unordered_set<int> obs;
        std::vector<int> obs_to_cam_idx;
        float confidence = 0;
        // std::cout << "Processing keypoint: " << keypoints_names[i] << std::endl;
        for (auto* f : features_list) {
          assert(f->keypoints.size() == 17 && "Keypoints size mismatch");
          if (f->keypoints[i].z() > player_params->extractor.min_keypoint_confidence) {
            // std::cout << "\tAdding: " << camera_calib_params_->camera_names[f->camera_index] << "--> "
            //           << f->keypoints[i].transpose() << std::endl;
            detections.emplace_back(f->keypoints[i].head<2>());
            obs.insert(static_cast<int>(detections.size()) - 1);
            obs_to_cam_idx.push_back(static_cast<int>(f->camera_index));
            confidence += f->keypoints[i].z();
          }
        }
        if (!detections.empty()) {
          triangulation::TriangulatedPoint pt = triangulators[player_id]->TriangulatePoint(
            detections, obs, obs_to_cam_idx, player_params->extractor.undistort_keypoints);
          pose->keypoints[i] = pt.position;
          pose->projection_error[i] = pt.reprojection_error;  // pt.covariance(0, 0);
          pose->confidences[i] = confidence / static_cast<float>(detections.size());
        }
      }
      features.estimated_players.push_back(pose);
    }
  }

  void OnPlayerDetection(PlayerBBoxDetection::SharedPtr detection) {
    auto& rect = detection->bbox;
    if (player_params->extractor.apply_rotation_correction) {
      player_params->InvRotateBBox(detection->camera_index, rect);

      // std::cout << "Player Detected: " << rect << std::endl;
    }
    if (detection->sequence_number > last_bbox_detections[detection->camera_index]->sequence_number ||
        detection->sequence_number < last_bbox_detections[detection->camera_index]->sequence_number - 30) {
      last_bbox_detections[detection->camera_index] = detection;
    }
    if (on_player_detected_callback != nullptr) {
      on_player_detected_callback(detection->sequence_number, detection->player_index, detection->camera_index, rect,
                                  detection->confidence);
    }
  }

  class ImageProcessingPool {
   public:
    using CallbackFunction = std::function<void(PlayerPoseExtractor::ImageData::SharedPtr, cudaStream_t)>;
    ImageProcessingPool(WorkerSharedContext* context) : context_(context) {}
    void Start() {
      cuda_streams.resize(kMaxThreads);
      for (int i = 0; i < kMaxThreads; ++i) {
        CUDA_CHECK(cudaStreamCreate(&cuda_streams[i]));
        workers.emplace_back([this, i]() {
          while (true) {
            Job job;
            {
              std::unique_lock<std::mutex> lock(data_mutex);
              cond_var.wait(lock, [this]() { return !image_queue.empty() || is_stopped; });
              if (is_stopped) {
                break;
              }
              job = std::move(image_queue.front());
              image_queue.pop();
            }
            auto image_data = job.image_data;
            if (image_data == nullptr || image_data->processed) {
              continue;
            }
            // process image_data

            job.process_callback(image_data, cuda_streams[i]);
          }
        });
      }
    }
    ~ImageProcessingPool() { Stop(); }

    void Enqueue(PlayerPoseExtractor::ImageData::SharedPtr image_data, CallbackFunction process_callback) {
      std::scoped_lock<std::mutex> lock(data_mutex);
      image_queue.push({image_data, process_callback});
      cond_var.notify_one();
    }
    void Stop() {
      {
        is_stopped = true;
        cond_var.notify_all();
      }
      for (auto& worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
      workers.clear();
    }

   private:
    WorkerSharedContext* context_;
    class Job {
     public:
      PlayerPoseExtractor::ImageData::SharedPtr image_data;
      ImageProcessingPool::CallbackFunction process_callback;
    };
    std::queue<Job> image_queue;
    std::vector<std::thread> workers;
    static constexpr int kMaxThreads{4};
    std::condition_variable cond_var;
    std::mutex data_mutex;
    bool is_stopped{false};
    std::vector<cudaStream_t> cuda_streams;
  };

  void EnqueueImage(PlayerPoseExtractor::ImageData::SharedPtr image_data,
                    ImageProcessingPool::CallbackFunction process_callback) {
    image_processing_pool->Enqueue(std::move(image_data), process_callback);
  }

  std::shared_ptr<ImageProcessingPool> image_processing_pool;
  std::mutex data_mutex;
  std::mutex history_mutex;
  size_t last_frame_id{0};
  std::map<PlayerPoseParameters::PlayerCameraPair, size_t> last_detections_frameids;
  std::map<PlayerPoseParameters::PlayerCameraPair, PlayerPoseDetection::SharedPtr> last_detections_percamera;
  std::map<PlayerPoseParameters::PlayerCameraPair, std::vector<PlayerPoseDetection::SharedPtr>> detections_history;
  std::vector<PlayerFrameFeatures> latest_features;
  std::vector<PlayerFrameFeatures> latest_features_fetched;
  std::vector<PlayerBBoxDetection::SharedPtr> last_bbox_detections;

  int total_processed{0};
  std::chrono::high_resolution_clock::time_point last_time;

  int outdated_diff;

  using TriangulatorType = triangulation::Triangulation<Eigen::Vector2f>;

  std::map<int, std::shared_ptr<TriangulatorType>> triangulators;
  PlayerPoseParameters::SharedPtr player_params;
  calibration::CameraCalibrationParameters::SharedPtr camera_calib_params;

  PlayerPoseExtractor::PlayerDetectedCallback on_player_detected_callback;
  PlayerPoseExtractor::PlayerPose2DCallback on_player_pose2d_callback;
  PlayerPoseExtractor::PlayerPose3DCallback on_player_pose3d_callback;

  std::string log_file_name;
  player_pose::datalogger::DataWriter::SharedPtr datawriter;
};
}  // namespace perception
