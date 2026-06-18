// Confidential, Copyright 2025, Sony AI, All rights reserved
#include "player_pose/PlayerPoseExtractor.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cuda_common/TRTEngine.hpp>
#include <mutex>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "ace_loggers/ace_loggers.hpp"

///// Internals
#include "internal/ExtractorWorker.hpp"

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

class PlayerPoseExtractor::PlayerPoseExtractorImpl {
 public:
  explicit PlayerPoseExtractorImpl(const std::string& log_name) {
    parameters = std::make_shared<PlayerPoseParameters>();
    shared_context = std::make_shared<WorkerSharedContext>(::datalogger::ConstructSubLogName(log_name, "player_pose"));
  }
  ~PlayerPoseExtractorImpl() = default;

  bool Start(const std::string& camera_calib_params_path, const std::string& player_params_path,
             const std::string& model_path) {
    parameters->Initialize(player_params_path);
    parameters->PrintParameters();

    camera_calibration_params = std::make_shared<calibration::CameraCalibrationParameters>();
    if (camera_calibration_params->Initialize(camera_calib_params_path)) {
      camera_calibration_params->PrintParameters();
    } else {
      LOG(ERROR) << "Failed to initialize camera calibration parameters." << std::endl;
      return false;
    }

    parameters->InitializeCameras(camera_calibration_params);

    shared_context->outdated_diff = parameters->extractor.outdated_diff;

    shared_context->Initialize(camera_calibration_params, parameters);

    // now find camera index from calibration
    for (auto& r : parameters->players) {
      auto& cams = r.second.cameras;
      for (auto& cam : cams) {
        for (size_t j = 0; j < camera_calibration_params->camera_names.size(); ++j) {
          if (camera_calibration_params->camera_names[j] == cam.second.camera_name) {
            cam.second.camera_calib_index = j;
            break;
          }
        }
      }
    }

    for (size_t i = 0; i < parameters->extractor.workers_count; ++i) {
      auto worker = std::make_shared<ExtractorWorker>(i, shared_context, model_path);
      worker->Start();
      workers.push_back(worker);
    }
    return true;
  }

  const std::vector<PlayerFrameFeatures>& GetLastDetections() { return shared_context->GetAndClearDetections(); }
  void Stop() {
    for (const auto& worker : workers) {
      worker->Stop();
    }
    workers.clear();
  }

  bool OnImagesCallback(size_t seq_id, std::vector<PlayerPoseExtractor::ImageData::SharedPtr>& images) {
    if (!PlayerPoseParameters::CheckSampling(seq_id, parameters->extractor.sampling)) {
      return false;  // early skip
    }
    auto result = std::any_of(workers.begin(), workers.end(),
                              [&](std::shared_ptr<ExtractorWorker>& worker) { return worker->Assign(seq_id, images); });

    if (!result) {
      LOG(WARNING) << "Failed to process: " << seq_id << ", not enough workers to handle the job";
    }
    return result;
  }
  /////////

  std::vector<std::shared_ptr<ExtractorWorker>> workers;

  PlayerPoseParameters::SharedPtr parameters;
  triangulation::TriangulationParameters::SharedPtr triangulator_params;
  WorkerSharedContext::SharedPtr shared_context;
  calibration::CameraCalibrationParameters::SharedPtr camera_calibration_params;
};

PlayerPoseExtractor::PlayerPoseExtractor(const std::string& log_name) {
  impl_ = std::make_shared<PlayerPoseExtractorImpl>(log_name);
}
PlayerPoseExtractor::~PlayerPoseExtractor() { Stop(); }

void PlayerPoseExtractor::Start(const std::string& camera_calib_params_path, const std::string& Player_params_path,
                                const std::string& model_path) {
  impl_->Start(camera_calib_params_path, Player_params_path, model_path);
}
void PlayerPoseExtractor::Stop() { impl_->Stop(); }

bool PlayerPoseExtractor::OnImages(size_t seq_id, std::vector<PlayerPoseExtractor::ImageData::SharedPtr>& images) {
  return impl_->OnImagesCallback(seq_id, images);
}
void PlayerPoseExtractor::SetPlayerDetectedCallback(PlayerDetectedCallback callback) {
  impl_->shared_context->on_player_detected_callback = callback;
}
void PlayerPoseExtractor::SetPlayerPose2DCallback(PlayerPose2DCallback callback) {
  impl_->shared_context->on_player_pose2d_callback = callback;
}
void PlayerPoseExtractor::SetPlayerPose3DCallback(PlayerPose3DCallback callback) {
  impl_->shared_context->on_player_pose3d_callback = callback;
}

const std::vector<PlayerFrameFeatures>& PlayerPoseExtractor::GetLastDetections() {
  if (impl_ == nullptr) {
    throw std::runtime_error("PlayerPoseExtractor::GetLastDetections() - not initialized!");
  }
  return impl_->GetLastDetections();
}

PlayerPoseDetection::SharedPtr PlayerPoseExtractor::GetLastDetectionForCamera(int index) {
  if (impl_ == nullptr) {
    throw std::runtime_error("PlayerPoseExtractor::GetLastDetectionForCamera() - not initialized!");
  }
  return impl_->shared_context->GetLastDetections(index, 0);
}
PlayerBBoxDetection::SharedPtr PlayerPoseExtractor::GetLastBBoxDetectionForCamera(int index) {
  return impl_->shared_context->last_bbox_detections[index];
}

calibration::CameraCalibrationParameters::SharedPtr PlayerPoseExtractor::GetCameraCalibrationParameters() {
  return impl_->camera_calibration_params;
}
PlayerPoseParameters::SharedPtr PlayerPoseExtractor::GetPlayerParameters() { return impl_->parameters; }
::datalogger::DataWriter::SharedPtr PlayerPoseExtractor::GetDataLogger() { return impl_->shared_context->datawriter; }

void PlayerPoseExtractor::SetDataLoggerFileName(const std::string& name) {
  std::lock_guard<std::mutex> lock(impl_->shared_context->data_mutex);
  impl_->shared_context->datawriter->SetFileName(name);
}

}  // namespace perception
