// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "racket_pose_estimation/RacketPoseExtractor.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <opencv2/cudawarping.hpp>

#include "ace_loggers/ace_loggers.hpp"
#include "aps/SyncedImagesCallback.hpp"
#include "racket_pose_estimation/RacketInference.hpp"
#include "racket_pose_estimation/RacketPoseFitter.hpp"

///// Internals
#include "internal/ExtractorWorker.hpp"

namespace perception {

class RacketPoseExtractor::RacketPoseExtractorImpl {
 public:
  explicit RacketPoseExtractorImpl(const std::string& log_name) {
    parameters = std::make_shared<RacketParameters>();
    shared_context = std::make_shared<WorkerSharedContext>(
      ::datalogger::ConstructSubLogName(::datalogger::ConstructFullLogName(log_name), "racket_pose_estimator"));
  }
  ~RacketPoseExtractorImpl() = default;

  bool Start(const std::string& camera_calib_params_path, const std::string& racket_params_path,
             const std::string& model_path, bool perform_estimation) {
    parameters->Initialize(racket_params_path);
    parameters->PrintParameters();

    camera_calibration_params = std::make_shared<calibration::CameraCalibrationParameters>();
    if (camera_calibration_params->Initialize(camera_calib_params_path)) {
      camera_calibration_params->PrintParameters();
    } else {
      LOG(ERROR) << "Failed to initialize camera calibration parameters." << std::endl;
      return false;
    }

    shared_context->Initialize(camera_calibration_params, parameters);

    // now find camera index from calibration
    std::map<std::string, int> cam_name_idx;
    for (size_t j = 0; j < camera_calibration_params->camera_names.size(); ++j) {
      cam_name_idx[camera_calibration_params->camera_names[j]] = static_cast<int>(j);
    }

    // now find camera index from calibration
    for (auto& r : parameters->rackets) {
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
      auto worker = std::make_shared<ExtractorWorker>(i, model_path, shared_context, perform_estimation);
      worker->Start();
      workers.push_back(worker);
    }
    return true;
  }

  const std::vector<RacketFrameFeatures>& GetLastDetections() { return shared_context->GetAndClearDetections(); }
  void Stop() {
    for (auto& worker : workers) {
      worker->Stop();
    }
    workers.clear();
  }

  bool OnImagesCallback(size_t seq_id, const std::vector<RacketPoseExtractor::ImageData>& images, bool async) {
    if (!RacketParameters::CheckSampling(seq_id, shared_context->racket_params->extractor.sampling)) {
      return false;  // early skip
    }
    auto result = std::any_of(workers.begin(), workers.end(), [&](std::shared_ptr<ExtractorWorker>& worker) {
      return worker->Assign(seq_id, images, async);
    });

    if (!result) {
      LOG(WARNING) << "Failed to process: " << seq_id << ", not enough workers to handle the job";
    }
    return result;
  }
  /////////

  calibration::CameraCalibrationParameters::SharedPtr camera_calibration_params;

  std::vector<std::shared_ptr<ExtractorWorker>> workers;

  RacketParameters::SharedPtr parameters;
  WorkerSharedContext::SharedPtr shared_context;
};

RacketPoseExtractor::RacketPoseExtractor(const std::string& log_name) {
  impl_ = std::make_shared<RacketPoseExtractorImpl>(log_name);
}
RacketPoseExtractor::~RacketPoseExtractor() { Stop(); }

bool RacketPoseExtractor::Start(const std::string& camera_calib_params_path, const std::string& Racket_params_path,
                                const std::string& model_path, bool perform_estimation) {
  return impl_->Start(camera_calib_params_path, Racket_params_path, model_path, perform_estimation);
}
void RacketPoseExtractor::Stop() { impl_->Stop(); }

const std::vector<RacketFrameFeatures>& RacketPoseExtractor::GetLastDetections() {
  if (impl_ == nullptr) {
    throw std::runtime_error("RacketPoseExtractor::GetLastDetections() - not initialized!");
  }
  return impl_->GetLastDetections();
}
bool RacketPoseExtractor::OnImages(size_t seq_id, const std::vector<ImageData>& images, bool async) {
  return impl_->OnImagesCallback(seq_id, images, async);
}
void RacketPoseExtractor::SetROI(size_t racket_idx, size_t camera_idx, const Eigen::Vector4i& roi) {
  impl_->shared_context->SetROI(racket_idx, camera_idx, roi);
}
bool RacketPoseExtractor::GetROI(size_t racket_idx, size_t camera_idx, Eigen::Vector4i& roi) {
  return impl_->shared_context->GetROI(racket_idx, camera_idx, roi);
}

void RacketPoseExtractor::SetRacketPose3DCallback(RacketPose3DCallback callback) {
  impl_->shared_context->on_racket_pose3d_callback = callback;
}
void RacketPoseExtractor::SetRacketDetection2DCallback(RacketDetection2DCallback callback) {
  impl_->shared_context->on_racket_detection2d_callback = callback;
}
RacketPoseFeatures::SharedPtr RacketPoseExtractor::GetLastDetectionForCamera(size_t camera_idx) {
  if (impl_ == nullptr) {
    throw std::runtime_error("RacketPoseExtractor::GetLastDetectionForCamera() - not initialized!");
  }
  return impl_->shared_context->GetLastDetections(camera_idx, 0);
}

calibration::CameraCalibrationParameters::SharedPtr RacketPoseExtractor::GetCameraCalibrationParameters() {
  return impl_->camera_calibration_params;
}
RacketParameters::SharedPtr RacketPoseExtractor::GetRacketParameters() { return impl_->parameters; }
::datalogger::DataWriter::SharedPtr RacketPoseExtractor::GetDataLogger() { return impl_->shared_context->datawriter; }
void RacketPoseExtractor::SetDataLoggerFileName(const std::string& name) {
  std::lock_guard<std::mutex> lock(impl_->shared_context->data_mutex);
  impl_->shared_context->datawriter->SetFileName(name);
}
}  // namespace perception
