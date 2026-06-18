// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <ace_rt_profiles/profile_manager.hpp>
#include <utility>

#include "WorkerSharedContext.hpp"
#include "ace_loggers/ace_loggers.hpp"

namespace perception {

class ExtractorWorker {
  std::thread worker_thread_;
  bool is_done_{true};
  int worker_index_;
  std::mutex mutex_;
  std::mutex data_mutex_;
  std::condition_variable cond_;
  bool work_available_{false};
  bool perform_estimation_;
  using JobType = std::pair<size_t, std::vector<RacketPoseExtractor::ImageData>>;
  JobType job_;
  static constexpr std::chrono::milliseconds kWaitDuration{100};

  RacketFrameFeatures latest_detections_internal_;
  bool new_data_available_{false};

  std::map<size_t, RacketPoseFitter::SharedPtr> keypoint_fitters_;

  WorkerSharedContext::SharedPtr shared_context_;

  RacketInference::SharedPtr inference_;

 public:
  ExtractorWorker(int worker_index, const std::string& model_path, WorkerSharedContext::SharedPtr shared_context,
                  bool perform_estimation)
    : worker_index_(worker_index), perform_estimation_(perform_estimation), shared_context_(shared_context) {
    inference_ = std::make_shared<RacketInference>();
    if (!inference_->Initialize(model_path, shared_context->racket_params)) {
      LOG(ERROR) << "Failed to initialize tflite model." << std::endl;
    }
    if (perform_estimation_) {
      for (auto racket : shared_context_->racket_params->rackets) {
        auto fitter = std::make_shared<RacketPoseFitter>();
        fitter->SetCalibration(shared_context_->camera_calib_params);
        fitter->Initialize(shared_context_->racket_params);
        // fitter->SetCameras(racket.second.camera_indicies);
        keypoint_fitters_[racket.second.racket_id] = fitter;
      }
    }
  }
  ~ExtractorWorker() { Stop(); }

  bool Start() {
    is_done_ = false;
    worker_thread_ = std::thread(std::bind(&ExtractorWorker::WorkerThread, this));
    return true;
  }
  bool Stop() {
    is_done_ = true;
    work_available_ = false;
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    return true;
  }

  [[nodiscard]] bool IsReady() const { return !work_available_; }
  [[nodiscard]] bool Assign(size_t seq_id, const std::vector<RacketPoseExtractor::ImageData>& images, bool async) {
    if (work_available_) {
      return false;
    }
    std::lock_guard<std::mutex> lock(data_mutex_);
    job_ = JobType(seq_id, std::move(images));
    work_available_ = true;
    if (async) {
      cond_.notify_all();
    } else {
      RunInference();
    }
    return true;
  }

  [[nodiscard]] bool NewDataAvailable() const { return new_data_available_; }

  [[nodiscard]] bool GetDetections(RacketFrameFeatures& features) {
    if (new_data_available_ && !work_available_) {
      // std::lock_guard<std::mutex> lock(data_mutex_);
      features = latest_detections_internal_;
      new_data_available_ = false;
    }
    return true;
  }

  void RunInference() {
    auto sequence_number = job_.first;
    latest_detections_internal_.sequence_number = sequence_number;
    latest_detections_internal_.features.clear();
    latest_detections_internal_.estimated_rackets.clear();
    if (job_.second.empty()) {
      work_available_ = false;
      new_data_available_ = true;
      return;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    if (job_.second.size() != shared_context_->racket_params->cameras_names.size()) {
      LOG(WARNING) << "Running inference on a limited number of cameras: " << job_.second.size() << "/"
                   << shared_context_->racket_params->cameras_names.size();
    }
    std::vector<cv::cuda::GpuMat> frames;
    std::vector<cv::Mat> in_frames;
    const cv::Mat* first = nullptr;
    std::vector<Eigen::Vector2i> offsets;
    for (const auto& img : job_.second) {
      auto camera_index = img.first;

      for (const auto& racket : shared_context_->racket_params->rackets) {
        auto image = img.second;
        auto racket_cam_it = std::find_if(
          racket.second.cameras.begin(), racket.second.cameras.end(),
          [camera_index](const auto& cam_roi) { return cam_roi.second.camera_calib_index == camera_index; });
        if (racket_cam_it == racket.second.cameras.end()) {
          continue;
        }
        if (!first) {
          first = &img.second;
        }

        Eigen::Vector4i roi = Eigen::Vector4i(0, 0, image.cols - 1, image.rows - 1);
        if (shared_context_->GetROI(racket.second.racket_id, camera_index, roi)) {
          // crop the image
          roi.x() = std::max<int>(0, roi.x() - roi.x() % 2);
          roi.x() = std::min<int>(image.cols, roi.x());
          roi.y() = std::max<int>(0, roi.y() - roi.y() % 2);
          roi.y() = std::min<int>(image.rows, roi.y());
          roi.z() = std::min<int>(roi.z(), std::max(0, image.cols - roi.x()));
          roi.w() = std::min<int>(roi.w(), std::max(0, image.rows - roi.y()));
          if (roi.z() < 10 || roi.w() < 10) {
            continue;
          }
          cv::Rect rect(roi.x(), roi.y(), roi.z(), roi.w());
          // respect bayer format
          image = image(rect);
        }
        if (image.dims == 3) {
          // convert to 2D image with x channels
          image = cv::Mat(cv::Size(image.size[1], image.size[0]), CV_8UC(image.size[2]), image.data);
        }
        offsets.emplace_back(roi.x(), roi.y());
        // process the image so it can be used in inference
        auto frame = inference_->PreProcessImage(image);
        if (shared_context_->racket_params->extractor.debug_output) {
          cv::Mat img;
          frame.download(img);
          img.convertTo(img, CV_8UC3, 255.0F);
          cv::imshow("image_in_" + std::to_string(camera_index), img);
          cv::waitKey(1);
        }
        frames.push_back(frame);
        auto result = std::make_shared<RacketPoseFeatures>();
        result->player_index = racket.second.racket_id;
        result->camera_index = camera_index;
        result->sequence_number = sequence_number;
        latest_detections_internal_.features.push_back(result);
      }
    }
    job_.second.clear();
    if (!frames.empty()) {
      // LOG(INFO) << "Inference for: " << frames.size();
      inference_->DetectRacketKeypoints(frames, latest_detections_internal_);
      // LOG(INFO) << "Total features found: " << latest_detections_internal_.features.size();
      // update results using offsets
      for (size_t i = 0; i < offsets.size(); ++i) {
        if (latest_detections_internal_.features[i] != nullptr) {
          latest_detections_internal_.features[i]->OffsetBy(offsets[i].x(), offsets[i].y());
        }
      }

      if (perform_estimation_) {
        shared_context_->TriangulateDetections(latest_detections_internal_);

        for (auto& it : keypoint_fitters_) {
          auto fitter_idx = it.first;
          auto& fitter = it.second;
          auto& racket = shared_context_->racket_params->rackets[fitter_idx];

          std::vector<std::vector<RacketPoseFeatures::Keypoint>> keypoints_vec;
          // iterate over the triangulated points and fit the orientation based on them
          // LOG(INFO) << "Total Rackets found: " << latest_detections_internal_.estimated_rackets.size();
          for (auto& pose : latest_detections_internal_.estimated_rackets) {
            if (pose->racket_id != fitter_idx) {
              continue;
            }
            std::vector<int> cam_indicies;
            // match with the camera indicies
            for (auto& cam : racket.cameras) {
              // if (cam_idx == -1) {
              //   continue;
              // }
              RacketPoseFeatures::SharedPtr detection;
              auto cam_idx = cam.second.camera_calib_index;
              for (auto& feature : latest_detections_internal_.features) {
                if (feature != nullptr && cam_idx == static_cast<int>(feature->camera_index) &&
                    feature->player_index == racket.racket_id) {
                  detection = feature;
                  break;
                }
              }

              if (!detection) {
                // try to get it from the shared context
                auto detections = shared_context_->GetLastDetections(racket.racket_id, cam_idx, 0);
                if (detections != nullptr) {
                  detection = detections;
                }
              }
              if (detection == nullptr) {
                continue;
              }
              // LOG(INFO) << "detection found: " << detection->camera_index;
              cam_indicies.push_back(cam_idx);
              // std::vector<Eigen::Vector2f> keypoints;
              // for (auto& kp : detection->keypoints) {
              //   keypoints.emplace_back(kp.x(), kp.y());
              // }
              keypoints_vec.push_back(detection->keypoints);
            }
            if (!keypoints_vec.empty()) {
              // LOG(INFO) << "Fitting for: " << cam_indicies;
              fitter->SetCameras(cam_indicies);
              pose->orientation =
                fitter->FitPoints(pose->position, keypoints_vec, pose->iterations_count, pose->orientation_error);
              // TODO(yamen): calculate confidence here based on the orientation error/axis confidence
              pose->orientation_confidence =
                static_cast<float>(cam_indicies.size()) / static_cast<float>(racket.cameras.size());
            }
          }
        }
      }

      shared_context_->AddDetection(start_time, latest_detections_internal_);
    }
    work_available_ = false;
    new_data_available_ = true;
  }

  void WorkerThread() {
    const auto& manager = ace_rt_profiles::ProfileManager::GetInstance();

    const std::string rt_profile = shared_context_->racket_params->extractor.rt_profile;
    const std::vector<int> affinity = manager.GetAffinity(rt_profile);
    if (!affinity.empty()) {
      auto affinity_index = worker_index_ % affinity.size();

      const ace_rt_profiles::RTSubProfile rt_subprofile = {.rt_profile = rt_profile, .affinity_index = affinity_index};
      manager.Apply(rt_subprofile);
      LOG(INFO) << "ExtractorWorker[" << worker_index_ << "] Started at CPU affinity: " << affinity[affinity_index];
    } else {
      LOG(WARNING) << "ExtractorWorker[" << worker_index_ << "] Started without cpu affinity assignment!";
    }

    using clock = std::chrono::high_resolution_clock;
    double time_acc = 0;
    int total = 0;
    while (!is_done_) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!cond_.wait_for(lock, kWaitDuration, [&]() -> bool { return work_available_; })) {
        continue;
      }
      auto time_now = clock::now();
      RunInference();
      time_acc +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - time_now).count());
      total += 1;
      if (total >= 100) {
        time_acc /= static_cast<double>(total);
        LOG(INFO) << worker_index_ << ": Average Inference Time: " << time_acc << "ms";
        time_acc = 0;
        total = 0;
      }
    }
    LOG(INFO) << "ExtractorWorker[" << worker_index_ << "] Done";
  }
};
}  // namespace perception
