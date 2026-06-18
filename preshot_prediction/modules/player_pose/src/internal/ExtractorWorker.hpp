// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <ace_rt_profiles/profile_manager.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <utility>

#include "WorkerSharedContext.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "player_pose/PersonDetector.hpp"

namespace perception {

class ExtractorWorker {
  std::thread worker_thread_;
  bool is_done_{true};
  int worker_index_;
  std::mutex mutex_;
  std::mutex data_mutex_;
  std::condition_variable cond_;
  std::mutex image_work_mutex_;
  std::condition_variable image_work_done_;
  std::vector<cudaStream_t> cuda_streams_;
  std::map<std::pair<int, int>, cv::cuda::GpuMat> uploaded_images_;
  bool work_available_{false};
  using JobType = std::pair<size_t, std::vector<PlayerPoseExtractor::ImageData::SharedPtr>>;
  JobType job_;
  static constexpr std::chrono::milliseconds kWaitDuration{100};

  PlayerFrameFeatures latest_detections_internal_;
  bool new_data_available_{false};

  WorkerSharedContext::SharedPtr shared_context_;

  PersonDetector::SharedPtr detector_;

 public:
  ExtractorWorker(int worker_index, WorkerSharedContext::SharedPtr shared_context, const std::string& model_path)
    : worker_index_(worker_index), shared_context_(shared_context) {
    detector_ = std::make_shared<PersonDetector>();
    if (!detector_->Initialize(shared_context->player_params, model_path)) {
      LOG(ERROR) << "Failed to initialize person detector." << std::endl;
    }

    detector_->SetPlayerDetectedCallback(
      [this](PlayerBBoxDetection::SharedPtr detection) { shared_context_->OnPlayerDetection(detection); });
  }
  ~ExtractorWorker() { Stop(); }

  bool Start() {
    is_done_ = false;
    worker_thread_ = std::thread(std::bind(&ExtractorWorker::WorkerThread, this));
    cuda_streams_.resize(8);
    for (auto& stream : cuda_streams_) {
      CUDA_CHECK(cudaStreamCreate(&stream));
    }
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
  bool Assign(size_t seq_id, std::vector<PlayerPoseExtractor::ImageData::SharedPtr>& images) {
    if (work_available_) {
      return false;
    }
    std::scoped_lock<std::mutex> lock(data_mutex_);

    std::sort(images.begin(), images.end(),
              [](const PlayerPoseExtractor::ImageData::SharedPtr& a,
                 const PlayerPoseExtractor::ImageData::SharedPtr& b) { return a->camera_index < b->camera_index; });
    job_ = JobType(seq_id, std::move(images));
    // static auto cb = [this](PlayerPoseExtractor::ImageData::SharedPtr image_data, cudaStream_t stream) {
    //   this->ProcessImage(image_data, stream);
    // };
    // for (auto& img : job_.second) {
    //   // shared_context_->EnqueueImage(img, cb);
    // }
    // LOG(INFO) << worker_index_ << " - Assigning : " << seq_id;
    work_available_ = true;
    cond_.notify_all();
    return true;
  }

  void ProcessImage(PlayerPoseExtractor::ImageData::SharedPtr image_data, int player_idx, const cv::Rect2i& roi,
                    cudaStream_t stream) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto& image = image_data->image;
    auto src_offset = image.step * roi.y + roi.x;

    auto camera_index = image_data->camera_index;
    auto key = std::make_pair(player_idx, camera_index);
    if (uploaded_images_.find(key) == uploaded_images_.end()) {
      uploaded_images_[key] = cv::cuda::GpuMat(roi.height, roi.width, image.type(), cv::Scalar(0));
    }
    image_data->uploaded_image = uploaded_images_[key];
    cudaMemcpy2DAsync(image_data->uploaded_image.ptr(), image_data->uploaded_image.step,
                      image.data + src_offset * image.elemSize(), image.step, roi.width * image.elemSize(), roi.height,
                      cudaMemcpyHostToDevice, stream);
    if (roi.width != image.cols || roi.height != image.rows) {
      image_data->cropped = true;
    }

    image_data->processed = true;
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // image_work_done_.notify_one();
    // auto t2 = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    // LOG(INFO) << worker_index_ << " - Upload Image Time: " << duration * 1e-3 << " ms";
  }

  [[nodiscard]] bool NewDataAvailable() const { return new_data_available_; }

  bool GetDetections(PlayerFrameFeatures& features) {
    if (new_data_available_ && !work_available_) {
      // std::scoped_lock<std::mutex> lock(data_mutex_);
      features = latest_detections_internal_;
      new_data_available_ = false;
    }
    return true;
  }
  void RunInference() {
    auto sequence_number = job_.first;
    latest_detections_internal_.sequence_number = sequence_number;
    latest_detections_internal_.features.clear();
    latest_detections_internal_.estimated_players.clear();

    auto start_time = std::chrono::high_resolution_clock::now();
    // LOG(INFO) << worker_index_ << " - Started Inference: " << sequence_number;
    if (job_.second.empty()) {
      work_available_ = false;
      new_data_available_ = true;
      // LOG(INFO) << worker_index_ << " - Job is empty for: " << sequence_number;

      return;
    }

    if (job_.second.size() != shared_context_->player_params->cameras_names.size()) {
      LOG(WARNING) << sequence_number << " Running inference on a limited number of cameras: " << job_.second.size()
                   << "/" << shared_context_->player_params->cameras_names.size();
    }
    {
      /*
      for (const auto& img : job_.second) {
        ProcessImage(img,0);
      }
      std::unique_lock<std::mutex> lock(image_work_mutex_);
      image_work_done_.wait(lock, [this]() {
        for (const auto& img : job_.second) {
          if (!img->processed) {
            return false;
          }
        }
        return true;
      });*/
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    for (const auto& player : shared_context_->player_params->players) {
      std::vector<cv::cuda::GpuMat> frames;
      std::vector<cv::Rect2i> rois;
      // process the images for this player
      for (const auto& img : job_.second) {
        auto camera_index = img->camera_index;
        const auto& image = img->uploaded_image;

        auto player_cam_it = std::find_if(
          player.second.cameras.begin(), player.second.cameras.end(),
          [camera_index](const auto& cam_roi) { return cam_roi.second.camera_calib_index == camera_index; });
        if (player_cam_it == player.second.cameras.end()) {
          continue;
        }
        const auto& player_cam = *player_cam_it;
        const auto& roi = player_cam.second.roi;

        ProcessImage(img, player.first, roi, cuda_streams_[frames.size()]);
        frames.push_back(img->uploaded_image);
        rois.push_back(roi);
      }
      if (frames.empty()) {
        continue;
      }
      for(int i=0;i<frames.size();++i) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_streams_[i]));
      }
      // extract person
      // add a request to detect the player in the frame (asynchronously)
      std::vector<PlayerBBoxDetection::SharedPtr> results;
      for (int i = 0; i < frames.size(); ++i) {
        results.push_back(std::make_shared<PlayerBBoxDetection>());
      }
      bool correct_rotation = shared_context_->player_params->extractor.apply_rotation_correction;
      bool success = detector_->ExtractPerson(frames, nullptr, results, correct_rotation);

      // fetch last available player detection we have
      if (success) {
        for (int i = 0; i < results.size(); ++i) {
          auto& person = results[i];
          auto camera_index = job_.second[i]->camera_index;
          if (person->confidence == 0 || person->keypoints.empty()) {
            continue;
          }

          if (job_.second[i]->cropped || correct_rotation) {
            auto cam_idx = shared_context_->player_params->cameras_map[camera_index];
            const auto& cam = shared_context_->player_params->cameras[cam_idx];
            const auto& roi = rois[i];
            // std::cout<<"Applying offset: "<<cam.roi.x<<","<<cam.roi.y<<" for camera: "<<cam.name<<std::endl;
            person->bbox.x() += roi.x;
            person->bbox.y() += roi.y;
            for (auto& kp : person->keypoints) {
              kp.x() += roi.x;
              kp.y() += roi.y;
            }
          }
          auto result = std::make_shared<PlayerPoseDetection>();
          result->player_index = player.second.player_id;
          result->camera_index = camera_index;
          result->sequence_number = sequence_number;
          result->bbox = std::move(person->bbox);
          result->keypoints = std::move(person->keypoints);
          latest_detections_internal_.features.emplace_back(result);
        }
      }
    }
    job_.second.clear();

    // if (shared_context_->player_params->extractor.apply_rotation_correction) {
    //   shared_context_->player_params->InvRotateDetections(latest_detections_internal_);
    // }
    // auto t4 = std::chrono::high_resolution_clock::now();
    // perform temporal filtering to reduce noisy detections
    for (auto& feature : latest_detections_internal_.features) {
      shared_context_->AverageDetection(feature);
    }
    // auto t5 = std::chrono::high_resolution_clock::now();
    // triangulate detections
    shared_context_->TriangulateDetections(latest_detections_internal_);
    // auto t6 = std::chrono::high_resolution_clock::now();

    // add them to the shared context
    shared_context_->AddDetection(start_time, latest_detections_internal_);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> total_time = end_time - start_time;
    // std::chrono::duration<double, std::milli> preprocess_time = t2 - t1;
    // std::chrono::duration<double, std::milli> inference_time = t3 - t2;
    // std::chrono::duration<double, std::milli> postprocess_time = t6 - t3;
    // std::cout << "Extractor Worker Times (ms): preprocess: " << preprocess_time.count()
    //           << ", inference: " << inference_time.count() << ", postprocess: " << postprocess_time.count()
    //           << ", total: " << total_time.count() << std::endl;

    // LOG(INFO) << worker_index_ << " - Done Inference: " << latest_detections_internal_.sequence_number;
    work_available_ = false;
    new_data_available_ = true;
  }

  void WorkerThread() {
    const auto& manager = ace_rt_profiles::ProfileManager::GetInstance();

    const std::string rt_profile = shared_context_->player_params->extractor.rt_profile;
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
