// Confidential, Copyright 2025, Sony AI, All rights reserved
#include "player_pose/PersonDetector.hpp"

#include <condition_variable>
#include <list>
#include <mutex>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <thread>

#include "player_pose/YoloObjectDetector.hpp"

namespace perception {

class PersonDetector::PersonDetectorImpl {
 public:
  PersonDetectorImpl() { yolo_detector = std::make_shared<YoloObjectDetector>(); }
  bool Initialize(PlayerPoseParameters::SharedPtr params, const std::string& model_path) {
    LOG(INFO) << "Person Detector: Initialization using model: " << model_path;
    player_params = params;
    is_done = false;
    if (!yolo_detector->Initialize(model_path, std::dynamic_pointer_cast<YoloConfig>(player_params->person_detector),
                                   player_params)) {
      return false;
    }
    worker_thread = std::thread(std::bind(&PersonDetectorImpl::WorkerThread, this));
    return true;
  }
  ~PersonDetectorImpl() {
    is_done = true;
    if (worker_thread.joinable()) {
      worker_thread.join();
    }
  }

  void AddBBoxDetectionRequest(size_t seq_id, size_t player_index, size_t cam_index, cv::cuda::GpuMat& image,
                               bool correct_rotation) {
    {
      auto key = std::pair(player_index, cam_index);
      std::lock_guard<std::mutex> lock(bbox_mutex);
      if (last_bbox_detection_percamera.find(key) != last_bbox_detection_percamera.end()) {
        auto current = last_bbox_detection_percamera[key]->sequence_number;
        if (seq_id < current &&
            seq_id > current - 30) {  // check if there is no major skip (e.g. reset of sequence ids)
          return;                     // early skip
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(worker_mutex);
      for (auto it = work_queue.begin(); it != work_queue.end(); ++it) {
        if (it->detection[0]->camera_index == cam_index && it->detection[0]->player_index == player_index) {
          work_queue.erase(it);
          break;
        }
      }
      /**/
      PlayerBBoxDetection::SharedPtr person = std::make_shared<PlayerBBoxDetection>();
      person->camera_index = cam_index;
      person->sequence_number = seq_id;
      person->player_index = player_index;

      WorkOrder wo;
      wo.correct_rotation = correct_rotation;
      wo.image = {image};
      wo.detection = {person};

      work_queue.emplace_back(wo);
      cond.notify_all();
    }
  }
  bool AddBBoxDetection(PlayerBBoxDetection::SharedPtr detection) {
    std::lock_guard<std::mutex> lock(bbox_mutex);
    auto key = std::pair(detection->player_index, detection->camera_index);
    if (last_bbox_detection_percamera.find(key) == last_bbox_detection_percamera.end()) {
      last_bbox_detection_percamera[key] = detection;
      return true;
    }

    auto current = last_bbox_detection_percamera[key]->sequence_number;
    if (detection->sequence_number > current ||
        detection->sequence_number < current - 30) {  // check if there is no major skip (e.g. reset of sequence ids)
      last_bbox_detection_percamera[key] = detection;
      return true;
    }
    return false;
  }
  bool GetLastBBoxDetection(size_t player_index, size_t camera_index, PlayerBBoxDetection::SharedPtr& person) {
    std::lock_guard<std::mutex> lock(bbox_mutex);
    auto key = std::pair(player_index, camera_index);
    if (last_bbox_detection_percamera.find(key) == last_bbox_detection_percamera.end()) {
      return false;
    }
    person = last_bbox_detection_percamera[key];
    return true;
  }

  bool ExtractPerson(const std::vector<cv::cuda::GpuMat>& images, std::vector<cv::cuda::GpuMat*>* extracted,
                     std::vector<PlayerBBoxDetection::SharedPtr>& detections, bool correct_rotation) {
    float ratio = 1;
    // auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<cv::cuda::GpuMat> processed;
    std::vector<std::pair<float, cv::Size2i>> ratios;
    for (int i = 0; i < images.size(); i++) {
      auto input = images[i];
      if (correct_rotation) {
        input = player_params->RotateImage(detections[i]->camera_index, input);
      }
      float ratio = 1;
      processed.push_back(yolo_detector->ProcessImage(input, ratio, false));
      ratios.push_back({1.0f / ratio, cv::Size2i(input.cols, input.rows)});
    }
    // auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<YoloObject>> all_results;
    if (!yolo_detector->DetectObjects(processed, all_results, ratios)) {
      return false;
    }
    // auto t4 = std::chrono::high_resolution_clock::now();

    // std::cout << "Person Found:" << all_results.size() << std::endl;
    // std::cout << input.size() << std::endl;
    // std::cout << results[0].rect << std::endl;
    // crop extracted person from input
    for (int i = 0; i < images.size(); i++) {
      auto& detection = detections[i];
      auto& results = all_results[i];
      if (results.empty()) {
        detection->keypoints.clear();
        detection->confidence = 0;
        continue;
      }
      detection->confidence = results[0].confidence;
      const int inflation_w = static_cast<int>(static_cast<float>(results[0].rect.width) *
                                               player_params->person_detector->bbox_inflation.x());
      const int inflation_h = static_cast<int>(static_cast<float>(results[0].rect.height) *
                                               player_params->person_detector->bbox_inflation.y());
      auto rect = PlayerBBoxDetection::Boundingbox(results[0].rect.x, results[0].rect.y, results[0].rect.width,
                                                   results[0].rect.height);
      rect.x() -= inflation_w / 2;
      rect.y() -= inflation_h / 2;
      rect.z() += inflation_w;
      rect.w() += inflation_h;

      if (correct_rotation) {
        player_params->InvRotateBBox(detection->camera_index, rect);
      }

      auto br = Eigen::Vector2i(rect.x() + rect.z(), rect.y() + rect.w());
      detection->bbox.x() = std::max<int>(0, rect.x());
      detection->bbox.y() = std::max<int>(0, rect.y());
      detection->bbox.z() = std::min<int>(images[i].cols - detection->bbox.x() - 1, br.x() - detection->bbox.x());
      detection->bbox.w() = std::min<int>(images[i].rows - detection->bbox.y() - 1, br.y() - detection->bbox.y());
      detection->keypoints = std::move(results[0].keypoints);
      if (correct_rotation) {
        for (auto& kp : detection->keypoints) {
          auto tmp = std::vector<cv::Point2f>{cv::Point2f(kp.x(), kp.y())};
          player_params->InvRotateKeypoints(detection->camera_index, tmp);
          kp.x() = tmp[0].x;
          kp.y() = tmp[0].y;
        }
      }
      if (extracted && (*extracted)[i]) {
        images[i](results[0].rect).copyTo(*(*extracted)[i]);
      }
    }
    // for (int i = to_remove.size() - 1; i >= 0; --i) {
    //   detections.erase(detections.begin() + to_remove[i]);
    // }
    // auto t5 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> rotation_time = t2 - t1;
    // std::chrono::duration<double, std::milli> inference_time = t4 - t2;
    // std::chrono::duration<double, std::milli> postprocess_time = t5 - t4;
    // std::cout << "Person detection times (ms): preprocess: " << rotation_time.count()
    //           << ", inference: " << inference_time.count() << ", postprocess: " << postprocess_time.count()
    //           << ", total: " << (rotation_time + inference_time + postprocess_time).count() << std::endl;
    return true;
  }

  void AverageDetection(PlayerBBoxDetection::SharedPtr detection) {
    auto key = std::pair(detection->player_index, detection->camera_index);
    bbox_detection_history.try_emplace(key, std::vector<PlayerBBoxDetection::SharedPtr>());
    auto& history = bbox_detection_history[key];
    history.push_back(detection);

    auto total_weight = player_params->person_detector->latest_weight;
    auto confidence = detection->confidence * total_weight;
    Eigen::Vector4f bbox = detection->bbox.cast<float>() * total_weight;
    auto weight = 1 - player_params->person_detector->latest_weight;
    auto max_distance = static_cast<float>(player_params->person_detector->history_difference);
    for (size_t i = 0; i < history.size() - 1; ++i) {
      if (history[i]->confidence < 0.01) {
        continue;
      }
      auto dist = std::abs<float>(static_cast<float>(history[i]->sequence_number) -
                                  static_cast<float>(detection->sequence_number));
      if (dist > max_distance) {
        auto it = history.begin();
        std::advance(it, i);
        history.erase(it);
        --i;
      } else {
        auto w = weight * (1 - dist / max_distance);
        confidence += history[i]->confidence * w;
        bbox += history[i]->bbox.cast<float>() * w;
        total_weight += w;
      }
    }
    detection->confidence = confidence / total_weight;
    detection->bbox = (bbox / total_weight).cast<int>();
  }

  void WorkerThread() {
    static constexpr std::chrono::milliseconds kWaitDuration{100};
    // using clock = std::chrono::high_resolution_clock;
    LOG(INFO) << "Shared Worker Started";
    while (!is_done) {
      std::unique_lock<std::mutex> lock(worker_mutex);
      if (!cond.wait_for(lock, kWaitDuration, [&]() -> bool { return !work_queue.empty(); })) {
        continue;
      }
      // auto time_now = clock::now();
      auto queue = std::move(work_queue);
      lock.unlock();
      for (auto& p : queue) {
        if (ExtractPerson(p.image, nullptr, p.detection, p.correct_rotation)) {
          for (auto& detection : p.detection) {
            AverageDetection(detection);
            if (AddBBoxDetection(detection) && on_player_detected_callback != nullptr) {
              on_player_detected_callback(detection);
            }
          }
        }
      }
      // std::cout << "Detection [" << queue.size()
      //           << "]: " << std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - time_now).count()
      //           << std::endl;
    }
    LOG(INFO) << "Shared Worker Done";
  }

  std::thread detector_thread;
  std::thread worker_thread;
  std::mutex bbox_mutex;
  std::mutex worker_mutex;
  PlayerPoseParameters::SharedPtr player_params;
  class WorkOrder {
   public:
    std::vector<cv::cuda::GpuMat> image;
    std::vector<PlayerBBoxDetection::SharedPtr> detection;
    bool correct_rotation{false};
  };
  std::list<WorkOrder> work_queue;
  std::condition_variable cond;
  bool is_done{false};
  std::map<PlayerPoseParameters::PlayerCameraPair, PlayerBBoxDetection::SharedPtr> last_bbox_detection_percamera;
  std::map<PlayerPoseParameters::PlayerCameraPair, std::vector<PlayerBBoxDetection::SharedPtr>> bbox_detection_history;
  YoloObjectDetector::SharedPtr yolo_detector;

  PersonDetector::PlayerDetectedCallback on_player_detected_callback;
};

PersonDetector::PersonDetector() { impl_ = std::make_shared<PersonDetectorImpl>(); }
bool PersonDetector::Initialize(PlayerPoseParameters::SharedPtr player_params, const std::string& model_path) {
  return impl_->Initialize(player_params, model_path);
}

void PersonDetector::AddBBoxDetectionRequest(size_t seq_id, size_t player_index, size_t cam_index,
                                             cv::cuda::GpuMat& image, bool correct_rotation) {
  impl_->AddBBoxDetectionRequest(seq_id, player_index, cam_index, image, correct_rotation);
}
void PersonDetector::AddBBoxDetection(PlayerBBoxDetection::SharedPtr detection) { impl_->AddBBoxDetection(detection); }
bool PersonDetector::GetLastBBoxDetection(size_t player_index, size_t camera_index,
                                          PlayerBBoxDetection::SharedPtr& person) {
  return impl_->GetLastBBoxDetection(player_index, camera_index, person);
}

void PersonDetector::SetPlayerDetectedCallback(PlayerDetectedCallback callback) {
  impl_->on_player_detected_callback = callback;
}
bool PersonDetector::ExtractPerson(const std::vector<cv::cuda::GpuMat>& input,
                                   std::vector<cv::cuda::GpuMat*>* extracted,
                                   std::vector<PlayerBBoxDetection::SharedPtr>& detection, bool correct_rotation) {
  return impl_->ExtractPerson(input, extracted, detection, correct_rotation);
}

}  // namespace perception
