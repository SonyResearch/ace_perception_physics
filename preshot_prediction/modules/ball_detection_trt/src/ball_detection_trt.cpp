// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include "ball_detection_trt/ball_detection_trt.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace ball_detection_trt {

BallDetectionTRT::BallDetectionTRT(const rclcpp::NodeOptions& options)
  : rclcpp::Node("ball_detection_trt", options),
    params_ptr_(std::make_shared<BallDetectionParameters>()),
    camera_calib_params_ptr_(std::make_shared<calibration::CameraCalibrationParameters>()) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }
  init_ = false;

  const std::string& params_path = declare_parameter<std::string>("params_path");
  const std::string& camera_calib_params_path = declare_parameter<std::string>("camera_calib_params_path");

  // Initialize ball pose-estimation.
  if (Initialize(params_path, camera_calib_params_path)) {
    LOG(INFO) << "Initialized ball_detection_trt." << std::endl;
  } else {
    LOG(ERROR) << "Failed to initialize ball_detection_trt." << std::endl;
    throw rclcpp::exceptions::InvalidNodeError();
  }
}

BallDetectionTRT::~BallDetectionTRT() {
  LOG(INFO) << "Stopping threads for ball detection.\n";
  canceled_.store(true);
  cv_shared_queue_.notify_all();
  for (auto& t : worker_threads_) {
    if (t.Joinable()) {
      t.Join();  // Wait until all jobs are done.
    }
  }
  for (auto& t : publisher_threads_) {
    if (t.Joinable()) {
      t.Join();  // Wait until all jobs are done.
    }
  }

  // Clear image buffer (due to zero copy, otherwise middleware does not know they are available again).
  {
    std::lock_guard<std::mutex> lock(shared_queue_mutex_);
    shared_queue_.clear();
  }
  LOG(INFO) << "All threads for ball detection stopped.\n";
}

bool BallDetectionTRT::Initialize(const std::string& params_path, const std::string& camera_calib_params_path) {
  if (!params_ptr_->Initialize(params_path)) {
    return false;
  }
  params_ptr_->PrintParameters();
  if (!camera_calib_params_ptr_->Initialize(camera_calib_params_path)) {
    return false;
  }
  camera_calib_params_ptr_->PrintParameters();

  publisher_mutexes_ = std::make_unique<std::mutex[]>(params_ptr_->device_ids.size());
  cv_publisher_ = std::make_unique<std::condition_variable[]>(params_ptr_->device_ids.size());
  detections_ready_ = std::vector<std::atomic<bool>>(params_ptr_->device_ids.size());
  device_messages_.resize(params_ptr_->device_ids.size());
  for (auto& ready_flag : detections_ready_) {
    ready_flag.store(false);
  }

  canceled_.store(false);

  const auto& manager = ace_rt_profiles::ProfileManager::GetInstance();
  const auto cpu_affinity = manager.GetAffinity(params_ptr_->rt_profile);
  if (cpu_affinity.size() < params_ptr_->device_ids.size()) {
    LOG(WARNING) << "CPU Affinity list size is less than cameras count, performance might be impacted!";
  }

  worker_threads_ = manager.MakeThreads(params_ptr_->rt_profile, params_ptr_->device_ids.size(),
                                        ace_rt_profiles::AffinityMode::kSequential);
  publisher_threads_ = manager.MakeThreads(params_ptr_->rt_profile, params_ptr_->device_ids.size(),
                                           ace_rt_profiles::AffinityMode::kSequential);

  for (unsigned int i = 0; i < params_ptr_->device_ids.size(); ++i) {
    auto params = std::make_shared<trt_ball_detector::BallDetectorParameters>();
    params->onnx_engine_path = params_ptr_->model_engine_path;
    params->device_id = params_ptr_->device_ids[i];
    params->batch_size = params_ptr_->batch_size;

    auto ball_detector_ptr = std::make_shared<trt_ball_detector::BallDetector>();
    ball_detectors_ptr_.push_back(ball_detector_ptr);

    worker_threads_[i].Run([i, this, params]() {
      if (!ball_detectors_ptr_[i]->Initialize(params)) {
        LOG(ERROR) << "Failed to initialize BallDetector for device index " << i << ".\n";
        return;
      }
      WorkerThread(i);
    });
    publisher_threads_[i].Run([i, this]() { PublishThread(i); });
  }

  for (auto camera_index = 0u; camera_index < params_ptr_->camera_names.size(); ++camera_index) {
    const std::string& camera_name = params_ptr_->camera_names[camera_index];
    auto it = camera_calib_params_ptr_->cameras.find(camera_name);
    if (it == camera_calib_params_ptr_->cameras.end()) {
      LOG(ERROR) << "Unable to find camera calibration parameters for camera " << camera_name << ".\n";
      return false;
    }

    // Create a separate publisher for camera.
    {
      std::string topic = std::string("/sensors/" + camera_name + "/" + "ball_detection");
      pubs_.emplace_back(this->create_publisher<ace_interfaces::msg::BallsWithAttributes>(topic, 1));
    }
    // and image subscriper for camera
    {
      const auto callback = [&, camera_index](ace_interfaces::msg::ImageData::ConstSharedPtr message) {
        this->ProcessImage(message, camera_index);
      };

      const std::string topic = std::string("/sensors/" + camera_name + "/" + "image");
      subs_.emplace_back(this->create_subscription<ace_interfaces::msg::ImageData>(topic, 1, callback));
    }
  }

  init_ = true;
  return true;
}

void BallDetectionTRT::ProcessImage(const ace_interfaces::msg::ImageData::ConstSharedPtr& message,
                                    unsigned int camera_index) {
  if (!init_) {
    LOG(WARNING) << "Received image message before initialization. Ignoring.\n";
    return;
  }
  if (!BallDetectionParameters::CheckSampling(static_cast<int>(message->header.sequence_number / 5),
                                              params_ptr_->sampling)) {
    return;  // early skip
  }

  {
    std::lock_guard<std::mutex> lock(shared_queue_mutex_);
    // Cap the queue at batch_size * num_devices: drop the oldest frame when consumers
    // have fallen behind. This ensures we always infer on the freshest available frames
    // instead of accumulating unbounded backlog of stale images.
    const size_t max_queue_size = static_cast<size_t>(params_ptr_->batch_size) * params_ptr_->device_ids.size();
    if (shared_queue_.size() >= max_queue_size) {
      shared_queue_.pop_front();
    }
    shared_queue_.push_back({message, camera_index});
  }
  cv_shared_queue_.notify_all();
}

void BallDetectionTRT::WorkerThread(unsigned int device_index) {
  auto ball_detector_ptr = ball_detectors_ptr_[device_index];
  size_t batch_size = params_ptr_->batch_size;

  // flush the queue at the beginning to avoid processing stale messages from before initialization
  {
    std::lock_guard<std::mutex> lock(shared_queue_mutex_);
    shared_queue_.clear();
  }
  while (!canceled_.load()) {
    std::vector<ace_interfaces::msg::ImageData::ConstSharedPtr> messages;
    std::vector<unsigned int> camera_indecies;
    {
      std::unique_lock<std::mutex> lock(shared_queue_mutex_);
      cv_shared_queue_.wait_for(lock, kMessagesWaitDuration, [&, batch_size]() {
        return shared_queue_.size() >= batch_size || canceled_.load();
      });

      if (canceled_.load()) {
        break;
      }

      if (shared_queue_.size() < batch_size) {
        continue;
      }

      messages.reserve(batch_size);
      camera_indecies.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        messages.push_back(shared_queue_.front().message);
        camera_indecies.push_back(shared_queue_.front().camera_index);
        shared_queue_.pop_front();
      }
    }

    std::vector<cv::Mat> bayer_images;
    // review number of messages that are on the latest sequence ID
    auto sequence_id = messages.back()->header.sequence_number;
    int first_message_index = messages.size() - 1;
    while (first_message_index > 0) {
      if (messages[first_message_index - 1]->header.sequence_number != sequence_id ||
          messages.size() - first_message_index == this->params_ptr_->batch_size) {
        // std::cout << first_message_index
        //           << " Breaking due to: " << messages[first_message_index - 1]->header.sequence_number
        //           << "!=" << sequence_id << std::endl;
        // std::cout << "   or " << messages.size() - first_message_index << std::endl;
        break;
      }
      --first_message_index;
    }
    if (first_message_index > 0) {
      std::cout << "Dropping: " << first_message_index << " messages" << std::endl;
    }

    bayer_images.reserve(messages.size() - first_message_index);
    for (size_t i = first_message_index; i < messages.size(); ++i) {
      auto message = messages[i];
      const void* data_ptr =
        reinterpret_cast<const void*>(message->data.data() + message->step * message->offset_y + message->offset_x);
      cv::Mat bayer_img(message->height, message->width, CV_8UC1, const_cast<void*>(data_ptr));
      bayer_images.push_back(bayer_img);
    }
    if (ball_detector_ptr->EncodeImages(sequence_id, bayer_images)) {
      std::lock_guard<std::mutex> lock(publisher_mutexes_[device_index]);
      // notify publisher thread about the results being ready to parse.
      device_messages_[device_index].resize(messages.size() - first_message_index);
      for (size_t i = first_message_index; i < messages.size(); ++i) {
        device_messages_[device_index][i - first_message_index].camera_index = camera_indecies[i];
        device_messages_[device_index][i - first_message_index].header = messages[i]->header;
      }
      detections_ready_[device_index].store(true);
      cv_publisher_[device_index].notify_one();
    }
  }
}

void BallDetectionTRT::PublishThread(unsigned int device_index) {
  auto ball_detector_ptr = ball_detectors_ptr_[device_index];
  while (!canceled_.load()) {
    std::unique_lock<std::mutex> lock(publisher_mutexes_[device_index]);
    cv_publisher_[device_index].wait_for(lock, kMessagesWaitDuration,
                                         [&]() { return detections_ready_[device_index].load() || canceled_.load(); });

    if (canceled_.load()) {
      break;
    }
    if (!detections_ready_[device_index].load()) {
      continue;
    }
    detections_ready_[device_index].store(false);
    auto curr_messages = std::move(device_messages_[device_index]);
    lock.unlock();
    // Read the results, and publish.
    const auto& detections = ball_detector_ptr->GetDecodingResults();
    // Collect all outgoing messages locally to push to the publish queue
    // in a single lock acquisition with one notify, rather than once per detection.
    for (size_t i = 0; i < detections.size(); ++i) {
      const auto& detection = detections[i];
      const auto camera_index = curr_messages[i].camera_index;
      ace_interfaces::msg::BallsWithAttributes message_out;
      message_out.header.stamp = curr_messages[i].header.stamp;
      message_out.header.sequence_number = curr_messages[i].header.sequence_number;

      bool valid_detction =
        !detection->ball_positions.empty() && detection->confidences[0] > params_ptr_->minimum_confidence;

      if (valid_detction) {
        message_out.balls.emplace_back();
        auto& ball_msg = message_out.balls.back();
        Eigen::Vector2f pos;

        pos = detection->ball_positions[0];
        float radius = detection->radius[0];
        ball_msg.ball.center.x = static_cast<double>(pos.x() * 1440.0F);
        ball_msg.ball.center.y = static_cast<double>(pos.y() * 1080.0F);
        ball_msg.ball.radius = static_cast<double>(radius * 1440.0F);
      }
      pubs_[camera_index]->publish(message_out);
    }
  }
}

}  // namespace ball_detection_trt
