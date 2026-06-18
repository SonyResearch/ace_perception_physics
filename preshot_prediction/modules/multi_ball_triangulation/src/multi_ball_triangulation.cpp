// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "multi_ball_triangulation/multi_ball_triangulation.hpp"

#include <ace_rt_profiles/profile_manager.hpp>

#include "ace_interfaces/msg/point_with_covariance.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "dot_projector/dot_projector.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "triangulation/datalogger.hpp"

namespace multi_ball_triangulation {

MultiBallTriangulation::MultiBallTriangulation(const rclcpp::NodeOptions& options)
  : rclcpp::Node("multi_ball_triangulation", options),
    params_ptr_(std::make_shared<MultiBallTriangulationParameters>()),
    camera_calib_params_ptr_(std::make_shared<calibration::CameraCalibrationParameters>()) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }
  init_ = false;

  const std::string& params_path = declare_parameter<std::string>("params_path");
  const std::vector<std::string>& ball_detection_params_paths =
    declare_parameter<std::vector<std::string>>("ball_detection_params_paths");
  const std::string& camera_calib_params_path = declare_parameter<std::string>("camera_calib_params_path");

  if (Initialize(params_path, ball_detection_params_paths, camera_calib_params_path)) {
    LOG(INFO) << "Initialized ball triangulation.\n";
  } else {
    LOG(INFO) << "Failed to initialize ball triangulation.\n";
  }

  // Launch worker thread
  canceled_.store(false);
  worker_thread_ = std::thread(std::bind(&MultiBallTriangulation::WorkerThread, this));
  worker_thread_.detach();

  // Launch status reporter thread
  status_thread_ = std::thread(std::bind(&MultiBallTriangulation::ReportStatus, this));
  status_thread_.detach();

  // Launch calib reload thread
  reload_needed_.store(false);
  reload_thread_ = std::thread(std::bind(&MultiBallTriangulation::ReloadCalibrationThread, this));
  reload_thread_.detach();
}

MultiBallTriangulation::~MultiBallTriangulation() {
  canceled_.store(true);
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  if (status_thread_.joinable()) {
    status_thread_.join();
  }
  if (reload_thread_.joinable()) {
    reload_thread_.join();
  }
}

bool MultiBallTriangulation::Initialize(const std::string& params_path,
                                        const std::vector<std::string>& ball_detection_params_paths,
                                        const std::string& camera_calib_params_path) {
  if (!params_ptr_->Initialize(params_path)) {
    return false;
  }
  params_ptr_->PrintParameters();

  LOG(INFO) << "-------------------------------------------------------" << std::endl;
  camera_names_.clear();
  for (const auto& ball_detection_params_path : ball_detection_params_paths) {
    vision_common::BallDetectionParameters ball_detection_params;
    if (!ball_detection_params.Initialize(ball_detection_params_path)) {
      return false;
    }
    LOG(INFO) << "Loading " << ball_detection_params.camera_names.size() << " cameras" << std::endl;
    LOG(INFO) << " - from: " << ball_detection_params_path << std::endl;
    camera_names_.insert(camera_names_.end(), ball_detection_params.camera_names.begin(),
                         ball_detection_params.camera_names.end());
  }

  // Sort and remove duplicate camera names
  std::sort(camera_names_.begin(), camera_names_.end());
  camera_names_.erase(std::unique(camera_names_.begin(), camera_names_.end()), camera_names_.end());

  LOG(INFO) << "-------------------------------------------------------" << std::endl;
  LOG(INFO) << "camera_names:" << std::endl;
  for (const auto& camera_name : camera_names_) {
    LOG(INFO) << " - " << camera_name << std::endl;
  }
  LOG(INFO) << "-------------------------------------------------------" << std::endl;

  if (!camera_calib_params_ptr_->Initialize(camera_calib_params_path)) {
    return false;
  }
  camera_calib_params_ptr_->PrintParameters();

  // Initialize triangulator.
  auto datawriter_ptr = std::make_shared<triangulation::datalogger::DataWriter>(
    ::datalogger::ConstructFullLogName("multiball_triangulation"), params_ptr_->ToString(),
    camera_calib_params_ptr_->ToString());
  triangulator_ptr_ = std::make_unique<triangulation::Triangulation<Eigen::Vector3f>>(
    params_ptr_, camera_calib_params_ptr_, std::move(datawriter_ptr));
  triangulator_ptr_->SetCameras(camera_names_);
  triangulator_ptr_->SetTriangulationFilter([this](const triangulation::TriangulatedPoint& triangulated_point) {
    return this->IsPointInVOI(triangulated_point);
  });

  // Initialize dot_projector
  dot_projector_ptr_ = std::make_unique<dot_projector::DotProjector>();
  if (!dot_projector_ptr_->SetParameters(params_ptr_)) {
    return false;
  }

  num_cameras_ = static_cast<int>(camera_names_.size());

  camera_buffers_ptr_ = std::make_unique<CameraBuffers>(num_cameras_, params_ptr_->camera_buffer_length,
                                                        params_ptr_->use_variable_response_time);
  worker_queue_ptr_ = std::make_unique<WorkerQueue>(params_ptr_->worker_queue_length);

  // Set up ball detection callbacks.
  for (int camera_index = 0; camera_index < num_cameras_; ++camera_index) {
    auto cb = [&, camera_index](const ace_interfaces::msg::BallsWithAttributes::SharedPtr message) {
      this->BallDetectionCallback(camera_index, message);
    };

    const std::string& camera_name = camera_names_[camera_index];
    subs_.push_back(this->create_subscription<ace_interfaces::msg::BallsWithAttributes>(
      std::string(this->get_namespace()) + "/" + camera_name + "/ball_detection", 1, cb));
  }

  rclcpp::QoS qos_persistent(rclcpp::KeepLast(1));
  qos_persistent.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos_persistent.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

  // Set up calibration reload callback
  calib_reload_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    std::string(this->get_namespace()) + "/calibration/reload_state", qos_persistent,
    std::bind(&MultiBallTriangulation::ReloadCalibrationCallback, this, std::placeholders::_1));

  // Set up LoggerControl callback
  loggercontrol_sub_ = this->create_subscription<ace_interfaces::msg::LoggerControl>(
    "/logger/control", qos_persistent,
    std::bind(&MultiBallTriangulation::LoggerControlCallback, this, std::placeholders::_1));

  execution_mode_sub_ = this->create_subscription<std_msgs::msg::String>(
    "/execution_mode", qos_persistent, [&](std_msgs::msg::String::SharedPtr message) {
      LOG(INFO) << "Perception execution mode is: " << message->data << std::endl;
      params_ptr_->SetExecutionMode(message->data);
    });

  // Set up publisher.
  std::string topic = std::string(this->get_namespace()) + "/" + this->get_name() + "/" + "points";
  pub_ = this->create_publisher<ace_interfaces::msg::PointsWithCovariance>(topic, 1);

  init_ = true;

  return true;
}
bool MultiBallTriangulation::IsPointInVOI(const triangulation::TriangulatedPoint& triangulated_point) {
  return !(params_ptr_->voi_filter_enable &&
           ((triangulated_point.position.array() < params_ptr_->voi_filter_min_corner.array()).any() ||
            (triangulated_point.position.array() > params_ptr_->voi_filter_max_corner.array()).any()));
}

void MultiBallTriangulation::TriangulateBallPositions(const CameraBuffers::GroupsOfSyncedItems::mapped_type& detections,
                                                      ace_interfaces::msg::PointsWithCovariance& message) {
  const auto start_time = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::vector<Eigen::Vector3f>> ball_detections(num_cameras_);
  for (const auto& [camera_index, balls_with_attributes_ptr] : detections) {
    for (const auto& ball_with_attributes : balls_with_attributes_ptr->balls) {
      ball_detections[camera_index].emplace_back(static_cast<float>(ball_with_attributes.ball.center.x),
                                                 static_cast<float>(ball_with_attributes.ball.center.y),
                                                 static_cast<float>(ball_with_attributes.ball.radius));
    }
  }

  triangulator_ptr_->GetDataLogger() << message.header.sequence_number << message.header.stamp << start_time;
  std::vector<triangulation::TriangulatedPoint> triangulated_points =
    triangulator_ptr_->TriangulatePoints(ball_detections);
  if (static_cast<int>(triangulated_points.size()) > params_ptr_->max_triangulated_balls) {
    LOG(WARNING) << "Skipping, max number of triangulated points was reached!.\n";
    return;
  }

  std::vector<Eigen::Vector2f> marker_2d_points;
  std::vector<Eigen::Vector3f> marker_3d_points;
  for (const auto& triangulated_point : triangulated_points) {
    message.points.emplace_back();
    ace_interfaces::msg::PointWithCovariance& ball_msg = message.points.back();

    ball_msg.position.x = triangulated_point.position[0];
    ball_msg.position.y = triangulated_point.position[1];
    ball_msg.position.z = triangulated_point.position[2];
    memcpy(ball_msg.covariance.data(), triangulated_point.covariance.data(), sizeof(ball_msg.covariance));
    ball_msg.num_cameras = static_cast<int>(triangulated_point.observation_indices.size());

    if (params_ptr_->projector_enable) {
      for (size_t obs_i = 0; obs_i < triangulated_point.observation_indices.size(); ++obs_i) {
        const auto& [camera_index, ball_index] = triangulated_point.observation_indices[obs_i];
        const auto& marker_detections = detections.at(camera_index)->balls[ball_index].markers;
        marker_2d_points.clear();
        marker_2d_points.reserve(marker_detections.size());
        for (const auto& marker : marker_detections) {
          marker_2d_points.emplace_back(marker.center.x, marker.center.y);
        }

        const std::string& camera_name = camera_names_[camera_index];
        const calibration::Camera& camera = camera_calib_params_ptr_->cameras[camera_name];
        marker_3d_points.clear();
        dot_projector_ptr_->ProjectMarkers(
          Eigen::Map<const Eigen::Matrix2Xf>(reinterpret_cast<const float*>(marker_2d_points.data()), 2,
                                             static_cast<int>(marker_2d_points.size())),
          camera, triangulated_point.position, marker_3d_points);
        if (!marker_3d_points.empty()) {
          ball_msg.marker_groups.emplace_back();
          auto& points_group_msg = ball_msg.marker_groups.back();

          points_group_msg.camera_name = camera_name;
          for (const auto& marker_3d_point : marker_3d_points) {
            points_group_msg.points.emplace_back();
            auto& point_msg = points_group_msg.points.back();

            point_msg.x = marker_3d_point[0];
            point_msg.y = marker_3d_point[1];
            point_msg.z = marker_3d_point[2];
          }
        }
      }
    }
  }
}

void MultiBallTriangulation::WorkerThread() {
  LOG(INFO) << "Starting worker thread.\n";

  // Dedicate CPU cores to this process
  ace_rt_profiles::ProfileManager::GetInstance().Apply("multi_ball_triangulation");

  // Consumer variables
  uint64_t next_sequence_number = std::numeric_limits<uint64_t>::min();
  CameraBuffers::GroupsOfSyncedItems groups_of_items;
  const uint64_t seq_reset_thresh =
    static_cast<uint64_t>(10.0 / std::chrono::duration<double>(params_ptr_->timeout_ms).count());

  while (!canceled_.load()) {
    groups_of_items.clear();
    if (!worker_queue_ptr_->GetFrontAndPop(groups_of_items, params_ptr_->timeout_ms)) {
      ++timeout_counter_;
      LOG(WARNING) << "Ball triangulation timed out.\n";
      continue;
    }
    for (const typename CameraBuffers::GroupsOfSyncedItems::value_type& synced_items : groups_of_items) {
      ++total_triangulations_counter_;

      const uint64_t& curr_sequence_number = synced_items.first;
      const int& num_detections = static_cast<int>(synced_items.second.size());
      if (curr_sequence_number < next_sequence_number) {
        if (curr_sequence_number + seq_reset_thresh >= next_sequence_number) {
          ++overwritten_detections_counter_;
          LOG(WARNING) << "Overwriting " << num_detections
                       << " unprocessed ball detections with sequence_number=" << curr_sequence_number << std::endl;
          continue;
        }
        // Reset as we are mostly replying from a dataset
        LOG(ERROR) << "Resetting sequence_number to " << curr_sequence_number << std::endl;
      }
      next_sequence_number = curr_sequence_number + 1;

      if (num_detections != num_cameras_) {
        ++limited_triangulations_counter_;
        LOG(WARNING) << "Performing triangulation on limited number of ball detections (" << num_detections << ")"
                     << std::endl;
      } else {
        ++complete_triangulations_counter_;
      }

      // Triangulate position
      ace_interfaces::msg::PointsWithCovariance message;
      message.header.sequence_number = curr_sequence_number;
      message.header.stamp = synced_items.second.begin()->second->header.stamp;
      TriangulateBallPositions(synced_items.second, message);

      pub_->publish(message);
    }
  }

  LOG(INFO) << "Exiting worker thread.\n";
}

void MultiBallTriangulation::BallDetectionCallback(
  const int& camera_index, const ace_interfaces::msg::BallsWithAttributes::ConstSharedPtr& msg_ptr) {
  CameraBuffers::GroupsOfSyncedItems groups_of_items;  // No need to guard a temp variable
  while (!camera_buffers_ptr_->Push(groups_of_items, camera_index, msg_ptr)) {
    const std::string& camera_name = camera_names_[camera_index];
    camera_buffers_ptr_->Pop(groups_of_items, std::numeric_limits<uint64_t>::max());
    LOG(ERROR) << "Dropping " << groups_of_items.size() << " detection groups as camera \"" << camera_name
               << "\" reported outdated sequence_number=" << GetSequenceNumber(msg_ptr) << std::endl;
    groups_of_items.clear();
  }
  worker_queue_ptr_->PushBack(groups_of_items);
}

void MultiBallTriangulation::ReloadCalibrationCallback(const std_msgs::msg::Bool::ConstSharedPtr& msg_ptr) {
  if (!msg_ptr->data) {
    return;
  }
  LOG(INFO) << "Camera calibration parameters got externally updated." << std::endl;
  {
    std::lock_guard<std::mutex> lock(reload_mutex_);
    reload_needed_.store(true);
  }
  reload_.notify_one();
}

void MultiBallTriangulation::ReloadCalibrationThread() {
  std::unique_lock<std::mutex> lock(reload_mutex_);
  auto tmp = std::make_shared<calibration::CameraCalibrationParameters>(*camera_calib_params_ptr_);

  while (!canceled_.load()) {
    if (!reload_.wait_for(lock, kReloadTimeout, [this] { return reload_needed_.load(); })) {
      continue;
    }
    LOG(INFO) << "Reloading camera calibration parameters..." << std::endl;
    reload_needed_.store(false);

    // While we are writing, other threads might read corrupted calibration params
    // To avoid that, first write into a tmp variable, then swap
    tmp->ReadParametersFromFile();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      camera_calib_params_ptr_.swap(tmp);
      triangulator_ptr_->SetCameraCalibrationParameters(camera_calib_params_ptr_);
    }
    LOG(INFO) << "Reloaded camera calibration parameters from file!" << std::endl;
  }
}

void MultiBallTriangulation::LoggerControlCallback(const ace_interfaces::msg::LoggerControl::ConstSharedPtr& msg_ptr) {
  const auto log_prefix = std::string(reinterpret_cast<const char*>(msg_ptr->log_prefix.data()));
  const auto log_name = ::datalogger::ConstructSubLogName(log_prefix, "multiball_triangulation");
  {
    std::lock_guard<std::mutex> lock(mutex_);
    triangulator_ptr_->GetDataLogger()->SetFileName(log_name);
  }
  LOG(INFO) << "Log filename got externally updated to:\n - \"" << log_prefix << "\"" << std::endl;
}

void MultiBallTriangulation::ReportStatus() {
  // Print every nth time a header.
  const int n = 10;
  int print_counter = 1;

  while (!canceled_.load()) {
    std::this_thread::sleep_for(kStatusReportInterval);

    if (--print_counter == 0) {
      LOG(INFO) << "----------------------------------------------------------------------------------------------"
                << std::endl;
      LOG(INFO) << "    # total | # complete |  # limited |     # zero ||| # outdated | # overwritten | # timeouts"
                << std::endl;
      print_counter = n;
    }

    LOG(INFO) << " " << std::setfill(' ') << std::setw(10) << total_triangulations_counter_ << " | " << std::setw(10)
              << complete_triangulations_counter_ << " | " << std::setw(10) << limited_triangulations_counter_ << " | "
              << std::setw(10) << zero_triangulations_counter_ << " ||| " << std::setw(10)
              << outdated_detections_counter_ << " | " << std::setw(13) << overwritten_detections_counter_ << " | "
              << std::setw(10) << timeout_counter_ << std::endl;
  }
}

}  // namespace multi_ball_triangulation
