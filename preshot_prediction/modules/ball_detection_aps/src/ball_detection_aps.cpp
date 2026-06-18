// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ball_detection_aps/ball_detection_aps.hpp"

#include <ace_rt_profiles/profile_manager.hpp>
#include <memory>

#include "ace_loggers/logging.hpp"
#include "ball_detector/ball_detector_opencv.hpp"
#include "ball_detector/datalogger.hpp"
#include "cuda_common/cuda_common.hpp"
#include "dot_detector/dot_detector_opencv.hpp"
#include "vision_common/vision_common.hpp"

namespace ball_detection_aps {

/*
 * Update distorted mask using undistorted shape
 */
inline void UpdateMaskUsingShape(cv::Mat& dst_mask_dist, const Eigen::Matrix2Xf& undistorted_points,
                                 const calibration::Camera& camera) {
  // Compute shape_hull
  cv::Mat shape_hull;
  cv::convexHull(vision_common::EigenToCV(*const_cast<Eigen::Matrix2Xf*>(&undistorted_points)), shape_hull);
  shape_hull.convertTo(shape_hull, CV_32S);

  // Fill undistorted mask
  cv::Mat valid_mask_undist(camera.resolution[1], camera.resolution[0], CV_8U, cv::Scalar(0));
  cv::fillPoly(valid_mask_undist, shape_hull, 255);

  // Update input mask
  cv::Mat valid_mask_dist;
  camera.DistortImage(valid_mask_undist, valid_mask_dist);
  cv::bitwise_and(dst_mask_dist, valid_mask_dist, dst_mask_dist);
}

BallDetectionAPS::BallDetectionAPS(const rclcpp::NodeOptions& options)
  : rclcpp::Node("ball_detection_aps", options),
    params_ptr_(std::make_shared<BallDetectionAPSParameters>()),
    camera_calib_params_ptr_(std::make_shared<calibration::CameraCalibrationParameters>()) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }
  init_ = false;
  debug_ = DebugMode::kOff;

  const std::string& params_path = declare_parameter<std::string>("params_path");
  const std::string& camera_calib_params_path = declare_parameter<std::string>("camera_calib_params_path");
  const std::string& debug = declare_parameter<std::string>("debug");
  const bool& ignore_gpu_checks = declare_parameter<bool>("ignore_gpu_checks");
  if (ignore_gpu_checks) {
    LOG(WARNING) << "Ignoring GPU checks.\033[0m\n";
  }

  // Initialize ball detection.
  if (Initialize(params_path, camera_calib_params_path, debug, ignore_gpu_checks)) {
    LOG(INFO) << "Initialized ball detection.\n";
  } else {
    LOG(ERROR) << "Failed to initialize the ball detection.\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }
}

BallDetectionAPS::~BallDetectionAPS() {
  LOG(INFO) << "Stopping threads for ball detection.\n";

  canceled_.store(true);
  for (auto& t : threads_) {
    if (t.Joinable()) {
      t.Join();  // Wait until all jobs are done.
    }
  }

  // Clear image buffer (due to zero copy, otherwise middleware does not now they are available again).
  for (auto camera_index = 0u; camera_index < params_ptr_->camera_names.size(); ++camera_index) {
    messages_[camera_index].reset();
  }

  // stop debug callback
  while (debug_timer_ != nullptr && !debug_timer_->is_canceled()) {
    debug_timer_->cancel();
  }

  // Close windows if in debug mode.
  if (debug_ == DebugMode::kDetection) {
    cv::destroyAllWindows();
  }

  LOG(INFO) << "All threads for ball detection stopped.\n";
}

bool BallDetectionAPS::UpdateParameterValue(const ParameterID parameter_id, const int value) {
  if (!init_) {
    LOG(ERROR) << "\"UpdateParameterValue\" is only available in in debug mode.\n";
    return false;
  }

  switch (parameter_id) {
    case ParameterID::kHLow:
      params_ptr_->hsv_lower_boundary[0] = static_cast<double>(value);
      params_ptr_->hsv_upper_boundary[0] =
        std::max(params_ptr_->hsv_lower_boundary[0], params_ptr_->hsv_upper_boundary[0]);
      break;

    case ParameterID::kHHigh:
      params_ptr_->hsv_upper_boundary[0] = static_cast<double>(value);
      params_ptr_->hsv_lower_boundary[0] =
        std::min(params_ptr_->hsv_lower_boundary[0], params_ptr_->hsv_upper_boundary[0]);
      break;

    case ParameterID::kSLow:
      params_ptr_->hsv_lower_boundary[1] = static_cast<double>(value);
      params_ptr_->hsv_upper_boundary[1] =
        std::max(params_ptr_->hsv_lower_boundary[1], params_ptr_->hsv_upper_boundary[1]);
      break;

    case ParameterID::kSHigh:
      params_ptr_->hsv_upper_boundary[1] = static_cast<double>(value);
      params_ptr_->hsv_lower_boundary[1] =
        std::min(params_ptr_->hsv_lower_boundary[1], params_ptr_->hsv_upper_boundary[1]);
      break;

    case ParameterID::kVLow:
      params_ptr_->hsv_lower_boundary[2] = static_cast<double>(value);
      params_ptr_->hsv_upper_boundary[2] =
        std::max(params_ptr_->hsv_lower_boundary[2], params_ptr_->hsv_upper_boundary[2]);
      break;

    case ParameterID::kVHigh:
      params_ptr_->hsv_upper_boundary[2] = static_cast<double>(value);
      params_ptr_->hsv_lower_boundary[2] =
        std::min(params_ptr_->hsv_lower_boundary[2], params_ptr_->hsv_upper_boundary[2]);
      break;

    default:
      LOG(ERROR) << "Invalid parameter_id (" << static_cast<int>(parameter_id) << ").\n";
      return false;
  }

  return true;
}

bool BallDetectionAPS::Initialize(const std::string& params_path, const std::string& camera_calib_params_path,
                                  const std::string& debug, const bool& ignore_gpu_checks) {
  if (!params_ptr_->Initialize(params_path)) {
    return false;
  }
  params_ptr_->PrintParameters();

  if (!camera_calib_params_ptr_->Initialize(camera_calib_params_path)) {
    return false;
  }
  camera_calib_params_ptr_->PrintParameters();

  if (debug == "true" || debug == "detection") {
    LOG(WARNING) << "Running ball detection in DEBUG=" << debug
                 << " mode. Do not run it in debug mode for experiments.\n";

    debug_ = DebugMode::kDetection;

    cv::namedWindow(debug_window_name_);
    cv::createTrackbar("h_low", debug_window_name_, nullptr, 179);
    cv::setTrackbarPos("h_low", debug_window_name_, static_cast<int>(params_ptr_->hsv_lower_boundary[0]));
    cv::createTrackbar("h_high", debug_window_name_, nullptr, 179);
    cv::setTrackbarPos("h_high", debug_window_name_, static_cast<int>(params_ptr_->hsv_upper_boundary[0]));
    cv::createTrackbar("s_low", debug_window_name_, nullptr, 255);
    cv::setTrackbarPos("s_low", debug_window_name_, static_cast<int>(params_ptr_->hsv_lower_boundary[1]));
    cv::createTrackbar("s_high", debug_window_name_, nullptr, 255);
    cv::setTrackbarPos("s_high", debug_window_name_, static_cast<int>(params_ptr_->hsv_upper_boundary[1]));
    cv::createTrackbar("v_low", debug_window_name_, nullptr, 255);
    cv::setTrackbarPos("v_low", debug_window_name_, static_cast<int>(params_ptr_->hsv_lower_boundary[2]));
    cv::createTrackbar("v_high", debug_window_name_, nullptr, 255);
    cv::setTrackbarPos("v_high", debug_window_name_, static_cast<int>(params_ptr_->hsv_upper_boundary[2]));
    debug_timer_ = this->create_wall_timer(std::chrono::milliseconds(30), [this]() { DebugCallback(); });
  } else if (debug == "mask") {
    LOG(WARNING) << "Running ball detection in DEBUG=" << debug
                 << " mode. Do not run it in debug mode for experiments.\n";

    debug_ = DebugMode::kValidMask;
  } else if (!debug.empty()) {
    LOG(ERROR) << "Ignoring invalid DEBUG=" << debug << " mode.\n";
  }

  // Start a thread for each camera.
  canceled_.store(false);

  ball_detector_ptrs_.clear();
  dot_detector_ptrs_.clear();
  mutexes_ = std::make_unique<std::mutex[]>(params_ptr_->camera_names.size());
  condition_variable_messages_available_ =
    std::make_unique<std::condition_variable[]>(params_ptr_->camera_names.size());

  const auto& manager = ace_rt_profiles::ProfileManager::GetInstance();
  const auto cpu_affinity = manager.GetAffinity(params_ptr_->rt_profile);
  if (cpu_affinity.size() < params_ptr_->camera_names.size()) {
    LOG(WARNING) << "CPU Affinity list size is less than cameras count, performance might be impacted!";
  }

  threads_ = manager.MakeThreads(params_ptr_->rt_profile, params_ptr_->camera_names.size(),
                                 ace_rt_profiles::AffinityMode::kSequential);

  for (auto camera_index = 0u; camera_index < params_ptr_->camera_names.size(); ++camera_index) {
    const std::string& camera_name = params_ptr_->camera_names[camera_index];
    auto it = camera_calib_params_ptr_->cameras.find(camera_name);
    if (it == camera_calib_params_ptr_->cameras.end()) {
      LOG(ERROR) << "Unable to find camera calibration parameters for camera " << camera_name << ".\n";
      return false;
    }
    const auto& camera = it->second;

    // Initially the whole image plane is valid
    valid_masks_.emplace_back(camera.resolution[1], camera.resolution[0], CV_8U, cv::Scalar(255));

    if (params_ptr_->border_filter_enable) {
      // Compute 3D points unprojection on the undistorted plane
      Eigen::Matrix<float, 2, 4> distorted_points;
      vision_common::ComputeAABBFromMinMaxCorners<float, 2>(
        distorted_points, Eigen::Vector2f::Zero(), Eigen::Vector2f({camera.resolution[0], camera.resolution[1]}));

      Eigen::Matrix2Xf undistorted_points = camera.UndistortPointsVectorized(distorted_points);
      Eigen::Matrix3Xf border_filter_corners = params_ptr_->border_filter_plane_depth * camera.camera_matrix.inverse() *
                                               undistorted_points.colwise().homogeneous();

      // Subtract the margins
      Eigen::Matrix<float, 2, 4> margin_offsets;
      vision_common::ComputeAABBFromMinMaxCorners<float, 2>(
        margin_offsets,
        Eigen::Vector2f({-params_ptr_->border_filter_margin_size, -params_ptr_->border_filter_margin_size}),
        Eigen::Vector2f({params_ptr_->border_filter_margin_size, params_ptr_->border_filter_margin_size}));
      border_filter_corners.block<2, 4>(0, 0) -= margin_offsets;

      // Project the points on the undistorted plane
      undistorted_points = (camera.camera_matrix * border_filter_corners).colwise().hnormalized();

      // Update corresponding mask
      UpdateMaskUsingShape(valid_masks_[camera_index], undistorted_points, camera);
    }

    // Create a separate publisher for camera.
    {
      std::string topic = std::string(this->get_namespace()) + "/" + camera_name + "/" + "ball_detection";
      pubs_.emplace_back(this->create_publisher<ace_interfaces::msg::BallsWithAttributes>(topic, 1));

      if (debug_ == DebugMode::kDetection || debug_ == DebugMode::kValidMask) {
        topic += "_mask";
        debug_pubs_.emplace_back(this->create_publisher<ace_interfaces::msg::ImageData>(topic, 1));
      }
    }

    // Start worker thread for camera.
    const int cuda_device_id = cuda_common::GetCudaDeviceID(
      params_ptr_->cuda_device_uuids[camera_index % params_ptr_->cuda_device_uuids.size()], ignore_gpu_checks);
    if (cuda_device_id < 0) {
      return false;
    }

    auto datawriter_ptr = std::make_shared<ball_detector::datalogger::DataWriter>(
      ::datalogger::ConstructFullLogName("ball_detector_" + camera_name), camera_name, params_ptr_->ToString());
    ball_detector_ptrs_.emplace_back(std::make_unique<ball_detector::BallDetectorOpenCV>());
    if (!ball_detector_ptrs_[camera_index]->SetDataWriter(std::move(datawriter_ptr))) {
      return false;
    }
    if (!ball_detector_ptrs_[camera_index]->SetParameters(params_ptr_)) {
      return false;
    }

    // dot_detector_ptrs_.emplace_back(std::make_unique<dot_detector::DotDetectorTensorRT>());
    dot_detector_ptrs_.emplace_back(std::make_unique<dot_detector::DotDetectorOpenCV>(params_ptr_));
    if (!dot_detector_ptrs_[camera_index]->SetParameters(params_ptr_)) {
      return false;
    }

    messages_available_.push_back(false);
    messages_.push_back(nullptr);

    threads_[camera_index].Run([camera_index, cuda_device_id, this]() { WorkerThread(camera_index, cuda_device_id); });
    // Add subscriber to camera images that feeds worker thread.
    {
      const auto callback = [&, camera_index](ace_interfaces::msg::ImageData::ConstSharedPtr message) {
        this->ProcessImage(message, camera_index);
      };

      const std::string topic = std::string(this->get_namespace()) + "/" + camera_name + "/" + "image";
      subs_.emplace_back(this->create_subscription<ace_interfaces::msg::ImageData>(topic, 1, callback));
    }
  }

  // Set up LoggerControl callback
  rclcpp::QoS qos_persistent(rclcpp::KeepLast(1));
  qos_persistent.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos_persistent.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

  using LoggerControlPtr = const ace_interfaces::msg::LoggerControl::ConstSharedPtr;
  loggercontrol_sub_ = this->create_subscription<ace_interfaces::msg::LoggerControl>(
    "/logger/control", qos_persistent,
    [this](LoggerControlPtr const& payload) { return LoggerControlCallback(payload); });

  using RequestPtr = ace_interfaces::srv::ImageData::Request::SharedPtr;
  using ResponsePtr = ace_interfaces::srv::ImageData::Response::SharedPtr;

  image_data_service_ = create_service<ace_interfaces::srv::ImageData>(
    "/sensors/ball_detection/image_data",
    [this](RequestPtr request, ResponsePtr response) { ImageDataService(std::move(request), std::move(response)); });

  init_ = true;

  return true;
}

void BallDetectionAPS::ProcessImage(const ace_interfaces::msg::ImageData::ConstSharedPtr& message,
                                    const unsigned int camera_index) {
  std::unique_lock<std::mutex> lock(mutexes_[camera_index]);
  messages_[camera_index] = message;
  messages_available_[camera_index] = true;
  lock.unlock();
  condition_variable_messages_available_[camera_index].notify_all();
}

namespace {
void FillMessage(ace_interfaces::msg::ImageData& out_msg, const ace_interfaces::msg::ImageData::ConstSharedPtr& in_msg,
                 const cv::Mat& in_img) {
  out_msg.header.stamp = in_msg->header.stamp;
  out_msg.header.sequence_number = in_msg->header.sequence_number;
  out_msg.frame_rate = in_msg->frame_rate;
  out_msg.encoding = std::array<unsigned char, 20>{"mono8"};
  out_msg.height = static_cast<uint16_t>(in_img.rows);
  out_msg.width = static_cast<uint16_t>(in_img.cols);
  out_msg.offset_x = 0UL;
  out_msg.offset_y = 0UL;
  out_msg.step = static_cast<uint16_t>(in_img.step);
  std::memcpy(out_msg.data.data(), in_img.data, out_msg.height * out_msg.step);
}
}  // namespace

void BallDetectionAPS::WorkerThread(const unsigned int camera_index, const int cuda_device_id) {
  std::vector<std::pair<cv::Point2f, float>> ball_detections;
  std::vector<std::pair<cv::RotatedRect, float>> marker_detections;
  cv::cuda::GpuMat bgr_img;
  cv::Mat bayer_img;
  cv::Mat debug_mask;

  const auto& ball_detector_ptr = ball_detector_ptrs_[camera_index].get();
  if (!ball_detector_ptr->SetCudaDeviceID(cuda_device_id) ||
      !dot_detector_ptrs_[camera_index]->SetCudaDeviceID(cuda_device_id)) {
    LOG(ERROR) << "Unable to set cuda_device_id=" << cuda_device_id << " for camera "
               << params_ptr_->camera_names[camera_index] << ".\n";
    return;
  }

  while (!canceled_.load()) {
    std::unique_lock<std::mutex> lock(mutexes_[camera_index]);
    if (!condition_variable_messages_available_[camera_index].wait_for(
          lock, kMessagesWaitDuration, [&]() -> bool { return messages_available_[camera_index]; })) {
      continue;
    }

    // Copy pointer to image and return lock.
    const ace_interfaces::msg::ImageData::ConstSharedPtr message = messages_[camera_index];
    messages_available_[camera_index] = false;
    lock.unlock();

    // Prepare output message.
    const auto start_time = std::chrono::high_resolution_clock::now();
    ace_interfaces::msg::BallsWithAttributes message_out;
    message_out.header.stamp = message->header.stamp;
    message_out.header.sequence_number = message->header.sequence_number;

    // Convert ROS 2 message to OpenCV image
    const void* data_ptr =
      reinterpret_cast<const void*>(message->data.data() + message->step * message->offset_y + message->offset_x);
    // NOLINT
    bayer_img = cv::Mat(message->height, message->width, CV_8UC1, const_cast<void*>(data_ptr), message->step);

    ball_detections.clear();
    ball_detector_ptr->SetFrameRate(message->frame_rate);
    ball_detector_ptr->SetBayerImage(bayer_img);
    ball_detector_ptr->GetDataLogger() << message->header.sequence_number << message->header.stamp << start_time;
    ball_detector_ptr->DetectBalls(ball_detections, valid_masks_[camera_index]);

    // Provide the bgr_img to the dot detector
    ball_detector_ptr->GetBgrImage(bgr_img);
    dot_detector_ptrs_[camera_index]->SetBgrImage(bgr_img);

    for (auto& [ball_center, ball_radius] : ball_detections) {
      message_out.balls.emplace_back();
      auto& ball_msg = message_out.balls.back();
      ball_msg.ball.center.x = ball_center.x;
      ball_msg.ball.center.y = ball_center.y;
      ball_msg.ball.radius = ball_radius;

      // Detect markers
      marker_detections.clear();
      dot_detector_ptrs_[camera_index]->DetectMarkers(ball_center, ball_radius, marker_detections);

      // Append marker detections to the message
      ball_msg.markers.reserve(marker_detections.size());
      for (auto& [marker_rect, marker_radius] : marker_detections) {
        ball_msg.markers.emplace_back();
        auto& marker_msg = ball_msg.markers.back();
        marker_msg.center.x = marker_rect.center.x;
        marker_msg.center.y = marker_rect.center.y;
        marker_msg.radius = marker_radius;
      }
    }

    pubs_[camera_index]->publish(std::move(message_out));

    if (debug_ == DebugMode::kDetection || debug_ == DebugMode::kValidMask) {
      if (debug_ == DebugMode::kValidMask) {
        debug_mask = valid_masks_[camera_index];
      } else {
        ball_detector_ptr->GetDetectionMask(debug_mask);
      }
      auto loaned_msg = debug_pubs_[camera_index]->borrow_loaned_message();
      auto& debug_message_out = loaned_msg.get();
      FillMessage(debug_message_out, message, debug_mask);
      debug_pubs_[camera_index]->publish(std::move(loaned_msg));
    }
  }
}

void BallDetectionAPS::ImageDataService(ace_interfaces::srv::ImageData::Request::SharedPtr request,
                                        ace_interfaces::srv::ImageData::Response::SharedPtr response) {
  const auto camera_name = std::string(reinterpret_cast<char*>(&request->camera_name[0]));
  const auto it = std::find(params_ptr_->camera_names.begin(), params_ptr_->camera_names.end(), camera_name);
  if (it == params_ptr_->camera_names.end()) {
    LOG(WARNING) << "Cannot find camera_name=\"" << camera_name << "\" specified in the request!";
    response->is_valid = false;
    return;
  }

  const auto camera_index = std::distance(params_ptr_->camera_names.begin(), it);
  const auto img_type = static_cast<ImageType>(request->img_type);
  if (img_type == ImageType::kDetection) {
    const std::scoped_lock<std::mutex> lock(mutexes_[camera_index]);

    const auto& message = messages_[camera_index];
    if (message == nullptr) {
      LOG(WARNING) << "Input image is missing. Is APS system running?";
      response->is_valid = false;
      return;
    }
    const auto& ball_detector_ptr = ball_detector_ptrs_[camera_index].get();

    cv::Mat detection_mask;
    ball_detector_ptr->GetDetectionMask(detection_mask);
    FillMessage(response->img, message, detection_mask);
    LOG(INFO) << "Handled a request with camera_name=\"" << camera_name << "\" and img_type=" << request->img_type;
    response->is_valid = true;
    return;
  }

  LOG(WARNING) << "Request with img_type=" << request->img_type << " is not supported!";
  response->is_valid = false;
}

void BallDetectionAPS::LoggerControlCallback(const ace_interfaces::msg::LoggerControl::ConstSharedPtr& msg_ptr) {
  const auto log_prefix = std::string(reinterpret_cast<const char*>(msg_ptr->log_prefix.data()));

  for (auto camera_index = 0u; camera_index < params_ptr_->camera_names.size(); ++camera_index) {
    const auto& ball_detector_ptr = ball_detector_ptrs_[camera_index].get();
    const std::string& camera_name = params_ptr_->camera_names[camera_index];
    const auto log_name = ::datalogger::ConstructSubLogName(log_prefix, "ball_detector_" + camera_name);
    {
      const std::lock_guard lock(mutexes_[camera_index]);
      ball_detector_ptr->GetDataLogger()->SetFileName(log_name);
    }
    LOG(INFO) << "Log filename got externally updated to:\n - \"" << log_prefix << "\"\n";
  }
}

void BallDetectionAPS::DebugCallback() {
  const int h_low = cv::getTrackbarPos("h_low", debug_window_name_);
  params_ptr_->hsv_lower_boundary[0] = static_cast<double>(h_low);
  if (params_ptr_->hsv_upper_boundary[0] !=
      std::max(params_ptr_->hsv_lower_boundary[0], params_ptr_->hsv_upper_boundary[0])) {
    params_ptr_->hsv_upper_boundary[0] =
      std::max(params_ptr_->hsv_lower_boundary[0], params_ptr_->hsv_upper_boundary[0]);
    cv::setTrackbarPos("h_high", debug_window_name_, static_cast<int>(params_ptr_->hsv_upper_boundary[0]));
  }

  const int h_high = cv::getTrackbarPos("h_high", debug_window_name_);
  params_ptr_->hsv_upper_boundary[0] = static_cast<double>(h_high);
  if (params_ptr_->hsv_lower_boundary[0] !=
      std::min(params_ptr_->hsv_lower_boundary[0], params_ptr_->hsv_upper_boundary[0])) {
    params_ptr_->hsv_lower_boundary[0] =
      std::min(params_ptr_->hsv_lower_boundary[0], params_ptr_->hsv_upper_boundary[0]);
    cv::setTrackbarPos("h_low", debug_window_name_, static_cast<int>(params_ptr_->hsv_lower_boundary[0]));
  }

  const int s_low = cv::getTrackbarPos("s_low", debug_window_name_);
  params_ptr_->hsv_lower_boundary[1] = static_cast<double>(s_low);
  if (params_ptr_->hsv_upper_boundary[1] !=
      std::max(params_ptr_->hsv_lower_boundary[1], params_ptr_->hsv_upper_boundary[1])) {
    params_ptr_->hsv_upper_boundary[1] =
      std::max(params_ptr_->hsv_lower_boundary[1], params_ptr_->hsv_upper_boundary[1]);
    cv::setTrackbarPos("s_high", debug_window_name_, static_cast<int>(params_ptr_->hsv_upper_boundary[1]));
  }

  const int s_high = cv::getTrackbarPos("s_high", debug_window_name_);
  params_ptr_->hsv_upper_boundary[1] = static_cast<double>(s_high);
  if (params_ptr_->hsv_lower_boundary[1] !=
      std::min(params_ptr_->hsv_lower_boundary[1], params_ptr_->hsv_upper_boundary[1])) {
    params_ptr_->hsv_lower_boundary[1] =
      std::min(params_ptr_->hsv_lower_boundary[1], params_ptr_->hsv_upper_boundary[1]);
    cv::setTrackbarPos("s_low", debug_window_name_, static_cast<int>(params_ptr_->hsv_lower_boundary[1]));
  }

  const int v_low = cv::getTrackbarPos("v_low", debug_window_name_);
  params_ptr_->hsv_lower_boundary[2] = static_cast<double>(v_low);
  if (params_ptr_->hsv_upper_boundary[2] !=
      std::max(params_ptr_->hsv_lower_boundary[2], params_ptr_->hsv_upper_boundary[2])) {
    params_ptr_->hsv_upper_boundary[2] =
      std::max(params_ptr_->hsv_lower_boundary[2], params_ptr_->hsv_upper_boundary[2]);
    cv::setTrackbarPos("v_high", debug_window_name_, static_cast<int>(params_ptr_->hsv_upper_boundary[2]));
  }

  const int v_high = cv::getTrackbarPos("v_high", debug_window_name_);
  params_ptr_->hsv_upper_boundary[2] = static_cast<double>(v_high);
  if (params_ptr_->hsv_lower_boundary[2] !=
      std::min(params_ptr_->hsv_lower_boundary[2], params_ptr_->hsv_upper_boundary[2])) {
    params_ptr_->hsv_lower_boundary[2] =
      std::min(params_ptr_->hsv_lower_boundary[2], params_ptr_->hsv_upper_boundary[2]);
    cv::setTrackbarPos("v_low", debug_window_name_, static_cast<int>(params_ptr_->hsv_lower_boundary[2]));
  }

  const int pressed_key = cv::waitKey(30);
  if ((pressed_key & 0xff) == 's') {
    LOG(INFO) << "Key \"s\" was pressed on keyboard, storing data to parameter file.\n";
    params_ptr_->WriteParametersToFile();

    LOG(INFO) << "New parameters:\n";
    params_ptr_->PrintParameters();
  }
}

}  // namespace ball_detection_aps
