// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "racket_pose_estimation/ros/RacketPoseEstimator.hpp"

#include <filesystem>
#include <memory>

#include "ace_loggers/ace_loggers.hpp"
#include "racket_pose_estimation/RacketPoseExtractor.hpp"
#include "racket_pose_estimation/ros/RacketPoseROSNode.hpp"

namespace perception {

RacketPoseEstimator::RacketPoseEstimator(const rclcpp::NodeOptions &options)
  : rclcpp::Node("racket_pose_vive", options) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }
  init_ = false;
}

RacketPoseEstimator::~RacketPoseEstimator() { Destroy(); }

void RacketPoseEstimator::Initialize() {
  auto camera_config_path = declare_parameter<std::string>("camera_config_path");
  auto racket_config_path = declare_parameter<std::string>("racket_config_path");
  auto racket_detector = declare_parameter<std::string>("racket_detector");
  tune_parameters_ = declare_parameter<std::string>("tune") == "true";

  node_ = std::make_shared<perception::RacketPoseROSNode>();
  extractor_ = std::make_shared<perception::RacketPoseExtractor>(std::filesystem::path(racket_config_path).stem());
  extractor_->Start(camera_config_path, racket_config_path, racket_detector, true);

  node_->StartWithNode(shared_from_this(), extractor_);
  init_ = true;

  if (tune_parameters_) {
    auto parameters = extractor_->GetRacketParameters();
    cv::namedWindow(tune_window_name_);

    cv::createTrackbar("extractor.confidence_threshold", tune_window_name_, nullptr, 1e2);
    cv::setTrackbarPos("extractor.confidence_threshold", tune_window_name_,
                       static_cast<int>(parameters->extractor.confidence_threshold * 1e2));
    cv::createTrackbar("extractor.nms_threshold", tune_window_name_, nullptr, 1e2);
    cv::setTrackbarPos("extractor.nms_threshold", tune_window_name_,
                       static_cast<int>(parameters->extractor.nms_threshold * 1e2));
    cv::createTrackbar("extractor.latest_weight", tune_window_name_, nullptr, 1e2);
    cv::setTrackbarPos("extractor.latest_weight", tune_window_name_,
                       static_cast<int>(parameters->extractor.latest_weight * 1e2));
    cv::createTrackbar("extractor.history_difference", tune_window_name_, nullptr, 1e2);
    cv::setTrackbarPos("extractor.history_difference", tune_window_name_, parameters->extractor.history_difference);

    cv::createTrackbar("fitter.initial_rate", tune_window_name_, nullptr, 1e3);
    cv::setTrackbarPos("fitter.initial_rate", tune_window_name_,
                       static_cast<int>(parameters->fitter.initial_rate * 1e3));
    cv::createTrackbar("fitter.max_iterations", tune_window_name_, nullptr, 3e3);
    cv::setTrackbarPos("fitter.max_iterations", tune_window_name_, static_cast<int>(parameters->fitter.max_iterations));
    cv::createTrackbar("fitter.min_axis_length", tune_window_name_, nullptr, 1e2);
    cv::setTrackbarPos("fitter.min_axis_length", tune_window_name_,
                       static_cast<int>(parameters->fitter.min_axis_length));
    cv::createTrackbar("fitter.max_error", tune_window_name_, nullptr, 1e3);
    cv::setTrackbarPos("fitter.max_error", tune_window_name_, static_cast<int>(parameters->fitter.max_error * 1e3));

    tune_timer_ =
      this->create_wall_timer(std::chrono::milliseconds(30), std::bind(&RacketPoseEstimator::TunerCallback, this));
  }
}
void RacketPoseEstimator::Destroy() {
  node_->Stop();
  extractor_.reset();

  // stop callback
  while (tune_timer_ != nullptr && !tune_timer_->is_canceled()) {
    tune_timer_->cancel();
  }

  if (tune_parameters_) {
    cv::destroyAllWindows();
  }
}

void RacketPoseEstimator::TunerCallback() {
  auto parameters = extractor_->GetRacketParameters();
  int value;

  value = cv::getTrackbarPos("extractor.confidence_threshold", tune_window_name_);
  parameters->extractor.confidence_threshold = static_cast<float>(value) * 1e-2F;
  value = cv::getTrackbarPos("extractor.nms_threshold", tune_window_name_);
  parameters->extractor.nms_threshold = static_cast<float>(value) * 1e-2F;
  value = cv::getTrackbarPos("extractor.latest_weight", tune_window_name_);
  parameters->extractor.latest_weight = static_cast<float>(value) * 1e-2F;
  value = cv::getTrackbarPos("extractor.history_difference", tune_window_name_);
  parameters->extractor.history_difference = value;

  value = cv::getTrackbarPos("fitter.initial_rate", tune_window_name_);
  parameters->fitter.initial_rate = static_cast<float>(value) * 1e-3F;
  value = cv::getTrackbarPos("fitter.max_iterations", tune_window_name_);
  parameters->fitter.max_iterations = value;
  value = cv::getTrackbarPos("fitter.min_axis_length", tune_window_name_);
  parameters->fitter.min_axis_length = static_cast<float>(value);
  value = cv::getTrackbarPos("fitter.max_error", tune_window_name_);
  parameters->fitter.max_error = static_cast<float>(value) * 1e-3F;

  int pressed_key = cv::waitKey(10);
  if ((pressed_key & 0xff) == 's') {
    // LOG(INFO) << "Key \"s\" was pressed on keyboard, storing data to parameter file.\n";
    // params_ptr_->WriteParametersToFile();

    // LOG(INFO) << "New parameters:\n";
    // params_ptr_->PrintParameters();
  }
}
}  // namespace perception
