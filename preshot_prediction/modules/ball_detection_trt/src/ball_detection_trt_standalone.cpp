// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <ace_rt_profiles/profile_manager.hpp>

#include "ace_loggers/ace_loggers.hpp"
#include "ball_detection_trt/ball_detection_trt.hpp"

int main(int argc, char* argv[]) {
  setvbuf(stdout, nullptr, _IONBF, BUFSIZ);
  ace_rt_profiles::ProfileManager::GetInstance().Apply("perception_default_ros");
  rclcpp::init(argc, argv);

  google::AllowCommandLineReparsing();
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(false);
  options.automatically_declare_parameters_from_overrides(false);
  options.start_parameter_event_publisher(false);
  options.start_parameter_services(false);

  rclcpp::spin(std::make_shared<ball_detection_trt::BallDetectionTRT>(options));
  rclcpp::shutdown();

  return 0;
}
