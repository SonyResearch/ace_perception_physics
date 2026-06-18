// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include <ace_rt_profiles/profile_manager.hpp>

#include "ace_loggers/ace_loggers.hpp"
#include "aps/multi_camera.hpp"

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
  // TODO(cv3d): IPC is useful, but does not work yet with loaned messages yet
  // See https://github.com/ros2/rclcpp/issues/1769
  // options.use_intra_process_comms(true);
  rclcpp::spin(std::make_shared<aps::MultiCamera>(options));
  rclcpp::shutdown();

  return 0;
}
