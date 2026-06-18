// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include <ace_rt_profiles/profile_manager.hpp>

#include "ace_loggers/ace_loggers.hpp"
#include "multi_ball_triangulation/multi_ball_triangulation.hpp"

// NOLINTNEXTLINE(bugprone-exception-escape)
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
  // intraprocess communication allowed only with volatile durability
  // options.use_intra_process_comms(true);
  rclcpp::spin(std::make_shared<multi_ball_triangulation::MultiBallTriangulation>(options));
  rclcpp::shutdown();

  return 0;
}
