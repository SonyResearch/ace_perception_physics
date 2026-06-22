// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <glog/logging.h>
#include "evs/metavision_recorder.hpp"

int main(int argc, char* argv[]) {
  setvbuf(stdout, nullptr, _IONBF, BUFSIZ);
  rclcpp::init(argc, argv);

  google::AllowCommandLineReparsing();
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(false);
  options.automatically_declare_parameters_from_overrides(false);
  options.start_parameter_event_publisher(false);
  options.start_parameter_services(false);
  options.use_intra_process_comms(true);
  rclcpp::spin(std::make_shared<evs::MetavisionRecorder>(options));
  rclcpp::shutdown();

  return 0;
}
