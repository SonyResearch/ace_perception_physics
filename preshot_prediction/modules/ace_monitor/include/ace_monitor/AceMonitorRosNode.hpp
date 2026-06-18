// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <rclzmq/rclzmq.hpp>

namespace ace_monitor {

class AceMonitorRosNode {
 public:
  AceMonitorRosNode();
  ~AceMonitorRosNode();

  static void Initialize();
  static void Shutdown();
  static std::shared_ptr<rclzmq::Node> Node();
};
}  // namespace ace_monitor
