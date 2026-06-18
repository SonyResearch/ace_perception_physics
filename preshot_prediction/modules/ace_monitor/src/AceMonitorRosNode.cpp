// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ace_monitor/AceMonitorRosNode.hpp"

#include "ace_loggers/ace_loggers.hpp"
#include "arena_native_ros/ArenaNativeAPI.hpp"

namespace ace_monitor {

AceMonitorRosNode::AceMonitorRosNode() = default;
AceMonitorRosNode::~AceMonitorRosNode() { Shutdown(); }

void AceMonitorRosNode::Initialize() { arena::ArenaNativeAPI::RefROS(); }

void AceMonitorRosNode::Shutdown() { arena::ArenaNativeAPI::UnrefROS(); }

std::shared_ptr<rclzmq::Node> AceMonitorRosNode::Node() { return arena::ArenaNativeAPI::GetROSNode(); }
}  // namespace ace_monitor
