// Standalone shim: alias rclzmq -> rclcpp.
#pragma once

#include <rclcpp/rclcpp.hpp>

namespace rclzmq {
using Node = ::rclcpp::Node;
using QoS = ::rclcpp::QoS;
using KeepLast = ::rclcpp::KeepLast;
using KeepAll = ::rclcpp::KeepAll;

template <typename M>
using Subscription = ::rclcpp::Subscription<M>;

template <typename M>
using Publisher = ::rclcpp::Publisher<M>;

inline bool ok() { return ::rclcpp::ok(); }
inline void init(int argc, char** argv) { ::rclcpp::init(argc, argv); }
inline void shutdown() { ::rclcpp::shutdown(); }
inline void spin_some(::rclcpp::Node::SharedPtr node) { ::rclcpp::spin_some(node); }
}  // namespace rclzmq
