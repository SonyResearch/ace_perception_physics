// Standalone shim: ArenaNativeAPI now owns a plain rclcpp::Node + executor thread.
#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>

namespace arena {

class ArenaNativeAPI {
 public:
  static void RefROS() {
    auto& self = Instance();
    if (self.ref_count_++ == 0) {
      if (!::rclcpp::ok()) {
        int argc = 0;
        char** argv = nullptr;
        ::rclcpp::init(argc, argv);
      }
      self.node_ = std::make_shared<::rclcpp::Node>("ace_monitor");
      self.exec_ = std::make_shared<::rclcpp::executors::SingleThreadedExecutor>();
      self.exec_->add_node(self.node_);
      self.running_ = true;
      self.thread_ = std::thread([&self]() {
        while (self.running_.load()) {
          self.exec_->spin_some(std::chrono::milliseconds(50));
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      });
    }
  }

  static void UnrefROS() {
    auto& self = Instance();
    if (self.ref_count_ == 0) {
      return;
    }
    if (--self.ref_count_ == 0) {
      self.running_ = false;
      if (self.thread_.joinable()) {
        self.thread_.join();
      }
      if (self.exec_ && self.node_) {
        self.exec_->remove_node(self.node_);
      }
      self.exec_.reset();
      self.node_.reset();
    }
  }

  static std::shared_ptr<::rclcpp::Node> GetROSNode() { return Instance().node_; }

 private:
  ArenaNativeAPI() = default;
  static ArenaNativeAPI& Instance() {
    static ArenaNativeAPI inst;
    return inst;
  }
  std::shared_ptr<::rclcpp::Node> node_;
  std::shared_ptr<::rclcpp::executors::SingleThreadedExecutor> exec_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  int ref_count_{0};
};

}  // namespace arena
