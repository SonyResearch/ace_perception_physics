// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "ace_interfaces/msg/image_data.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "rclcpp/rclcpp.hpp"

namespace aps {

class ZeroCopySubscriber : public rclcpp::Node {
 public:
  using SharedPtr = std::shared_ptr<ZeroCopySubscriber>;
  using ConstSharedPtr = std::shared_ptr<const ZeroCopySubscriber>;

  explicit ZeroCopySubscriber(const rclcpp::NodeOptions& options);
  ~ZeroCopySubscriber() override;

  bool IsInitialized() const { return init_; };
  int AddSubscription(std::string topic);
  bool GetLatestMessage(int handle, ace_interfaces::msg::ImageData::SharedPtr* message);
  bool ReleaseLatestMessage(int handle);

 private:
  // general
  bool init_;
  bool Initialize();

  // subscribers
  const int max_subscriptions_ = 50;
  int num_subs_;
  std::vector<rclcpp::Subscription<ace_interfaces::msg::ImageData>::SharedPtr> subs_;
  std::vector<ace_interfaces::msg::ImageData::SharedPtr> latest_msgs_;
  std::vector<ace_interfaces::msg::ImageData::SharedPtr> locked_msgs_;
  std::unique_ptr<std::mutex[]> m_msgs_;
};

class ZeroCopySubscriberPy {
 public:
  using SharedPtr = std::shared_ptr<ZeroCopySubscriberPy>;
  using ConstSharedPtr = std::shared_ptr<const ZeroCopySubscriberPy>;

  explicit ZeroCopySubscriberPy();

  [[nodiscard]] bool IsInitialized() const { return init_; };
  int AddSubscription(std::string topic);
  std::tuple<bool, uint64_t, std::string, pybind11::array> GetLatestMessage(int handle);
  bool ReleaseLatestMessage(int handle);

 private:
  // general
  bool init_;
  bool Initialize();

  ZeroCopySubscriber::SharedPtr zero_copy_subscriber_;

  // thread
  std::thread thread_;
  void WorkerThread();
};

}  // namespace aps
