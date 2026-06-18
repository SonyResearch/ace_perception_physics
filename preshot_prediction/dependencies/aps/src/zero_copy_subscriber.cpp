// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "aps/zero_copy_subscriber.hpp"

#include <memory>

#include "ace_loggers/ace_loggers.hpp"

namespace py = pybind11;

namespace aps {

ZeroCopySubscriber::ZeroCopySubscriber(const rclcpp::NodeOptions& options)
  : rclcpp::Node("aps_zero_copy_subscriber", options) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }
  init_ = false;

  if (Initialize()) {
    LOG(INFO) << "Initialized APS zero copy subscriber.\n";
  } else {
    LOG(ERROR) << "Failed to initialize APS zero copy subscriber.\n";
    return;
  }
}

ZeroCopySubscriber::~ZeroCopySubscriber() { LOG(INFO) << "Stopping APS zero copy subscriber.\n"; }

int ZeroCopySubscriber::AddSubscription(const std::string topic) {
  if (!init_) {
    LOG(ERROR) << "Zero copy subscriber is not initialized.\n";
    return -1;
  }

  if (num_subs_ >= max_subscriptions_) {
    LOG(ERROR) << "Zero copy subscriber reached maxmimum number of subscribers.\n";
    return -1;
  }

  int handle = num_subs_++;

  // Add new field to temporarily keep pointer to messages
  latest_msgs_.emplace_back(nullptr);
  locked_msgs_.emplace_back(nullptr);

  // Callback for temporarily storing pointers
  auto cb = [&, handle](ace_interfaces::msg::ImageData::SharedPtr message) {
    std::lock_guard<std::mutex> lck(m_msgs_[handle]);
    latest_msgs_[handle] = message;
  };
  subs_.push_back(this->create_subscription<ace_interfaces::msg::ImageData>(topic, 1, cb));

  return handle;
}

bool ZeroCopySubscriber::GetLatestMessage(int handle, ace_interfaces::msg::ImageData::SharedPtr* message) {
  if (!init_ || handle >= num_subs_ || locked_msgs_[handle] != nullptr) {
    return false;
  }

  std::lock_guard<std::mutex> lck(m_msgs_[handle]);
  if (latest_msgs_[handle] == nullptr) {
    return false;
  }
  locked_msgs_[handle] = latest_msgs_[handle];
  *message = locked_msgs_[handle];

  return true;
}

bool ZeroCopySubscriber::ReleaseLatestMessage(int handle) {
  if (!init_) {
    LOG(ERROR) << "Zero copy subscriber is not initialized.\n";
    return false;
  }

  locked_msgs_[handle] = nullptr;

  return true;
}

bool ZeroCopySubscriber::Initialize() {
  num_subs_ = 0;
  m_msgs_ = std::make_unique<std::mutex[]>(max_subscriptions_);

  init_ = true;

  return true;
}

ZeroCopySubscriberPy::ZeroCopySubscriberPy() {
  if (!google::IsGoogleLoggingInitialized()) {
    std::string name = "aps_zero_copy_subscriber_py";
    google::InitGoogleLogging(name.c_str());
  }
  init_ = false;

  // initialize ZeroCopySubscriberPy
  zero_copy_subscriber_ = nullptr;
  if (Initialize()) {
    LOG(INFO) << "Initialized zero copy subscriber.\n";
  } else {
    LOG(ERROR) << "Failed to initialize zero copy subscriber.\n";
    return;
  }
}

int ZeroCopySubscriberPy::AddSubscription(const std::string topic) {
  if (!init_) {
    LOG(ERROR) << "Zero copy subscriber is not initialized.\n";
    return -1;
  }

  return zero_copy_subscriber_->AddSubscription(topic);
}

std::tuple<bool, uint64_t, std::string, py::array> ZeroCopySubscriberPy::GetLatestMessage(int handle) {
  if (!init_) {
    LOG(ERROR) << "Zero copy subscriber is not initialized.\n";
    return std::make_tuple(false, 0, "", py::array());
  }

  ace_interfaces::msg::ImageData::SharedPtr message;
  if (!zero_copy_subscriber_->GetLatestMessage(handle, &message)) {
    return std::make_tuple(false, 0, "", py::array());
  }

  py::str dummy_data_owner;
  void* data_ptr =
    reinterpret_cast<void*>(message->data.data() + message->step * message->offset_y + message->offset_x);
  return std::make_tuple(true, message->header.sequence_number,
                         std::string(reinterpret_cast<char*>(&(message->encoding[0]))),
                         py::array(py::dtype::of<uint8_t>(), {message->height, message->width},
                                   {static_cast<size_t>(message->step), sizeof(uint8_t)}, data_ptr, dummy_data_owner));
}

bool ZeroCopySubscriberPy::ReleaseLatestMessage(int handle) {
  if (!init_) {
    LOG(ERROR) << "Zero copy subscriber is not initialized.\n";
    return false;
  }

  return zero_copy_subscriber_->ReleaseLatestMessage(handle);
}

bool ZeroCopySubscriberPy::Initialize() {
  thread_ = std::thread(std::bind(&ZeroCopySubscriberPy::WorkerThread, this));

  // Wait for node to be initialized
  int counter = 0;
  while (zero_copy_subscriber_ == nullptr || !zero_copy_subscriber_->IsInitialized()) {
    usleep(100000);  // sleep for 100ms

    if (++counter > 10) {
      return false;
    }
  }

  init_ = true;

  return true;
}

// thread
void ZeroCopySubscriberPy::WorkerThread() {
  setvbuf(stdout, nullptr, _IONBF, BUFSIZ);
  rclcpp::init(0, nullptr);

  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(false);
  options.automatically_declare_parameters_from_overrides(false);
  options.start_parameter_event_publisher(false);
  options.start_parameter_services(false);
  options.use_intra_process_comms(true);
  zero_copy_subscriber_ = std::make_shared<ZeroCopySubscriber>(options);
  rclcpp::spin(zero_copy_subscriber_);
  rclcpp::shutdown();
}

}  // namespace aps
