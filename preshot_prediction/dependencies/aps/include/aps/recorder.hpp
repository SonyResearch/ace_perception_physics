// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <memory>
#include <string>

#include "ace_interfaces/msg/file_image.hpp"
#include "ace_interfaces/msg/image_data.hpp"
#include "aps/jpeg_encoder.hpp"
#include "aps/recorder_parameters.hpp"
#include "rclcpp/rclcpp.hpp"

namespace aps {

class Recorder : public rclcpp::Node {
 public:
  explicit Recorder(const rclcpp::NodeOptions& options);
  ~Recorder() override;

 private:
  // general
  bool init_;
  bool Initialize(const std::string& params_path, const std::string& recording_type, const std::string& name,
                  const std::string& topics, const std::string& topic_filter, const bool& ignore_gpu_checks,
                  const std::string& record_path, int sampling, int quality, bool publish_events);

  std::vector<aps::JpegEncoder::SharedPtr> jpeg_encoders_;
  std::vector<rclcpp::Subscription<ace_interfaces::msg::ImageData>::SharedPtr> subs_;
  std::vector<rclcpp::Publisher<ace_interfaces::msg::FileImage>::SharedPtr> pubs_;

  static std::string ImageCallbackRecordRaw(ace_interfaces::msg::ImageData::SharedPtr message, const std::string& path,
                                            RecorderParameters& params);
  static std::string ImageCallbackRecordJpeg(const ace_interfaces::msg::ImageData::ConstSharedPtr& message,
                                             const std::string& path, aps::JpegEncoder::SharedPtr jpeg_encoder,
                                             const std::string& type_ending, RecorderParameters& params);

  void PublishImageEvent(ace_interfaces::msg::ImageData::SharedPtr message, const std::string& path, int index);

  RecorderParameters params_;
};

}  // namespace aps
