// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "aps/recorder.hpp"

#include <unistd.h>

#include <filesystem>
#include <ios>
#include <sstream>

#include "ace_interfaces/msg/file_image.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "cuda_common/cuda_common.hpp"

namespace aps {

Recorder::Recorder(const rclcpp::NodeOptions& options) : rclcpp::Node("recorder", options) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }
  init_ = false;

  const std::string& params_path = declare_parameter<std::string>("params_path");
  const std::string& recording_type = declare_parameter<std::string>("recording_type");
  const std::string& name = declare_parameter<std::string>("name");
  const std::string& topics = declare_parameter<std::string>("topics");
  const std::string& topic_filter = declare_parameter<std::string>("topic_filter");
  const bool& ignore_gpu_checks = declare_parameter<bool>("ignore_gpu_checks");
  const std::string& record_path = declare_parameter<std::string>("record_path");
  const bool& publish_events = declare_parameter<bool>("publish_events");
  int sampling = static_cast<int>(declare_parameter<int>("sampling"));
  int quality = static_cast<int>(declare_parameter<int>("quality"));
  if (ignore_gpu_checks) {
    LOG(WARNING) << "Ignoring GPU checks.\033[0m\n";
  }

  // Initialize recorder.
  if (Initialize(params_path, recording_type, name, topics, topic_filter, ignore_gpu_checks, record_path, sampling,
                 quality, publish_events)) {
    LOG(INFO) << "Initialized recorder.\n";
  } else {
    LOG(ERROR) << "Failed to initialize recorder.\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }
}

Recorder::~Recorder() { LOG(INFO) << "Stopping recorder.\n"; }

bool Recorder::Initialize(const std::string& params_path, const std::string& recording_type, const std::string& name,
                          const std::string& topics, const std::string& topic_filter, const bool& ignore_gpu_checks,
                          const std::string& record_path, int sampling, int quality, bool publish_events) {
  if (!params_.Initialize(params_path)) {
    return false;
  }
  params_.PrintParameters();

  params_.sampling = sampling;
  params_.quality = quality;
  params_.publish_events = publish_events;

  // Create folder in /var/tmp/recordings with timestamp
  std::time_t t = std::time(nullptr);
  std::tm* now = std::localtime(&t);
  std::string str_path;

  if (record_path.empty()) {
    std::stringstream path;

    path << "/var/tmp/recordings/" << now->tm_year + 1900         //
         << std::setfill('0') << std::setw(2) << now->tm_mon + 1  //
         << std::setfill('0') << std::setw(2) << now->tm_mday     //
         << "_"                                                   //
         << std::setfill('0') << std::setw(2) << now->tm_hour     //
         << std::setfill('0') << std::setw(2) << now->tm_min      //
         << std::setfill('0') << std::setw(2) << now->tm_sec;
    str_path = path.str();
    if (!name.empty()) {
      str_path += "_" + name;
    }
    str_path += "/aps/";
  } else {
    str_path = record_path;
  }

  std::vector<std::string> topic_vec;
  if (topics == "all") {
    for (int trial = 0; trial < 10; trial++) {  // Some trials, before giving up
      // Wait for topics to be discovered.
      usleep(500000);
      auto topics_ros = this->get_topic_names_and_types();
      for (const auto& topic_ros : topics_ros) {  // Iterate over all topics.
        if (topic_ros.second[0] == "ace_interfaces/msg/ImageData" &&
            topic_ros.first.find(topic_filter) != std::string::npos) {
          topic_vec.push_back(topic_ros.first);
        }
      }
      if (!topic_vec.empty()) {
        LOG(INFO) << "Found " << topic_vec.size() << " topics:";
        for (const auto& topic : topic_vec) {
          LOG(INFO) << " - " << topic;
        }
        break;
      }
      LOG(INFO) << "No topics found to record! Retrying...";
    }
  } else {
    std::stringstream ss(topics);
    while (ss.good()) {
      std::string substr;
      std::getline(ss, substr, ',');
      topic_vec.push_back(substr);
    }
  }

  if (topic_vec.empty()) {
    return false;
  }

  for (size_t topic_index = 0; topic_index < topic_vec.size(); ++topic_index) {
    const auto& topic = topic_vec[topic_index];
    // Parse topic to find camera name.
    int cam_name_end = static_cast<int>(topic.find("/image"));
    int cam_name_start = static_cast<int>(topic.substr(0, cam_name_end).rfind('/'));
    if (cam_name_end - cam_name_start - 1 > 0) {
      // Create an empty folder for each camera.
      std::string cam_name = topic.substr(cam_name_start + 1, cam_name_end - cam_name_start - 1);
      std::string path_cam = str_path;
      path_cam += "/";
      path_cam += cam_name;
      std::filesystem::remove_all(path_cam);
      std::filesystem::create_directories(path_cam);

      pubs_.push_back(this->create_publisher<ace_interfaces::msg::FileImage>(topic + "/jpeg", 1));
      if (recording_type == "raw") {
        size_t index = subs_.size();
        auto cb = [&, path_cam, cam_name, index](ace_interfaces::msg::ImageData::SharedPtr message) {
          std::string file_path = aps::Recorder::ImageCallbackRecordRaw(message, path_cam, params_);
          if (params_.publish_events && !file_path.empty()) {
            std::stringstream path;
            path << cam_name << "/" << file_path;
            PublishImageEvent(message, path.str(), static_cast<int>(index));
          }
        };

        subs_.push_back(this->create_subscription<ace_interfaces::msg::ImageData>(topic, 1, cb));
      } else {
        std::string type_ending = "jpg";
        if (recording_type != "jpg" && recording_type != "png") {
          LOG(WARNING) << "Unknown recording type " << recording_type << ", using default type jpg.\n";
        }

        JpegEncoder::SharedPtr jpeg_encoder;

        const int& cuda_device_id = cuda_common::GetCudaDeviceID(
          params_.cuda_device_uuids[static_cast<int>(topic_index % params_.cuda_device_uuids.size())],
          ignore_gpu_checks);
        if (cuda_device_id < 0) {
          return false;
        }

        if (recording_type == "png") {
          // De-bayer only.
          jpeg_encoder = std::make_shared<JpegEncoder>(cuda_device_id, true, params_.quality);
          type_ending = "png";
        } else {
          // De-bayer and jpeg encoding.
          jpeg_encoder = std::make_shared<JpegEncoder>(cuda_device_id, false, params_.quality);
        }

        jpeg_encoders_.push_back(jpeg_encoder);
        size_t index = subs_.size();
        auto cb = [&, path_cam, cam_name, jpeg_encoder, type_ending,
                   index](ace_interfaces::msg::ImageData::SharedPtr message) {
          std::string file_path =
            aps::Recorder::ImageCallbackRecordJpeg(message, path_cam, jpeg_encoder, type_ending, params_);
          if (params_.publish_events && !file_path.empty()) {
            std::stringstream path;
            path << cam_name << "/" << file_path;
            PublishImageEvent(message, path.str(), static_cast<int>(index));
          }
        };

        subs_.push_back(this->create_subscription<ace_interfaces::msg::ImageData>(topic, 1, cb));
      }
    } else {
      LOG(WARNING) << "Could not recognize camera name from topic " << topic << ", skipping topic.\n";
    }
  }

  init_ = true;

  return true;
}

std::string Recorder::ImageCallbackRecordRaw(ace_interfaces::msg::ImageData::SharedPtr message, const std::string& path,
                                             RecorderParameters& params) {
  if (message->header.sequence_number % params.sampling != 0) {
    return "";
  }
  // Convert image to OpenCV.
  void* data_ptr =
    reinterpret_cast<void*>(message->data.data() + message->step * message->offset_y + message->offset_x);
  cv::Mat image = cv::Mat(message->height, message->width, CV_8UC1, data_ptr, message->step);

  // Compute filename.
  std::stringstream filename;
  filename << "img" << std::setfill('0') << std::setw(8) << message->header.sequence_number << ".png";
  std::string str_filename(filename.str());

  // Store png image.
  cv::imwrite(path + "/" + str_filename, image);
  return str_filename;
}

std::string Recorder::ImageCallbackRecordJpeg(const ace_interfaces::msg::ImageData::ConstSharedPtr& message,
                                              const std::string& path, aps::JpegEncoder::SharedPtr jpeg_encoder,
                                              const std::string& type_ending, RecorderParameters& params) {
  if (message->header.sequence_number % params.sampling != 0) {
    return "";
  }
  // Compute filename.
  std::stringstream filename;
  filename << "img" << std::setfill('0') << std::setw(8) << message->header.sequence_number << "." << type_ending;
  std::string str_filename(filename.str());

  // De-bayer image, encode as jpeg (default) and store image.
  jpeg_encoder->SaveImage(message, path + "/" + str_filename);
  return str_filename;
}

void Recorder::PublishImageEvent(ace_interfaces::msg::ImageData::SharedPtr message, const std::string& path,
                                 int index) {
  ace_interfaces::msg::FileImage msg;
  msg.header = message->header;
  msg.path = path;

  pubs_[index]->publish(msg);
}

}  // namespace aps
