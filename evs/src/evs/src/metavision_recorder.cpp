// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "evs/metavision_recorder.hpp"

// #include "ace_loggers/ace_loggers.hpp"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_decoder.h"
#include "metavision/hal/facilities/i_erc_module.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_trigger_in.h"
#include "metavision/hal/facilities/i_trigger_out.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"

namespace evs {

MetavisionRecorder::MetavisionRecorder(const rclcpp::NodeOptions& options)
  : rclcpp::Node("metavision_recorder", options) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }
  init_ = false;
  canceled_.store(false);

  const std::string& multi_camera_params_path = declare_parameter<std::string>("multi_camera_params_path");
  const std::string& camera_params_path = declare_parameter<std::string>("camera_params_path");
  const std::string& name = declare_parameter<std::string>("name");

  // Initialize recorder.
  if (Initialize(multi_camera_params_path, camera_params_path, name)) {
    LOG(INFO) << "Initialized metavision recorder.\n";
  } else {
    LOG(ERROR) << "Failed to initialize metavision recorder.\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }

  // Start recording.
  LOG(INFO) << "Starting metavision recording.\n";
  StartRecording();
}

MetavisionRecorder::~MetavisionRecorder() {
  // Stop recording.
  LOG(INFO) << "Stopping metavision recording.\n";
  StopRecording();
}

bool MetavisionRecorder::Initialize(const std::string& multi_camera_params_path, const std::string& camera_params_path,
                                    const std::string& name) {
  // Read parameters.
  multi_camera_params_ = std::make_shared<MultiCameraParameters>();
  if (!multi_camera_params_->Initialize(multi_camera_params_path)) {
    return false;
  }
  multi_camera_params_->PrintParameters();

  camera_params_ = std::make_shared<CameraParameters>();
  if (!camera_params_->Initialize(camera_params_path)) {
    return false;
  }
  camera_params_->PrintParameters();

  // Create folder in /tmp/recordings with timestamp.
  std::time_t t = std::time(nullptr);
  std::tm* now = std::localtime(&t);
  char path[250];
  std::sprintf(path, "/tmp/recordings/%04d%02d%02d_%02d%02d%02d", now->tm_year + 1900, now->tm_mon + 1, now->tm_mday,
               now->tm_hour, now->tm_min, now->tm_sec);
  recording_path_ = std::string(path);
  if (!name.empty()) {
    recording_path_ += "_" + name;
  }
  recording_path_ += "/evs/";
  std::filesystem::remove_all(recording_path_);
  std::filesystem::create_directories(recording_path_);

  // Initialize slave cameras.
  for (const auto& slave_serial_number : multi_camera_params_->slave_serial_numbers) {
    if (!InitializeCamera(slave_serial_number, false)) {
      LOG(ERROR) << "Failed to initialize slave camera " << slave_serial_number << ".\n";
      return false;
    }
  }

  // Initialize master camera.
  usleep(100000);  // Sleep to ensure all slave cameras are ready, then start master camera.
  if (!InitializeCamera(multi_camera_params_->master_serial_number, true)) {
    LOG(ERROR) << "Failed to initialize master camera " << multi_camera_params_->master_serial_number << ".\n";
    return false;
  }

  return true;
}

bool MetavisionRecorder::InitializeCamera(const std::string& serial_number, bool is_master) {
  try {
    cams_[serial_number] = Metavision::DeviceDiscovery::open(serial_number);
  } catch (Metavision::HalException& e) {
    LOG(ERROR) << e.what() << std::endl;
    return false;
  }

  auto* event_stream_ptr = cams_[serial_number]->get_facility<Metavision::I_EventsStream>();
  auto* cam_sync = cams_[serial_number]->get_facility<Metavision::I_CameraSynchronization>();
  auto* biases_ptr = cams_[serial_number]->get_facility<Metavision::I_LL_Biases>();

  // Apply camera parameters.
  biases_ptr->set("bias_diff", camera_params_->bias_diff);
  biases_ptr->set("bias_diff_off", camera_params_->bias_diff_off);
  biases_ptr->set("bias_diff_on", camera_params_->bias_diff_on);
  biases_ptr->set("bias_fo", camera_params_->bias_fo_p);
  biases_ptr->set("bias_hpf", camera_params_->bias_hpf);
  biases_ptr->set("bias_refr", camera_params_->bias_refr);

  // Set master/slave mode.
  if (is_master) {
    if (cam_sync->set_mode_master()) {
      std::cout << "Set mode Master successful. Remember to start the slave first." << std::endl;
    } else {
      std::cerr << "Could not set Master mode. Master/slave might not be supported by your camera" << std::endl;
      return false;
    }
  } else {
    cam_sync->set_mode_slave();
  }

  // Enable external sync/trigger events.
  if (is_master) {
    auto* trigger_in_ptr = cams_[serial_number]->get_facility<Metavision::I_TriggerIn>();
    trigger_in_ptr->enable(Metavision::I_TriggerIn::Channel::Main);

    auto* trigger_in_decoder_ptr =
      cams_[serial_number]->get_facility<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
    if (trigger_in_decoder_ptr) {
      // Add dummy decoder to record external trigger events.
      trigger_in_decoder_ptr->add_event_buffer_callback(
        [](const Metavision::EventExtTrigger*, const Metavision::EventExtTrigger*) {});
    } else {
      LOG(ERROR) << "Failed to obtain trigger_in decoder.\n";
      return false;
    }
  }

  // Enable recording.
  if (event_stream_ptr) {
    event_stream_ptr->log_raw_data(recording_path_ + "evs" + serial_number + ".raw");
  } else {
    LOG(ERROR) << "Failed to obtain event stream.\n";
    return false;
  }

  // Event buffer needs to be polled regularly.
  cam_threads_.emplace_back([this, event_stream_ptr]() {
    int64_t num_bytes;
    while (!canceled_.load()) {
      event_stream_ptr->poll_buffer();
      event_stream_ptr->get_latest_raw_data(num_bytes);
    }
  });

  return true;
}

void MetavisionRecorder::StartRecording() {
  // Start all slave cameras.
  for (const auto& serial_number : multi_camera_params_->slave_serial_numbers) {
    auto* event_stream_ptr = cams_[serial_number]->get_facility<Metavision::I_EventsStream>();

    event_stream_ptr->start();
  }

  // Start master camera.
  {
    usleep(100000);
    auto* event_stream_ptr =
      cams_[multi_camera_params_->master_serial_number]->get_facility<Metavision::I_EventsStream>();

    event_stream_ptr->start();
  }
}

void MetavisionRecorder::StopRecording() {
  // Stop polling threads.
  canceled_.store(true);
  for (auto& cam_thread : cam_threads_) {
    if (cam_thread.joinable()) {
      cam_thread.join();
    }
  }

  // Stop master camera.
  {
    auto* event_stream_ptr =
      cams_[multi_camera_params_->master_serial_number]->get_facility<Metavision::I_EventsStream>();

    event_stream_ptr->stop();
  }

  // Stop all slave cameras.
  for (const auto& serial_number : multi_camera_params_->slave_serial_numbers) {
    auto* event_stream_ptr = cams_[serial_number]->get_facility<Metavision::I_EventsStream>();

    event_stream_ptr->stop();
  }
}

}  // namespace evs
