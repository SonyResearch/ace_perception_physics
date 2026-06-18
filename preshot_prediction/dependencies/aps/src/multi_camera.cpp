// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "aps/multi_camera.hpp"

#include <unistd.h>

#include <ace_rt_profiles/profile_manager.hpp>
#include <fstream>

#include "ace_loggers/ace_loggers.hpp"

namespace aps {

MultiCamera::MultiCamera(const rclcpp::NodeOptions& options)
  : rclcpp::Node("multi_camera", options), system_ptr_(Spinnaker::System::GetInstance()) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }

  // Initialize parameters
  const std::string& params_path = declare_parameter<std::string>("params_path");
  params_ = std::make_shared<MultiCameraParameters>();
  if (!params_->Initialize(params_path)) {
    LOG(ERROR) << "Failed to initialize multi camera system.\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }
  params_->PrintParameters();

  // Initialize camera parameters
  const std::string& camera_params_path = declare_parameter<std::string>("camera_params_path");
  camera_params_ = std::make_shared<CameraParameters>();
  if (!camera_params_->Initialize(camera_params_path)) {
    LOG(ERROR) << "Failed to load camera parameters " << std::endl;
    throw rclcpp::exceptions::InvalidNodeError();
  }
  camera_params_->PrintParameters();

  // Check USBFS  memory size
  std::ifstream usbfs_memory_file{std::string(kUSBFSMemorySizeFile)};
  if (!usbfs_memory_file) {
    LOG(ERROR) << "Failed to check USBFS memory size.\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }
  std::string usbfs_memory_size_str;
  std::getline(usbfs_memory_file, usbfs_memory_size_str);
  const int usbfs_memory_size = std::stoi(usbfs_memory_size_str);
  if (usbfs_memory_size < kMinUSBFSMemorySize) {
    LOG(ERROR) << "USBFS memory size is too small (" << usbfs_memory_size << "MB). Increase the size to a minimum of "
               << kMinUSBFSMemorySize << "MB.\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }
  LOG(INFO) << "UBSFS memory size is " << usbfs_memory_size << "MB.\n";

  Spinnaker::InterfaceList interfaces = system_ptr_->GetInterfaces();
  for (size_t i = 0; i < interfaces.GetSize(); i++) {
    Spinnaker::InterfacePtr interface_ptr = interfaces[i];
    interface_ptr->RegisterEventHandler(interface_event_handler_);
  }

  // Add arrival callback
  std::function<void(const std::string&)> device_arrival_callback = [&](const std::string& serial_number) -> void {
    this->DeviceArrivalCallback(serial_number);
  };
  if (!interface_event_handler_.RegisterDeviceArrivalCallback(device_arrival_callback)) {
    LOG(ERROR) << "Failed to register device arrival callback" << std::endl;
    throw rclcpp::exceptions::InvalidNodeError();
  }

  // Add removal callback
  std::function<void(const std::string&)> device_removal_callback = [&](const std::string& serial_number) -> void {
    this->DeviceRemovalCallback(serial_number);
  };
  if (!interface_event_handler_.RegisterDeviceRemovalCallback(device_removal_callback)) {
    LOG(ERROR) << "Failed to register device remocal callback" << std::endl;
    throw rclcpp::exceptions::InvalidNodeError();
  }

  // Initialize multi camera system.
  if (Initialize()) {
    LOG(INFO) << "Initialized multi camera system.\n";
  } else {
    LOG(ERROR) << "Failed to initialize multi camera system.\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }

  // Add callbacks.
  rclcpp::QoS qos(rclcpp::KeepLast(camera_params_->manual_stream_buffer_count - 1));
  for (size_t camera_idx = 0; camera_idx < cams_.size(); ++camera_idx) {
    auto& serial_number = params_->serial_numbers[camera_idx];
    auto& cam_ptr = cams_[serial_number];
    std::string topic = std::string(this->get_namespace()) + "/aps" + serial_number + "/image";
    pubs_.push_back(this->create_publisher<ace_interfaces::msg::ImageData>(topic, qos));
    pubs_stats_.push_back(this->create_publisher<ace_interfaces::msg::Statistics>(topic + "/info", qos));
    stats_.emplace_back(ace_interfaces::msg::Statistics(), clock::now());

    std::function<void(const std::string&)> device_callback = [&, camera_idx](const std::string& event_name) -> void {
      this->DeviceCallback(camera_idx, event_name);
    };
    if (!cam_ptr->RegisterDeviceCallback(device_callback)) {
      LOG(ERROR) << "Failed to register device event callback for " << serial_number << std::endl;
      throw rclcpp::exceptions::InvalidNodeError();
    }

    std::function<void(Spinnaker::ImagePtr&)> cb = [&, camera_idx](Spinnaker::ImagePtr& image_ptr) -> void {
      this->ImageCallback(camera_idx, image_ptr);
    };
    if (!cam_ptr->RegisterImageCallback(cb)) {
      LOG(ERROR) << "Failed to register image callback for \"aps" << serial_number << "\"" << std::endl;
      throw rclcpp::exceptions::InvalidNodeError();
    }
  }

  {
    std::string topic = std::string(this->get_namespace()) + "/aps_sync";
    sync_pub_ = this->create_publisher<std_msgs::msg::Float32>(topic, qos);
  }

  // Start acquisition.
  if (!StartAcquisition()) {
    throw rclcpp::exceptions::InvalidNodeError();
  }
  if (!params_->first_as_master) {
    // EtherCAT subscriber
    auto ecat_cb = [&](std_msgs::msg::Float32::SharedPtr message) {
      auto time_now = clock::now();
      LOG(INFO) << "EtherCAT countdown value: " << message->data << " seconds" << std::endl;

      if (ecat_restart_time_ < time_now) {
        LOG(INFO) << "A new restart command arrived!" << std::endl;
        auto dt = std::chrono::duration<double>(message->data + 0.1);
        ecat_restart_time_ = time_now + std::chrono::duration_cast<clock::duration>(dt);
        if (!StopAcquisition()) {
          throw rclcpp::exceptions::InvalidNodeError();
        }
        rclcpp::sleep_for(std::chrono::milliseconds(200));
        if (!StartAcquisition()) {
          throw rclcpp::exceptions::InvalidNodeError();
        }
      }
    };
    ecat_restart_time_ = clock::now();
    ecat_sync_sub_ = this->create_subscription<std_msgs::msg::Float32>("/ecat_sync", 1, ecat_cb);

    LOG(ERROR) << "Multi-camera system is in slave mode!\n";
    LOG(ERROR) << "Please run the external trigger from now on ...\n";
  }
}

MultiCamera::~MultiCamera() {
  StopAcquisition();

  // Cameras need to be de-initialized before spinner system is released.
  for (auto& cam : cams_) {
    cam.second.reset();
  }

  system_ptr_->ReleaseInstance();
}

bool MultiCamera::Initialize() {
  init_ = false;

  const std::vector<int> affinity = ace_rt_profiles::ProfileManager::GetInstance().GetAffinity(params_->rt_profile);
  if (affinity.size() < params_->serial_numbers.size()) {
    LOG(WARNING)
      << "CPU Affinity list size is smaller than camera count, this will lead to a potential performance drop!";
  }
  // Initialize cameras.
  int index = 0;
  for (auto it = params_->serial_numbers.begin(); it < params_->serial_numbers.end(); ++it) {
    auto affinity_index = index % affinity.size();
    ace_rt_profiles::RTSubProfile rt_subprofile = {.rt_profile = params_->rt_profile, .affinity_index = affinity_index};
    if (!InitializeCamera(*it, it == params_->serial_numbers.begin() && params_->first_as_master, rt_subprofile)) {
      LOG(ERROR) << "Failed to initialize camera " << *it << ".\n";
      return false;
    }
    // In initial stage, even master is handled as a slave
    if (!SetAcqusitionParameters(cams_[*it], it == params_->serial_numbers.begin() && params_->first_as_master)) {
      LOG(ERROR) << "Failed to set the acquisition control for camera " << *it << " (inital stage).\n";
      return false;
    }
    ++index;
    rclcpp::sleep_for(std::chrono::microseconds(200));
  }

  // Set up external trigger service.
  if (params_->first_as_master && params_->software_trigger) {
    trigger_service_ = create_service<ace_interfaces::srv::Trigger>(
      "/trigger", std::bind(&MultiCamera::TriggerService, this, std::placeholders::_1, std::placeholders::_2));
    LOG(WARNING) << "Enabled external trigger service.\n";
  }

  // Obtain exact framerate.
  frame_rate_ = 0.0;
  sequence_number_to_ns_ = 0.0;
  stats_pub_rate_ = std::numeric_limits<uint64_t>::max();
  stats_print_rate_ = std::numeric_limits<uint64_t>::max();
  callback_max_time_us_ = std::numeric_limits<uint64_t>::max();
  if (!params_->software_trigger) {
    frame_rate_ = params_->first_as_master ? cams_[params_->serial_numbers[0]]->GetAcquisitionFrameRate()
                                           : camera_params_->acquisition_frame_rate;
    LOG(INFO) << "Resulting frame rate is " << frame_rate_ << ".\n";
    sequence_number_to_ns_ = 1e9 / frame_rate_;
    stats_pub_rate_ = static_cast<uint64_t>(std::ceil(0.1 * frame_rate_));
    stats_print_rate_ = static_cast<uint64_t>(10 * frame_rate_);
    callback_max_time_us_ = static_cast<uint64_t>(kCallbackTimeRatio * sequence_number_to_ns_ * 1e-3);
  }

  // Set up camera timing checks.
  if (params_->software_trigger) {
    frame_interval_ns_ = std::numeric_limits<uint64_t>::max();
  } else {
    frame_interval_ns_ = static_cast<uint64_t>(sequence_number_to_ns_);
  }
  previous_seq_nums_.resize(cams_.size(), 0);
  previous_ts_.resize(cams_.size(), 0);
  timing_check_start_seq_num_ = static_cast<uint64_t>(frame_rate_ * kTimingCheckStartPeriod);
  frames_to_skip_.resize(cams_.size(), 0);
  state_counters_.resize(cams_.size(), 0);

  init_ = true;

  return init_;
}

bool MultiCamera::InitializeCamera(const std::string& serial_number, bool is_master,
                                   const ace_rt_profiles::RTSubProfile& rt_subprofile) {
  if (cams_.find(serial_number) != cams_.end()) {
    LOG(WARNING) << "Camera " << serial_number << " is already initialized.\n";
    return false;
  }

  // initialize camera
  cams_[serial_number] = std::make_shared<BlackflySCamera>(serial_number);
  auto& cam_ptr = cams_[serial_number];
  if (!cam_ptr->Initialize(rt_subprofile)) {
    return false;
  }

  // apply camera parameters
  if (!cam_ptr->ApplyParameters(camera_params_, is_master)) {
    return false;
  }

  // Track device error event
  if (!cam_ptr->SetEventSelector(Spinnaker::EventSelector_Error)) {
    return false;
  }
  if (!cam_ptr->SetEventNotification(Spinnaker::EventNotification_On)) {
    return false;
  }

  return true;
}

bool MultiCamera::SetAcqusitionParameters(BlackflySCamera::SharedPtr& cam_ptr, bool is_master) {
  // set up digital io and acquisition control
  // digital io control
  if (!cam_ptr->SetLineSelector(Spinnaker::LineSelector_Line2)) {
    return false;
  }
  if (!cam_ptr->Set3V3Enabled(false)) {
    return false;
  }
  if (!cam_ptr->SetLineMode(is_master ? Spinnaker::LineMode_Output : Spinnaker::LineMode_Input)) {
    return false;
  }

  if (is_master && !cam_ptr->SetLineSource(Spinnaker::LineSource_ExposureActive)) {
    return false;
  }

  // acquisition control
  if (!cam_ptr->SetTriggerSelector(Spinnaker::TriggerSelector_FrameStart)) {
    return false;
  }
  const auto trigger_mode =
    (params_->software_trigger || (!is_master)) ? Spinnaker::TriggerMode_On : Spinnaker::TriggerMode_Off;
  if (!cam_ptr->SetTriggerMode(trigger_mode)) {
    return false;
  }
  if (!cam_ptr->SetTriggerOverlap(is_master ? Spinnaker::TriggerOverlap_Off : Spinnaker::TriggerOverlap_ReadOut)) {
    return false;
  }
  if (!cam_ptr->SetTriggerSource(is_master ? Spinnaker::TriggerSource_Software : Spinnaker::TriggerSource_Line2)) {
    return false;
  }
  if (!is_master && !cam_ptr->SetTriggerActivation(Spinnaker::TriggerActivation_RisingEdge)) {
    return false;
  }
  if (!is_master && !cam_ptr->SetTriggerDelay(9.0)) {
    return false;
  }

  if (!cam_ptr->SetLineSelector(Spinnaker::LineSelector_Line1)) {
    return false;
  }
  if (!cam_ptr->SetLineMode(Spinnaker::LineMode_Output)) {
    return false;
  }
  if (!cam_ptr->SetUserOutputSelector(Spinnaker::UserOutputSelector_UserOutput0)) {
    return false;
  }
  if (!cam_ptr->SetUserOutputValue(!is_master)) {
    return false;
  }  // low valid, disable all cameras and takes EtherCAT as master
  if (!cam_ptr->SetLineSource(Spinnaker::LineSource_UserOutput0)) {
    return false;
  }

  return true;
}

bool MultiCamera::StartAcquisition() {
  if (!init_) {
    LOG(ERROR) << "Please initialize the node first!\n";
    return false;
  }

  // Reset sequence IDs & timestamps
  std::fill(previous_seq_nums_.begin(), previous_seq_nums_.end(), 0);
  std::fill(previous_ts_.begin(), previous_ts_.end(), 0);

  // Start all cameras (master last).
  bool return_value = true;
  // NOLINTNEXTLINE
  for (auto it = params_->serial_numbers.rbegin(); it != params_->serial_numbers.rend(); ++it) {
    return_value &= cams_[*it]->StartAcquisition();
  }

  // if (params_->first_as_master) {
  //   // Start generating sync signals
  //   return_value &= SetAcqusitionParameters(cams_[params_->serial_numbers[0]], true);
  // }

  if (return_value) {
    LOG(INFO) << "Started acquisition of cameras.\n";
  } else {
    LOG(ERROR) << "Failed to start acquisition of cameras.\n";
  }
  return return_value;
}

bool MultiCamera::StopAcquisition() {
  if (!init_) {
    LOG(ERROR) << "Please initialize the node first!\n";
    return false;
  }

  bool return_value = true;

  if (params_->first_as_master) {
    // Stop generating sync signals
    return_value &= SetAcqusitionParameters(cams_[params_->serial_numbers[0]], false);
  }

  // Stop all cameras (master first).
  for (auto& serial_number : params_->serial_numbers) {
    return_value &= cams_[serial_number]->StopAcquisition();
  }

  if (return_value) {
    LOG(INFO) << "Stopped acquisition of cameras.\n";
  } else {
    LOG(ERROR) << "Failed to stop acquisition of cameras.\n";
  }
  return return_value;
}

bool MultiCamera::ResetDevices() {
  if (!init_) {
    LOG(ERROR) << "Please initialize the node first!\n";
    return false;
  }

  // Restart all cameras (master first).
  bool return_value = true;
  for (auto& serial_number : params_->serial_numbers) {
    return_value &= cams_[serial_number]->ResetDevice();
  }

  if (return_value) {
    LOG(INFO) << "Restarted cameras.\n";
  } else {
    LOG(ERROR) << "Failed to restart cameras.\n";
  }
  return return_value;
}

void MultiCamera::UpdateAndPrintStatistics(const size_t& camera_idx, const uint64_t& curr_seq_num,
                                           const uint64_t& curr_timestamp) {
  // Both update and statiscs blocks need to be performed with the same mutex lock
  std::lock_guard<std::mutex> lock(counter_mutex_);

  // Update current sequence number and time stamp
  previous_seq_nums_[camera_idx] = curr_seq_num;
  previous_ts_[camera_idx] = curr_timestamp;

  // Compute sync statistics
  auto count_functor = [&curr_seq_num](const uint64_t& prev_seq_num) { return prev_seq_num == curr_seq_num; };
  const int counter_idx =
    static_cast<int>(std::count_if(previous_seq_nums_.begin(), previous_seq_nums_.end(), count_functor)) - 1;
  const auto curr_counter = ++state_counters_[counter_idx];

  if (counter_idx != 0) {
    // Only first counter is allowed to print/publish statistics
    return;
  }
  const auto sync_state = static_cast<float>(state_counters_[cams_.size() - 1]) / static_cast<float>(curr_counter);

  if (curr_counter % stats_pub_rate_ == 0) {
    // Publish sync statistics
    sync_msg_.data = sync_state;
    sync_pub_->publish(sync_msg_);
  }

  if (curr_counter % stats_print_rate_ == 0) {
    // Print sync statistics
    std::stringstream state_stream;
    state_stream << "Sync states: ";
    for (auto& state_counter : state_counters_) {
      state_stream << std::setw(8) << state_counter;
    }
    state_stream << std::endl;

    if (sync_state > 1.0F) {
      LOG(ERROR) << "Sync logic is broken! Is mutex locked properly?" << std::endl;
      LOG(FATAL) << state_stream.str() << std::flush;
    } else if (sync_state <= 0.8F) {
      LOG(ERROR) << state_stream.str() << std::flush;
    } else if (sync_state <= 0.9F) {
      LOG(WARNING) << state_stream.str() << std::flush;
    } else {
      LOG(INFO) << state_stream.str() << std::flush;
    }
    std::fill(state_counters_.begin(), state_counters_.end(), 0);
  }
}

void MultiCamera::DeviceArrivalCallback(const std::string& serial_number) {
  LOG(INFO) << "Camera " << serial_number << " got connected" << std::endl;
  if (cams_.find(serial_number) == cams_.end()) {
    return;
  }

  LOG(WARNING) << "Please reinitialize the system" << std::endl;
}

void MultiCamera::DeviceRemovalCallback(const std::string& serial_number) {
  std::stringstream ss;
  ss << "Camera " << serial_number << " got disconnected" << std::endl;

  if (cams_.find(serial_number) != cams_.end()) {
    LOG(ERROR) << ss.str();
  } else {
    LOG(WARNING) << ss.str();
  }
}

void MultiCamera::DeviceCallback(const size_t& camera_idx, const std::string& event_name) {
  LOG(WARNING) << "Camera " << params_->serial_numbers[camera_idx] << " had " << event_name << " event" << std::endl;
}

bool CheckSampling(size_t sample_id, float modular) {
  return static_cast<int>(static_cast<float>(sample_id) - modular * floor(static_cast<float>(sample_id) / modular)) ==
         0;
}

void MultiCamera::ImageCallback(const size_t& camera_idx, Spinnaker::ImagePtr& image_ptr) {
  auto start_time = clock::now();
  auto curr_seq_num = image_ptr->GetFrameID();
  auto curr_timestamp = image_ptr->GetTimeStamp();
  if (curr_seq_num <= previous_seq_nums_[camera_idx]) {
    return;
  }
  if (frames_to_skip_[camera_idx] > 0) {
    --frames_to_skip_[camera_idx];
    return;
  }

  if (!CheckSampling(curr_seq_num, params_->sampling)) {
    return;
  }
  if (params_->software_trigger) {
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(start_time - trigger_time_).count();
    LOG(INFO) << "Current frame for \"aps" << params_->serial_numbers[camera_idx] << "\" took " << elapsed_us
              << "us from trigger timer" << std::endl;
  }

  // Check time interval between messages
  if (!params_->software_trigger && curr_seq_num > timing_check_start_seq_num_) {
    const uint64_t expected_ts =
      previous_ts_[camera_idx] + (curr_seq_num - previous_seq_nums_[camera_idx]) * frame_interval_ns_;
    uint64_t delta_ts = expected_ts > curr_timestamp ? expected_ts - curr_timestamp : curr_timestamp - expected_ts;

    if (kTimingMaxDeviationNs < delta_ts) {
      LOG(WARNING) << "Time between images for camera \"" << params_->serial_numbers[camera_idx]
                   << "\" deviated from expected value by " << delta_ts << "ns (max: " << kTimingMaxDeviationNs
                   << "ns)\n";
    }
  }
  try {
    auto loaned_msg = pubs_[camera_idx]->borrow_loaned_message();
    if (!loaned_msg.is_valid()) {
      LOG(WARNING) << "Failed to obatain a loaned message for camera \"" << params_->serial_numbers[camera_idx];
      return;
    }
    auto& msg = loaned_msg.get();

    msg.header.stamp = rclcpp::Time(static_cast<int64_t>(static_cast<double>(curr_seq_num) * sequence_number_to_ns_));
    msg.header.sequence_number = static_cast<uint64_t>(static_cast<double>(curr_seq_num) * (1000.0 / frame_rate_));
    msg.frame_rate = frame_rate_;
    msg.encoding = std::array<unsigned char, 20>{"bayer_rggb8"};

    msg.height = static_cast<uint16_t>(image_ptr->GetHeight());
    msg.width = static_cast<uint16_t>(image_ptr->GetWidth());
    msg.offset_x = static_cast<uint16_t>(image_ptr->GetXOffset());
    msg.offset_y = static_cast<uint16_t>(image_ptr->GetYOffset());
    msg.step = static_cast<uint16_t>(image_ptr->GetStride());

    std::memcpy(msg.data.data(), image_ptr->GetData(), msg.height * msg.step);

    pubs_[camera_idx]->publish(std::move(loaned_msg));
    stats_[camera_idx].first.framerate++;

    auto now = clock::now();
    auto stats_dt = static_cast<double>(
                      std::chrono::duration_cast<std::chrono::milliseconds>(now - stats_[camera_idx].second).count()) /
                    1e3;
    if (stats_dt >= 1) {
      stats_[camera_idx].first.stamp = msg.header.stamp;
      stats_[camera_idx].first.framerate = static_cast<uint16_t>(stats_[camera_idx].first.framerate / stats_dt);
      pubs_stats_[camera_idx]->publish(stats_[camera_idx].first);
      stats_[camera_idx].first.framerate = 0;
      stats_[camera_idx].second = now;
    }

    UpdateAndPrintStatistics(camera_idx, curr_seq_num, curr_timestamp);
    auto end_time = clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    if (static_cast<uint64_t>(elapsed_us) > callback_max_time_us_) {
      frames_to_skip_[camera_idx] += static_cast<uint64_t>(elapsed_us) / callback_max_time_us_;
      LOG(WARNING) << "Current frame for \"aps" << params_->serial_numbers[camera_idx] << "\" took " << elapsed_us
                   << "us! Skipping next frame... " << std::endl;
    }
  } catch (std::exception& e) {
    LOG(WARNING) << "Exception occurred for camera \"" << params_->serial_numbers[camera_idx] << "\": " << e.what();
  }
}

void MultiCamera::TriggerService(ace_interfaces::srv::Trigger::Request::SharedPtr /*request*/,
                                 ace_interfaces::srv::Trigger::Response::SharedPtr response) {
  trigger_time_ = clock::now();
  cams_[params_->serial_numbers[0]]->GenerateSoftwareTrigger();
  response->trigger_executed = true;
}

}  // namespace aps
