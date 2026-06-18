// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "aps/single_camera.hpp"

#include <fstream>

#include "ace_loggers/ace_loggers.hpp"

namespace aps {

SingleCamera::SingleCamera(const rclcpp::NodeOptions& options)
  : rclcpp::Node("single_camera", options),
    system_ptr_(Spinnaker::System::GetInstance()),
    serial_number_(declare_parameter<std::string>("serial_number")) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }

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

  // Initialize camera.
  if (Initialize()) {
    LOG(INFO) << "Initialized single camera " << serial_number_ << ".\n";
  } else {
    LOG(ERROR) << "Failed to initialize single camera " << serial_number_ << ".\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }

  // Add callbacks.
  std::function<void(Spinnaker::ImagePtr&)> cb = [&](Spinnaker::ImagePtr& image_ptr) -> void {
    this->ImageCallback(image_ptr);
  };
  if (!cam_->RegisterImageCallback(cb)) {
    LOG(ERROR) << "Failed to register image callback for " << serial_number_ << std::endl;
    throw rclcpp::exceptions::InvalidNodeError();
  }

  // Start acquisition
  if (StartAcquisition()) {
    LOG(INFO) << "Started acquisition of camera " << serial_number_ << ".\n";
  } else {
    LOG(ERROR) << "Failed to start acquisition of camera " << serial_number_ << ".\n";
    throw rclcpp::exceptions::InvalidNodeError();
  }
}

SingleCamera::~SingleCamera() {
  if (StopAcquisition()) {
    LOG(INFO) << "Stopped acquisition of camera " << serial_number_ << ".\n";
  } else {
    LOG(ERROR) << "Failed to stop acquisition of camera " << serial_number_ << ".\n";
  }

  // Camera need to be de-initialized before spinner system is released.
  cam_.reset();
  system_ptr_->ReleaseInstance();
}

bool SingleCamera::Initialize() {
  init_ = false;

  // Initialize camera
  cam_ = std::make_shared<BlackflySCamera>(serial_number_);
  if (!cam_->Initialize(std::nullopt)) {
    return false;
  }

  if (!cam_->ApplyParameters(camera_params_, true)) {
    return false;
  }

  // Set up acquisition control to run single camera independently.
  if (!cam_->SetTriggerSelector(Spinnaker::TriggerSelector_FrameStart)) {
    return false;
  }
  if (!cam_->SetTriggerMode(Spinnaker::TriggerMode_Off)) {
    return false;
  }
  if (!cam_->SetTriggerOverlap(Spinnaker::TriggerOverlap_Off)) {
    return false;
  }
  if (!cam_->SetTriggerSource(Spinnaker::TriggerSource_Software)) {
    return false;
  }

  std::string topic = std::string(this->get_namespace()) + "/aps" + serial_number_ + "/image";
  rclcpp::QoS qos(rclcpp::KeepLast(camera_params_->manual_stream_buffer_count - 1));
  pub_ = this->create_publisher<ace_interfaces::msg::ImageData>(topic, qos);
  // Track device error event
  if (!cam_->SetEventSelector(Spinnaker::EventSelector_Error)) {
    return false;
  }
  if (!cam_->SetEventNotification(Spinnaker::EventNotification_On)) {
    return false;
  }

  // Obtain exact framerate.
  frame_rate_ = cam_->GetAcquisitionFrameRate();
  LOG(INFO) << "Resulting frame rate is " << frame_rate_ << ".\n";
  sequence_number_to_ns_ = 1e9 / frame_rate_;

  // Add device callback
  std::function<void(const std::string&)> device_callback = [&](const std::string& event_name) -> void {
    this->DeviceCallback(event_name);
  };
  if (!cam_->RegisterDeviceCallback(device_callback)) {
    LOG(ERROR) << "Failed to register device event callback for " << serial_number_ << std::endl;
    throw rclcpp::exceptions::InvalidNodeError();
  }

  // Add image callback
  std::function<void(Spinnaker::ImagePtr&)> image_callback = [&](Spinnaker::ImagePtr& image_ptr) -> void {
    this->ImageCallback(image_ptr);
  };
  if (!cam_->RegisterImageCallback(image_callback)) {
    LOG(ERROR) << "Failed to register image callback for " << serial_number_ << std::endl;
    throw rclcpp::exceptions::InvalidNodeError();
  }

  init_ = true;

  return true;
}

bool SingleCamera::StartAcquisition() {
  if (!init_) {
    return false;
  }

  return cam_->StartAcquisition();
}

bool SingleCamera::StopAcquisition() {
  if (!init_) {
    return false;
  }

  return cam_->StopAcquisition();
}

void SingleCamera::DeviceArrivalCallback(const std::string& serial_number) {
  LOG(INFO) << "Camera " << serial_number << " got connected" << std::endl;
  if (serial_number != serial_number_) {
    return;
  }

  LOG(WARNING) << "Please reinitialize the system" << std::endl;
}

void SingleCamera::DeviceRemovalCallback(const std::string& serial_number) {
  if (serial_number == serial_number_) {
    LOG(ERROR) << "Camera " << serial_number << " got disconnected" << std::endl;
  } else {
    LOG(WARNING) << "Camera " << serial_number << " got disconnected" << std::endl;
  }
}

void SingleCamera::DeviceCallback(const std::string& event_name) {
  LOG(WARNING) << "Camera " << serial_number_ << " had " << event_name << " event" << std::endl;
}

void SingleCamera::ImageCallback(Spinnaker::ImagePtr& image_ptr) {
  auto loaned_msg = pub_->borrow_loaned_message();
  auto& msg = loaned_msg.get();

  auto curr_seq_num = image_ptr->GetFrameID();
  msg.header.stamp = rclcpp::Time(static_cast<int64_t>(static_cast<double>(curr_seq_num) * sequence_number_to_ns_));
  msg.header.sequence_number = curr_seq_num;
  msg.frame_rate = frame_rate_;
  msg.encoding = std::array<unsigned char, 20>{"bayer_rggb8"};

  msg.height = static_cast<uint16_t>(image_ptr->GetHeight());
  msg.width = static_cast<uint16_t>(image_ptr->GetWidth());
  msg.offset_x = static_cast<uint16_t>(image_ptr->GetXOffset());
  msg.offset_y = static_cast<uint16_t>(image_ptr->GetYOffset());
  msg.step = static_cast<uint16_t>(image_ptr->GetStride());

  std::memcpy(msg.data.data(), image_ptr->GetData(), msg.height * msg.step);

  pub_->publish(std::move(loaned_msg));
}

}  // namespace aps
