// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "aps/blackfly_s_camera.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace aps {

template <typename T, typename = typename std::enable_if<std::is_enum<T>::value, T>::type>
inline std::ostream& operator<<(std::ostream& os, const T& value) {
  os << static_cast<uint64_t>(value);
  return os;
}

template <typename T, typename U>
inline bool SetValue(U& node, const T& value) {
  node.SetValue(value);
  return true;
}

template <typename T, typename U>
inline bool SetValue(Spinnaker::GenApi::IEnumerationT<U>& node, const T& value) {
  node.SetValue(static_cast<U>(value));
  return true;
}

template <typename T, typename U>
inline T GetValue(U& node) {
  return static_cast<const T>(node.GetValue());
}

template <typename U>
inline std::string GetValue(U& node) {
  return static_cast<const char*>(node.GetValue());
}

BlackflySCamera::BlackflySCamera(const std::string& serial_number)
  : cam_ptr_(CreateCameraPtr(serial_number)),
    device_(&cam_ptr_->GetTLDeviceNodeMap()),
    data_stream_(cam_ptr_ == nullptr ? nullptr : &cam_ptr_->GetTLStreamNodeMap()),
    image_event_handler_(serial_number),
    device_event_handler_(serial_number) {
  if (!google::IsGoogleLoggingInitialized()) {
    const std::string logging_name = serial_number + "_camera";
    google::InitGoogleLogging(logging_name.c_str());
  }

  init_ = false;
}

BlackflySCamera::~BlackflySCamera() {
  if (cam_ptr_) {
    cam_ptr_->UnregisterEventHandler(image_event_handler_);
    cam_ptr_->UnregisterEventHandler(device_event_handler_);
    cam_ptr_->DeInit();
  }
}

bool BlackflySCamera::Initialize(const std::optional<ace_rt_profiles::RTSubProfile>& rt_subprofile) {
  if (cam_ptr_ == nullptr) {
    return false;
  }

  try {
    image_event_handler_.SetRTSubProfile(rt_subprofile);
    cam_ptr_->Init();

    // Start/Stop acquisition in case camera was previously not stopped properly.
    cam_ptr_->BeginAcquisition();
    cam_ptr_->EndAcquisition();

    cam_ptr_->RegisterEventHandler(image_event_handler_);
    cam_ptr_->RegisterEventHandler(device_event_handler_);
  } catch (Spinnaker::Exception& e) {
    LOG(ERROR) << e.what();
    return false;
  }

  init_ = true;

  return init_;
}

bool BlackflySCamera::RegisterImageCallback(std::function<void(Spinnaker::ImagePtr&)> callback) {
  if (!init_) {
    return false;
  }

  return image_event_handler_.RegisterImageCallback(callback);
}

bool BlackflySCamera::RegisterDeviceCallback(std::function<void(const std::string&)> callback) {
  if (!init_) {
    return false;
  }

  return device_event_handler_.RegisterDeviceCallback(callback);
}

std::string BlackflySCamera::GetSerialNumber() const { return GetValue<std::string>(device_.DeviceSerialNumber); }

bool BlackflySCamera::ApplyParameters(const CameraParameters::ConstSharedPtr& params, const bool& is_master) {
  if (!init_) {
    return false;
  }

  // apply acquisition control
  if (!SetAcquisitionMode(params->acquisition_mode)) {
    return false;
  }
  if (!SetExposureAuto(params->exposure_auto)) {
    return false;
  }
  if (params->exposure_auto == Spinnaker::ExposureAuto_Off) {
    if (!SetExposureMode(params->exposure_mode)) {
      return false;
    }
    if (params->exposure_mode == Spinnaker::ExposureMode_Timed) {
      if (!SetExposureTime(params->exposure_time)) {
        return false;
      }
    }
  }

  // Initially enable acquisition frame rate - this will be adjusted later
  if (!SetAcquisitionFrameRateEnabled(true)) {
    return false;
  }

  // Ensure ALL cameras can handle the acquisition frame rate
  // This helps detecting cameras with USB2 connections, as they will fail here
  if (!SetAcquisitionFrameRate(params->acquisition_frame_rate)) {
    return false;
  }

  // Always disable acquisition frame rate for slave cameras as it will interfere with external trigger signal
  if (!SetAcquisitionFrameRateEnabled(is_master && params->acquisition_frame_rate_enable)) {
    return false;
  }

  // apply analog control
  if (!SetGainAuto(params->gain_auto)) {
    return false;
  }
  if (params->gain_auto == Spinnaker::GainAuto_Off) {
    if (!SetGain(params->gain)) {
      return false;
    }
  }
  if (!SetBlackLevelClampingEnabled(params->black_level_clamping_enable)) {
    return false;
  }
  if (params->black_level_clamping_enable) {
    if (!SetBlackLevel(params->black_level)) {
      return false;
    }
  }
  if (!SetBalanceWhiteAuto(params->balance_white_auto)) {
    return false;
  }
  if (params->balance_white_auto == Spinnaker::BalanceWhiteAuto_Off) {
    if (!SetBalanceRatioSelector(Spinnaker::BalanceRatioSelector_Blue)) {
      return false;
    }
    if (!SetBalanceRatio(params->balance_ratio_blue)) {
      return false;
    }

    if (!SetBalanceRatioSelector(Spinnaker::BalanceRatioSelector_Red)) {
      return false;
    }
    if (!SetBalanceRatio(params->balance_ratio_red)) {
      return false;
    }
  }
  if (!SetGammaEnabled(params->gamma_enable)) {
    return false;
  }
  if (params->gamma_enable) {
    if (!SetGamma(params->gamma)) {
      return false;
    }
  }

  // apply image format control
  if (!SetWidth(params->width)) {
    return false;
  }
  if (!SetHeight(params->height)) {
    return false;
  }
  if (!SetOffsetX(params->offset_x)) {
    return false;
  }
  if (!SetOffsetY(params->offset_y)) {
    return false;
  }
  if (!SetPixelFormat(params->pixel_format)) {
    return false;
  }
  if (!SetAdcBitDepth(params->adc_bit_depth)) {
    return false;
  }

  // apply chunk data control
  if (!SetChunkModeActivated(params->chunk_mode_active)) {
    return false;
  }
  if (params->chunk_mode_active) {
    // exposure time
    if (!SetChunkSelector(Spinnaker::ChunkSelector_ExposureTime)) {
      return false;
    }
    if (!SetChunkEnabled(params->chunk_exposure_time)) {
      return false;
    }

    // timestamp
    if (!SetChunkSelector(Spinnaker::ChunkSelector_Timestamp)) {
      return false;
    }
    if (!SetChunkEnabled(params->chunk_timestamp)) {
      return false;
    }
  }

  // apply stream buffer control
  if (!SetStreamBufferCountMode(params->stream_buffer_count_mode)) {
    return false;
  }
  if (!SetStreamBufferCountManual(params->manual_stream_buffer_count)) {
    return false;
  }
  if (!SetStreamBufferHandlingMode(params->stream_buffer_handling_mode)) {
    return false;
  }

  return true;
}

cv::Mat BlackflySCamera::ConvertImageWithoutCopying(Spinnaker::ImagePtr& image_ptr) {
  int height = static_cast<int>(image_ptr->GetHeight() + image_ptr->GetYPadding());
  int width = static_cast<int>(image_ptr->GetWidth() + image_ptr->GetXPadding());

  cv::Mat cv_img = cv::Mat(height, width, CV_8UC1, image_ptr->GetData(), image_ptr->GetStride());  // no copy

  return cv_img;
}

// device control
bool BlackflySCamera::ResetDevice() {
  cam_ptr_->DeviceReset.Execute();
  return true;
}

// events
bool BlackflySCamera::SetEventSelector(const EventSelector& value) {
  return SetNodeValue(cam_ptr_->EventSelector, value);
}

bool BlackflySCamera::SetEventNotification(const EventNotification& value) {
  return SetNodeValue(cam_ptr_->EventNotification, value);
}

// acquisition control
bool BlackflySCamera::SetAcquisitionMode(const AcquisitionMode& value) {
  return SetNodeValue(cam_ptr_->AcquisitionMode, value);
}

bool BlackflySCamera::StartAcquisition() {
  return RunWithChecks([this]() { cam_ptr_->BeginAcquisition(); });
}

bool BlackflySCamera::StopAcquisition() {
  return RunWithChecks([this]() { cam_ptr_->EndAcquisition(); });
}

bool BlackflySCamera::AbortAcquisition() {
  return RunWithChecks([this]() { cam_ptr_->AcquisitionAbort(); });
}

bool BlackflySCamera::SetExposureMode(const ExposureMode& value) { return SetNodeValue(cam_ptr_->ExposureMode, value); }

bool BlackflySCamera::SetExposureTime(const double& value) {
  return SetNodeValue(cam_ptr_->ExposureTime, value, true, 5e-3 * value);
}

bool BlackflySCamera::SetExposureAuto(const ExposureAuto& value) { return SetNodeValue(cam_ptr_->ExposureAuto, value); }

bool BlackflySCamera::SetAcquisitionFrameRate(const double& value) {
  return SetNodeValue(cam_ptr_->AcquisitionFrameRate, value, true, 5e-3 * value);
}

bool BlackflySCamera::SetAcquisitionFrameRateEnabled(const bool& value) {
  return SetNodeValue(cam_ptr_->AcquisitionFrameRateEnable, value);
}

bool BlackflySCamera::SetTriggerSelector(const TriggerSelector& value) {
  return SetNodeValue(cam_ptr_->TriggerSelector, value);
}

bool BlackflySCamera::SetTriggerMode(const TriggerMode& value) { return SetNodeValue(cam_ptr_->TriggerMode, value); }

bool BlackflySCamera::GenerateSoftwareTrigger() {
  return RunWithChecks([this]() { cam_ptr_->TriggerSoftware(); });
}

bool BlackflySCamera::SetTriggerSource(const TriggerSource& value) {
  return SetNodeValue(cam_ptr_->TriggerSource, value);
}

bool BlackflySCamera::SetTriggerActivation(const TriggerActivation& value) {
  return SetNodeValue(cam_ptr_->TriggerActivation, value);
}

bool BlackflySCamera::SetTriggerOverlap(const TriggerOverlap& value) {
  return SetNodeValue(cam_ptr_->TriggerOverlap, value);
}

bool BlackflySCamera::SetTriggerDelay(const double& value) { return SetNodeValue(cam_ptr_->TriggerDelay, value); }

bool BlackflySCamera::SetSensorShutterMode(const SensorShutterMode& value) {
  return SetNodeValue(cam_ptr_->SensorShutterMode, value);
}

AcquisitionMode BlackflySCamera::GetAcquisitionMode() const {
  return GetNodeValue<AcquisitionMode>(cam_ptr_->AcquisitionMode);
}

uint64_t BlackflySCamera::GetAcquisitionFrameCount() const {
  return GetNodeValue<uint64_t>(cam_ptr_->AcquisitionFrameCount);
}

ExposureMode BlackflySCamera::GetExposureMode() const { return GetNodeValue<ExposureMode>(cam_ptr_->ExposureMode); }

double BlackflySCamera::GetExposureTime() const { return GetNodeValue<double>(cam_ptr_->ExposureTime); }

ExposureAuto BlackflySCamera::GetExposureAuto() const { return GetNodeValue<ExposureAuto>(cam_ptr_->ExposureAuto); }

double BlackflySCamera::GetAcquisitionFrameRate() const {
  return GetNodeValue<double>(cam_ptr_->AcquisitionResultingFrameRate);
}

bool BlackflySCamera::IsAcquisitionFrameRateEnabled() const {
  return GetNodeValue<bool>(cam_ptr_->AcquisitionFrameRateEnable);
}

TriggerSelector BlackflySCamera::GetTriggerSelector() const {
  return GetNodeValue<TriggerSelector>(cam_ptr_->TriggerSelector);
}

TriggerMode BlackflySCamera::GetTriggerMode() const { return GetNodeValue<TriggerMode>(cam_ptr_->TriggerMode); }

TriggerSource BlackflySCamera::GetTriggerSource() const { return GetNodeValue<TriggerSource>(cam_ptr_->TriggerSource); }

TriggerActivation BlackflySCamera::GetTriggerActivation() const {
  return GetNodeValue<TriggerActivation>(cam_ptr_->TriggerActivation);
}

TriggerOverlap BlackflySCamera::GetTriggerOverlap() const {
  return GetNodeValue<TriggerOverlap>(cam_ptr_->TriggerOverlap);
}

double BlackflySCamera::GetTriggerDelay() const { return GetNodeValue<double>(cam_ptr_->TriggerDelay); }

SensorShutterMode BlackflySCamera::GetSensorShutterMode() const {
  return GetNodeValue<SensorShutterMode>(cam_ptr_->SensorShutterMode);
}

// analog control
bool BlackflySCamera::SetGain(const double& value) { return SetNodeValue(cam_ptr_->Gain, value, true, 5e-3 * value); }

bool BlackflySCamera::SetGainAuto(const GainAuto& value) { return SetNodeValue(cam_ptr_->GainAuto, value); }

bool BlackflySCamera::SetBlackLevel(const double& value) { return SetNodeValue(cam_ptr_->BlackLevel, value); }

bool BlackflySCamera::SetBlackLevelSelector(const BlackLevelSelector& value) {
  return SetNodeValue(cam_ptr_->BlackLevelSelector, value);
}

bool BlackflySCamera::SetBlackLevelClampingEnabled(const bool& value) {
  return SetNodeValue(cam_ptr_->BlackLevelClampingEnable, value);
}

bool BlackflySCamera::SetBalanceRatioSelector(const BalanceRatioSelector& value) {
  return SetNodeValue(cam_ptr_->BalanceRatioSelector, value);
}

bool BlackflySCamera::SetBalanceRatio(const double& value) {
  return SetNodeValue(cam_ptr_->BalanceRatio, value, true, 5e3 * value);
}

bool BlackflySCamera::SetBalanceWhiteAuto(const BalanceWhiteAuto& value) {
  return SetNodeValue(cam_ptr_->BalanceWhiteAuto, value);
}

bool BlackflySCamera::SetGamma(const double& value) { return SetNodeValue(cam_ptr_->Gamma, value, true, 5e3 * value); }

bool BlackflySCamera::SetGammaEnabled(const bool& value) { return SetNodeValue(cam_ptr_->GammaEnable, value); }

double BlackflySCamera::GetGain() const { return GetNodeValue<double>(cam_ptr_->Gain); }

GainAuto BlackflySCamera::GetGainAuto() const { return GetNodeValue<GainAuto>(cam_ptr_->GainAuto); }

double BlackflySCamera::GetBlackLevel() const { return GetNodeValue<double>(cam_ptr_->BlackLevel); }

BlackLevelSelector BlackflySCamera::GetBlackLevelSelector() const {
  return GetNodeValue<BlackLevelSelector>(cam_ptr_->BlackLevelSelector);
}

bool BlackflySCamera::IsBlackLevelClampingEnabled() const {
  return GetNodeValue<bool>(cam_ptr_->BlackLevelClampingEnable);
}

BalanceRatioSelector BlackflySCamera::GetBalanceRatioSelector() const {
  return GetNodeValue<BalanceRatioSelector>(cam_ptr_->BalanceRatioSelector);
}

double BlackflySCamera::GetBalanceRatio() const { return GetNodeValue<double>(cam_ptr_->BalanceRatio); }

BalanceWhiteAuto BlackflySCamera::GetBalanceWhiteAuto() const {
  return GetNodeValue<BalanceWhiteAuto>(cam_ptr_->BalanceWhiteAuto);
}

double BlackflySCamera::GetGamma() const { return GetNodeValue<double>(cam_ptr_->Gamma); }

bool BlackflySCamera::IsGammaEnabled() const { return GetNodeValue<bool>(cam_ptr_->GammaEnable); }

// image format control
bool BlackflySCamera::SetWidth(const uint64_t& value) { return SetNodeValue(cam_ptr_->Width, value); }

bool BlackflySCamera::SetHeight(const uint64_t& value) { return SetNodeValue(cam_ptr_->Height, value); }

bool BlackflySCamera::SetOffsetX(const uint64_t& value) { return SetNodeValue(cam_ptr_->OffsetX, value); }

bool BlackflySCamera::SetOffsetY(const uint64_t& value) { return SetNodeValue(cam_ptr_->OffsetY, value); }

bool BlackflySCamera::SetPixelFormat(const PixelFormat& value) { return SetNodeValue(cam_ptr_->PixelFormat, value); }

bool BlackflySCamera::SetIspEnabled(const bool& value) { return SetNodeValue(cam_ptr_->IspEnable, value); }

bool BlackflySCamera::SetAdcBitDepth(const AdcBitDepth& value) { return SetNodeValue(cam_ptr_->AdcBitDepth, value); }

uint64_t BlackflySCamera::GetWidthMax() const { return GetNodeValue<int64_t>(cam_ptr_->WidthMax); }

uint64_t BlackflySCamera::GetHeightMax() const { return GetNodeValue<int64_t>(cam_ptr_->HeightMax); }

uint64_t BlackflySCamera::GetWidth() const { return GetNodeValue<int64_t>(cam_ptr_->Width); }

uint64_t BlackflySCamera::GetHeight() const { return GetNodeValue<int64_t>(cam_ptr_->Height); }

uint64_t BlackflySCamera::GetOffsetX() const { return GetNodeValue<int64_t>(cam_ptr_->OffsetX); }

uint64_t BlackflySCamera::GetOffsetY() const { return GetNodeValue<int64_t>(cam_ptr_->OffsetY); }

PixelFormat BlackflySCamera::GetPixelFormat() const { return GetNodeValue<PixelFormat>(cam_ptr_->PixelFormat); }

bool BlackflySCamera::IsIspEnabled() const { return GetNodeValue<bool>(cam_ptr_->IspEnable); }

AdcBitDepth BlackflySCamera::GetAdcBitDepth() const { return GetNodeValue<AdcBitDepth>(cam_ptr_->AdcBitDepth); }

// digital IO control
bool BlackflySCamera::SetLineSelector(const LineSelector& value) { return SetNodeValue(cam_ptr_->LineSelector, value); }

bool BlackflySCamera::SetLineMode(const LineMode& value) { return SetNodeValue(cam_ptr_->LineMode, value); }

bool BlackflySCamera::SetUserOutputSelector(const UserOutputSelector& value) {
  return SetNodeValue(cam_ptr_->UserOutputSelector, value);
}
bool BlackflySCamera::SetUserOutputValue(const bool& value) { return SetNodeValue(cam_ptr_->UserOutputValue, value); }

bool BlackflySCamera::Set3V3Enabled(const bool& value) { return SetNodeValue(cam_ptr_->V3_3Enable, value); }

bool BlackflySCamera::SetLineSource(const LineSource& value) { return SetNodeValue(cam_ptr_->LineSource, value); }

LineSelector BlackflySCamera::GetLineSelector() const { return GetNodeValue<LineSelector>(cam_ptr_->LineSelector); }

LineMode BlackflySCamera::GetLineMode() const { return GetNodeValue<LineMode>(cam_ptr_->LineMode); }

bool BlackflySCamera::Is3V3Enabled() const { return GetNodeValue<bool>(cam_ptr_->V3_3Enable); }

LineSource BlackflySCamera::GetLineSource() const { return GetNodeValue<LineSource>(cam_ptr_->LineSource); }

UserOutputSelector BlackflySCamera::GetUserOutputSelector() const {
  return GetNodeValue<UserOutputSelector>(cam_ptr_->UserOutputSelector);
}
bool BlackflySCamera::GetUserOutputValue() const { return GetNodeValue<bool>(cam_ptr_->UserOutputValue); };

// chunk data control
bool BlackflySCamera::SetChunkModeActivated(const bool& value) {
  return SetNodeValue(cam_ptr_->ChunkModeActive, value);
}

bool BlackflySCamera::SetChunkSelector(const ChunkSelector& value) {
  return SetNodeValue(cam_ptr_->ChunkSelector, value);
}

bool BlackflySCamera::SetChunkEnabled(const bool& value) { return SetNodeValue(cam_ptr_->ChunkEnable, value); }

bool BlackflySCamera::IsChunkModeActivated() const { return GetNodeValue<bool>(cam_ptr_->V3_3Enable); }

ChunkSelector BlackflySCamera::GetChunkSelector() const { return GetNodeValue<ChunkSelector>(cam_ptr_->ChunkSelector); }

bool BlackflySCamera::IsChunkEnabled() const { return GetNodeValue<bool>(cam_ptr_->ChunkEnable); }

// buffer handling control
bool BlackflySCamera::SetStreamBufferCountManual(const uint64_t& value) {
  return SetNodeValue(data_stream_.StreamBufferCountManual, value);
}

bool BlackflySCamera::SetStreamBufferCountMode(const StreamBufferCountMode& value) {
  return SetNodeValue(data_stream_.StreamBufferCountMode, value);
}

bool BlackflySCamera::SetStreamBufferHandlingMode(const StreamBufferHandlingMode& value) {
  return SetNodeValue(data_stream_.StreamBufferHandlingMode, value);
}

uint64_t BlackflySCamera::GetStreamBufferCountManual() const {
  return GetNodeValue<uint64_t>(data_stream_.StreamBufferCountManual);
}

StreamBufferCountMode BlackflySCamera::GetStreamBufferCountMode() const {
  return GetNodeValue<StreamBufferCountMode>(data_stream_.StreamBufferCountMode);
}

StreamBufferHandlingMode BlackflySCamera::GetStreamBufferHandlingMode() const {
  return GetNodeValue<StreamBufferHandlingMode>(data_stream_.StreamBufferHandlingMode);
}

bool BlackflySCamera::SetPixelFormat(const std::vector<PixelFormat>& values) {
  return std::any_of(values.begin(), values.end(), [this](const PixelFormat& value) { return SetPixelFormat(value); });
}

bool BlackflySCamera::SetGainSelector(const GainSelector& value) { return SetNodeValue(cam_ptr_->GainSelector, value); }

GainSelector BlackflySCamera::GetGainSelector() const { return GetNodeValue<GainSelector>(cam_ptr_->GainSelector); }

// auxiliary functions
template <typename T>
inline bool IsEqual(const T& a, const T& b, const T& /*diffTolerance*/ = {}) {
  return a == b;
}

template <>
inline bool IsEqual(const double& a, const double& b, const double& diff_tolerance) {
  return std::abs(a - b) <= diff_tolerance;
}

template <typename T, typename U>
inline bool BlackflySCamera::SetNodeValue(U& node, const T& value, const bool& check_equal,
                                          const T& diff_tolerance) const {
  if (!init_) {
    throw std::runtime_error("Not initalized!");
  }

  try {
    if (!SetValue(node, value)) {
      LOG(ERROR) << "Cannot set value for node=" << node.GetName();
      return false;
    }
    if (check_equal && !IsEqual(GetValue<T>(node), value, diff_tolerance)) {
      LOG(ERROR) << "[" << node.GetName() << "]: " << GetValue<T>(node) << " != " << value;
      return false;
    }
    return true;
  } catch (Spinnaker::Exception& e) {
    LOG(ERROR) << e.what();
    return false;
  }
}

template <typename T, typename U>
inline T BlackflySCamera::GetNodeValue(U& node) const {
  if (!init_) {
    throw std::runtime_error("Not initalized!");
  }

  try {
    return GetValue<T>(node);
  } catch (Spinnaker::Exception& e) {
    LOG(ERROR) << e.what();
    throw std::runtime_error(e.what());
  }
}

inline bool BlackflySCamera::RunWithChecks(const std::function<void()>& func) const {
  if (!init_) {
    throw std::runtime_error("Not initalized!");
  }

  try {
    func();
    return true;
  } catch (Spinnaker::Exception& e) {
    LOG(ERROR) << e.what();
    return false;
  }
}

Spinnaker::CameraPtr BlackflySCamera::CreateCameraPtr(const std::string serial_number) {
  Spinnaker::CameraPtr return_value = nullptr;

  try {
    Spinnaker::SystemPtr system_ptr = Spinnaker::System::GetInstance();

    // Find all cameras.
    Spinnaker::CameraList cams = system_ptr->GetCameras();
    if (cams.GetBySerial(serial_number)) {
      return_value = cams.GetBySerial(serial_number);
    } else {
      LOG(ERROR) << "Failed to find camera (" << serial_number << ")\n";
    }
  } catch (Spinnaker::Exception& e) {
    LOG(ERROR) << e.what();
  }

  return return_value;
}

}  // namespace aps
