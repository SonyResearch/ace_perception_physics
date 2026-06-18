// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <Spinnaker.h>

#include <ace_rt_profiles/rt_sub_profile.hpp>
#include <functional>
#include <string>

#include "SpinGenApi/SpinnakerGenApi.h"
#include "aps/blackfly_s_device_event_handler.hpp"
#include "aps/blackfly_s_image_event_handler.hpp"
#include "aps/camera_parameters.hpp"
#include "opencv2/opencv.hpp"

namespace aps {

using EventSelector = Spinnaker::EventSelectorEnums;
using EventNotification = Spinnaker::EventNotificationEnums;
using AcquisitionMode = Spinnaker::AcquisitionModeEnums;
using AdcBitDepth = Spinnaker::AdcBitDepthEnums;
using BalanceRatioSelector = Spinnaker::BalanceRatioSelectorEnums;
using BalanceWhiteAuto = Spinnaker::BalanceWhiteAutoEnums;
using BlackLevelSelector = Spinnaker::BlackLevelSelectorEnums;
using ChunkSelector = Spinnaker::ChunkSelectorEnums;
using ExposureAuto = Spinnaker::ExposureAutoEnums;
using ExposureMode = Spinnaker::ExposureModeEnums;
using GainAuto = Spinnaker::GainAutoEnums;
using GainSelector = Spinnaker::GainSelectorEnums;
using LineMode = Spinnaker::LineModeEnums;
using LineSelector = Spinnaker::LineSelectorEnums;
using LineSource = Spinnaker::LineSourceEnums;
using UserOutputSelector = Spinnaker::UserOutputSelectorEnums;
using PixelFormat = Spinnaker::PixelFormatEnums;
using SensorShutterMode = Spinnaker::SensorShutterModeEnums;
using StreamBufferCountMode = Spinnaker::StreamBufferCountModeEnum;
using StreamBufferHandlingMode = Spinnaker::StreamBufferHandlingModeEnum;
using StreamType = Spinnaker::StreamTypeEnum;
using TriggerActivation = Spinnaker::TriggerActivationEnums;
using TriggerMode = Spinnaker::TriggerModeEnums;
using TriggerOverlap = Spinnaker::TriggerOverlapEnums;
using TriggerSelector = Spinnaker::TriggerSelectorEnums;
using TriggerSource = Spinnaker::TriggerSourceEnums;

class BlackflySCamera {
 public:
  using SharedPtr = std::shared_ptr<BlackflySCamera>;
  using ConstSharedPtr = std::shared_ptr<const BlackflySCamera>;

  explicit BlackflySCamera(const std::string& serial_number);
  BlackflySCamera(const BlackflySCamera&) = delete;

  ~BlackflySCamera();

  bool Initialize(const std::optional<ace_rt_profiles::RTSubProfile>& rt_subprofile);
  bool RegisterImageCallback(std::function<void(Spinnaker::ImagePtr&)> callback);
  bool RegisterDeviceCallback(std::function<void(const std::string&)> callback);
  [[nodiscard]] std::string GetSerialNumber() const;
  bool ApplyParameters(const CameraParameters::ConstSharedPtr& params, const bool& is_master = false);
  static cv::Mat ConvertImageWithoutCopying(Spinnaker::ImagePtr& image_ptr);

  // device control
  bool ResetDevice();

  // events
  bool SetEventSelector(const EventSelector& value);
  bool SetEventNotification(const EventNotification& value);

  // acquisition control
  bool SetAcquisitionMode(const AcquisitionMode& value);
  bool StartAcquisition();
  bool StopAcquisition();
  bool AbortAcquisition();
  bool SetExposureMode(const ExposureMode& value);
  bool SetExposureTime(const double& value);
  bool SetExposureAuto(const ExposureAuto& value);
  bool SetAcquisitionFrameRate(const double& value);
  bool SetAcquisitionFrameRateEnabled(const bool& value);
  bool SetTriggerSelector(const TriggerSelector& value);
  bool SetTriggerMode(const TriggerMode& value);
  bool GenerateSoftwareTrigger();
  bool SetTriggerSource(const TriggerSource& value);
  bool SetTriggerActivation(const TriggerActivation& value);
  bool SetTriggerOverlap(const TriggerOverlap& value);
  bool SetTriggerDelay(const double& value);
  bool SetSensorShutterMode(const SensorShutterMode& value);

  [[nodiscard]] AcquisitionMode GetAcquisitionMode() const;
  [[nodiscard]] uint64_t GetAcquisitionFrameCount() const;
  [[nodiscard]] ExposureMode GetExposureMode() const;
  [[nodiscard]] double GetExposureTime() const;
  [[nodiscard]] ExposureAuto GetExposureAuto() const;
  [[nodiscard]] double GetAcquisitionFrameRate() const;
  [[nodiscard]] bool IsAcquisitionFrameRateEnabled() const;
  [[nodiscard]] TriggerSelector GetTriggerSelector() const;
  [[nodiscard]] TriggerMode GetTriggerMode() const;
  [[nodiscard]] TriggerSource GetTriggerSource() const;
  [[nodiscard]] TriggerActivation GetTriggerActivation() const;
  [[nodiscard]] TriggerOverlap GetTriggerOverlap() const;
  [[nodiscard]] double GetTriggerDelay() const;
  [[nodiscard]] SensorShutterMode GetSensorShutterMode() const;

  // analog control
  bool SetGain(const double& value);
  bool SetGainSelector(const GainSelector& value);
  bool SetGainAuto(const GainAuto& value);
  bool SetBlackLevel(const double& value);
  bool SetBlackLevelSelector(const BlackLevelSelector& value);
  bool SetBlackLevelClampingEnabled(const bool& value);
  bool SetBalanceRatioSelector(const BalanceRatioSelector& value);
  bool SetBalanceRatio(const double& value);
  bool SetBalanceWhiteAuto(const BalanceWhiteAuto& value);
  bool SetGamma(const double& value);
  bool SetGammaEnabled(const bool& value);

  [[nodiscard]] double GetGain() const;
  [[nodiscard]] GainSelector GetGainSelector() const;
  [[nodiscard]] GainAuto GetGainAuto() const;
  [[nodiscard]] double GetBlackLevel() const;
  [[nodiscard]] BlackLevelSelector GetBlackLevelSelector() const;
  [[nodiscard]] bool IsBlackLevelClampingEnabled() const;
  [[nodiscard]] BalanceRatioSelector GetBalanceRatioSelector() const;
  [[nodiscard]] double GetBalanceRatio() const;
  [[nodiscard]] BalanceWhiteAuto GetBalanceWhiteAuto() const;
  [[nodiscard]] double GetGamma() const;
  [[nodiscard]] bool IsGammaEnabled() const;

  // image format control
  bool SetWidth(const uint64_t& value);
  bool SetHeight(const uint64_t& value);
  bool SetOffsetX(const uint64_t& value);
  bool SetOffsetY(const uint64_t& value);
  bool SetPixelFormat(const PixelFormat& value);
  bool SetPixelFormat(const std::vector<PixelFormat>& values);
  bool SetIspEnabled(const bool& value);
  bool SetAdcBitDepth(const AdcBitDepth& value);

  [[nodiscard]] uint64_t GetWidthMax() const;
  [[nodiscard]] uint64_t GetHeightMax() const;
  [[nodiscard]] uint64_t GetWidth() const;
  [[nodiscard]] uint64_t GetHeight() const;
  [[nodiscard]] uint64_t GetOffsetX() const;
  [[nodiscard]] uint64_t GetOffsetY() const;
  [[nodiscard]] PixelFormat GetPixelFormat() const;
  [[nodiscard]] bool IsIspEnabled() const;
  [[nodiscard]] AdcBitDepth GetAdcBitDepth() const;

  // digital IO control
  bool SetLineSelector(const LineSelector& value);
  bool SetLineMode(const LineMode& value);
  bool Set3V3Enabled(const bool& value);
  bool SetLineSource(const LineSource& value);
  bool SetUserOutputSelector(const UserOutputSelector& value);
  bool SetUserOutputValue(const bool& value);

  [[nodiscard]] LineSelector GetLineSelector() const;
  [[nodiscard]] LineMode GetLineMode() const;
  [[nodiscard]] bool Is3V3Enabled() const;
  [[nodiscard]] LineSource GetLineSource() const;
  [[nodiscard]] UserOutputSelector GetUserOutputSelector() const;
  [[nodiscard]] bool GetUserOutputValue() const;

  // chunk data control
  bool SetChunkModeActivated(const bool& value);
  bool SetChunkSelector(const ChunkSelector& value);
  bool SetChunkEnabled(const bool& value);

  [[nodiscard]] bool IsChunkModeActivated() const;
  [[nodiscard]] ChunkSelector GetChunkSelector() const;
  [[nodiscard]] bool IsChunkEnabled() const;

  // buffer handling control
  bool SetStreamBufferCountManual(const uint64_t& value);
  bool SetStreamBufferCountMode(const StreamBufferCountMode& value);
  bool SetStreamBufferHandlingMode(const StreamBufferHandlingMode& value);

  [[nodiscard]] uint64_t GetStreamBufferCountManual() const;
  [[nodiscard]] StreamBufferCountMode GetStreamBufferCountMode() const;
  [[nodiscard]] StreamBufferHandlingMode GetStreamBufferHandlingMode() const;

 private:
  bool init_;

  const Spinnaker::CameraPtr cam_ptr_;
  const Spinnaker::TransportLayerDevice device_;
  const Spinnaker::TransportLayerStream data_stream_;
  BlackflySImageEventHandler image_event_handler_;
  BlackflySDeviceEventHandler device_event_handler_;

  // auxiliary functions
  template <typename T, typename U>
  inline bool SetNodeValue(U& node, const T& value, const bool& check_equal = true, const T& diff_tolerance = {}) const;

  template <typename T, typename U>
  inline T GetNodeValue(U& node) const;

  inline bool RunWithChecks(const std::function<void()>& func) const;

  static Spinnaker::CameraPtr CreateCameraPtr(std::string serial_number);
};

}  // namespace aps
