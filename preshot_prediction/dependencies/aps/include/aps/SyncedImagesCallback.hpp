#pragma once

#include <thread>

#include "ace_interfaces/msg/image_data.hpp"
#include "calibration/camera_calibration_parameters.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_common/synced_circular_buffers.hpp"

namespace aps {

class SyncedImagesCallback {
 public:
  using ImageType = ace_interfaces::msg::ImageData::SharedPtr;
  [[nodiscard]] static inline size_t GetSequenceNumber(const ImageType& img) { return img->header.sequence_number; }

  using SharedPtr = std::shared_ptr<SyncedImagesCallback>;
  using ConstSharedPtr = std::shared_ptr<const SyncedImagesCallback>;

  using ImageBuffers = utils::SyncedCircularBuffers<ImageType, size_t, GetSequenceNumber>;
  using ImagesCallback = std::function<void(const ImageBuffers::GroupsOfSyncedItems&)>;

 protected:
  class SyncedImagesCallbackImpl;
  std::shared_ptr<SyncedImagesCallbackImpl> impl_;

 public:
  SyncedImagesCallback();
  virtual ~SyncedImagesCallback() = default;

  void SetCallback(ImagesCallback callback);
  [[nodiscard]] calibration::CameraCalibrationParameters::SharedPtr GetCalibrationParameters();

  void Initialize(std::shared_ptr<rclcpp::Node> node,
                  calibration::CameraCalibrationParameters::SharedPtr camera_parameters,
                  const std::set<std::string>& camera_mask, int camera_buffer_length, int worker_queue_length,
                  bool use_variable_response_time);

  void Stop();
};

}  // namespace aps
