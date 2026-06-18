// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "aps/SyncedImagesCallback.hpp"

#include "ace_loggers/ace_loggers.hpp"
#include "vision_common/concurrent_circular_buffer.hpp"
namespace aps {

class SyncedImagesCallback::SyncedImagesCallbackImpl {
 public:
  SyncedImagesCallbackImpl() : is_done(false) {}
  virtual ~SyncedImagesCallbackImpl() { Stop(); }

  virtual void OnMessageArrived(size_t index, ace_interfaces::msg::ImageData::SharedPtr message) {
    ImageBuffers::GroupsOfSyncedItems groups_of_items;  // No need to guard a temp variable
    while (!image_buffers_ptr->Push(groups_of_items, index, message)) {
      const std::string& camera_name = camera_names[static_cast<int>(index)];
      image_buffers_ptr->Pop(groups_of_items, std::numeric_limits<uint64_t>::max());
      LOG(ERROR) << "Dropping " << groups_of_items.size() << " detection groups as camera \"" << camera_name
                 << "\" reported outdated sequence_number=" << GetSequenceNumber(message) << std::endl;
      groups_of_items.clear();
    }
    if (!groups_of_items.empty()) {
      worker_queue_ptr->PushBack(groups_of_items);
    }
  }

  bool Initialize(std::shared_ptr<rclcpp::Node> node, calibration::CameraCalibrationParameters::SharedPtr camera_params,
                  const std::set<std::string>& camera_mask, int camera_buffer_length, int worker_queue_length,
                  bool use_variable_response_time) {
    ros_node = node;
    std::cout << "Initializing Node" << std::endl;
    camera_calibration_params = camera_params;

    image_buffers_ptr = std::make_unique<ImageBuffers>(camera_calibration_params->cameras.size(), camera_buffer_length,
                                                       use_variable_response_time);
    worker_queue_ptr = std::make_unique<WorkerQueue>(worker_queue_length);

    worker_thread = std::thread(std::bind(&SyncedImagesCallbackImpl::WorkerThread, this));
    for (size_t index = 0; index < camera_calibration_params->cameras.size(); ++index) {
      auto it = camera_calibration_params->cameras.begin();
      std::advance(it, index);
      if (camera_mask.find(it->first) == camera_mask.end()) {
        continue;
      }
      camera_names[static_cast<int>(index)] = it->first;
      std::cout << "Camera: " << it->first << " has index of: " << index << std::endl;
      std::string topic = "/sensors/" + it->first + "/image";
      std::cout << "Subscribing to: " << topic << std::endl;
      auto cb = [&, index](ace_interfaces::msg::ImageData::SharedPtr message) { OnMessageArrived(index, message); };
      subs.push_back(ros_node->create_subscription<ace_interfaces::msg::ImageData>(topic, 1, cb));
    }
    return true;
  }
  void Stop() {
    is_done = true;
    if (worker_thread.joinable()) {
      worker_thread.join();
    }
    subs.clear();
    ros_node = nullptr;
  }

  void WorkerThread() const {
    ImageBuffers::GroupsOfSyncedItems groups_of_items;
    while (!is_done.load()) {
      groups_of_items.clear();
      if (!worker_queue_ptr->GetFrontAndPop(groups_of_items, kMessagesWaitDuration)) {
        continue;
      }
      if (callback == nullptr) {
        continue;
      }

      callback(groups_of_items);
    }
  }
  /////////

  using WorkerQueue = utils::ConcurrentCircularBuffer<SyncedImagesCallback::ImageBuffers::GroupsOfSyncedItems>;

  std::shared_ptr<rclcpp::Node> ros_node;
  calibration::CameraCalibrationParameters::SharedPtr camera_calibration_params;
  std::vector<rclcpp::Subscription<ace_interfaces::msg::ImageData>::SharedPtr> subs;

  std::unique_ptr<SyncedImagesCallback::ImageBuffers> image_buffers_ptr;
  std::unique_ptr<WorkerQueue> worker_queue_ptr;

  std::thread worker_thread;

  SyncedImagesCallback::ImagesCallback callback;

  std::map<int, std::string> camera_names;
  std::atomic<bool> is_done;
  static constexpr std::chrono::milliseconds kMessagesWaitDuration{100};
};

SyncedImagesCallback::SyncedImagesCallback() { impl_ = std::make_shared<SyncedImagesCallbackImpl>(); }

calibration::CameraCalibrationParameters::SharedPtr SyncedImagesCallback::GetCalibrationParameters() {
  return impl_->camera_calibration_params;
}
void SyncedImagesCallback::SetCallback(SyncedImagesCallback::ImagesCallback callback) { impl_->callback = callback; }

void SyncedImagesCallback::Initialize(std::shared_ptr<rclcpp::Node> node,
                                      calibration::CameraCalibrationParameters::SharedPtr camera_parameters,
                                      const std::set<std::string>& camera_mask, int camera_buffer_length,
                                      int worker_queue_length, bool use_variable_response_time) {
  impl_->Initialize(node, camera_parameters, camera_mask, camera_buffer_length, worker_queue_length,
                    use_variable_response_time);
}
void SyncedImagesCallback::Stop() { impl_->Stop(); }

}  // namespace aps
