// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "ace_interfaces/msg/image_data.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/opencv.hpp"

namespace aps {

class JpegEncoder {
 public:
  using SharedPtr = std::shared_ptr<JpegEncoder>;
  using ConstSharedPtr = std::shared_ptr<const JpegEncoder>;

  explicit JpegEncoder(int cuda_device_id, bool debayer_only = false, int quality = 70);
  ~JpegEncoder();

  void SaveImage(const ace_interfaces::msg::ImageData::ConstSharedPtr& message, const std::string& filename);

 private:
  const int cuda_device_id_;
  const bool debayer_only_;
  // cv::cuda::HostMem host_mem_;

  ace_interfaces::msg::ImageData::ConstSharedPtr message_;
  std::string filename_;
  int quality_;
  bool message_available_;

  std::thread t_;
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<bool> canceled_;
  void WorkerThread();
};

}  // namespace aps
