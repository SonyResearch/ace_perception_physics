// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <metavision/sdk/base/events/event_cd.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <string>
#include <vector>

#include "evs_ball_cpp/imageCHW.hpp"

class FrameGeneration {
 public:
  explicit FrameGeneration(std::vector<std::vector<cv::cuda::HostMem>>* event_frames, std::queue<int>* index_queue,
                           int dt_accumulate_us, int camera_id, int height, int width, int channels, int num_parts,
                           std::queue<int64>* sp_ts_queue, float trigger_freq);
  void Process(const Metavision::EventCD* begin, const Metavision::EventCD* end);

 private:
  int camera_id_;
  int flag_index_;
  ImageCHW merged_img_;

  std::vector<std::vector<cv::cuda::HostMem>>* event_frames_;
  std::queue<int>* index_queue_;
  int dt_accumulate_us_;

  int img_count_;
  int64 ts_start_frame_;
  bool skip_;

  int height_;
  int width_;
  int num_parts_;
  int channels_;
  int64 ts_aps_;

  std::queue<int64>* sp_ts_queue_;
  int count_trigger_;
  int last_count_;
  float trigger_freq_;
};
