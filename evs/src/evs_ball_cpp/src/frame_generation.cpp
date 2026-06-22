// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <glog/logging.h>
#include "evs_ball_cpp/frame_generation.hpp"

FrameGeneration::FrameGeneration(std::vector<std::vector<cv::cuda::HostMem>> *event_frames,
                                 std::queue<int> *index_queue, int dt_accumulate_us, int camera_id, int height,
                                 int width, int channels, int num_parts, std::queue<int64> *sp_ts_queue,
                                 float trigger_freq) {
  event_frames_ = event_frames;
  index_queue_ = index_queue;

  height_ = height;
  width_ = width;
  num_parts_ = num_parts;
  channels_ = channels;
  camera_id_ = camera_id;

  dt_accumulate_us_ = dt_accumulate_us;
  sp_ts_queue_ = sp_ts_queue;

  trigger_freq_ = trigger_freq;

  merged_img_ = ImageCHW(channels_, height_, width_);
  merged_img_.ResetToZero();

  ts_start_frame_ = 0;  // should be 0
  img_count_ = 0;
  ts_aps_ = 0;
  skip_ = true;  // skip_ first trigger (when using APS as master)
  count_trigger_ = 0;
  last_count_ = 0;
}

void FrameGeneration::Process(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
  for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
    if (!sp_ts_queue_->empty()) {
      if ((skip_) && (ev->t > sp_ts_queue_->front())) {
        // skip_ first trigger
        skip_ = false;
        sp_ts_queue_->pop();
      } else if ((!skip_) && (ev->t >= sp_ts_queue_->front()) && (ts_start_frame_ == 0)) {
        LOG(INFO) << "First ts_: " << sp_ts_queue_->front() << " id " << camera_id_;
        ts_start_frame_ = sp_ts_queue_->front();
        sp_ts_queue_->pop();
      }
    }

    if (!(skip_) && !(sp_ts_queue_->empty()) && (ts_start_frame_ > 0)) {
      if ((count_trigger_ % 2) == 0) {
        ts_aps_ = sp_ts_queue_->front();
      }
      count_trigger_++;
      sp_ts_queue_->pop();
    }

    if ((ts_start_frame_ > 0) && (ts_start_frame_ <= ev->t) && (ev->t >= (dt_accumulate_us_ + ts_start_frame_))) {
      if (ts_aps_ > 0) {
        img_count_ = count_trigger_ * static_cast<int>(1000.0 / trigger_freq_);
        ts_aps_ = 0;
      } else {
        img_count_ = img_count_ + dt_accumulate_us_ / 1000;
      }
      // If image_count is repeated due to tirgger signal, skip it
      if (img_count_ > last_count_) {
        flag_index_ = (img_count_) % num_parts_;
        index_queue_->push(img_count_);

        memcpy(event_frames_->at(flag_index_)[camera_id_].data, merged_img_.GetDataPointer(),
               channels_ * width_ * height_ * sizeof(half_float::half));
        last_count_ = img_count_;
      }

      // Use EVS time as start for next frame
      ts_start_frame_ = ts_start_frame_ + dt_accumulate_us_;
      merged_img_.ResetToZero();
    }

    if ((ts_start_frame_ > 0) && (ev->t >= ts_start_frame_)) {
      if (ev->p == 0) {
        merged_img_.At(0, ev->y, ev->x)++;
      }
      if ((ev->p == 1) &&
          (merged_img_.At(1, ev->y, ev->x) < static_cast<half_float::half>(static_cast<float>(ev->t - ts_start_frame_) /
                                                                           static_cast<float>(dt_accumulate_us_)) *
                                               255)) {
        merged_img_.At(1, ev->y, ev->x) = static_cast<half_float::half>(
          (static_cast<float>(ev->t - ts_start_frame_) / static_cast<float>(dt_accumulate_us_)) *
          255);  // Norm to range 0-255
      }
    }
  }
}
