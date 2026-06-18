// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <chrono>

namespace ace_monitor {
class FPSCalculator {
 public:
  using Clock = std::chrono::high_resolution_clock;
  using Seconds = std::chrono::duration<float>;
  using Timepoint = Clock::time_point;

 private:
  int frames_acc_{0};
  int fps_{0};
  int last_fps_{0};
  Timepoint start_;

 public:
  FPSCalculator() { Reset(); }
  void AddFrame() { frames_acc_++; }
  void Update() {
    Timepoint curr_time = Clock::now();

    auto delta_time = std::chrono::duration_cast<Seconds>(curr_time - start_);
    if (delta_time.count() > 1) {
      last_fps_ = fps_;
      fps_ = static_cast<int>(static_cast<float>(frames_acc_) / delta_time.count());
      start_ = curr_time;
      frames_acc_ = 0;
    }
  }
  void Reset() {
    frames_acc_ = 0;
    fps_ = 0;
    last_fps_ = 0;
    start_ = Clock::now();
  }
  [[nodiscard]] int CurrentFPS() const { return fps_; }
  [[nodiscard]] int LastFPS() const { return last_fps_; }
};
}  // namespace ace_monitor
