// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <functional>
#include <thread>

namespace ace_rt_profiles {

/**
 * A simple wrapper for std::thread.
 * Represents a thread executing under a real time profile
 * A profile consists of settings such as cpu affinity, scheduler policy, etc
 */
class RTThread {
 public:
  /**
   * Constructor
   *
   * A new thread is only spawned once RTThread::Run is called
   *
   * @param callback Function to execute INSIDE THE NEW THREAD before user code runs
   */
  explicit RTThread(std::function<void()> callback) : callback_(std::move(callback)) {}

  /**
   * Wrapper around std::thread constructor.
   * Executes the callback stored in this class before running user code.
   * The callback executes inside the new thread.
   */
  template <class F, class... Args>
  void Run(F &&f, Args &&...args) {
    thread_ = std::thread([this, f = std::forward<F>(f), ... args = std::forward<Args>(args)]() mutable {
      if (callback_) {
        callback_();
      }

      std::invoke(std::move(f), std::move(args)...);
    });
  }

  /** See std::thread::joinable */
  [[nodiscard]] bool Joinable() const noexcept { return thread_.joinable(); }

  /** See std::thread::join */
  void Join() { thread_.join(); }

 private:
  /** The underlying thread we're wrapping. */
  std::thread thread_;

  /** A callback that will execute before user code once Start is called. */
  const std::function<void()> callback_;
};

}  // namespace ace_rt_profiles
