// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <condition_variable>
#include <mutex>

#include "vision_common/circular_buffer.hpp"

namespace utils {

/* A thread-safe version of CircularBuffer.
 */
template <typename T>
class ConcurrentCircularBuffer : private CircularBuffer<T> {
 public:
  explicit ConcurrentCircularBuffer(const std::size_t& capacity) : CircularBuffer<T>(capacity) {}

  void PushBack(const T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    push_.wait(lock, [this] { return !CircularBuffer<T>::IsFull(); });
    CircularBuffer<T>::PushBack(item);
    pop_.notify_one();
  }

  bool PushBack(const T& item, const std::chrono::milliseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!push_.wait_for(lock, timeout, [this] { return !CircularBuffer<T>::IsFull(); })) {
      return false;
    }
    CircularBuffer<T>::PushBack(item);
    pop_.notify_one();
    return true;
  }

  void GetFrontAndPop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    pop_.wait(lock, [this] { return !CircularBuffer<T>::IsEmpty(); });
    item = CircularBuffer<T>::GetFront();
    CircularBuffer<T>::PopFront();
    push_.notify_one();
  }

  bool GetFrontAndPop(T& item, const std::chrono::milliseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!pop_.wait_for(lock, timeout, [this] { return !CircularBuffer<T>::IsEmpty(); })) {
      return false;
    }
    item = CircularBuffer<T>::GetFront();
    CircularBuffer<T>::PopFront();
    push_.notify_one();
    return true;
  }

  void PushFront(const T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    push_.wait(lock, [this] { return !CircularBuffer<T>::IsFull(); });
    CircularBuffer<T>::PushFront(item);
    pop_.notify_one();
  }

  bool PushFront(const T& item, const std::chrono::milliseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!push_.wait_for(lock, timeout, [this] { return !CircularBuffer<T>::IsFull(); })) {
      return false;
    }
    CircularBuffer<T>::PushFront(item);
    pop_.notify_one();
    return true;
  }

  void GetBackAndPop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    pop_.wait(lock, [this] { return !CircularBuffer<T>::IsEmpty(); });
    item = CircularBuffer<T>::GetBack();
    CircularBuffer<T>::PopBack();
    push_.notify_one();
  }

  bool GetBackAndPop(T& item, const std::chrono::milliseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!pop_.wait_for(lock, timeout, [this] { return !CircularBuffer<T>::IsEmpty(); })) {
      return false;
    }
    item = CircularBuffer<T>::GetBack();
    CircularBuffer<T>::PopBack();
    push_.notify_one();
    return true;
  }

  [[nodiscard]] std::size_t GetSize() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return CircularBuffer<T>::GetSize();
  }

  [[nodiscard]] bool IsEmpty() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return CircularBuffer<T>::IsEmpty();
  }

  [[nodiscard]] bool IsFull() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return CircularBuffer<T>::IsFull();
  }

  virtual ~ConcurrentCircularBuffer() = default;

 private:
  mutable std::mutex mutex_;
  std::condition_variable push_;
  std::condition_variable pop_;
};

}  // namespace utils
