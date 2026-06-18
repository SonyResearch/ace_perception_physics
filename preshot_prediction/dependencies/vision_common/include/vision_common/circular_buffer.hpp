// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <vector>

namespace utils {

/* A circular buffer on top of std::vector.
 *
 * Written with only speed in mind
 * It is the caller responsibility to:
 * - check for full/empty when pushing/poping
 * - manage thread-safety
 */
template <typename T>
class CircularBuffer {
 public:
  explicit CircularBuffer(const std::size_t& capacity)
    // additional element is needed to differentiate full and empty states
    : capacity_(capacity + 1), buffer_(capacity_) {}

  void PushBack(const T& item) {
    back_ = (back_ + 1) % capacity_;
    buffer_.at(back_) = item;
  }

  void PopFront() { front_ = (front_ + 1) % capacity_; }

  void PushFront(const T& item) {
    front_ = ((capacity_ - 1) + front_) % capacity_;
    buffer_.at(front_) = item;
  }

  void PopBack() { back_ = ((capacity_ - 1) + back_) % capacity_; }

  const T& GetFront() const { return buffer_[front_]; }

  const T& GetBack() const { return buffer_[back_]; }

  [[nodiscard]] std::size_t GetSize() const noexcept { return ((capacity_ - front_) + back_ + 1) % capacity_; }

  [[nodiscard]] bool IsEmpty() const noexcept { return front_ == (back_ + 1) % capacity_; }

  [[nodiscard]] bool IsFull() const noexcept { return front_ == (back_ + 2) % capacity_; }

  virtual ~CircularBuffer() = default;

 private:
  const std::size_t capacity_;
  std::vector<T> buffer_;
  std::size_t front_ = 1;
  std::size_t back_ = 0;
};

}  // namespace utils
