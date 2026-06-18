// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <ostream>

#include "vision_common/circular_buffer.hpp"

namespace utils {

/* A thread-safe container for multi-circular buffers.
 *
 * - Synced via a monotonically-increasing sequence (MIS) function
 * - Items are grouped and popped when:
 *   - A buffer is full, up-to that buffer's front
 *   - All buffers received a same-sequence item while in opportunistic mode, up-to that item's sequence
 */
template <typename T, typename MISType, MISType (*GetMISIdentifier)(const T&)>
class SyncedCircularBuffers {
 public:
  using SyncedItems = std::map<std::size_t, T>;
  using GroupsOfSyncedItems = std::map<MISType, std::map<std::size_t, T>>;

  SyncedCircularBuffers(const std::size_t& total_buffers, const std::size_t& buffer_capacity,
                        const bool& is_opportunistic)
    : buffers_(total_buffers, CircularBuffer<T>(buffer_capacity)), is_opportunistic_(is_opportunistic) {}

  // It is the caller responsibility to clear groups_of_items before passing
  bool Push(GroupsOfSyncedItems& groups_of_items, const std::size_t& buffer_index, const T& item) {
    std::lock_guard<std::mutex> lock(mutex_);

    CircularBuffer<T>& curr_buffer = buffers_[buffer_index];
    const MISType& item_id = GetMISIdentifier(item);
    if (!curr_buffer.IsEmpty() && item_id <= GetMISIdentifier(curr_buffer.GetBack())) {
      return false;
    }
    if (curr_buffer.IsFull()) {
      PopAlreadyLocked(groups_of_items, GetMISIdentifier(curr_buffer.GetFront()));
    }
    curr_buffer.PushBack(item);
    if (is_opportunistic_) {
      if (std::all_of(buffers_.begin(), buffers_.end(), [&item_id](const CircularBuffer<T>& buffer) {
            return !buffer.IsEmpty() && item_id <= GetMISIdentifier(buffer.GetBack());
          })) {
        PopAlreadyLocked(groups_of_items, item_id);
      }
    }
    return true;
  }

  // It is the caller responsibility to clear groups_of_items before passing
  void Pop(GroupsOfSyncedItems& groups_of_items, const MISType& up_to_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    PopAlreadyLocked(groups_of_items, up_to_id);
  }

  friend std::ostream& operator<<(std::ostream& io, const SyncedCircularBuffers& self) {
    std::lock_guard<std::mutex> lock(self.mutex_);

    io << std::string(70, '=') << std::endl;
    for (std::size_t loop_index = 0; loop_index < 2; loop_index++) {
      for (const CircularBuffer<T>& curr_buffer : self.buffers_) {
        if (curr_buffer.IsEmpty()) {
          io << std::setw(5) << "X";
        } else if (loop_index == 0) {
          io << std::setw(5) << GetMISIdentifier(curr_buffer.GetFront());
        } else {
          io << std::setw(5) << GetMISIdentifier(curr_buffer.GetBack());
        }
      }
      io << std::endl;
    }
    io << std::string(70, '=') << std::endl;
    return io;
  }

 private:
  // It is the caller responsibility to clear groups_of_items before passing
  // This function assumes the mutex is already locked
  void PopAlreadyLocked(GroupsOfSyncedItems& groups_of_items, const MISType& up_to_id) {
    const std::size_t& total_buffers = buffers_.size();

    for (std::size_t buffer_index = 0; buffer_index < total_buffers; buffer_index++) {
      CircularBuffer<T>& curr_buffer = buffers_[buffer_index];
      while (!curr_buffer.IsEmpty() && GetMISIdentifier(curr_buffer.GetFront()) <= up_to_id) {
        const T& item = curr_buffer.GetFront();
        const MISType& item_id = GetMISIdentifier(item);
        groups_of_items[item_id][buffer_index] = item;
        curr_buffer.PopFront();
      }
    }
  }

  mutable std::mutex mutex_;
  std::vector<CircularBuffer<T>> buffers_;
  const bool is_opportunistic_;
};

}  // namespace utils
