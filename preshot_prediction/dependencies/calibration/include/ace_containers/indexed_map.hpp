// Vendored stub of ace_containers::IndexedMap.
// The original ace_containers package is not available in this build.
//
// IndexedMap behaves like an ordered associative map that also remembers the
// insertion order of its keys, so that values can be looked up either by key
// or by their numeric index.  This minimal implementation supports the subset
// of the API used in this workspace (operator[], at(key), at(index),
// GetIndex(key), contains, find, begin/end iteration, size, clear).

#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ace_containers {

template <typename Key, typename Value, typename Compare = std::less<Key>,
          typename Allocator = std::allocator<std::pair<const Key, Value>>>
class IndexedMap {
 public:
  using map_type = std::map<Key, Value, Compare, Allocator>;
  using key_type = typename map_type::key_type;
  using mapped_type = typename map_type::mapped_type;
  using value_type = typename map_type::value_type;
  using iterator = typename map_type::iterator;
  using const_iterator = typename map_type::const_iterator;
  using size_type = typename map_type::size_type;

  Value& operator[](const Key& k) {
    auto it = data_.find(k);
    if (it == data_.end()) {
      order_.push_back(k);
      return data_[k];
    }
    return it->second;
  }

  Value& at(const Key& k) { return data_.at(k); }
  const Value& at(const Key& k) const { return data_.at(k); }

  // Index-based access using insertion order.
  Value& at(std::size_t idx) { return data_.at(order_.at(idx)); }
  const Value& at(std::size_t idx) const { return data_.at(order_.at(idx)); }

  // Index-based operator[] using insertion order (for integer types).
  template <typename I, typename = std::enable_if_t<std::is_integral_v<I> && !std::is_same_v<I, Key>>>
  Value& operator[](I idx) { return data_.at(order_.at(static_cast<std::size_t>(idx))); }
  template <typename I, typename = std::enable_if_t<std::is_integral_v<I> && !std::is_same_v<I, Key>>>
  const Value& operator[](I idx) const { return data_.at(order_.at(static_cast<std::size_t>(idx))); }

  std::size_t GetIndex(const Key& k) const {
    for (std::size_t i = 0; i < order_.size(); ++i) {
      if (order_[i] == k) return i;
    }
    throw std::out_of_range("IndexedMap::GetIndex: key not found");
  }

  bool contains(const Key& k) const { return data_.find(k) != data_.end(); }

  iterator find(const Key& k) { return data_.find(k); }
  const_iterator find(const Key& k) const { return data_.find(k); }

  iterator begin() { return data_.begin(); }
  iterator end() { return data_.end(); }
  const_iterator begin() const { return data_.begin(); }
  const_iterator end() const { return data_.end(); }
  const_iterator cbegin() const { return data_.cbegin(); }
  const_iterator cend() const { return data_.cend(); }

  size_type size() const noexcept { return data_.size(); }
  bool empty() const noexcept { return data_.empty(); }

  void clear() {
    data_.clear();
    order_.clear();
  }

  std::pair<iterator, bool> insert(const value_type& v) {
    auto res = data_.insert(v);
    if (res.second) order_.push_back(v.first);
    return res;
  }

  std::pair<iterator, bool> emplace(const Key& k, const Value& v) {
    auto res = data_.emplace(k, v);
    if (res.second) order_.push_back(k);
    return res;
  }

 private:
  map_type data_;
  std::vector<Key> order_;
};

}  // namespace ace_containers
