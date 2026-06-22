// SPDX-License-Identifier: MIT
// Drop-in replacement for ace_containers::IndexedMap using std::map.
#pragma once

#include <map>

namespace ace_containers {

template <typename Key, typename Value>
class IndexedMap : public std::map<Key, Value> {
 public:
  using std::map<Key, Value>::map;
};

}  // namespace ace_containers
