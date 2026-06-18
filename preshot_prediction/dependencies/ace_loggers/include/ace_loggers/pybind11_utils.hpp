// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <nlohmann/json.hpp>

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace pybind11::detail {

template <>
struct type_caster<nlohmann::json> {
 public:
  PYBIND11_TYPE_CASTER(nlohmann::json, _("dict"));

  // Python -> C++
  // NOLINTNEXTLINE(readability-identifier-naming)
  bool load(handle src, bool) {
    if (py::isinstance<py::dict>(src)) {
      value = nlohmann::json::object();
      for (const auto& [py_key, py_value] : src.cast<py::dict>()) {
        value[py_key.cast<std::string>()] = py_value.cast<nlohmann::json>();
      }
      return true;
    }
    if (py::isinstance<py::list>(src)) {
      value = nlohmann::json::array();
      for (const auto& py_item : src.cast<py::list>()) {
        value.push_back(py_item.cast<nlohmann::json>());
      }
      return true;
    }
    if (src.is_none()) {
      value = nlohmann::json();
      return true;
    }
    if (py::isinstance<py::bool_>(src)) {
      value = src.cast<bool>();
      return true;
    }
    if (py::isinstance<py::int_>(src)) {
      value = src.cast<int>();
      return true;
    }
    if (py::isinstance<py::float_>(src)) {
      value = src.cast<float>();
      return true;
    }
    if (py::isinstance<py::str>(src)) {
      value = src.cast<std::string>();
      return true;
    }
    return false;
  }

  // C++ -> Python
  // NOLINTNEXTLINE(readability-identifier-naming)
  static handle cast(const nlohmann::json& src, return_value_policy policy, handle parent) {
    if (src.is_object()) {
      py::dict py_dict;
      for (const auto& item : src.items()) {
        py_dict[item.key().c_str()] = py::cast(item.value(), policy, py_dict);
      }
      return py_dict.release();
    }
    if (src.is_array()) {
      py::list py_list;
      for (const auto& item : src) {
        py_list.append(py::cast(item, policy, py_list));
      }
      return py_list.release();
    }
    if (src.is_null()) {
      return py::none().release();
    }
    if (src.is_boolean()) {
      return py::cast(src.get<bool>(), policy, parent).release();
    }
    if (src.is_number_integer()) {
      return py::cast(src.get<int32_t>(), policy, parent).release();
    }
    if (src.is_number_float()) {
      return py::cast(src.get<float>(), policy, parent).release();
    }
    if (src.is_string()) {
      return py::cast(src.get<std::string>(), policy, parent).release();
    }

    throw std::runtime_error("Unsupported JSON type");
  }
};
}  // namespace pybind11::detail
