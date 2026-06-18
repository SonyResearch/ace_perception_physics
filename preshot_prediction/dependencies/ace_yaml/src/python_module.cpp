// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include <algorithm>

#include "ace_yaml/ace_yaml.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = ::pybind11;

py::object ToPyObj(const YAML::Node& node) {
  if (node.IsMap()) {
    py::dict py_dict;
    for (auto it = node.begin(); it != node.end(); ++it) {
      const py::str key = it->first.as<std::string>().c_str();
      py_dict[key] = ToPyObj(it->second);
    }
    return std::move(py_dict);
  }
  if (node.IsSequence()) {
    py::list py_list;
    for (auto it2 = node.begin(); it2 != node.end(); ++it2) {
      py_list.append(ToPyObj(*it2));
    }
    return std::move(py_list);
  }
  if (node.IsScalar()) {
    auto str_value_lower = node.as<std::string>();
    std::transform(str_value_lower.begin(), str_value_lower.end(), str_value_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (str_value_lower == "true") {
      return py::bool_(true);
    }
    if (str_value_lower == "false") {
      return py::bool_(false);
    }
    try {
      return py::int_(node.as<int>());
    } catch (const YAML::BadConversion& e) {
    }
    try {
      return py::float_(node.as<float>());
    } catch (const YAML::BadConversion& e) {
    }
    return py::str(node.as<std::string>());
  }
  return py::none();
}

PYBIND11_MODULE(python_module, m) {
  m.attr("INHERITANCE_NODE_NAME") = py::str(ace_yaml::kInheritanceNodeName);

  (void)py::class_<YAML::Node, std::shared_ptr<YAML::Node>>(m, "yaml_node").def(py::init<>());

  m.def("full_load",
        [](const py::object& fp) {
          const auto& file_path = fp.attr("name").cast<std::string>();
          YAML::Node nodes;
          {
            py::gil_scoped_release release;
            nodes = ace_yaml::LoadFile(file_path);
          }
          return ToPyObj(nodes);
        })
    .def("load_string", &ace_yaml::LoadString)
    .def("to_python", [](const YAML::Node& node) { return ToPyObj(node); });
}
