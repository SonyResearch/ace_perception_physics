// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_loggers/ace_loggers.hpp"
#include "ace_loggers/pybind11_utils.hpp"
#include "pybind11/chrono.h"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = ::pybind11;

namespace datalogger {

template <typename T>
class ContextManager {
 public:
  static typename T::SharedPtr Enter(const typename T::SharedPtr &self) { return self; }

  static void Exit(const typename T::SharedPtr &self, const py::object &, const py::object &, const py::object &) {
    self->Release();
  }
};

PYBIND11_MODULE(datalogger_pybind, m) {
  py::class_<DataWriter, DataWriter::SharedPtr>(m, "DataWriter")
    .def(py::init<std::string, nlohmann::json>(), py::arg("file_name"), py::arg("metadata"))
    .def("__enter__", &ContextManager<DataWriter>::Enter)
    .def("__exit__", &ContextManager<DataWriter>::Exit)
    .def("write", &DataWriter::Write<int32_t>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<uint32_t>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<int64_t>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<uint64_t>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<double>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<std::string>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<std::chrono::time_point<std::chrono::high_resolution_clock>>,
         py::arg("data").noconvert())
    .def("write", &DataWriter::Write<Eigen::MatrixXd>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<Eigen::MatrixXf>, py::arg("data").noconvert());

  py::class_<DataReader, DataReader::SharedPtr>(m, "DataReader")
    .def(py::init<std::string>(), py::arg("file_name"))
    .def("__enter__", &ContextManager<DataReader>::Enter)
    .def("__exit__", &ContextManager<DataReader>::Exit)
    .def_property_readonly("module_name", &DataReader::GetModuleName)
    .def_property_readonly("log_version", &DataReader::GetLogVersion)
    .def("get_metadata", &DataReader::GetMetadata, py::return_value_policy::take_ownership)
    .def("eof", &DataReader::IsEndOfFile)
    .def("tell", &DataReader::GetPosition)
    .def("read_int32", &DataReader::ReadAndReturn<int32_t>)
    .def("read_uint32", &DataReader::ReadAndReturn<uint32_t>)
    .def("read_int64", &DataReader::ReadAndReturn<int64_t>)
    .def("read_uint64", &DataReader::ReadAndReturn<uint64_t>)
    .def("read_float64", &DataReader::ReadAndReturn<double>)
    .def("read_str", &DataReader::ReadAndReturn<std::string>)
    .def("read_time", &DataReader::ReadAndReturn<std::chrono::time_point<std::chrono::high_resolution_clock>>)
    .def("read_mat2d_float64", &DataReader::ReadAndReturn<Eigen::MatrixXd>)
    .def("read_mat2d_float32", &DataReader::ReadAndReturn<Eigen::MatrixXf>);

  m.def("construct_log_path", &datalogger::ConstructFullLogName);
  m.def("construct_sub_log_path", &datalogger::ConstructSubLogName);
  m.def("is_loader_defined", &datalogger::IsLoaderDefined, py::arg("module_name"));
  m.def("get_module_name", &datalogger::GetModuleName, py::arg("path"));
  m.def(
    "get_loader",
    [](const std::string &path, py::args args) -> py::object {
      const auto module_name = GetModuleName(path);
      if (module_name.empty()) {
        return py::none();
      }
      py::module module = py::module::import((module_name + ".datalogger_utils").c_str());
      py::object RosLoader = module.attr("RosLoader");
      return RosLoader(path, *args);
    },
    py::arg("path"));
}  // PYBIND11_MODULE

}  // namespace datalogger
