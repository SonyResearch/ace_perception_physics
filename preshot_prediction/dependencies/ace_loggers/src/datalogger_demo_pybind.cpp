// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ace_loggers/datalogger_demo.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = ::pybind11;

namespace datalogger_demo {

PYBIND11_MODULE(datalogger_demo_pybind, m) {
  py::module::import("ace_loggers.datalogger_pybind");

  py::class_<HasPrimitivesStruct, HasPrimitivesStruct::SharedPtr>(m, "HasPrimitivesStruct")
    .def(py::init())
    .def(py::init<int64_t, double>())
    .def_readwrite("var_int", &HasPrimitivesStruct::var_int)
    .def_readwrite("var_float", &HasPrimitivesStruct::var_float);

  py::class_<HasPointerStruct, HasPointerStruct::SharedPtr, HasPrimitivesStruct>(m, "HasPointerStruct")
    .def(py::init())
    .def(py::init<int64_t, double, std::string>())
    .def_readwrite("var_str", &HasPointerStruct::var_str);

  py::class_<DataWriter, DataWriter::SharedPtr, ::datalogger::DataWriter>(m, "DataWriter")
    .def(py::init<std::string>())
    .def("write", &DataWriter::Write<std::string>, py::arg("data").noconvert())  // TODO(cv3d): Support inheritance
    .def("write", &DataWriter::Write<HasPointerStruct>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<HasPrimitivesStruct>, py::arg("data").noconvert());

  py::class_<DataReader, DataReader::SharedPtr, ::datalogger::DataReader>(m, "DataReader")
    .def(py::init<std::string>())
    .def("read_str", &DataReader::ReadAndReturn<std::string>)  // TODO(cv3d): Support inheritance
    .def("read", &DataReader::Read<HasPointerStruct>, py::arg("data").noconvert())
    .def("read", &DataReader::Read<HasPrimitivesStruct>, py::arg("data").noconvert());

}  // PYBIND11_MODULE

}  // namespace datalogger_demo
