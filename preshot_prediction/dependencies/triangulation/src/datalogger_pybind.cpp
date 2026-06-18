// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "triangulation/datalogger.hpp"

namespace py = ::pybind11;

namespace triangulation::datalogger {

PYBIND11_MODULE(datalogger_pybind, m) {
  py::module::import("ace_loggers.datalogger_pybind");
  py::module::import("triangulation.python_module");

  py::class_<DataWriter, DataWriter::SharedPtr, ::datalogger::DataWriter>(m, "DataWriter")
    .def(py::init<std::string, std::string, std::string>())
    .def("write", &DataWriter::Write<std::vector<std::vector<Eigen::Vector2f>>>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<std::vector<std::vector<Eigen::Vector3f>>>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<std::vector<std::vector<std::pair<int, int>>>>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<TriangulatedPoint>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<std::vector<TriangulatedPoint>>, py::arg("data").noconvert());

  py::class_<DataReader, DataReader::SharedPtr, ::datalogger::DataReader>(m, "DataReader")
    .def(py::init<std::string>())
    .def("read_uint32", &DataReader::ReadAndReturn<uint32_t>)
    .def("read_uint64", &DataReader::ReadAndReturn<uint64_t>)
    .def("read_timestamp", &DataReader::ReadAndReturn<std::pair<int32_t, uint32_t>>)
    .def("read_2d_input", &DataReader::ReadAndReturn<std::vector<std::vector<Eigen::Vector2f>>>)
    .def("read_3d_input", &DataReader::ReadAndReturn<std::vector<std::vector<Eigen::Vector3f>>>)
    .def("read_correspondence_input", &DataReader::ReadAndReturn<std::vector<std::vector<std::pair<int, int>>>>)
    .def("read", &DataReader::Read<TriangulatedPoint>, py::arg("data").noconvert())
    .def("read_triangulation_output", &DataReader::ReadAndReturn<std::vector<TriangulatedPoint>>);

}  // PYBIND11_MODULE

}  // namespace triangulation::datalogger
