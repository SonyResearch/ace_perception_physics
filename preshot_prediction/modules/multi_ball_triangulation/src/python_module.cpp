// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "dot_projector/dot_projector.hpp"
#include "multi_ball_triangulation/multi_ball_triangulation.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace bt = multi_ball_triangulation;
namespace py = ::pybind11;

PYBIND11_MODULE(python_module, m) {
  py::module::import("triangulation.python_module");
  py::module::import("dot_projector.python_module");

  py::class_<bt::MultiBallTriangulationParameters, triangulation::TriangulationParameters,
             dot_projector::IDotProjectorParameters, bt::MultiBallTriangulationParameters::SharedPtr>(
    m, "MultiBallTriangulationParameters", py::multiple_inheritance())
    .def(py::init())
    .def("initialize", &bt::MultiBallTriangulationParameters::Initialize)
    .def("is_initialized", &bt::MultiBallTriangulationParameters::IsInitialized)
    .def("print_parameters", &bt::MultiBallTriangulationParameters::PrintParameters)
    .def("read_parameters_from_file", &bt::MultiBallTriangulationParameters::ReadParametersFromFile)
    .def("write_parameters_to_file", &bt::MultiBallTriangulationParameters::WriteParametersToFile)
    .def_readwrite("timeout_ms", &bt::MultiBallTriangulationParameters::timeout_ms)
    .def_readwrite("use_variable_response_time", &bt::MultiBallTriangulationParameters::use_variable_response_time)
    .def_readwrite("camera_buffer_length", &bt::MultiBallTriangulationParameters::camera_buffer_length)
    .def_readwrite("worker_queue_length", &bt::MultiBallTriangulationParameters::worker_queue_length);
}
