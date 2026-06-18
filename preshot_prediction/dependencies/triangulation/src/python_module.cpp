// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "triangulation/triangulation.hpp"

namespace py = ::pybind11;
namespace tri = ::triangulation;
namespace cal = ::calibration;

PYBIND11_MODULE(python_module, m) {
  py::enum_<tri::TriangulationParameters::GhostBallsPolicy>(m, "GhostBallsPolicy")
    .value("none", tri::TriangulationParameters::GhostBallsPolicy::kNone)
    .value("drop", tri::TriangulationParameters::GhostBallsPolicy::kDrop)
    .value("merge", tri::TriangulationParameters::GhostBallsPolicy::kMerge)
    .export_values();

  py::class_<tri::TriangulatedPoint, tri::TriangulatedPoint::SharedPtr>(m, "TriangulatedPoint")
    .def(py::init())
    .def_readwrite("position", &tri::TriangulatedPoint::position)
    .def_readwrite("covariance", &tri::TriangulatedPoint::covariance)
    .def_readwrite("observation_indices", &tri::TriangulatedPoint::observation_indices)
    .def_readwrite("reprojection_error", &tri::TriangulatedPoint::reprojection_error);

  py::class_<tri::TriangulationParameters, tri::TriangulationParameters::SharedPtr>(m, "TriangulationParameters")
    .def(py::init())
    .def_readwrite("geometric_error_threshold", &tri::TriangulationParameters::geometric_error_threshold)
    .def_readwrite("radius_error_threshold", &tri::TriangulationParameters::radius_error_threshold)
    .def_readwrite("covariance_model", &tri::TriangulationParameters::covariance_model)
    .def_readwrite("use_nonlinear_refinement", &tri::TriangulationParameters::use_nonlinear_refinement)
    .def_readwrite("tol_position", &tri::TriangulationParameters::tol_position)
    .def_readwrite("tol_reprojection_error", &tri::TriangulationParameters::tol_reprojection_error)
    .def_readwrite("observation_variance", &tri::TriangulationParameters::observation_variance)
    .def_readwrite("max_iterations", &tri::TriangulationParameters::max_iterations)
    .def_readwrite("similarity_threshold", &tri::TriangulationParameters::similarity_threshold)
    .def_readwrite("ghost_balls_policy", &tri::TriangulationParameters::ghost_balls_policy);

  py::class_<tri::Triangulation<Eigen::Vector2f>, tri::Triangulation<Eigen::Vector2f>::SharedPtr>(m, "Triangulation")
    .def(py::init<tri::TriangulationParameters::SharedPtr, cal::CameraCalibrationParameters::SharedPtr,
                  ::datalogger::DataWriter::SharedPtr>())
    .def("set_cameras", &tri::Triangulation<Eigen::Vector2f>::SetCameras)
    .def("triangulate_uncertain_points", &tri::Triangulation<Eigen::Vector2f>::TriangulateUncertainPoints)
    .def("triangulate_points", &tri::Triangulation<Eigen::Vector2f>::TriangulatePoints)
    .def("triangulate_sorted_points", &tri::Triangulation<Eigen::Vector2f>::TriangulateSortedPoints);
  py::class_<tri::Triangulation<Eigen::Vector3f>, tri::Triangulation<Eigen::Vector3f>::SharedPtr>(m, "Triangulation3")
    .def(py::init<tri::TriangulationParameters::SharedPtr, cal::CameraCalibrationParameters::SharedPtr,
                  ::datalogger::DataWriter::SharedPtr>())
    .def("set_cameras", &tri::Triangulation<Eigen::Vector3f>::SetCameras)
    .def("triangulate_uncertain_points", &tri::Triangulation<Eigen::Vector3f>::TriangulateUncertainPoints)
    .def("triangulate_points", &tri::Triangulation<Eigen::Vector3f>::TriangulatePoints)
    .def("triangulate_sorted_points", &tri::Triangulation<Eigen::Vector3f>::TriangulateSortedPoints);
}
