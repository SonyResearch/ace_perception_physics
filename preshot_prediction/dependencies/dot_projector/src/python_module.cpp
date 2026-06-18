// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "dot_projector/dot_projector.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace dp = dot_projector;
namespace py = ::pybind11;

PYBIND11_MODULE(python_module, m) {
  py::class_<dp::IDotProjectorParameters, dp::IDotProjectorParameters::SharedPtr>(m, "IDotProjectorParameters")
    .def(py::init())
    .def_readwrite("projector_enable", &dp::IDotProjectorParameters::projector_enable)
    .def_readwrite("projector_ball_radius", &dp::IDotProjectorParameters::projector_ball_radius);

  py::class_<dp::DotProjector, dp::DotProjector::SharedPtr>(m, "DotProjector")
    .def(py::init())
    .def("set_parameters", &dp::DotProjector::SetParameters)
    .def("project_markers",
         [](dp::DotProjector::SharedPtr& self,
            const py::array_t<float, py::array::c_style | py::array::forcecast>& distorted_points,
            const calibration::Camera& camera,
            const py::array_t<float, py::array::c_style | py::array::forcecast>& triangulated_point) {
           if (distorted_points.ndim() != 2 || distorted_points.shape(1) != 2) {
             throw std::runtime_error("distorted_points must be a 2D-points list");
           }
           const Eigen::Map<const Eigen::Matrix2Xf> distorted_points_eigen(distorted_points.data(), 2,
                                                                           distorted_points.shape(0));
           const Eigen::Vector3f triangulated_point_eigen(triangulated_point.data());

           std::vector<Eigen::Vector3f> projected_points;
           self->ProjectMarkers(distorted_points_eigen, camera, triangulated_point_eigen, projected_points);
           return py::array_t<float, py::array::c_style>({projected_points.size(), static_cast<size_t>(3)},
                                                         reinterpret_cast<const float*>(projected_points.data()));
         });
}
