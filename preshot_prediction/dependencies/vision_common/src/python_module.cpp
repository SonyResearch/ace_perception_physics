// Confidential, Copyright 2024, Sony AI, All rights reserved.

// TODO(cv3d): Implement me

#include "pybind11/complex.h"
#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "vision_common/estimators/SpinEstimator.hpp"
#include "vision_common/estimators/VelocityEstimator.hpp"
namespace py = pybind11;

PYBIND11_MODULE(python_module, m) {
  py::class_<vision_common::SpinEstimator, vision_common::SpinEstimator::SharedPtr>(m, "spin_estimator")
    .def(py::init<>())
    .def("initialize_yaml", &vision_common::SpinEstimator::InitializeYaml)
    .def("initialize", &vision_common::SpinEstimator::Initialize)
    .def("estimate_spin",
         [](vision_common::SpinEstimator* self, double timestamp, const std::vector<double>& quat) {
           return self->EstimateSpin(timestamp, Eigen::Quaterniond(quat[3], quat[0], quat[1], quat[2]));
         })
    .def("get_last_spin", &vision_common::SpinEstimator::GetLastSpin)
    .def("get_last_covariance", &vision_common::SpinEstimator::GetLastCovariance)
    .def("reset", &vision_common::SpinEstimator::Reset);

  py::class_<vision_common::VelocityEstimator, vision_common::VelocityEstimator::SharedPtr>(m, "velocity_estimator")
    .def(py::init<>())
    .def("initialize_yaml", &vision_common::VelocityEstimator::InitializeYaml)
    .def("initialize", &vision_common::VelocityEstimator::Initialize)
    .def("estimate_velocity", &vision_common::VelocityEstimator::EstimateVelocity)
    .def("get_last_velocity", &vision_common::VelocityEstimator::GetLastVelocity)
    .def("get_last_covariance", &vision_common::VelocityEstimator::GetLastCovariance)
    .def("reset", &vision_common::VelocityEstimator::Reset);
}
