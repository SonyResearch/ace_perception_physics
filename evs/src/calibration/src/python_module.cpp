// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "calibration/camera_calibration_parameters.hpp"
#include "calibration/robot_calibration_parameters.hpp"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = ::pybind11;
namespace cal = ::calibration;

namespace pybind11::detail {
template <typename Key, typename Value>
struct type_caster<ace_containers::IndexedMap<Key, Value>>
  : map_caster<ace_containers::IndexedMap<Key, Value>, Key, Value> {};
}  // namespace pybind11::detail

PYBIND11_MODULE(python_module, m) {
  py::class_<cal::Camera, cal::Camera::SharedPtr>(m, "Camera")
    .def(py::init())
    .def_readwrite("resolution", &cal::Camera::resolution)
    .def_property(
      "camera_model", [](const cal::Camera& self) { return cal::CameraModelToString(self.camera_model); },
      [](cal::Camera& self, const std::string& value) {
        self.camera_model = cal::StringToCameraModel(value);
        return true;
      })
    .def_readwrite("camera_matrix", &cal::Camera::camera_matrix)
    .def_readonly("camera_matrix_inv", &cal::Camera::camera_matrix_inv)
    .def_readwrite("distortion_enable", &cal::Camera::distortion_enable)
    .def_property(
      "distortion_model", [](const cal::Camera& self) { return cal::DistortionModelToString(self.distortion_model); },
      [](cal::Camera& self, const std::string& value) {
        self.distortion_model = cal::StringToDistortionModel(value);
        return true;
      })
    .def_readwrite("distortion_coeffs", &cal::Camera::distortion_coeffs)
    .def_readwrite("T_camera_world", &cal::Camera::T_camera_world)
    .def_readonly("T_world_camera", &cal::Camera::T_world_camera)
    .def_readonly("T_camera_origin", &cal::Camera::T_camera_origin)
    .def_readonly("T_origin_camera", &cal::Camera::T_origin_camera)
    .def_readonly("projection_matrix", &cal::Camera::projection_matrix)
    .def_readwrite("rotation", &cal::Camera::rotation)
    .def("distort_point", [](const cal::Camera& self, const Eigen::Vector2f& p_u) { return self.DistortPoint(p_u); })
    .def("distort_points", &cal::Camera::DistortPoints)
    .def("undistort_point", &cal::Camera::UndistortPoint)
    .def("undistort_points", &cal::Camera::UndistortPoints)
    .def("project_point", &cal::Camera::ProjectPointPybind)
    .def("project_points", &cal::Camera::ProjectPointsPybind)
    .def("project_point_extended", &cal::Camera::ProjectPointExtendedPybind)
    .def("update_internal_parameters", &cal::Camera::UpdateInternalParameters);

  py::class_<cal::MultiViewData, cal::MultiViewData::SharedPtr>(m, "MultiViewData")
    .def(py::init())
    .def_readonly("fundamental_matrix", &cal::MultiViewData::fundamental_matrix)
    .def_readonly("epipole_i", &cal::MultiViewData::epipole_i)
    .def_readonly("epipole_j", &cal::MultiViewData::epipole_j);

  py::class_<cal::CameraCalibrationParameters, cal::CameraCalibrationParameters::SharedPtr>(
    m, "CameraCalibrationParameters")
    .def(py::init())
    .def("initialize", &cal::CameraCalibrationParameters::Initialize)
    .def("is_initialized", &cal::CameraCalibrationParameters::IsInitialized)
    .def("print_parameters", &cal::CameraCalibrationParameters::PrintParameters)
    .def("read_parameters_from_file", &cal::CameraCalibrationParameters::ReadParametersFromFile)
    .def("write_parameters_to_file", &cal::CameraCalibrationParameters::WriteParametersToFile)
    .def_readonly("camera_names", &cal::CameraCalibrationParameters::camera_names)
    .def_readonly("cameras", &cal::CameraCalibrationParameters::cameras)
    .def_readonly("multiview_data", &cal::CameraCalibrationParameters::multiview_data)
    .def_readwrite("T_world_origin", &cal::CameraCalibrationParameters::T_world_origin)
    .def_readonly("T_origin_world", &cal::CameraCalibrationParameters::T_origin_world)
    .def("update_internal_parameters", &cal::CameraCalibrationParameters::UpdateInternalParameters);

  py::class_<cal::RobotCalibrationParameters, cal::RobotCalibrationParameters::SharedPtr>(m,
                                                                                          "RobotCalibrationParameters")
    .def(py::init())
    .def("initialize", &cal::RobotCalibrationParameters::Initialize)
    .def("is_initialized", &cal::RobotCalibrationParameters::IsInitialized)
    .def("print_parameters", &cal::RobotCalibrationParameters::PrintParameters)
    .def("read_parameters_from_file", &cal::RobotCalibrationParameters::ReadParametersFromFile)
    .def("write_parameters_to_file", &cal::RobotCalibrationParameters::WriteParametersToFile)
    .def_readwrite("T_robot_origin", &cal::RobotCalibrationParameters::T_robot_origin)
    .def_readonly("T_origin_robot", &cal::RobotCalibrationParameters::T_origin_robot)
    .def("update_internal_parameters", &cal::RobotCalibrationParameters::UpdateInternalParameters);
}
