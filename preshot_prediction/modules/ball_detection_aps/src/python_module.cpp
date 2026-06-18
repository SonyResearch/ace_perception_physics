// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ball_detection_aps/ball_detection_aps.hpp"
#include "ball_detector/ball_detector_opencv.hpp"
#include "dot_detector/dot_detector_opencv.hpp"
#include "dot_detector/dot_detector_opencv_parameters.hpp"
#include "eigen3/Eigen/Eigen"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace bdaps = ball_detection_aps;
namespace bd = ball_detector;
namespace dd = dot_detector;
namespace py = ::pybind11;

PYBIND11_MODULE(python_module, m) {
  py::module::import("ball_detector.python_module");
  py::module::import("dot_detector.python_module");

  py::enum_<bdaps::ImageType>(m, "ImageType")
    .value("RAW", bdaps::ImageType::kRaw)
    .value("DETECTION", bdaps::ImageType::kDetection)
    .value("VALID_MASK", bdaps::ImageType::kValidMask)
    .export_values();

  py::class_<bdaps::BallDetectionAPSParameters, bd::IBallDetectorParameters, dd::IDotDetectorParameters,
             dd::DotDetectorOpenCVParameters, bdaps::BallDetectionAPSParameters::SharedPtr>(
    m, "BallDetectionParameters", py::multiple_inheritance())
    .def(py::init())
    .def("initialize", &bdaps::BallDetectionAPSParameters::Initialize)
    .def("is_initialized", &bdaps::BallDetectionAPSParameters::IsInitialized)
    .def("print_parameters", &bdaps::BallDetectionAPSParameters::PrintParameters)
    .def("read_parameters_from_file", &bdaps::BallDetectionAPSParameters::ReadParametersFromFile)
    .def("write_parameters_to_file", &bdaps::BallDetectionAPSParameters::WriteParametersToFile)
    .def_readwrite("cuda_device_uuids", &bdaps::BallDetectionAPSParameters::cuda_device_uuids)
    .def_readwrite("blur_kernel_size", &bdaps::BallDetectionAPSParameters::blur_kernel_size)
    .def_readwrite("hsv_lower_boundary", &bdaps::BallDetectionAPSParameters::hsv_lower_boundary)
    .def_readwrite("hsv_upper_boundary", &bdaps::BallDetectionAPSParameters::hsv_upper_boundary)
    .def_readwrite("motion_filter_enable", &bdaps::BallDetectionAPSParameters::motion_filter_enable)
    .def_readwrite("motion_filter_delay", &bdaps::BallDetectionAPSParameters::motion_filter_delay)
    .def_readwrite("motion_filter_lower_boundary", &bdaps::BallDetectionAPSParameters::motion_filter_lower_boundary)
    .def_readwrite("motion_filter_upper_boundary", &bdaps::BallDetectionAPSParameters::motion_filter_upper_boundary)
    .def_readwrite("min_circularity_ratio", &bdaps::BallDetectionAPSParameters::min_circularity_ratio)
    .def_readwrite("min_radius", &bdaps::BallDetectionAPSParameters::min_radius)
    .def_readwrite("camera_names", &bdaps::BallDetectionAPSParameters::camera_names);
}
