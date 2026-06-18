// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include <algorithm>

#include "trt_ball_detector/ball_detector.hpp"
////
#include <vision_common/pybind11/opencv.hpp>

#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = ::pybind11;

PYBIND11_MODULE(python_module, m) {
  py::class_<trt_ball_detector::BallDetectorParameters, trt_ball_detector::BallDetectorParameters::SharedPtr>(
    m, "ball_detector_parameters")
    .def(py::init<>())
    .def_readwrite("onnx_engine_path", &trt_ball_detector::BallDetectorParameters::onnx_engine_path)
    .def_readwrite("batch_size", &trt_ball_detector::BallDetectorParameters::batch_size)
    .def_readwrite("device_id", &trt_ball_detector::BallDetectorParameters::device_id)
    .def_readwrite("copy_heatmap", &trt_ball_detector::BallDetectorParameters::copy_heatmap);

  py::class_<trt_ball_detector::BallDetector, trt_ball_detector::BallDetector::SharedPtr>(m, "ball_detector")
    .def(py::init<>())
    .def("initialize", &trt_ball_detector::BallDetector::Initialize)
    .def("reset_encoded_images", &trt_ball_detector::BallDetector::ResetEncodedImages)
    .def("encode_images", &trt_ball_detector::BallDetector::EncodeImages)
    .def("get_decoding_results", &trt_ball_detector::BallDetector::GetDecodingResults);

  py::class_<trt_ball_detector::DecodingResult, trt_ball_detector::DecodingResult::SharedPtr>(m, "decoding_result")
    .def(py::init<>())
    .def_readonly("sequence_id", &trt_ball_detector::DecodingResult::sequence_id)
    .def_readonly("confidences", &trt_ball_detector::DecodingResult::confidences)
    .def_readonly("ball_positions", &trt_ball_detector::DecodingResult::ball_positions)
    .def_readonly("ball_velocities", &trt_ball_detector::DecodingResult::ball_velocities)
    .def_readonly("radius", &trt_ball_detector::DecodingResult::radius)
    .def_readonly("heatmap", &trt_ball_detector::DecodingResult::heatmap);
}
