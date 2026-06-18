// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ball_detector/datalogger.hpp"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = ::pybind11;

namespace ball_detector::datalogger {

PYBIND11_MODULE(datalogger_pybind, m) {
  py::module::import("cv2");
  py::module::import("ace_loggers.datalogger_pybind");
  py::module::import("ball_detector.python_module");

  py::class_<DataWriter, DataWriter::SharedPtr, ::datalogger::DataWriter>(m, "DataWriter")
    .def(py::init<std::string, std::string, std::string>(), py::arg("file_name"), py::arg("camera_name"),
         py::arg("ball_detector_params"))
    .def("write", &DataWriter::Write<cv::Mat>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<std::vector<std::pair<cv::Point2f, float>>>, py::arg("data").noconvert());

  py::class_<DataReader, DataReader::SharedPtr, ::datalogger::DataReader>(m, "DataReader")
    .def(py::init<std::string>(), py::arg("file_name"))
    .def("read_uint64", &DataReader::ReadAndReturn<uint64_t>)
    .def("read_timestamp", &DataReader::ReadAndReturn<std::pair<int32_t, uint32_t>>)
    .def("read_roi",
         [](DataReader::SharedPtr& self) {
           const auto& rect = self->ReadAndReturn<cv::Rect>();
           return py::make_tuple(py::make_tuple(rect.x, rect.y),
                                 py::make_tuple(rect.x + rect.width, rect.y + rect.height));
         })
    .def("read_image",
         [](DataReader::SharedPtr& self) {
           const auto& mat = self->ReadAndReturn<cv::Mat>();
           if (mat.depth() != CV_8U) {
             throw std::runtime_error("Only unsigned-char image is supported for now!");
           }
           std::vector<int> sizes(&mat.size[0], &mat.size[0] + mat.dims);
           std::vector<size_t> steps(&mat.step[0], &mat.step[0] + mat.dims);
           return py::array_t<uint8_t, py::array::c_style>(sizes, steps, mat.data);
         })
    .def("read_image", &DataReader::ReadAndReturn<cv::Mat>)
    .def("read_ball_center",
         [](DataReader::SharedPtr& self) {
           const auto& pt = self->ReadAndReturn<cv::Point2f>();
           return py::make_tuple(pt.x, pt.y);
         })
    .def("read_ball_radius", &DataReader::ReadAndReturn<float>);

}  // PYBIND11_MODULE

}  // namespace ball_detector::datalogger
