// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "player_pose/datalogger.hpp"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = ::pybind11;

namespace player_pose::datalogger {

PYBIND11_MODULE(datalogger_pybind, m) {
  py::module::import("ace_loggers.datalogger_pybind");
  py::module::import("player_pose.python_module");

  py::enum_<::datalogger::PlayerPoseDataType>(m, "type")
    .value("player_bbox_detection", ::datalogger::PlayerPoseDataType::kPlayerBBoxDetection)
    .value("player_pose_detection", ::datalogger::PlayerPoseDataType::kPlayerPoseDetection)
    .value("player_pose_estimate", ::datalogger::PlayerPoseDataType::kPlayerPoseEstimate)
    .value("player_frame_features", ::datalogger::PlayerPoseDataType::kPlayerFrameFeatures);

  py::class_<DataWriter, DataWriter::SharedPtr, ::datalogger::DataWriter>(m, "DataWriter")
    .def(py::init<std::string, std::vector<int>>(), py::arg("file_name"), py::arg("player_ids"))
    .def("write", &DataWriter::Write<perception::PlayerBBoxDetection>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<perception::PlayerPoseDetection>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<perception::PlayerPoseEstimate>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<perception::PlayerFrameFeatures>, py::arg("data").noconvert());

  py::class_<DataReader, DataReader::SharedPtr, ::datalogger::DataReader>(m, "DataReader")
    .def(py::init<std::string>(), py::arg("file_name"))
    .def("read_enum", &DataReader::ReadAndReturn<::datalogger::PlayerPoseDataType>)
    .def("read_bbox_detection", &DataReader::ReadAndReturn<perception::PlayerBBoxDetection>)
    .def("read_pose_detection", &DataReader::ReadAndReturn<perception::PlayerPoseDetection>)
    .def("read_pose_estimate", &DataReader::ReadAndReturn<perception::PlayerPoseEstimate>)
    .def("read_frame_features", &DataReader::ReadAndReturn<perception::PlayerFrameFeatures>);

}  // PYBIND11_MODULE

}  // namespace player_pose::datalogger
