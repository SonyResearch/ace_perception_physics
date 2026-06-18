// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "racket_pose_estimation/datalogger.hpp"

namespace py = ::pybind11;

namespace racket_pose_estimation::datalogger {

PYBIND11_MODULE(datalogger_pybind, m) {
  py::module::import("ace_loggers.datalogger_pybind");
  py::module::import("racket_pose_estimation.python_module");

  py::enum_<::datalogger::RacketPoseEstimationDataType>(m, "type")
    .value("racket_pose_features", ::datalogger::RacketPoseEstimationDataType::kRacketPoseFeatures)
    .value("racket_estimated_pose", ::datalogger::RacketPoseEstimationDataType::kRacketEstimatedPose)
    .value("racket_frame_features", ::datalogger::RacketPoseEstimationDataType::kRacketFrameFeatures);

  py::class_<DataWriter, DataWriter::SharedPtr, ::datalogger::DataWriter>(m, "DataWriter")
    .def(py::init<std::string, std::vector<int>>(), py::arg("file_name"), py::arg("racket_ids"))
    .def("write", &DataWriter::Write<perception::RacketPoseFeatures>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<perception::RacketEstimatedPose>, py::arg("data").noconvert())
    .def("write", &DataWriter::Write<perception::RacketFrameFeatures>, py::arg("data").noconvert());

  py::class_<DataReader, DataReader::SharedPtr, ::datalogger::DataReader>(m, "DataReader")
    .def(py::init<std::string>(), py::arg("file_name"))
    .def("read_enum", &DataReader::ReadAndReturn<::datalogger::RacketPoseEstimationDataType>)
    .def("read_pose_features", &DataReader::ReadAndReturn<perception::RacketPoseFeatures>)
    .def("read_estimated_pose", &DataReader::ReadAndReturn<perception::RacketEstimatedPose>)
    .def("read_frame_features", &DataReader::ReadAndReturn<perception::RacketFrameFeatures>);

}  // PYBIND11_MODULE

}  // namespace racket_pose_estimation::datalogger
