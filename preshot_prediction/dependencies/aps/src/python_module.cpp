// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "aps/camera_parameters.hpp"
#include "aps/multi_camera_parameters.hpp"
#include "aps/recorder_parameters.hpp"
#include "aps/zero_copy_subscriber.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = ::pybind11;

PYBIND11_MODULE(python_module, m) {
  py::class_<aps::CameraParameters, aps::CameraParameters::SharedPtr>(m, "CameraParameters")
    .def(py::init())
    .def("initialize", &aps::CameraParameters::Initialize)
    .def("is_initialized", &aps::CameraParameters::IsInitialized)
    .def("print_parameters", &aps::CameraParameters::PrintParameters)
    .def("read_parameters_from_file", &aps::CameraParameters::ReadParametersFromFile)
    .def("write_parameters_to_file", &aps::CameraParameters::WriteParametersToFile)
    .def_readwrite("acquisition_mode", &aps::CameraParameters::acquisition_mode)
    .def_readwrite("exposure_mode", &aps::CameraParameters::exposure_mode)
    .def_readwrite("exposure_time", &aps::CameraParameters::exposure_time)
    .def_readwrite("exposure_auto", &aps::CameraParameters::exposure_auto)
    .def_readwrite("acquisition_frame_rate", &aps::CameraParameters::acquisition_frame_rate)
    .def_readwrite("acquisition_frame_rate_enable", &aps::CameraParameters::acquisition_frame_rate_enable)
    .def_readwrite("gain", &aps::CameraParameters::gain)
    .def_readwrite("gain_auto", &aps::CameraParameters::gain_auto)
    .def_readwrite("black_level", &aps::CameraParameters::black_level)
    .def_readwrite("black_level_clamping_enable", &aps::CameraParameters::black_level_clamping_enable)
    .def_readwrite("balance_ratio_blue", &aps::CameraParameters::balance_ratio_blue)
    .def_readwrite("balance_ratio_red", &aps::CameraParameters::balance_ratio_red)
    .def_readwrite("balance_white_auto", &aps::CameraParameters::balance_white_auto)
    .def_readwrite("gamma", &aps::CameraParameters::gamma)
    .def_readwrite("gamma_enable", &aps::CameraParameters::gamma_enable)
    .def_readwrite("width", &aps::CameraParameters::width)
    .def_readwrite("height", &aps::CameraParameters::height)
    .def_readwrite("offset_x", &aps::CameraParameters::offset_x)
    .def_readwrite("offset_y", &aps::CameraParameters::offset_y)
    .def_readwrite("pixel_format", &aps::CameraParameters::pixel_format)
    .def_readwrite("adc_bit_depth", &aps::CameraParameters::adc_bit_depth)
    .def_readwrite("chunk_mode_active", &aps::CameraParameters::chunk_mode_active)
    .def_readwrite("chunk_exposure_time", &aps::CameraParameters::chunk_exposure_time)
    .def_readwrite("manual_stream_buffer_count", &aps::CameraParameters::manual_stream_buffer_count)
    .def_readwrite("stream_buffer_count_mode", &aps::CameraParameters::stream_buffer_count_mode)
    .def_readwrite("stream_buffer_handling_mode", &aps::CameraParameters::stream_buffer_handling_mode);

  py::class_<aps::MultiCameraParameters, aps::MultiCameraParameters::SharedPtr>(m, "MultiCameraParameters")
    .def(py::init())
    .def("initialize", &aps::MultiCameraParameters::Initialize)
    .def("is_initialized", &aps::MultiCameraParameters::IsInitialized)
    .def("print_parameters", &aps::MultiCameraParameters::PrintParameters)
    .def("read_parameters_from_file", &aps::MultiCameraParameters::ReadParametersFromFile)
    .def("write_parameters_to_file", &aps::MultiCameraParameters::WriteParametersToFile)
    .def_readwrite("software_trigger", &aps::MultiCameraParameters::software_trigger)
    .def_readwrite("first_as_master", &aps::MultiCameraParameters::first_as_master)
    .def_readwrite("serial_numbers", &aps::MultiCameraParameters::serial_numbers);

  py::class_<aps::ZeroCopySubscriberPy, aps::ZeroCopySubscriberPy::SharedPtr>(m, "ZeroCopySubscriber")
    .def(py::init<>())
    .def("is_initialized", &aps::ZeroCopySubscriberPy::IsInitialized)
    .def("add_subscription", &aps::ZeroCopySubscriberPy::AddSubscription)
    .def("get_latest_message", &aps::ZeroCopySubscriberPy::GetLatestMessage)
    .def("release_latest_message", &aps::ZeroCopySubscriberPy::ReleaseLatestMessage);

  py::class_<aps::RecorderParameters, aps::RecorderParameters::SharedPtr>(m, "RecorderParameters")
    .def(py::init())
    .def("initialize", &aps::RecorderParameters::Initialize)
    .def("is_initialized", &aps::RecorderParameters::IsInitialized)
    .def("print_parameters", &aps::RecorderParameters::PrintParameters)
    .def("read_parameters_from_file", &aps::RecorderParameters::ReadParametersFromFile)
    .def("write_parameters_to_file", &aps::RecorderParameters::WriteParametersToFile)
    .def_readwrite("cuda_device_uuids", &aps::RecorderParameters::cuda_device_uuids);
}
