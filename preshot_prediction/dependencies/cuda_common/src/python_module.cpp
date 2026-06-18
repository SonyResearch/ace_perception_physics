// Confidential, Copyright 2025, Sony AI, All rights reserved.

#include "cuda_common/TRTEngine.hpp"
#include "cuda_common/cuda_common.hpp"
#include "pybind11/complex.h"
#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "vision_common/pybind11/opencv.hpp"

namespace py = pybind11;

PYBIND11_MODULE(python_module, m) {
  m.def("get_cuda_device_id", &cuda_common::GetCudaDeviceID);

  py::enum_<cuda_common::TRTEngineOptions::Precision>(m, "EnginePrecision")
    .value("FP32", cuda_common::TRTEngineOptions::Precision::kFP32)
    .value("FP16", cuda_common::TRTEngineOptions::Precision::kFP16)
    .value("INT8", cuda_common::TRTEngineOptions::Precision::kINT8)
    .export_values();
  py::enum_<cuda_common::TRTEngineOptions::InputFormat>(m, "EngineInputFormat")
    .value("BHWC", cuda_common::TRTEngineOptions::InputFormat::kBHWC)
    .value("BCHW", cuda_common::TRTEngineOptions::InputFormat::kBCHW)
    .value("Array", cuda_common::TRTEngineOptions::InputFormat::kArray)
    .export_values();

  py::class_<cuda_common::TRTEngineOptions>(m, "TRTEngineOptions")
    .def(py::init<>())
    .def_readwrite("precision", &cuda_common::TRTEngineOptions::precision)
    .def_readwrite("input_format", &cuda_common::TRTEngineOptions::input_format)
    .def_readwrite("opt_batch_size", &cuda_common::TRTEngineOptions::opt_batch_size)
    .def_readwrite("max_batch_size", &cuda_common::TRTEngineOptions::max_batch_size)
    .def_readwrite("opt_width", &cuda_common::TRTEngineOptions::opt_width)
    .def_readwrite("opt_height", &cuda_common::TRTEngineOptions::opt_height)
    .def_readwrite("device_index", &cuda_common::TRTEngineOptions::device_index);

  py::class_<cuda_common::TRTEngine, cuda_common::TRTEngine::SharedPtr>(m, "TRTEngine")
    .def(py::init<const cuda_common::TRTEngineOptions&>())
    .def("build", &cuda_common::TRTEngine::Build)
    .def("load_network", &cuda_common::TRTEngine::LoadNetwork)
    .def("build_and_load", &cuda_common::TRTEngine::BuildAndLoadNetwork)
    .def("get_engine_name", &cuda_common::TRTEngine::GetEngineName)
    .def("get_options", &cuda_common::TRTEngine::GetOptions)
    .def("get_output", &cuda_common::TRTEngine::GetOutputVector)
    .def("warmup", &cuda_common::TRTEngine::Warmup)
    .def("set_input", &cuda_common::TRTEngine::SetInputVector)
    .def("run_inference", &cuda_common::TRTEngine::RunInference)
    .def("inference_single",
         [](cuda_common::TRTEngine* self, std::vector<cv::MatND>& inputs_vec) {
           for (size_t i = 0; i < inputs_vec.size(); ++i) {
             self->SetInputVector(static_cast<int>(i), {inputs_vec[i]});
           }
           return self->RunInference(1);
         })
    .def("inference_batch",
         [](cuda_common::TRTEngine* self, const std::vector<std::vector<cv::MatND>>& inputs_vec) {
           for (size_t i = 0; i < inputs_vec.size(); ++i) {
             self->SetInputVector(static_cast<int>(i), inputs_vec[i]);
           }
           return self->RunInference(static_cast<int>(inputs_vec[0].size()));
         })
    .def("get_input_dims",
         [](cuda_common::TRTEngine* self) {
           py::list dims;

           const auto& vec = self->GetInputDims();
           for (const auto& d : vec) {
             py::list x;
             for (int i = 0; i < d.nbDims; ++i) {
               x.append(d.d[i]);
             }

             dims.append(x);
           }
           return dims;
         })
    .def("get_output_dims",
         [](cuda_common::TRTEngine* self) {
           py::list dims;

           const auto& vec = self->GetOutputDims();
           for (const auto& d : vec) {
             py::list x;
             for (int i = 0; i < d.nbDims; ++i) {
               x.append(d.d[i]);
             }

             dims.append(x);
           }
           return dims;
         })
    .def("get_input_index", &cuda_common::TRTEngine::GetInputIndex)
    .def("get_output_index", &cuda_common::TRTEngine::GetOutputIndex);
}
