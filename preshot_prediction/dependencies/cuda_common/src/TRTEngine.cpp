// Confidential, Copyright 2025, Sony AI, All rights reserved.

#include "cuda_common/TRTEngine.hpp"

#include <cuda_fp16.h>  // __half, __half2float — FP16→FP32 conversion in UnpackPinnedOutputs
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <random>
#include <utility>

#include "NvOnnxParser.h"
#include "ace_loggers/ace_loggers.hpp"

namespace util {
uint32_t checksum(const std::string& path) {
  std::ifstream file;
  file.open(path);
  uint32_t checksum = 0;
  unsigned shift = 0;
  for (uint32_t ch = file.get(); file; ch = file.get()) {
    checksum += (ch << shift);
    shift += 8;
    if (shift == 32) {
      shift = 0;
    }
  }
  return checksum;
}

inline bool doesFileExist(const std::string& filepath) {
  std::ifstream f(filepath.c_str());
  return f.good();
}

inline bool checkCudaErrorCode(cudaError_t code) {
  if (code != 0) {
    std::string err_msg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                          "), with message: " + cudaGetErrorString(code);
    LOG(ERROR) << err_msg;
    return false;
  }
  return true;
}

std::vector<std::string> getFilesInDirectory(const std::string& dir_path) {
  std::vector<std::string> filepaths;
  for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
    filepaths.emplace_back(entry.path().string());
  }
  return filepaths;
}

// RAII guard: ensures the correct CUDA device is current for the duration of the scope.
// cudaSetDevice() is a per-thread property. Without re-asserting it at every CUDA/TRT
// entry point the active device can silently change when a thread switches between multiple
// TRTEngine instances targeting different GPUs, causing buffers to be freed on the wrong
// device or inference to run on the wrong GPU.
struct ScopedCudaDevice {
  explicit ScopedCudaDevice(int device_index) {
    // cv::cuda::setDevice(device_index);
    checkCudaErrorCode(cudaSetDevice(device_index));
  }
};
}  // namespace util

namespace cuda_common {

class TRTEngineGlobals {
 public:
  int tensor_cv_type_map_3[10];
  int tensor_cv_type_map_1[10];
  int tensor_size_type_map[10];

  TRTEngineGlobals() {
    tensor_cv_type_map_3[static_cast<int>(nvinfer1::DataType::kFLOAT)] = CV_32FC3;
    tensor_cv_type_map_3[static_cast<int>(nvinfer1::DataType::kHALF)] = CV_16FC3;
    tensor_cv_type_map_3[static_cast<int>(nvinfer1::DataType::kBOOL)] = CV_8UC3;
    tensor_cv_type_map_3[static_cast<int>(nvinfer1::DataType::kINT8)] = CV_8SC3;
    tensor_cv_type_map_3[static_cast<int>(nvinfer1::DataType::kINT32)] = CV_32SC3;
    tensor_cv_type_map_3[static_cast<int>(nvinfer1::DataType::kINT64)] = CV_64FC3;
    tensor_cv_type_map_3[static_cast<int>(nvinfer1::DataType::kUINT8)] = CV_8UC3;

    tensor_cv_type_map_1[static_cast<int>(nvinfer1::DataType::kFLOAT)] = CV_32F;
    tensor_cv_type_map_1[static_cast<int>(nvinfer1::DataType::kHALF)] = CV_16F;
    tensor_cv_type_map_1[static_cast<int>(nvinfer1::DataType::kBOOL)] = CV_8U;
    tensor_cv_type_map_1[static_cast<int>(nvinfer1::DataType::kINT8)] = CV_8S;
    tensor_cv_type_map_1[static_cast<int>(nvinfer1::DataType::kINT64)] = CV_64F;
    tensor_cv_type_map_1[static_cast<int>(nvinfer1::DataType::kINT32)] = CV_32S;
    tensor_cv_type_map_1[static_cast<int>(nvinfer1::DataType::kUINT8)] = CV_8U;

    tensor_size_type_map[static_cast<int>(nvinfer1::DataType::kFLOAT)] = 4;
    tensor_size_type_map[static_cast<int>(nvinfer1::DataType::kHALF)] = 2;
    tensor_size_type_map[static_cast<int>(nvinfer1::DataType::kBOOL)] = 1;
    tensor_size_type_map[static_cast<int>(nvinfer1::DataType::kINT8)] = 1;
    tensor_size_type_map[static_cast<int>(nvinfer1::DataType::kINT64)] = 8;
    tensor_size_type_map[static_cast<int>(nvinfer1::DataType::kINT32)] = 4;
    tensor_size_type_map[static_cast<int>(nvinfer1::DataType::kUINT8)] = 1;
  }
} g_trt_globals;

void TRTEngine::Logger::log(Severity severity, const char* msg) noexcept {
  // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
  // For the sake of this tutorial, will just log to the console.

  // Only log Warnings or more important.
  if (severity <= Severity::kWARNING) {
    LOG(WARNING) << msg;
  }
}

TRTEngine::TRTEngine(const TRTEngineOptions& options) : m_options_(std::move(options)) {}

TRTEngine::~TRTEngine() {
  // Bind the engine's device so that cudaFree/cudaFreeHost/cudaStreamDestroy all target
  // the correct GPU. Without this, if the owning thread has since switched its active device
  // (or the object is destroyed on a different thread), resources leak on the wrong device.
  util::ScopedCudaDevice guard(m_options_.device_index);
  // Free device-side GPU buffers
  for (auto& buffer : m_input_buffers_) {
    util::checkCudaErrorCode(cudaFree(buffer));
  }
  for (auto& buffer : m_output_buffers_) {
    util::checkCudaErrorCode(cudaFree(buffer));
  }
  // Free page-locked host staging buffers used for coalesced D2H copies
  for (void* buf : m_output_pinned_buffers_) {
    if (buf) {
      util::checkCudaErrorCode(cudaFreeHost(buf));
    }
  }
  // Destroy any streams still in the pool (would otherwise leak)
  for (auto s : cuda_streams_) {
    util::checkCudaErrorCode(cudaStreamDestroy(s));
  }
  m_input_buffers_.clear();
  m_output_buffers_.clear();
  m_output_pinned_buffers_.clear();
  cuda_streams_.clear();
  if (m_cuda_graph_exec_) {
    cudaGraphExecDestroy(m_cuda_graph_exec_);
    m_cuda_graph_exec_ = nullptr;
  }
}

bool TRTEngine::Build(std::string onnx_model_path) {
  // Only regenerate the engine file if it has not already been generated for the specified options
  m_engine_name_ = SerializeEngineOptions(m_options_, onnx_model_path);
  LOG(INFO) << "Searching for engine file with name: " << m_engine_name_;

  if (util::doesFileExist(m_engine_name_)) {
    LOG(INFO) << "TRTEngine found, not regenerating...";
    return true;
  }

  if (!util::doesFileExist(onnx_model_path)) {
    throw std::runtime_error("Could not find model at path: " + onnx_model_path);
  }

  // Was not able to find the engine file, generate...

  LOG(WARNING) << "\033[1;31mTRTEngine not found, generating. This could take a while...\033[0m \U000023F3";

  // Set the device index
  cv::cuda::setDevice(m_options_.device_index);
  auto ret = cudaSetDevice(m_options_.device_index);
  if (ret != 0) {
    int num_gp_us;
    cudaGetDeviceCount(&num_gp_us);
    auto err_msg = "Unable to set GPU device index to: " + std::to_string(m_options_.device_index) +
                   ". Note, your device has " + std::to_string(num_gp_us) + " CUDA-capable GPU(s).";
    throw std::runtime_error(err_msg);
  }
  // Create our engine builder.
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger_));
  if (!builder) {
    return false;
  }

  // Define an explicit batch size and then create the network (implicit batch size is deprecated).
  // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
  auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) {
    return false;
  }

  // Create a parser for reading the onnx file.
  auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger_));
  if (!parser) {
    return false;
  }

  // We are going to first read the onnx file into memory, then pass that buffer to the parser.
  // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
  std::ifstream file(onnx_model_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Unable to read engine file");
  }

  // Parse the buffer we read into memory.
  auto parsed = parser->parse(buffer.data(), buffer.size());
  if (!parsed) {
    return false;
  }

  // Ensure that all the inputs have the same batch size
  const auto num_inputs = network->getNbInputs();
  if (num_inputs < 1) {
    throw std::runtime_error("Error, model needs at least 1 input!");
  }
  const auto input0_batch = network->getInput(0)->getDimensions().d[0];
  for (int32_t i = 1; i < num_inputs; ++i) {
    if (network->getInput(i)->getDimensions().d[0] != input0_batch) {
      throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
    }
  }

  // Check to see if the model supports dynamic batch size or not
  bool does_support_dynamic_batch = false;
  if (input0_batch == -1) {
    does_support_dynamic_batch = true;
    LOG(INFO) << "Model supports dynamic batch size";
  } else {
    LOG(INFO) << "Model only supports fixed batch size of " << input0_batch;
    // If the model supports a fixed batch size, ensure that the maxBatchSize and optBatchSize were set correctly.
    if (m_options_.opt_batch_size != input0_batch || m_options_.max_batch_size != input0_batch) {
      throw std::runtime_error("Error, model only supports a fixed batch size of " + std::to_string(input0_batch) +
                               ". Must set Options.optBatchSize and Options.maxBatchSize to " +
                               std::to_string(input0_batch));
    }
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  // Register a single optimization profile
  nvinfer1::IOptimizationProfile* opt_profile = builder->createOptimizationProfile();
  for (int32_t i = 0; i < num_inputs; ++i) {
    // Must specify dimensions for all the inputs the model expects.
    auto* const input = network->getInput(i);
    const auto* const input_name = input->getName();
    auto dims = input->getDimensions();
    // int c_index = 1;
    int h_index = 2;
    int w_index = 3;

    if (m_options_.input_format == TRTEngineOptions::InputFormat::kBHWC) {
      h_index = 1;
      w_index = 2;
      // c_index = 3;
    }

    if (dims.d[w_index] == -1) {
      dims.d[w_index] = m_options_.opt_width;
    }
    if (dims.d[h_index] == -1) {
      dims.d[h_index] = m_options_.opt_height;
    }
    {
      LOG(INFO) << "Processing: " << input_name;
      LOG(INFO) << "\t#Dims: " << dims.nbDims;
      std::string dim_str;
      for (int i = 0; i < dims.nbDims; ++i) {
        dim_str += std::to_string(dims.d[i]) + ", ";
      }
      LOG(INFO) << "\tShape: " << dim_str;
    }
    // Specify the optimization profile`
    if (does_support_dynamic_batch) {
      dims.d[0] = 1;
      opt_profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, dims);
    } else {
      dims.d[0] = m_options_.opt_batch_size;
      opt_profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, dims);
    }
    dims.d[0] = m_options_.opt_batch_size;
    opt_profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, dims);
    dims.d[0] = m_options_.max_batch_size;
    opt_profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, dims);
  }
  config->addOptimizationProfile(opt_profile);

  // Set the precision level
  if (m_options_.precision == TRTEngineOptions::Precision::kFP16) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (m_options_.precision == TRTEngineOptions::Precision::kINT8) {
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
  } else if (m_options_.precision == TRTEngineOptions::Precision::kFP32) {
    // no need to set flags
  } else {
    throw std::runtime_error("TRTEngineOptions provided precision is Not supported");
  }

  // CUDA stream used for profiling by the builder.
  cudaStream_t profile_stream;
  util::checkCudaErrorCode(cudaStreamCreate(&profile_stream));
  config->setProfileStream(profile_stream);

  // Build the engine
  // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
  // Doing so will provide you with more information on why exactly it is failing.
  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return false;
  }

  // Write the engine to disk
  std::ofstream outfile(m_engine_name_, std::ofstream::binary);
  outfile.write(reinterpret_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));

  LOG(INFO) << "Success, saved engine to " << m_engine_name_;

  util::checkCudaErrorCode(cudaStreamDestroy(profile_stream));
  return true;
}

bool TRTEngine::LoadNetwork() {
  // Read the serialized model from disk
  std::ifstream file(m_engine_name_, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Unable to read engine file");
  }

  // Set the device index
  cv::cuda::setDevice(m_options_.device_index);
  auto ret = cudaSetDevice(m_options_.device_index);
  if (ret != 0) {
    int num_gp_us;
    cudaGetDeviceCount(&num_gp_us);
    auto err_msg = "Unable to set GPU device index to: " + std::to_string(m_options_.device_index) +
                   ". Note, your device has " + std::to_string(num_gp_us) + " CUDA-capable GPU(s).";
    throw std::runtime_error(err_msg);
  }

  // Create a runtime to deserialize the engine file.
  m_runtime_ = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger_)};
  if (!m_runtime_) {
    return false;
  }
  // Create an engine, a representation of the optimized model.
  m_engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!m_engine_) {
    // File is corrupt
    return false;
  }

  // The execution context contains all of the state associated with a particular invocation
  m_context_ = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine_->createExecutionContext());
  if (!m_context_) {
    return false;
  }

  int curr_input = 0;
  int curr_output = 0;

  int nb_input_tensors = 0;
  int nb_output_tensors = 0;
  for (int i = 0; i < m_engine_->getNbIOTensors(); ++i) {
    const auto* const tensor_name = m_engine_->getIOTensorName(i);
    const auto tensor_type = m_engine_->getTensorIOMode(tensor_name);

    if (tensor_type == nvinfer1::TensorIOMode::kINPUT) {
      nb_input_tensors++;
    } else {
      nb_output_tensors++;
    }
  }

  // Storage for holding the input and output buffers
  // This will be passed to TRTEngineOptions for inference
  m_input_buffers_.resize(nb_input_tensors);
  m_output_buffers_.resize(nb_output_tensors);

  // Allocate GPU memory for input and output buffers
  m_output_lengths_float_.clear();
  for (int i = 0; i < m_engine_->getNbIOTensors(); ++i) {
    const auto* const tensor_name = m_engine_->getIOTensorName(i);
    const auto tensor_type = m_engine_->getTensorIOMode(tensor_name);
    auto tensor_shape = m_engine_->getTensorShape(tensor_name);
    auto data_type = static_cast<int>(m_engine_->getTensorDataType(tensor_name));
    if (tensor_type == nvinfer1::TensorIOMode::kINPUT) {
      size_t input_length = 1;
      for (int d = 0; d < tensor_shape.nbDims; ++d) {
        if (d > 0) {
          input_length *= tensor_shape.d[d];
        }
      }
      // Allocate memory for the input
      // Allocate enough to fit the max batch size (we could end up using less later)
      auto input_type_size = g_trt_globals.tensor_size_type_map[data_type];
      auto input_data_size = input_length * input_type_size;
      LOG(INFO) << "Allocating: " << m_options_.max_batch_size << " x " << input_data_size;
      util::checkCudaErrorCode(cudaMalloc(&m_input_buffers_[curr_input], m_options_.max_batch_size * input_data_size));
      m_input_batch_size_ = static_cast<int32_t>(tensor_shape.d[0]);

      // Store the input dims for later use
      int64_t input_stride = 0;
      int64_t batch_size = 0;
      if (tensor_shape.d[0] == -1) {
        tensor_shape.d[0] = m_options_.max_batch_size;
      }
      if (m_options_.input_format == TRTEngineOptions::InputFormat::kBCHW) {
        batch_size = static_cast<int>(tensor_shape.d[0]);
        auto input_channels = static_cast<int>(tensor_shape.d[1]);
        auto input_width = static_cast<int>(tensor_shape.d[3]);
        input_stride = input_width * input_channels;
      } else if (m_options_.input_format == TRTEngineOptions::InputFormat::kBHWC) {
        batch_size = static_cast<int>(tensor_shape.d[0]);
        auto input_width = static_cast<int>(tensor_shape.d[2]);
        auto input_channels = static_cast<int>(tensor_shape.d[3]);
        input_stride = input_width * input_channels;
      } else if (m_options_.input_format == TRTEngineOptions::InputFormat::kArray) {
        batch_size = static_cast<int>(tensor_shape.d[0]);
        input_stride = 1;
      }
      {
        LOG(INFO) << "Input Name    : " << tensor_name;
        LOG(INFO) << "\tBatch size    : " << batch_size;
        LOG(INFO) << "\tData Type     : " << data_type;
        LOG(INFO) << "\tStride     : " << input_stride;
        std::string shape_str;
        for (int d = 0; d < tensor_shape.nbDims; ++d) {
          shape_str += std::to_string(tensor_shape.d[d]) + ", ";
        }
        LOG(INFO) << "\tInput size: " << shape_str;
      }

      input_name_map_[tensor_name] = curr_input;
      nvinfer1::Dims input_dims;
      input_dims.nbDims = tensor_shape.nbDims - 1;
      for (int d = 0; d < input_dims.nbDims; ++d) {
        input_dims.d[d] = tensor_shape.d[d + 1];
      }

      m_input_dims_.emplace_back(input_dims);
      m_input_tensor_names_.emplace_back(tensor_name);

      input_type_.push_back(data_type);

      auto* gpu_input = static_cast<unsigned char*>(m_input_buffers_[curr_input]);
      std::vector<GPUInput> batch_inputs;
      batch_inputs.reserve(batch_size);
      size_t offset = 0;
      for (int j = 0; j < batch_size; j++) {
        GPUInput input;
        input.data_type = data_type;
        input.dims = tensor_shape;
        input.stride = input_stride * input_type_size;
        input.size = input_data_size;
        input.offset = offset;
        input.ptr = gpu_input + offset;
        offset += input.size;
        batch_inputs.emplace_back(input);

        cv::cuda::GpuMatND::SizeArray size;
        for (int d = 0; d < input_dims.nbDims; ++d) {
          size.push_back(static_cast<int>(input_dims.d[d]));
        }
      }
      inputs_.emplace_back(batch_inputs);

      ++curr_input;
      m_using_external_input_.push_back(false);

    } else if (tensor_type == nvinfer1::TensorIOMode::kOUTPUT) {
      // The binding is an output
      uint32_t output_len_float = 1;
      output_name_map_[tensor_name] = curr_output;
      nvinfer1::Dims output_dims;
      output_dims.nbDims = tensor_shape.nbDims - 1;
      for (int d = 0; d < output_dims.nbDims; ++d) {
        output_dims.d[d] = tensor_shape.d[d + 1];
      }

      m_output_dims_.push_back(output_dims);
      m_output_tensor_names_.emplace_back(tensor_name);
      output_type_.push_back(data_type);
      {
        LOG(INFO) << "Output Name : " << tensor_name;
        LOG(INFO) << "\tData Type   : " << data_type;

        std::string shape_str;
        for (int j = 1; j < tensor_shape.nbDims; ++j) {
          // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
          output_len_float *= tensor_shape.d[j];
          shape_str += std::to_string(tensor_shape.d[j]) + ", ";
        }
        LOG(INFO) << "\tDimension   : " << shape_str;
      }

      m_output_lengths_float_.push_back(output_len_float);
      // Now size the output buffer appropriately, taking into account the max possible batch size (although we could
      // actually end up using less memory)
      util::checkCudaErrorCode(
        cudaMalloc(&m_output_buffers_[curr_output],
                   output_len_float * m_options_.max_batch_size * g_trt_globals.tensor_size_type_map[data_type]));
      m_using_external_output_.push_back(false);
      // Cache element size to avoid map lookup on the inference hot path.
      const size_t o_elem_size = static_cast<size_t>(g_trt_globals.tensor_size_type_map[data_type]);
      m_output_elem_sizes_.push_back(o_elem_size);
      // Allocate pinned (page-locked) host memory for coalesced async D2H copies.
      // Covers all max_batch_size batches contiguously: pinned directly DMA-able by the GPU.
      {
        void* pinned_ptr = nullptr;
        const size_t pinned_bytes = output_len_float * o_elem_size * static_cast<size_t>(m_options_.max_batch_size);
        util::checkCudaErrorCode(cudaMallocHost(&pinned_ptr, pinned_bytes));
        m_output_pinned_buffers_.push_back(pinned_ptr);
        m_output_pinned_bytes_.push_back(pinned_bytes);
      }
      ++curr_output;
    } else {
      throw std::runtime_error("Error, IO Tensor is neither an input or output!");
    }
  }

  // Set the address of the input and output buffers
  for (size_t i = 0; i < m_input_buffers_.size(); ++i) {
    bool status = m_context_->setTensorAddress(m_input_tensor_names_[i].c_str(), m_input_buffers_[i]);
    if (!status) {
      LOG(WARNING) << "Failed to set input tensor address for: " << m_input_tensor_names_[i];
      return false;
    }
  }
  for (size_t i = 0; i < m_output_buffers_.size(); ++i) {
    bool status = m_context_->setTensorAddress(m_output_tensor_names_[i].c_str(), m_output_buffers_[i]);
    if (!status) {
      LOG(WARNING) << "Failed to set output tensor address for: " << m_output_tensor_names_[i];
      return false;
    }
  }
  auto batch_count = m_options_.max_batch_size != -1 ? m_options_.max_batch_size : m_input_batch_size_;
  for (int batch = 0; batch < batch_count; ++batch) {
    // Batch
    std::vector<std::vector<float>> batch_outputs{};
    for (size_t output_binding = 0; output_binding < m_output_tensor_names_.size(); ++output_binding) {
      // We start at index m_inputDims.size() to account for the inputs in our m_buffers
      std::vector<float> output;
      auto output_len_float = m_output_lengths_float_[output_binding];
      output.resize(output_len_float);
      batch_outputs.emplace_back(std::move(output));
    }
    inference_output_.emplace_back(std::move(batch_outputs));
  }
  // Reset last-batch-size cache so shape-setting is not skipped on the first inference.
  m_last_batch_size_ = -1;
  // Pre-populate the stream pool to avoid cudaStreamCreate on the inference hot path.
  if (cuda_streams_.empty()) {
    constexpr int kPrewarmStreams = 2;
    for (int s = 0; s < kPrewarmStreams; ++s) {
      cudaStream_t prewarm_stream = nullptr;
      util::checkCudaErrorCode(cudaStreamCreate(&prewarm_stream));
      cuda_streams_.push_back(prewarm_stream);
    }
  }
  return true;
}

bool TRTEngine::BuildAndLoadNetwork(std::string onnx_model_path) {
  std::string file_name = SerializeEngineOptions(m_options_, onnx_model_path);
  bool engine_exist = util::doesFileExist(file_name);

  if (!Build(onnx_model_path)) {
    return false;
  }
  if (LoadNetwork()) {
    return true;
  }
  if (engine_exist) {
    LOG(WARNING) << "Attempting to rebuild the engine";
    // remove it, and rebuild it
    std::remove(file_name.c_str());
    // it shouldn't be recursive since the file was removed
    return BuildAndLoadNetwork(onnx_model_path);
  }
  return false;
}
void TRTEngine::Warmup(int duration_ms) {
  auto start = std::chrono::high_resolution_clock::now();
  while (true) {
    auto* stream = RunInferenceOnly(1);
    if (stream) {
      // util::checkCudaErrorCode(cudaStreamSynchronize(stream));
      cuda_streams_.push_back(stream);
    }
    auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    if (duration >= duration_ms) {
      break;
    }
  }
}
cudaStream_t TRTEngine::GetCudaStream() {
  cudaStream_t stream;
  // cv::cuda::setDevice alone is not sufficient; cudaSetDevice must also be called so that
  // cudaStreamCreate and subsequent TRT/CUDA calls land on the correct device.
  util::ScopedCudaDevice guard(m_options_.device_index);
  if (cuda_streams_.empty()) {
    util::checkCudaErrorCode(cudaStreamCreate(&stream));
    return stream;
  }
  stream = cuda_streams_.front();
  cuda_streams_.pop_front();
  return stream;
}

cudaStream_t TRTEngine::RunInferenceOnly(int batch_size) {
  if (batch_size == 0) {
    LOG(WARNING) << "RunInferenceOnly: batch_size is 0";
    return nullptr;
  }
  // Bind the engine's device before any TRT/CUDA API call. enqueueV3() internally issues CUDA
  // kernels; without the correct active device they would be launched on whichever GPU the
  // calling thread last used, silently producing wrong results or crashing.
  util::ScopedCudaDevice guard(m_options_.device_index);

  // Acquire a stream from the pool (creates one on first call if pool is empty).
  cudaStream_t inference_cuda_stream = GetCudaStream();
  if (!inference_cuda_stream) {
    return nullptr;
  }

  // Set input shapes only when batch size changes.
  // TRT persists setInputShape state on the context until changed, so constant-batch
  // hotpaths (e.g. always batch=4) avoid redundant API calls entirely.
  if (batch_size != m_last_batch_size_) {
    const auto num_inputs = m_input_dims_.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      const auto& dims = m_input_dims_[i];
      nvinfer1::Dims input_dims;
      input_dims.nbDims = 1 + dims.nbDims;
      input_dims.d[0] = batch_size;
      for (int d = 0; d < dims.nbDims; ++d) {
        input_dims.d[d + 1] = dims.d[d];
      }
      if (!m_context_->setInputShape(m_input_tensor_names_[i].c_str(), input_dims)) {
        LOG(WARNING) << "RunInferenceOnly: failed to set input shape for " << m_input_tensor_names_[i];
      }
    }
    if (!m_context_->allInputDimensionsSpecified()) {
      LOG(WARNING) << "RunInferenceOnly: not all input dimensions specified";
    }
    m_last_batch_size_ = batch_size;
  }

  if (!m_context_->enqueueV3(inference_cuda_stream)) {
    LOG(WARNING) << "RunInferenceOnly: enqueueV3 failed";
    cuda_streams_.push_back(inference_cuda_stream);  // return stream to pool before bailing
    return nullptr;
  }
  return inference_cuda_stream;
}
bool TRTEngine::CopyOutputsOnly(int batch_size, cudaStream_t inference_cuda_stream) {
  // Bind device so that cudaMemcpyAsync and cudaStreamSynchronize operate on the correct GPU.
  util::ScopedCudaDevice guard(m_options_.device_index);
  // Coalesced D2H: one cudaMemcpyAsync per output binding copies all batch_size batches in a
  // single DMA transfer to pinned (page-locked) host memory, maximising PCIe bandwidth and
  // reducing CUDA API call overhead from (batch * n_outputs) to n_outputs.
  const size_t n_outputs = m_output_tensor_names_.size();
  for (size_t output_binding = 0; output_binding < n_outputs; ++output_binding) {
    const size_t elem_size = m_output_elem_sizes_[output_binding];
    const size_t output_len = m_output_lengths_float_[output_binding];
    const size_t copy_bytes = output_len * elem_size * static_cast<size_t>(batch_size);
    if (!util::checkCudaErrorCode(cudaMemcpyAsync(m_output_pinned_buffers_[output_binding],
                                                  m_output_buffers_[output_binding], copy_bytes, cudaMemcpyDeviceToHost,
                                                  inference_cuda_stream))) {
      LOG(WARNING) << "CopyOutputsOnly: D2H failed for output " << output_binding;
      util::checkCudaErrorCode(cudaStreamSynchronize(inference_cuda_stream));
      cuda_streams_.push_back(inference_cuda_stream);
      return false;
    }
  }
  util::checkCudaErrorCode(cudaStreamSynchronize(inference_cuda_stream));
  cuda_streams_.push_back(inference_cuda_stream);
  // Unpack staging buffers: FP16→FP32 conversion or plain memcpy depending on tensor type.
  return UnpackPinnedOutputs(batch_size);
}
bool TRTEngine::RunInference(int batch_size) {
  auto* stream = RunInferenceOnly(batch_size);
  if (!stream) {
    return false;
  }
  if (!CopyOutputsOnly(batch_size, stream)) {
    return false;
  }
  return true;
}

bool TRTEngine::RunInferenceOnStream(int batch_size, cudaStream_t stream) {
  // Like RunInferenceOnly but on a caller-owned stream (not drawn from the internal pool).
  // The caller is responsible for stream synchronisation and lifetime.
  if (batch_size == 0) {
    LOG(WARNING) << "RunInferenceOnStream: batch_size is 0";
    return false;
  }
  // Bind device so enqueueV3() launches kernels on the correct GPU.
  util::ScopedCudaDevice guard(m_options_.device_index);
  // Same batch-size caching as RunInferenceOnly: skip setInputShape when batch hasn't changed.
  if (batch_size != m_last_batch_size_) {
    const auto num_inputs = m_input_dims_.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      const auto& dims = m_input_dims_[i];
      nvinfer1::Dims input_dims;
      input_dims.nbDims = 1 + dims.nbDims;
      input_dims.d[0] = batch_size;
      for (int d = 0; d < dims.nbDims; ++d) {
        input_dims.d[d + 1] = dims.d[d];
      }
      if (!m_context_->setInputShape(m_input_tensor_names_[i].c_str(), input_dims)) {
        LOG(WARNING) << "RunInferenceOnStream: failed to set input shape for " << m_input_tensor_names_[i];
        return false;
      }
    }
    if (!m_context_->allInputDimensionsSpecified()) {
      LOG(WARNING) << "RunInferenceOnStream: not all input dimensions specified";
    }
    m_last_batch_size_ = batch_size;
  }
  if (!m_context_->enqueueV3(stream)) {
    LOG(WARNING) << "RunInferenceOnStream: enqueueV3 failed";
    return false;
  }
  return true;
}

bool TRTEngine::CaptureGraph(cudaStream_t stream, int batch_size) {
  util::ScopedCudaDevice guard(m_options_.device_index);
  // Optionally prime input shapes so the context is valid for capture.
  if (batch_size != -1 && batch_size != m_last_batch_size_) {
    const auto num_inputs = m_input_dims_.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      const auto& dims = m_input_dims_[i];
      nvinfer1::Dims input_dims;
      input_dims.nbDims = 1 + dims.nbDims;
      input_dims.d[0] = batch_size;
      for (int d = 0; d < dims.nbDims; ++d) {
        input_dims.d[d + 1] = dims.d[d];
      }
      if (!m_context_->setInputShape(m_input_tensor_names_[i].c_str(), input_dims)) {
        LOG(WARNING) << "CaptureGraph: failed to set input shape for " << m_input_tensor_names_[i];
        return false;
      }
    }
    m_last_batch_size_ = batch_size;
  }
  if (m_last_batch_size_ == -1) {
    LOG(WARNING) << "CaptureGraph: input shapes have not been set; call with a valid batch_size first.";
    return false;
  }
  cudaGraph_t graph = nullptr;
  auto err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    LOG(WARNING) << "CaptureGraph: cudaStreamBeginCapture failed: " << cudaGetErrorString(err);
    return false;
  }
  const bool ok = m_context_->enqueueV3(stream);
  err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess || !ok || !graph) {
    LOG(WARNING) << "CaptureGraph: capture failed (enqueueV3=" << ok << ", endCapture=" << cudaGetErrorString(err)
                 << ")";
    if (graph) cudaGraphDestroy(graph);
    return false;
  }
  cudaGraphExec_t new_exec = nullptr;
  err = cudaGraphInstantiate(&new_exec, graph, nullptr, nullptr, 0);
  cudaGraphDestroy(graph);
  if (err != cudaSuccess) {
    LOG(WARNING) << "CaptureGraph: cudaGraphInstantiate failed: " << cudaGetErrorString(err);
    return false;
  }
  if (m_cuda_graph_exec_) {
    cudaGraphExecDestroy(m_cuda_graph_exec_);
  }
  m_cuda_graph_exec_ = new_exec;
  LOG(INFO) << "CaptureGraph: CUDA graph captured successfully (batch_size=" << m_last_batch_size_ << ")";
  return true;
}

bool TRTEngine::RunGraphOnStream(cudaStream_t stream) {
  if (!m_cuda_graph_exec_) {
    LOG(WARNING) << "RunGraphOnStream: no graph captured; call CaptureGraph() first.";
    return false;
  }
  util::ScopedCudaDevice guard(m_options_.device_index);
  const auto err = cudaGraphLaunch(m_cuda_graph_exec_, stream);
  if (err != cudaSuccess) {
    LOG(WARNING) << "RunGraphOnStream: cudaGraphLaunch failed: " << cudaGetErrorString(err);
    return false;
  }
  return true;
}

bool TRTEngine::CopyOutputsOnStream(int batch_size, cudaStream_t stream, std::vector<int> excluded_output_indices) {
  // Bind device so that cudaMemcpyAsync targets the correct GPU.
  util::ScopedCudaDevice guard(m_options_.device_index);
  // Coalesced D2H to pinned staging buffers: one cudaMemcpyAsync per output covers all batches.
  // The caller MUST:
  //   1. cudaStreamSynchronize(stream)
  //   2. UnpackPinnedOutputs(batch_size)
  // before reading GetOutputVector().
  const size_t n_outputs = m_output_tensor_names_.size();
  for (size_t output_binding = 0; output_binding < n_outputs; ++output_binding) {
    if (std::find(excluded_output_indices.begin(), excluded_output_indices.end(), output_binding) !=
        excluded_output_indices.end()) {
      continue;
    }
    const size_t elem_size = m_output_elem_sizes_[output_binding];
    const size_t output_len = m_output_lengths_float_[output_binding];
    const size_t copy_bytes = output_len * elem_size * static_cast<size_t>(batch_size);
    if (!util::checkCudaErrorCode(cudaMemcpyAsync(m_output_pinned_buffers_[output_binding],
                                                  m_output_buffers_[output_binding], copy_bytes, cudaMemcpyDeviceToHost,
                                                  stream))) {
      LOG(WARNING) << "CopyOutputsOnStream: D2H failed for output " << output_binding;
      return false;
    }
  }
  return true;
}

bool TRTEngine::UnpackPinnedOutputs(int batch_size, std::vector<int> excluded_output_indices) {
  // Copy pinned staging buffers into inference_output_ after the stream has been synchronised.
  // FP16 tensors are converted element-by-element to FP32; FP32 tensors use a single memcpy
  // per batch slot.  All other types fall back to a raw memcpy (data interpreted as bytes).
  const size_t n_outputs = m_output_tensor_names_.size();
  for (size_t output_binding = 0; output_binding < n_outputs; ++output_binding) {
    if (std::find(excluded_output_indices.begin(), excluded_output_indices.end(), output_binding) !=
        excluded_output_indices.end()) {
      continue;
    }
    const size_t elem_size = m_output_elem_sizes_[output_binding];
    const size_t output_len = m_output_lengths_float_[output_binding];
    const int type = output_type_[output_binding];
    const bool is_fp16 = (type == static_cast<int>(nvinfer1::DataType::kHALF));
    const bool is_fp32 = (type == static_cast<int>(nvinfer1::DataType::kFLOAT));

    for (int batch = 0; batch < batch_size; ++batch) {
      auto& output = inference_output_[batch][output_binding];
      const char* pinned_base = static_cast<const char*>(m_output_pinned_buffers_[output_binding]) +
                                static_cast<size_t>(batch) * output_len * elem_size;

      if (is_fp16) {
        // Convert __half → float in-place using the CUDA fp16 intrinsic.
        const auto* src = reinterpret_cast<const __half*>(pinned_base);
        for (uint32_t e = 0; e < output_len; ++e) {
          output[e] = __half2float(src[e]);
        }
      } else if (is_fp32) {
        std::memcpy(output.data(), pinned_base, output_len * sizeof(float));
      } else {
        // INT8, UINT8, INT32, etc.: copy raw bytes (same as the previous unconverted behaviour).
        std::memcpy(output.data(), pinned_base, output_len * elem_size);
      }
    }
  }
  return true;
}

bool TRTEngine::SetInputVectorGpu(int index, const std::vector<cv::cuda::GpuMat>& input_vector, cudaStream_t stream) {
  if (m_options_.input_format == TRTEngineOptions::InputFormat::kArray) {
    LOG(WARNING) << "===== Error =====";
    LOG(WARNING) << "Can't set GPU input vector for an array input!";
    return false;
  }
  if (input_vector.size() > static_cast<size_t>(m_options_.max_batch_size)) {
    LOG(WARNING) << "===== Error =====";
    LOG(WARNING) << "The batch size is larger than the model expects!";
    LOG(WARNING) << "Model max batch size: " << m_options_.max_batch_size;
    LOG(WARNING) << "Batch size provided to call to SetInputVector[" << index << "]: " << input_vector.size();
    return false;
  }
  // Ensure that if the model has a fixed batch size that is greater than 1, the input has the correct length
  if (m_input_batch_size_ != -1 && input_vector.size() != static_cast<size_t>(m_input_batch_size_)) {
    LOG(WARNING) << "===== Error =====";
    LOG(WARNING) << "The batch size is different from what the model expects!";
    LOG(WARNING) << "Model batch size: " << m_input_batch_size_;
    LOG(WARNING) << "Batch size provided to call to SetInputVector[" << index << "]: " << input_vector.size();
    return false;
  }

  const auto& dims = m_input_dims_[index];
  const auto& input = input_vector[0];
  auto width = input.cols;
  auto height = input.rows;
  auto channels = input.channels();
  bool failed = false;
  if (m_options_.input_format == TRTEngineOptions::InputFormat::kBCHW) {
    if (channels != dims.d[0] || height != dims.d[1] || width != dims.d[2]) {
      failed = true;
    }
  } else if (m_options_.input_format == TRTEngineOptions::InputFormat::kBHWC) {
    if (height != dims.d[0] || width != dims.d[1] || channels != dims.d[2]) {
      failed = true;
    }
  }
  if (failed) {
    LOG(WARNING) << "===== Error =====";
    LOG(WARNING) << "Input does not have correct size!";
    LOG(WARNING) << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ")";
    LOG(WARNING) << "Got: (" << channels << ", " << height << ", " << width << ")";
    LOG(WARNING) << "Ensure you resize your input image to the correct size";
    return false;
  }
  return InternalBlobFromGpuMats(input_vector, index, stream);
}

bool TRTEngine::SetInputVector(int index, const std::vector<cv::MatND>& input_vector) {
  if (input_vector.size() > static_cast<size_t>(m_options_.max_batch_size)) {
    LOG(WARNING) << "===== Error =====";
    LOG(WARNING) << "The batch size is larger than the model expects!";
    LOG(WARNING) << "Model max batch size: " << m_options_.max_batch_size;
    LOG(WARNING) << "Batch size provided to call to SetInputVector[" << index << "]: " << input_vector.size();
    return false;
  }
  // Ensure that if the model has a fixed batch size that is greater than 1, the input has the correct length
  if (m_input_batch_size_ != -1 && input_vector.size() != static_cast<size_t>(m_input_batch_size_)) {
    LOG(WARNING) << "===== Error =====";
    LOG(WARNING) << "The batch size is different from what the model expects!";
    LOG(WARNING) << "Model batch size: " << m_input_batch_size_;
    LOG(WARNING) << "Batch size provided to call to SetInputVector[" << index << "]: " << input_vector.size();
    return false;
  }

  const auto& dims = m_input_dims_[index];

  const auto& input = input_vector[0];
  if (m_options_.input_format == TRTEngineOptions::InputFormat::kArray) {
    if (input.dims != dims.nbDims) {
      // sometimes the last size is of 1, which can be ignored
      if (input.dims != dims.nbDims + 1 || input.size[dims.nbDims] != 1) {
        LOG(WARNING) << "===== Error =====";
        LOG(WARNING) << "Input[" << index << "] does not have correct dims!";
        LOG(WARNING) << "Expected: " << dims.nbDims;
        LOG(WARNING) << "Got: " << input.dims;
        LOG(WARNING) << "Ensure you resize your input image to the correct size";
        return false;
      }
    }
    for (int d = 0; d < dims.nbDims; ++d) {
      if (input.size[d] != dims.d[d]) {
        LOG(WARNING) << "===== Error =====";
        LOG(WARNING) << "Input[" << index << "][" << d << "] does not have correct size!";
        LOG(WARNING) << "Expected: " << dims.d[d];
        LOG(WARNING) << "Got: " << input.size[d];
        return false;
      }
    }
  } else {
    auto width = input.size[0];
    auto height = input.size[1];
    auto channels = input.dims > 2 ? input.size[2] : input.channels();
    bool failed = false;
    if (m_options_.input_format == TRTEngineOptions::InputFormat::kBCHW) {
      if (channels != dims.d[0] || height != dims.d[1] || width != dims.d[2]) {
        failed = true;
      }
    } else if (m_options_.input_format == TRTEngineOptions::InputFormat::kBHWC) {
      if (height != dims.d[0] || width != dims.d[1] || channels != dims.d[2]) {
        failed = true;
      }
    }
    if (failed) {
      LOG(WARNING) << "===== Error =====";
      LOG(WARNING) << "Input does not have correct size!";
      LOG(WARNING) << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ")";
      LOG(WARNING) << "Got: (" << channels << ", " << height << ", " << width << ")";
      LOG(WARNING) << "Ensure you resize your input image to the correct size";
      return false;
    }
  }
  // input data and perform the preprocessing
  return InternalBlobFromMats(input_vector, index);
}
bool TRTEngine::CopyToInputDirectly(int input_index, int batch_index, const void* data, size_t size, size_t dst_offset,
                                    cudaStream_t stream) {
  auto& batch_image = inputs_[input_index][batch_index];
  if (size > batch_image.size) {
    LOG(WARNING) << "===== Error =====";
    LOG(WARNING) << "CopyToInputDirectly() Attempting to copy data larger than input";
    LOG(WARNING) << size << ">" << batch_image.size;
    return false;
  }
  util::ScopedCudaDevice guard(m_options_.device_index);
  if (stream) {
    // Async on the caller-provided stream; caller must keep 'data' valid until the stream is synced.
    util::checkCudaErrorCode(cudaMemcpyAsync(batch_image.ptr + dst_offset, data, size, cudaMemcpyHostToDevice, stream));
  } else {
    util::checkCudaErrorCode(cudaMemcpy(batch_image.ptr + dst_offset, data, size, cudaMemcpyHostToDevice));
  }
  return true;
}

bool TRTEngine::InternalBlobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batch_input, size_t input_index,
                                        cudaStream_t stream) {
  if (batch_input.empty()) {
    return false;
  }
  util::ScopedCudaDevice guard(m_options_.device_index);

  auto width = batch_input[0].cols;
  auto height = batch_input[0].rows;
  // input data and perform the preprocessing
  auto input_type = input_type_[input_index];
  if (m_options_.input_format == TRTEngineOptions::InputFormat::kBCHW) {
    auto layer = width * height;
    auto type = g_trt_globals.tensor_cv_type_map_1[input_type];
    auto size = g_trt_globals.tensor_size_type_map[input_type];
    auto channels = static_cast<size_t>(m_input_dims_[input_index].d[0]);
    cv::cuda::GpuMat tmp_image(height, width, g_trt_globals.tensor_cv_type_map_3[input_type]);

    for (size_t img = 0; img < batch_input.size(); img++) {
      auto& batch_image = inputs_[input_index][img];
      const auto& image = batch_input[img];
      if (channels > 1) {
        std::vector<cv::cuda::GpuMat> input_channels;
        input_channels.reserve(channels);
        for (size_t i = 0; i < channels; ++i) {
          input_channels.emplace_back(height, width, type, &(batch_image.ptr[i * layer * size]));
        }
        // Use async variant when stream is supplied so the copy is ordered with other work
        if (stream) {
          cudaMemcpy2DAsync(tmp_image.ptr(), tmp_image.step, image.data, image.step,
                            width * channels * g_trt_globals.tensor_size_type_map[input_type], height,
                            cudaMemcpyDeviceToDevice, stream);
        } else {
          cudaMemcpy2D(tmp_image.ptr(), tmp_image.step, image.data, image.step,
                       width * channels * g_trt_globals.tensor_size_type_map[input_type], height,
                       cudaMemcpyDeviceToDevice);
        }
        // cv::cuda::split does not accept a stream directly; it uses the OpenCV stream wrapper.
        cv::cuda::Stream cv_stream = stream ? cv::cuda::StreamAccessor::wrapStream(stream) : cv::cuda::Stream::Null();
        cv::cuda::split(tmp_image, input_channels, cv_stream);  // HWC -> CHW
      } else {
        const auto& image = batch_input[img];
        if (stream) {
          cudaMemcpy2DAsync(batch_image.ptr, batch_image.stride, image.data, image.step,
                            width * channels * g_trt_globals.tensor_size_type_map[input_type], height,
                            cudaMemcpyDeviceToDevice, stream);
        } else {
          cudaMemcpy2D(batch_image.ptr, batch_image.stride, image.data, image.step,
                       width * channels * g_trt_globals.tensor_size_type_map[input_type], height,
                       cudaMemcpyDeviceToDevice);
        }
      }
    }
    return true;
  }
  if (m_options_.input_format == TRTEngineOptions::InputFormat::kBHWC) {
    auto channels = static_cast<int>(m_input_dims_[input_index].d[2]);
    for (size_t img = 0; img < batch_input.size(); img++) {
      const auto& image = batch_input[img];
      auto& batch_image = inputs_[input_index][img];
      if (stream) {
        cudaMemcpy2DAsync(batch_image.ptr, batch_image.stride, image.data, image.step,
                          width * channels * g_trt_globals.tensor_size_type_map[input_type], height,
                          cudaMemcpyDeviceToDevice, stream);
      } else {
        cudaMemcpy2D(batch_image.ptr, batch_image.stride, image.data, image.step,
                     width * channels * g_trt_globals.tensor_size_type_map[input_type], height,
                     cudaMemcpyDeviceToDevice);
      }
    }
    return true;
  }
  if (m_options_.input_format == TRTEngineOptions::InputFormat::kArray) {
    for (size_t img = 0; img < batch_input.size(); img++) {
      const auto& image = batch_input[img];
      auto& batch_image = inputs_[input_index][img];
      if (stream) {
        cudaMemcpyAsync(batch_image.ptr, image.data, batch_image.size, cudaMemcpyDeviceToDevice, stream);
      } else {
        cudaMemcpy(batch_image.ptr, image.data, batch_image.size, cudaMemcpyDeviceToDevice);
      }
    }
    return true;
  }
  return false;
}
bool TRTEngine::InternalBlobFromMats(const std::vector<cv::MatND>& batch_input, size_t input_index) {
  if (batch_input.empty()) {
    return false;
  }
  util::ScopedCudaDevice guard(m_options_.device_index);
  auto input_type = input_type_[input_index];
  if (m_options_.input_format == TRTEngineOptions::InputFormat::kBCHW) {
    auto width = batch_input[0].size[0];
    auto height = batch_input[0].size[1];
    auto layer = width * height;
    auto type = g_trt_globals.tensor_cv_type_map_1[input_type];
    auto size = g_trt_globals.tensor_size_type_map[input_type];
    auto channels = static_cast<size_t>(m_input_dims_[input_index].d[0]);
    cv::cuda::GpuMat tmp_image(height, width, g_trt_globals.tensor_cv_type_map_3[input_type]);

    for (size_t img = 0; img < batch_input.size(); img++) {
      auto& batch_image = inputs_[input_index][img];
      const auto& image = batch_input[img];
      if (channels > 1) {
        std::vector<cv::cuda::GpuMat> input_channels;
        input_channels.reserve(channels);
        for (size_t i = 0; i < channels; ++i) {
          input_channels.emplace_back(height, width, type, &(batch_image.ptr[i * layer * size]));
        }

        cudaMemcpy2D(tmp_image.ptr(), tmp_image.step, image.data, image.step,
                     width * channels * g_trt_globals.tensor_size_type_map[input_type], height, cudaMemcpyHostToDevice);
        cv::cuda::split(tmp_image, input_channels);  // HWC -> CHW
      } else {
        const auto& image = batch_input[img];
        cudaMemcpy2D(batch_image.ptr, batch_image.stride, image.data, batch_image.stride,
                     width * channels * g_trt_globals.tensor_size_type_map[input_type], height, cudaMemcpyHostToDevice);
      }
    }
  } else if (m_options_.input_format == TRTEngineOptions::InputFormat::kBHWC) {
    auto width = batch_input[0].size[0];
    auto height = batch_input[0].size[1];
    auto channels = static_cast<int>(m_input_dims_[input_index].d[2]);
    for (size_t img = 0; img < batch_input.size(); img++) {
      const auto& image = batch_input[img];
      auto& batch_image = inputs_[input_index][img];
      cudaMemcpy2D(batch_image.ptr, batch_image.stride, image.data, batch_image.stride,
                   width * channels * g_trt_globals.tensor_size_type_map[input_type], height, cudaMemcpyHostToDevice);
    }
  } else if (m_options_.input_format == TRTEngineOptions::InputFormat::kArray) {
    for (size_t img = 0; img < batch_input.size(); img++) {
      const auto& image = batch_input[img];
      auto& batch_image = inputs_[input_index][img];
      cudaMemcpy(batch_image.ptr, image.data, batch_image.size, cudaMemcpyHostToDevice);
    }
  }
  return true;
}

std::string TRTEngine::SerializeEngineOptions(const TRTEngineOptions& options, const std::string& onnx_model_path) {
  const auto filename_pos = onnx_model_path.find_last_of('/') + 1;
  std::stringstream engine_name;

  struct passwd* pw = getpwuid(getuid());
  std::string base_path = std::string(pw->pw_dir) + "/.local/trt_engines/";
  std::filesystem::create_directories(base_path);
  engine_name << base_path;
  engine_name << onnx_model_path.substr(filename_pos, onnx_model_path.find_last_of('.') - filename_pos) << ".engine";

  auto onnx_checksum = util::checksum(onnx_model_path);
  engine_name << "." << onnx_checksum;
  // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
  std::vector<std::string> device_names;
  GetDeviceNames(device_names);

  if (static_cast<size_t>(options.device_index) >= device_names.size()) {
    throw std::runtime_error("Error, provided device index is out of range: " + std::to_string(options.device_index));
  }

  auto device_name = device_names[options.device_index];
  // Remove spaces from the device name
  device_name.erase(std::remove_if(device_name.begin(), device_name.end(), ::isspace), device_name.end());

  engine_name << "." << device_name << "_" << options.device_index;

  // Serialize the specified options into the filename
  if (options.precision == TRTEngineOptions::Precision::kFP16) {
    engine_name << ".fp16";
  } else if (options.precision == TRTEngineOptions::Precision::kFP32) {
    engine_name << ".fp32";
  } else {
    engine_name << ".int8";
  }

  engine_name << "." << options.max_batch_size;
  engine_name << "." << options.opt_batch_size;

  return engine_name.str();
}

void TRTEngine::GetDeviceNames(std::vector<std::string>& device_names) {
  int num_gp_us;
  auto err_code = cudaGetDeviceCount(&num_gp_us);
  if (err_code != cudaSuccess) {
    LOG(ERROR) << "Error getting number of devices: Code=" << err_code << ", Message=" << cudaGetErrorString(err_code);
  }

  for (int device = 0; device < num_gp_us; device++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    device_names.emplace_back(prop.name);
  }
}

cv::Mat TRTEngine::ResizeKeepAspectRatioPadRightBottom(const cv::Mat& input, int width, int height,
                                                       const cv::Scalar& bgcolor, float& ratio) {
  if (width == input.cols && height == input.rows) {
    ratio = 1;
    return input;
  }
  ratio = std::min(static_cast<float>(width) / static_cast<float>(input.cols),
                   static_cast<float>(height) / static_cast<float>(input.rows));
  int unpad_w = static_cast<int>(static_cast<float>(input.cols) * ratio);
  int unpad_h = static_cast<int>(static_cast<float>(input.rows) * ratio);
  cv::Mat re(unpad_h, unpad_w, input.type());
  cv::resize(input, re, re.size());
  cv::Mat out(height, width, input.type(), bgcolor);
  re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

  return out;
}
cv::cuda::GpuMat TRTEngine::ResizeKeepAspectRatioPadRightBottomGPU(const cv::cuda::GpuMat& input, int width, int height,
                                                                   const cv::Scalar& bgcolor, float& ratio) {
  if (width == input.cols && height == input.rows) {
    ratio = 1;
    return input;
  }
  ratio = std::min(static_cast<float>(width) / static_cast<float>(input.cols),
                   static_cast<float>(height) / static_cast<float>(input.rows));
  int unpad_w = static_cast<int>(static_cast<float>(input.cols) * ratio);
  int unpad_h = static_cast<int>(static_cast<float>(input.rows) * ratio);
  cv::cuda::GpuMat re(unpad_h, unpad_w, input.type());
  cv::cuda::resize(input, re, re.size());
  cv::cuda::GpuMat out(height, width, input.type(), bgcolor);
  re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

  return out;
}

// ---------------------------------------------------------------------------
// Zero-copy GPU input
// ---------------------------------------------------------------------------

bool TRTEngine::SetExternalInputGpu(int index, void* gpu_ptr) {
  if (index < 0 || static_cast<size_t>(index) >= m_input_buffers_.size()) {
    LOG(WARNING) << "SetExternalInputGpu: invalid index " << index;
    return false;
  }
  if (!gpu_ptr) {
    LOG(WARNING) << "SetExternalInputGpu: null gpu_ptr for input " << index;
    return false;
  }
  if (!m_context_->setTensorAddress(m_input_tensor_names_[index].c_str(), gpu_ptr)) {
    LOG(WARNING) << "SetExternalInputGpu: setTensorAddress failed for " << m_input_tensor_names_[index];
    return false;
  }
  m_using_external_input_[index] = true;
  return true;
}

bool TRTEngine::RestoreInternalInputBuffer(int index) {
  if (index < 0 || static_cast<size_t>(index) >= m_input_buffers_.size()) {
    LOG(WARNING) << "RestoreInternalInputBuffer: invalid index " << index;
    return false;
  }
  if (!m_context_->setTensorAddress(m_input_tensor_names_[index].c_str(), m_input_buffers_[index])) {
    LOG(WARNING) << "RestoreInternalInputBuffer: setTensorAddress failed for " << m_input_tensor_names_[index];
    return false;
  }
  m_using_external_input_[index] = false;
  return true;
}

bool TRTEngine::SetExternalOutputGpu(int index, void* gpu_ptr) {
  if (index < 0 || static_cast<size_t>(index) >= m_output_buffers_.size()) {
    LOG(WARNING) << "SetExternalOutputGpu: invalid index " << index;
    return false;
  }
  if (!gpu_ptr) {
    LOG(WARNING) << "SetExternalOutputGpu: null gpu_ptr for output " << index;
    return false;
  }
  if (!m_context_->setTensorAddress(m_output_tensor_names_[index].c_str(), gpu_ptr)) {
    LOG(WARNING) << "SetExternalOutputGpu: setTensorAddress failed for " << m_output_tensor_names_[index];
    return false;
  }
  m_using_external_output_[index] = true;
  return true;
}

bool TRTEngine::RestoreInternalOutputBuffer(int index) {
  if (index < 0 || static_cast<size_t>(index) >= m_output_buffers_.size()) {
    LOG(WARNING) << "RestoreInternalOutputBuffer: invalid index " << index;
    return false;
  }
  if (!m_context_->setTensorAddress(m_output_tensor_names_[index].c_str(), m_output_buffers_[index])) {
    LOG(WARNING) << "RestoreInternalOutputBuffer: setTensorAddress failed for " << m_output_tensor_names_[index];
    return false;
  }
  m_using_external_output_[index] = false;
  return true;
}

// ---------------------------------------------------------------------------
// Async inference and GPU-resident output getters
// ---------------------------------------------------------------------------

cudaStream_t TRTEngine::RunInferenceAsync(int batch_size) { return RunInferenceOnly(batch_size); }

void TRTEngine::SyncStreamAndReturn(cudaStream_t stream) {
  // Bind device: cudaStreamSynchronize must target the GPU the stream was created on.
  util::ScopedCudaDevice guard(m_options_.device_index);
  util::checkCudaErrorCode(cudaStreamSynchronize(stream));
  cuda_streams_.push_back(stream);
}

cv::cuda::GpuMat TRTEngine::GetOutputGpuMat(int output_index, int batch_index) const {
  if (output_index < 0 || static_cast<size_t>(output_index) >= m_output_buffers_.size()) {
    LOG(WARNING) << "GetOutputGpuMat: invalid output_index " << output_index;
    return {};
  }
  if (batch_index < 0 || batch_index >= m_options_.max_batch_size) {
    LOG(WARNING) << "GetOutputGpuMat: invalid batch_index " << batch_index;
    return {};
  }
  const int type = output_type_[output_index];
  const auto elem_size = static_cast<size_t>(g_trt_globals.tensor_size_type_map[type]);
  const auto output_len = static_cast<size_t>(m_output_lengths_float_[output_index]);
  const int cv_type = g_trt_globals.tensor_cv_type_map_1[type];

  auto* base_ptr = static_cast<unsigned char*>(m_output_buffers_[output_index]);
  auto* batch_ptr = base_ptr + static_cast<size_t>(batch_index) * output_len * elem_size;

  return cv::cuda::GpuMat(1, static_cast<int>(output_len), cv_type, batch_ptr);
}

const void* TRTEngine::GetOutputRawGpu(int output_index) const {
  if (output_index < 0 || static_cast<size_t>(output_index) >= m_output_buffers_.size()) {
    LOG(WARNING) << "GetOutputRawGpu: invalid output_index " << output_index;
    return nullptr;
  }
  return m_output_buffers_[output_index];
}

size_t TRTEngine::GetOutputElementSize(int output_index) const {
  if (output_index < 0 || static_cast<size_t>(output_index) >= output_type_.size()) {
    return 0;
  }
  return static_cast<size_t>(g_trt_globals.tensor_size_type_map[output_type_[output_index]]);
}

void TRTEngine::TransformOutput(std::vector<std::vector<std::vector<float>>>& input,
                                std::vector<std::vector<float>>& output) {
  if (input.size() != 1) {
    throw std::logic_error("The feature vector has incorrect dimensions!");
  }

  output = std::move(input[0]);
}

void TRTEngine::TransformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output) {
  if (input.size() != 1 || input[0].size() != 1) {
    throw std::logic_error("The feature vector has incorrect dimensions!");
  }

  output = std::move(input[0][0]);
}
}  // namespace cuda_common
