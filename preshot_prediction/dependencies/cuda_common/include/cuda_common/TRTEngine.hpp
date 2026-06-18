// code is based on: https://github.com/cyrusbehr/tensorrt-cpp-api
// Confidential, Copyright 2025, Sony AI, All rights reserved

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

namespace cuda_common {

// Options for the network
class TRTEngineOptions {
 public:
  // Precision used for GPU inference
  enum class Precision {
    // Full precision floating point value
    kFP32,
    // Half prevision floating point value
    kFP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    kINT8,
  };
  enum class InputFormat { kBHWC, kBCHW, kArray };

  // Precision to use for GPU inference.
  Precision precision = Precision::kFP16;

  InputFormat input_format = InputFormat::kBCHW;
  // If INT8 precision is selected, must provide path to calibration dataset directory.
  std::string calibration_data_directory_path;
  // The batch size to be used when computing calibration data for INT8 inference.
  // Should be set to as large a batch number as your GPU will support.
  int32_t calibration_batch_size = 128;
  // The batch size which should be optimized for.
  int32_t opt_batch_size = 1;
  // Maximum allowable batch size
  int32_t max_batch_size = 16;

  int32_t opt_width = 640;

  int32_t opt_height = 640;

  // GPU device index
  int device_index = 0;
};

// ---------------------------------------------------------------------------
// Thread-safety contract
// ---------------------------------------------------------------------------
// Each TRTEngine instance is NOT thread-safe on its own: the internal
// IExecutionContext, stream pool (cuda_streams_), m_last_batch_size_, and
// output staging buffers are all unsynchronised mutable state.
//
// Safe multi-GPU / multi-thread pattern (one engine per thread):
//   1. Create one TRTEngine per worker thread, with options.device_index set
//      to the GPU assigned to that thread.
//   2. At thread start call cudaSetDevice(gpu_id) and cv::cuda::setDevice(gpu_id)
//      before touching any CUDA resource for the first time.
//   3. Call BuildAndLoadNetwork(), then run the inference loop entirely within
//      the same thread - never pass the engine object across thread boundaries.
//   4. Destroy the engine on the same thread that created it.
//
// Every public CUDA/TRT entry point in TRTEngine.cpp re-asserts the engine's
// device via util::ScopedCudaDevice, so stray cudaSetDevice() calls from
// other parts of the same thread are tolerated, but concurrent calls into the
// same instance from multiple threads are still undefined behaviour.
// ---------------------------------------------------------------------------
class TRTEngine {
  // Class to extend TensorRT logger
  class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
  };

 public:
  using SharedPtr = std::shared_ptr<TRTEngine>;
  using ConstSharedPtr = std::shared_ptr<const TRTEngine>;

  // Input format [input][batch][cv::cuda::GpuMat]
  using InputVector = std::vector<cv::cuda::GpuMatND>;
  using InferenceInput = std::vector<InputVector>;
  // Output format [batch][output][feature_vector] (CPU)
  using InferenceOutput = std::vector<std::vector<std::vector<float>>>;
  // Output format [output_index][batch_index] -> GpuMat wrapping raw on-GPU output buffer (zero-copy).
  using GpuInferenceOutput = std::vector<std::vector<cv::cuda::GpuMat>>;

  explicit TRTEngine(const TRTEngineOptions& options);
  ~TRTEngine();

  // Build the network
  bool Build(std::string onnx_model_path);
  // Load and prepare the network for inference
  bool LoadNetwork();

  // Safe version to build and load network. Will attempt to rebuild it if the current one is invalid
  bool BuildAndLoadNetwork(std::string onnx_model_path);

  void Warmup(int duration_ms);
  // Run inference.
  // Input format [input][batch][cv::cuda::GpuMat]
  // Output format [batch][output][feature_vector]
  bool RunInference(int batch_size);

  bool SetInputVector(int input_index, const std::vector<cv::MatND>& input);
  bool SetInputVectorGpu(int input_index, const std::vector<cv::cuda::GpuMat>& input,
                         cudaStream_t stream = nullptr);
  // Copy host data to the internal GPU input buffer for the given input and batch slot.
  // If stream is non-null the copy is enqueued asynchronously on that stream;
  // the caller is responsible for ensuring the host buffer remains valid until completion.
  // If stream is nullptr the copy is synchronous.
  bool CopyToInputDirectly(int index, int batch_index, const void* data, size_t size,
                           size_t dst_offset = 0, cudaStream_t stream = nullptr);

  // Zero-copy GPU input: redirect TensorRT's input tensor address to an externally managed GPU buffer.
  // The buffer must remain valid until the inference call completes.
  // Use RestoreInternalInputBuffer() to revert to the pre-allocated internal buffer.
  bool SetExternalInputGpu(int index, void* gpu_ptr);
  bool RestoreInternalInputBuffer(int index);

  // Zero-copy GPU output: redirect TensorRT's output tensor address to an externally managed GPU buffer.
  // TRT will write inference results directly into the provided pointer — no internal copy or staging needed.
  // The buffer must remain valid until the inference call (and any subsequent reads) complete.
  // Use RestoreInternalOutputBuffer() to revert to the pre-allocated internal buffer.
  bool SetExternalOutputGpu(int index, void* gpu_ptr);
  bool RestoreInternalOutputBuffer(int index);

  // --- Async / GPU-resident output API ---
  // Like RunInference() but does NOT copy outputs to CPU; returns the active CUDA stream (from the
  // internal pool). Outputs can be accessed zero-copy via GetOutputGpuMat() / GetOutputRawGpu() after
  // synchronisation. Call SyncStreamAndReturn() to synchronise and recycle the stream.
  cudaStream_t RunInferenceAsync(int batch_size);
  // Synchronise a stream obtained from RunInferenceAsync() and return it to the internal pool.
  void SyncStreamAndReturn(cudaStream_t stream);

  // Run inference on a caller-provided CUDA stream.
  // The caller owns the stream and is responsible for synchronising it.
  // Useful for ordering TRT inference with upstream H2D uploads or downstream D2H copies on the
  // same stream, removing the need for cross-stream synchronisation barriers.
  // Does NOT touch the internal stream pool.
  bool RunInferenceOnStream(int batch_size, cudaStream_t stream);

  // Capture the TRT enqueueV3 call into a CUDA graph for fast replay.
  // Eliminates the ~0.5ms CPU-side driver overhead of enqueueV3 on subsequent calls.
  //
  // Prerequisites (must all be true before calling):
  //   1. All tensor addresses are finalised (SetExternalInput/OutputGpu done or using internal buffers).
  //   2. Input shapes have been set at least once (i.e. RunInferenceOnStream or RunInference called
  //      with the same batch_size at least once, OR batch_size != -1 is passed here).
  //   3. The provided stream is idle (no in-flight work that touches the same tensors).
  // Passing batch_size != -1 calls setInputShape internally if the batch has not been set yet.
  // Returns true on success; false leaves any prior captured graph unchanged.
  bool CaptureGraph(cudaStream_t stream, int batch_size = -1);

  // Replay the captured CUDA graph on the given stream.
  // Inputs must already be at their (fixed) tensor addresses before calling.
  // Falls back to a warning + false if no graph has been captured yet.
  bool RunGraphOnStream(cudaStream_t stream);

  // Returns true if a graph has been successfully captured via CaptureGraph().
  bool IsGraphCaptured() const { return m_cuda_graph_exec_ != nullptr; }

  // Copy outputs to CPU (inference_output_) asynchronously on the given stream without synchronising.
  // Must be called after RunInferenceOnStream() completes on the same stream, or after any
  // explicit sync. The caller must synchronise the stream and then call UnpackPinnedOutputs()
  // before reading inference_output_.
  bool CopyOutputsOnStream(int batch_size, cudaStream_t stream, std::vector<int> excluded_output_indices = {});

  // Unpack pinned staging buffers into inference_output_ after the stream has been synced.
  // Must be called after CopyOutputsOnStream() + cudaStreamSynchronize().
  // Handles FP16 → FP32 conversion automatically.
  // (CopyOutputsOnly() calls this internally; no need to call explicitly with RunInference().)
  bool UnpackPinnedOutputs(int batch_size, std::vector<int> excluded_output_indices = {});

  // Returns a cv::cuda::GpuMat view (zero-copy) into the on-GPU output buffer.
  // Shape: 1 x output_length, CV type matching the tensor's data type.
  // Valid after RunInference() or after the stream from RunInferenceAsync() has been synchronised.
  [[nodiscard]] cv::cuda::GpuMat GetOutputGpuMat(int output_index, int batch_index = 0) const;
  // Returns the raw device pointer to the output buffer for output_index.
  [[nodiscard]] const void* GetOutputRawGpu(int output_index) const;

  // Returns the byte size of one element for the given output tensor (e.g. 4 for FP32, 2 for FP16).
  // Useful when computing offsets into the raw GPU output buffer.
  [[nodiscard]] size_t GetOutputElementSize(int output_index) const;

  const InferenceOutput& GetOutputVector() const { return inference_output_; }

  const TRTEngineOptions& GetOptions() { return m_options_; }

  // Utility method for resizing an image while maintaining the aspect ratio by adding padding to smaller dimension
  // after scaling While letterbox padding normally adds padding to top & bottom, or left & right sides, this
  // implementation only adds padding to the right or bottom side This is done so that it's easier to convert detected
  // coordinates (ex. YOLO model) back to the original reference frame.
  static cv::Mat ResizeKeepAspectRatioPadRightBottom(const cv::Mat& input, int width, int height,
                                                     const cv::Scalar& bgcolor, float& ratio);
  static cv::cuda::GpuMat ResizeKeepAspectRatioPadRightBottomGPU(const cv::cuda::GpuMat& input, int width, int height,
                                                                 const cv::Scalar& bgcolor, float& ratio);

  [[nodiscard]] const std::vector<nvinfer1::Dims>& GetInputDims() const { return m_input_dims_; };
  [[nodiscard]] const std::vector<nvinfer1::Dims>& GetOutputDims() const { return m_output_dims_; };
  [[nodiscard]] int GetInputIndex(const std::string& name) {
    auto it = input_name_map_.find(name);
    if (it == input_name_map_.end()) {
      return -1;
    }
    return it->second;
  }

  [[nodiscard]] int GetOutputIndex(const std::string& name) {
    auto it = output_name_map_.find(name);
    if (it == output_name_map_.end()) {
      return -1;
    }
    return it->second;
  }

  [[nodiscard]] const std::vector<std::string>& GetInputTensorNames() const { return m_input_tensor_names_; }
  [[nodiscard]] const std::vector<std::string>& GetOutputTensorNames() const { return m_output_tensor_names_; }

  // Utility method for transforming triple nested output array into 2D array
  // Should be used when the output batch size is 1, but there are multiple output feature vectors
  static void TransformOutput(std::vector<std::vector<std::vector<float>>>& input,
                              std::vector<std::vector<float>>& output);

  // Utility method for transforming triple nested output array into single array
  // Should be used when the output batch size is 1, and there is only a single output feature vector
  static void TransformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output);
  const std::string& GetEngineName() { return m_engine_name_; }

 private:
  cudaStream_t RunInferenceOnly(int batch_size);
  bool CopyOutputsOnly(int batch_size, cudaStream_t stream);
  // Converts the engine options into a string
  static std::string SerializeEngineOptions(const TRTEngineOptions& options, const std::string& onnx_model_path);

  static void GetDeviceNames(std::vector<std::string>& device_names);
  bool InternalBlobFromMats(const std::vector<cv::MatND>& batch_input, size_t input_index);
  bool InternalBlobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batch_input, size_t input_index,
                               cudaStream_t stream = nullptr);
  // Holds pointers to the input and output GPU buffers
  std::vector<void*> m_input_buffers_;
  std::vector<void*> m_output_buffers_;
  std::vector<uint32_t> m_output_lengths_float_;
  std::vector<nvinfer1::Dims> m_input_dims_;
  std::vector<nvinfer1::Dims> m_output_dims_;
  std::unordered_map<std::string, int> input_name_map_;
  std::unordered_map<std::string, int> output_name_map_;
  std::vector<std::string> m_input_tensor_names_;
  std::vector<std::string> m_output_tensor_names_;
  std::vector<int> input_type_;
  std::vector<int> output_type_;
  int32_t m_input_batch_size_;

  struct GPUInput {
    unsigned char* ptr;
    size_t size;
    size_t offset;
    int data_type;
    size_t stride;
    nvinfer1::Dims dims;
  };
  std::vector<std::vector<GPUInput>> inputs_;
  InferenceInput inference_input_;
  TRTEngine::InferenceOutput inference_output_;

  // Must keep IRuntime around for inference, see:
  // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
  std::unique_ptr<nvinfer1::IRuntime> m_runtime_ = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> m_engine_ = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context_ = nullptr;

  std::list<cudaStream_t> cuda_streams_;
  const TRTEngineOptions m_options_;
  Logger m_logger_;
  std::string m_engine_name_;

  cudaStream_t GetCudaStream();

  // Tracks which input tensors are currently using an external GPU buffer.
  std::vector<bool> m_using_external_input_;
  // Tracks which output tensors are currently using an external GPU buffer.
  std::vector<bool> m_using_external_output_;

  // Per-output element byte size (cached from TRTEngineGlobals; avoids map lookup on the hot path).
  std::vector<size_t> m_output_elem_sizes_;
  // Per-output-binding pinned (page-locked) host memory for coalesced async D2H copies.
  // Each buffer covers all max_batch_size batches contiguously:
  //   bytes = output_len_float * elem_size * max_batch_size
  // For FP16 tensors these hold raw __half values; UnpackPinnedOutputs() converts to float.
  std::vector<void*> m_output_pinned_buffers_;
  std::vector<size_t> m_output_pinned_bytes_;
  // Last batch size used for setInputShape; allows skipping redundant shape-setting calls
  // when the batch size is constant across inferences (common for encoder/decoder pipelines).
  int32_t m_last_batch_size_{-1};
  // CUDA graph executable captured from enqueueV3. nullptr until CaptureGraph() succeeds.
  cudaGraphExec_t m_cuda_graph_exec_{nullptr};
};
}  // namespace cuda_common
