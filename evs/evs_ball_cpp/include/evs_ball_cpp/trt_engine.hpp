// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <metavision/sdk/base/events/event_cd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <list>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <thread>
#include <tuple>
#include <valarray>

#include "half.hpp"
#include "rclcpp/rclcpp.hpp"

struct TRTDestroy {
  template <class T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

class TensorRTEngine {
 private:
  template <class T>
  using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

  // Instantiate engine and context empty pointers;
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  std::vector<nvinfer1::IExecutionContext*> context_;

  // Create the buffer and dimension vectors
  std::vector<std::vector<void*>> buffer_ptrs_;

  void BuildEngine(const std::string& model_path);

  void AllocateMemory(int batch_size);

  void PreprocessImage(std::vector<cv::cuda::HostMem>& event_frames, int stream_idx);
  void PostProcess(int stream_idx, cudaStream_t download_stream,
                   std::vector<std::unordered_map<std::string, float>>& ball_positions);

  static size_t GetSizeByDim(const nvinfer1::Dims& dims, size_t batch_size = 1);

  std::vector<cv::cuda::Stream> streams_;
  std::vector<cv::cuda::GpuMat> gpu_frames_;

  std::string model_path_;

  int width_;
  int height_;
  int num_bbox_;
  float vel_const_;

  int batch_size_;
  int num_cams_;
  int stream_size_;

 public:
  explicit TensorRTEngine(std::string model_path, int batch_size, int stream_size, int number_cams, int width,
                          int height, int num_bbox, float threshold_conf, float vel_const);
  void Predict(std::vector<cv::cuda::HostMem>& event_frames, int stream_idx, cudaStream_t main_streams[12],
               std::vector<std::unordered_map<std::string, float>>& predictions);
  void ReleaseMemory();

  float thresh_conf;

  std::vector<std::vector<float>> x, y, r, x_vel, y_vel;
  std::vector<std::vector<float>> confs;
  std::vector<std::vector<float>> confs_host;
  std::vector<std::vector<float>> x_sigma, y_sigma, r_sigma, x_vel_sigma, y_vel_sigma;
};
