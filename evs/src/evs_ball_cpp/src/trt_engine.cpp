// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "evs_ball_cpp/trt_engine.hpp"

#include <glog/logging.h>
#include "evs_ball_cpp/logger.hpp"

Logger g_logger;

TensorRTEngine::TensorRTEngine(std::string model_path, int batch_size, int stream_size, int number_cams, int width,
                               int height, int num_bbox, float threshold_conf, float vel_const) {
  stream_size_ = stream_size;
  batch_size_ = batch_size;
  num_cams_ = number_cams;
  model_path_ = model_path;
  width_ = width;
  height_ = height;
  num_bbox_ = num_bbox;
  thresh_conf = threshold_conf;
  vel_const_ = vel_const;

  auto input_size = cv::Size(width_, height_);

  // Build the engine and the context
  BuildEngine(model_path_);
  AllocateMemory(batch_size_);

  int count_gpu_frame = 0;
  for (int i = 0; i < num_cams_; i++) {
    for (int j = 0; j < stream_size; j++) {
      cv::cuda::Stream stream;
      streams_.push_back(stream);
      cv::cuda::GpuMat gpu_frame(input_size, CV_16FC2,
                                 static_cast<half_float::half *>(buffer_ptrs_[count_gpu_frame].front()));
      gpu_frames_.push_back(std::move(gpu_frame));
      count_gpu_frame++;
    }
    x.emplace_back(std::vector<float>());
    y.emplace_back(std::vector<float>());
    r.emplace_back(std::vector<float>());
    x_vel.emplace_back(std::vector<float>());
    y_vel.emplace_back(std::vector<float>());
    confs.emplace_back(std::vector<float>());
    x_sigma.emplace_back(std::vector<float>());
    y_sigma.emplace_back(std::vector<float>());
    r_sigma.emplace_back(std::vector<float>());
    x_vel_sigma.emplace_back(std::vector<float>());
    y_vel_sigma.emplace_back(std::vector<float>());
  }

  for (int i = 0; i < num_cams_; i++) {
    x[i].resize(1);
    y[i].resize(1);
    r[i].resize(1);
    x_vel[i].resize(1);
    y_vel[i].resize(1);
    confs[i].resize(36000);

    x_sigma[i].resize(1);
    y_sigma[i].resize(1);
    r_sigma[i].resize(1);
    x_vel_sigma[i].resize(1);
    y_vel_sigma[i].resize(1);
  }
}

void TensorRTEngine::Predict(std::vector<cv::cuda::HostMem> &event_frames, int stream_idx,
                             cudaStream_t main_streams[12],
                             std::vector<std::unordered_map<std::string, float>> &ball_positions) {
  int pre_idx = stream_idx;
  int inf_idx = (stream_idx + 1) % 3;
  int post_idx = (stream_idx + 2) % 3;

  for (int i = 0; i < num_cams_; i++) {
    cudaStreamSynchronize(main_streams[pre_idx * num_cams_ + i]);
  }
  PreprocessImage(event_frames, pre_idx);
  int off_set = inf_idx * num_cams_;
  for (int i = 0; i < num_cams_; i++) {
    streams_[off_set + i].waitForCompletion();
  }
  for (int i = 0; i < num_cams_; i++) {
    context_[inf_idx * num_cams_ + i]->enqueueV3(main_streams[inf_idx * num_cams_ + i]);
  }
  PostProcess(post_idx, main_streams[post_idx], ball_positions);
}

void TensorRTEngine::PreprocessImage(std::vector<cv::cuda::HostMem> &event_frames, int stream_idx) {
  int bs = 0;
  int off_set = stream_idx * num_cams_;
  for (auto &frame : event_frames) {
    gpu_frames_[off_set + bs].upload(frame, streams_[off_set + bs]);
    bs++;
  }
}

void TensorRTEngine::PostProcess(int stream_idx, cudaStream_t download_stream,
                                 std::vector<std::unordered_map<std::string, float>> &predictions) {
  // static conversion to int from unsigned int for pointer arithmetic
  for (int i = 0; i < num_cams_; i++) {
    int conf_size = static_cast<int>(confs[i].size());
    int box_size = static_cast<int>(x[i].size());
    int uncert_size = static_cast<int>(x_sigma[i].size());

    void *confs_start_ptr = buffer_ptrs_[i + stream_idx * num_cams_][2];
    void *boxes_start_ptr = buffer_ptrs_[i + stream_idx * num_cams_][1];
    void *uncerts_start_ptr = buffer_ptrs_[i + stream_idx * num_cams_][3];

    cudaMemcpyAsync(confs[i].data(), confs_start_ptr, conf_size * sizeof(float), cudaMemcpyDeviceToHost,
                    download_stream);
    int max_confidence_index =
      static_cast<int>(std::distance(confs[i].begin(), std::max_element(confs[i].begin(), confs[i].end())));
    if (confs[i][max_confidence_index] > thresh_conf) {
      int max_confidence_offset = static_cast<int>(num_bbox_ * max_confidence_index * sizeof(float));
      // Extract Predictions
      cudaMemcpyAsync(x[i].data(), static_cast<void *>(static_cast<uint8_t *>(boxes_start_ptr) + max_confidence_offset),
                      box_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      cudaMemcpyAsync(
        y[i].data(),
        static_cast<void *>(static_cast<uint8_t *>(boxes_start_ptr) + max_confidence_offset + 1 * sizeof(float)),
        box_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      cudaMemcpyAsync(
        r[i].data(),
        static_cast<void *>(static_cast<uint8_t *>(boxes_start_ptr) + max_confidence_offset + 2 * sizeof(float)),
        box_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      cudaMemcpyAsync(
        x_vel[i].data(),
        static_cast<void *>(static_cast<uint8_t *>(boxes_start_ptr) + max_confidence_offset + 3 * sizeof(float)),
        box_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      cudaMemcpyAsync(
        y_vel[i].data(),
        static_cast<void *>(static_cast<uint8_t *>(boxes_start_ptr) + max_confidence_offset + 4 * sizeof(float)),
        box_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);

      // Extract Quantified Proxy Uncertainties
      cudaMemcpyAsync(x_sigma[i].data(),
                      static_cast<void *>(static_cast<uint8_t *>(uncerts_start_ptr) + max_confidence_offset),
                      uncert_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      cudaMemcpyAsync(
        y_sigma[i].data(),
        static_cast<void *>(static_cast<uint8_t *>(uncerts_start_ptr) + max_confidence_offset + 1 * sizeof(float)),
        uncert_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      cudaMemcpyAsync(
        r_sigma[i].data(),
        static_cast<void *>(static_cast<uint8_t *>(uncerts_start_ptr) + max_confidence_offset + 2 * sizeof(float)),
        uncert_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      cudaMemcpyAsync(
        x_vel_sigma[i].data(),
        static_cast<void *>(static_cast<uint8_t *>(uncerts_start_ptr) + max_confidence_offset + 3 * sizeof(float)),
        uncert_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      cudaMemcpyAsync(
        y_vel_sigma[i].data(),
        static_cast<void *>(static_cast<uint8_t *>(uncerts_start_ptr) + max_confidence_offset + 4 * sizeof(float)),
        uncert_size * sizeof(float), cudaMemcpyDeviceToHost, download_stream);
      // Pass on message
      predictions.push_back({{"idx", i},
                             {"conf", confs[i][max_confidence_index]},
                             {"u", x[i][0] * static_cast<float>(width_)},
                             {"v", y[i][0] * static_cast<float>(height_)},
                             {"r", r[i][0] * static_cast<float>(width_)},
                             {"vel_x", x_vel[i][0] * vel_const_},
                             {"vel_y", y_vel[i][0] * vel_const_},
                             {"u_sigma", x_sigma[i][0] * static_cast<float>(width_)},
                             {"v_sigma", y_sigma[i][0] * static_cast<float>(height_)},
                             {"r_sigma", r_sigma[i][0] * static_cast<float>(width_)},
                             {"vel_x_sigma", x_vel_sigma[i][0] * vel_const_},
                             {"vel_y_sigma", y_vel_sigma[i][0] * vel_const_}});
    }
  }
}

void TensorRTEngine::BuildEngine(const std::string &model_path) {
  Logger logger;

  // parse trt file
  std::ifstream engine_file(model_path, std::ios::binary);
  if (!engine_file) {
    std::printf("Error opening engine file: %s\n", model_path.c_str());
  }
  engine_file.seekg(0, std::ios::end);
  int64 fsize = engine_file.tellg();
  engine_file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(fsize);
  engine_file.read(engine_data.data(), static_cast<int64>(engine_data.size()));
  if (!engine_file) {
    std::printf("Error loading engine file: %s\n", model_path.c_str());
  }

  runtime_ = createInferRuntime(logger);
  engine_ = runtime_->deserializeCudaEngine(static_cast<void *>(engine_data.data()), engine_data.size());
  for (int j = 0; j < stream_size_ * num_cams_; j++) {
    context_.emplace_back(engine_->createExecutionContext());
  }

  std::printf("build engine successfully \n");

  std::cout << "=============\nBindings :\n";
  int n = engine_->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    const auto *const name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT) {
      std::cout << "Input " << i << " : " << name << "dim: ";
    } else {
      std::cout << "Output " << i << " : " << name << "dim: ";
    }
  }
};

void TensorRTEngine::AllocateMemory(int batch_size) {
  buffer_ptrs_.resize(num_cams_ * stream_size_);  // n streams of shape [input, boxes, confs, uncerts]
  std::cout << "bindings: " << engine_->getNbIOTensors() << std::endl;

  for (int j = 0; j < stream_size_ * num_cams_; j++) {  // two streams
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
      buffer_ptrs_[j].emplace_back(nullptr);

      //  Allocate the binding size
      const auto *const name = engine_->getIOTensorName(i);
      auto binding_size = GetSizeByDim(engine_->getTensorShape(name), batch_size) * sizeof(float);
      if (i == 0) {
        binding_size = GetSizeByDim(engine_->getTensorShape(name), batch_size) * sizeof(half_float::half);
      }
      std::printf("binding size: %lu, name: %s\n", binding_size, name);
      cudaMalloc(static_cast<void **>(&buffer_ptrs_[j].back()), binding_size);
      context_[j]->setTensorAddress(name, buffer_ptrs_[j].back());
    }
  }
}

void TensorRTEngine::ReleaseMemory() {
  // Release all buffers
  for (auto &buffer_ptrs_thred : buffer_ptrs_) {
    for (void *buf : buffer_ptrs_thred) {
      cudaFree(buf);
    }
  }
}

size_t TensorRTEngine::GetSizeByDim(const nvinfer1::Dims &dims, const size_t batch_size) {
  size_t size = batch_size;

  for (size_t i = 1; i < dims.nbDims; ++i) {
    size *= dims.d[i];
  }
  return size;
}
