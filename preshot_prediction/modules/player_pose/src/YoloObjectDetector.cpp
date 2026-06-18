// Confidential, Copyright 2025, Sony AI, All rights reserved
#include "player_pose/YoloObjectDetector.hpp"

#include <omp.h>

#include <cuda_common/TRTEngine.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "player_pose/PlayerPoseParameters.hpp"

namespace perception {

class YoloObjectDetector::YoloObjectDetectorImpl {
 public:
  Eigen::Vector2i input_size;
  bool initialized{false};
  cuda_common::TRTEngine::SharedPtr model;
  YoloConfig::SharedPtr config;
  int output_index{0};
  nvinfer1::Dims output_dim;

  YoloObjectDetectorImpl() = default;
  virtual ~YoloObjectDetectorImpl() = default;
  bool Initialize(const std::string& onnx_model_path, YoloConfig::SharedPtr config,
                  PlayerPoseParameters::SharedPtr player_params) {
    this->config = config;
    {
      cuda_common::TRTEngineOptions options;
      options.opt_batch_size = player_params->person_detector->batch_size;
      options.max_batch_size = player_params->person_detector->batch_size;
      options.device_index = player_params->person_detector->device_index;

      std::cout << "YoloObjectDetector - Building TensorRT engine with batch size: "
                << player_params->person_detector->batch_size
                << " on device index: " << player_params->person_detector->device_index << std::endl;

      options.precision = cuda_common::TRTEngineOptions::Precision::kFP16;

      model = std::make_shared<cuda_common::TRTEngine>(options);
      auto succ = model->BuildAndLoadNetwork(onnx_model_path);
      if (!succ) {
        const std::string err_msg =
          "YoloObjectDetector - Error: Unable to build the TensorRT engine. "
          "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(err_msg);
      }
    }

    //::CreateModel(model_path, 6, tensorrt::TensorRT::InputFormat::kBCHW);
    initialized = true;
    input_size.x() = static_cast<Eigen::Vector2i::Scalar>(model->GetInputDims()[0].d[2]);
    input_size.y() = static_cast<Eigen::Vector2i::Scalar>(model->GetInputDims()[0].d[1]);
    auto output = model->GetOutputDims();
    output_index = model->GetOutputIndex("output0");
    output_dim = output[output_index];  // 1,channels,anchors
    return true;
  }
  void ParseResultsSingle(float* output_ptr, std::vector<YoloObject>& results,
                          const std::pair<float, cv::Size2i>& ratios) {
    // Preallocate vectors
    int num_proposal = static_cast<int>(output_dim.d[1]);
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels;
    std::vector<std::vector<Eigen::Vector3f>> keypoints;

    // Process matrix in original orientation (no transpose)
    cv::Mat outs(static_cast<int>(output_dim.d[0]), static_cast<int>(output_dim.d[1]), CV_32F, output_ptr);
    // Each column is a proposal, so loop over columns
    num_proposal = outs.cols;
    // std::cout << "Number of proposals: " << num_proposal << std::endl;
    for (int col_ind = 0; col_ind < num_proposal; col_ind++) {
      const auto& pdata = outs.col(col_ind);
      float conf = 0;
      int label = 0;
      if (config->type == YoloModelType::kClasses) {
        // auto* scores_ptr = pdata + 4;
        // auto* max_score_ptr = std::max_element(scores_ptr, scores_ptr + config->class_names.size());
        // conf = *max_score_ptr;
        // label = static_cast<int>(max_score_ptr - scores_ptr);
      } else {
        conf = pdata.at<float>(4);
      }
      if (conf > config->conf_threshold) {
        float x = pdata.at<float>(0);
        float y = pdata.at<float>(1);
        float w = pdata.at<float>(2);
        float h = pdata.at<float>(3);

        int x0 = std::clamp<int>(static_cast<int>((x - 0.5F * w) * ratios.first), 0, ratios.second.width);
        int y0 = std::clamp<int>(static_cast<int>((y - 0.5F * h) * ratios.first), 0, ratios.second.height);
        int x1 = std::clamp<int>(static_cast<int>((x + 0.5F * w) * ratios.first), x0, ratios.second.width);
        int y1 = std::clamp<int>(static_cast<int>((y + 0.5F * h) * ratios.first), y0, ratios.second.height);

        if (config->type == YoloModelType::kKeypoints) {
          std::vector<Eigen::Vector3f> kpts;
          kpts.reserve(config->num_kps * config->keypoint_length);
          for (int kpi = 0; kpi < config->num_kps; ++kpi) {
            auto kx = std::clamp<float>(
              static_cast<float>((pdata.at<float>(5 + kpi * config->keypoint_length + 0)) * ratios.first), 0,
              ratios.second.width);
            auto ky = std::clamp<float>(
              static_cast<float>((pdata.at<float>(5 + kpi * config->keypoint_length + 1)) * ratios.first), 0,
              ratios.second.height);
            kpts.emplace_back(kx, ky, pdata.at<float>(5 + kpi * config->keypoint_length + 2));
          }
          keypoints.push_back(kpts);
        }

        confidences.emplace_back(conf);
        boxes.emplace_back(x0, y0, x1 - x0, y1 - y0);
        labels.emplace_back(label);
      }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config->conf_threshold, config->nms_threshold, indices);
    size_t count = config->top_k > 0 ? std::min<size_t>(config->top_k, indices.size()) : indices.size();
    results.resize(count);
    size_t index = 0;
    for (auto& obj_idx : indices) {
      if (index >= count) {
        break;
      }
      results[index].label = labels[obj_idx];
      results[index].confidence = confidences[obj_idx];
      results[index].rect = boxes[obj_idx];
      if (config->type == YoloModelType::kKeypoints) {
        results[index].keypoints = keypoints[obj_idx];
      }
      ++index;
    }
  }
  void ParseResults(const cuda_common::TRTEngine::InferenceOutput& output_buffer,
                    std::vector<std::vector<YoloObject>>& all_results,
                    const std::vector<std::pair<float, cv::Size2i>>& ratios) {
    all_results.resize(output_buffer.size());
    for (int curr_result = 0; curr_result < static_cast<int>(all_results.size()); ++curr_result) {
      auto& results = all_results[curr_result];
      auto* output_ptr = const_cast<float*>(output_buffer[curr_result][output_index].data());
      ParseResultsSingle(output_ptr, results, ratios[curr_result]);
    }
  }
  bool DetectObjects(const std::vector<cv::cuda::GpuMat>& input, std::vector<std::vector<YoloObject>>& results,
                     const std::vector<std::pair<float, cv::Size2i>>& ratios) {
    if (!initialized) {
      return false;
    }
    if (false) {
      for (int i = 0; i < input.size(); ++i) {
        cv::Mat image;
        input[i].download(image);
        image.convertTo(image, CV_8UC3, 255.0F);
        cv::imwrite("image_" + std::to_string(i) + ".png", image);
      }
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    if (!model->SetInputVectorGpu(0, input)) {
      std::cout << "YoloObjectDetector::DetectObjects() - failed to SetInputVector" << std::endl;
      return false;
    }
    if (!model->RunInference(input.size())) {
      std::cout << "Failed to run inference" << std::endl;
      return false;
    }
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> inference_time = t2 - t1;

    ParseResults(model->GetOutputVector(), results, ratios);
    // auto t3 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> postprocess_time = t3 - t2;
    // std::cout << "Yolo inference time: " << inference_time.count()
    //           << " ms, postprocess time: " << postprocess_time.count()
    //           << " ms, total: " << (inference_time + postprocess_time).count() << " ms" << std::endl;
    return true;
  }

  virtual cv::cuda::GpuMat ProcessImage(const cv::cuda::GpuMat& input, float& ratio, bool is_rgb) {
    cv::cuda::GpuMat frame;
    cv::cuda::GpuMat tmp;
    const cv::cuda::GpuMat* ptr = &input;
    tmp =
      cuda_common::TRTEngine::ResizeKeepAspectRatioPadRightBottomGPU(*ptr, input_size.x(), input_size.y(), 0, ratio);
    if (!is_rgb && ptr->type() == CV_8UC1) {
      cv::cuda::cvtColor(tmp, frame, cv::COLOR_BayerBG2BGR);
      tmp = frame;
    }

    tmp.convertTo(frame, CV_32FC3, 1 / 255.0F);
    return frame;
  }
};

YoloObjectDetector::YoloObjectDetector() { impl_ = std::make_shared<YoloObjectDetectorImpl>(); }
bool YoloObjectDetector::Initialize(const std::string& model_path, YoloConfig::SharedPtr config,
                                    PlayerPoseParameters::SharedPtr player_params) {
  return impl_->Initialize(model_path, config, player_params);
}
bool YoloObjectDetector::DetectObjects(const std::vector<cv::cuda::GpuMat>& input,
                                       std::vector<std::vector<YoloObject>>& results,
                                       const std::vector<std::pair<float, cv::Size2i>>& ratios) {
  return impl_->DetectObjects(input, results, ratios);
}

cv::cuda::GpuMat YoloObjectDetector::ProcessImage(const cv::cuda::GpuMat& image, float& ratio, bool is_rgb) {
  return impl_->ProcessImage(image, ratio, is_rgb);
}
const Eigen::Vector2i& YoloObjectDetector::GetInputSize() { return impl_->input_size; }

}  // namespace perception
