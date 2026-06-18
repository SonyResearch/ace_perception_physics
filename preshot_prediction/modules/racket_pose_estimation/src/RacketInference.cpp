// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "racket_pose_estimation/RacketInference.hpp"

#include <cuda_common/TRTEngine.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "ace_loggers/ace_loggers.hpp"

namespace perception {

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  for (auto& x : v) {
    out << x << ", ";
  }
  out << "\b\b]";
  return out;
}
template <typename T, typename V>
std::ostream& operator<<(std::ostream& out, const std::unordered_map<T, V>& v) {
  out << "{";
  for (auto& x : v) {
    out << x.first << ":" << x.second << ", ";
  }
  out << "\b\b}";
  return out;
}

class RacketInference::RacketInferenceImpl {
 public:
  Eigen::Vector3i input_size;
  bool initialized{false};
  cuda_common::TRTEngine::SharedPtr model;
  int output_index{0};
  nvinfer1::Dims output_dim;

  RacketParameters::SharedPtr racket_params;

  RacketInferenceImpl() = default;

  bool InitializeModel(const std::string& onnx_model_path, RacketParameters::SharedPtr params) {
    racket_params = params;
    {
      cuda_common::TRTEngineOptions options;
      options.opt_batch_size = params->extractor.opt_batch_size;
      options.max_batch_size = params->extractor.max_batch_size;
      options.device_index = params->extractor.device_index;
      options.opt_width = 640;
      options.opt_height = 640;

      options.precision = cuda_common::TRTEngineOptions::Precision::kFP16;
      options.input_format = cuda_common::TRTEngineOptions::InputFormat::kBCHW;

      model = std::make_shared<cuda_common::TRTEngine>(options);
      auto succ = model->BuildAndLoadNetwork(onnx_model_path);
      if (!succ) {
        const std::string err_msg =
          "RacketInference - Error: Unable to build the TensorRT engine. "
          "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(err_msg);
      }
    }
    for (int i = 0; i < 200; ++i) {
      model->RunInference(1);
    }
    input_size.x() = static_cast<Eigen::Vector3i::Scalar>(model->GetInputDims()[0].d[2]);
    input_size.y() = static_cast<Eigen::Vector3i::Scalar>(model->GetInputDims()[0].d[1]);
    input_size.z() = static_cast<Eigen::Vector3i::Scalar>(model->GetInputDims()[0].d[0]);
    auto output = model->GetOutputDims();
    output_index = model->GetOutputIndex("output0");
    if (output_index == -1) {
      // check for v7
      output_index = model->GetOutputIndex("output");
    }
    if (output_index == -1) {
      throw std::runtime_error("Failed to find the correct output index for the model!");
    }
    output_dim = output[output_index];
    initialized = true;

    LOG(INFO) << "Input size: " << input_size.x() << "x" << input_size.y() << "@" << input_size.z();
    LOG(INFO) << "Output size: " << output_dim.d[0] << "," << output_dim.d[1] << "," << output_dim.d[2];

    return true;
  }
};

RacketInference::RacketInference() { impl_ = std::make_shared<RacketInferenceImpl>(); }

bool RacketInference::Initialize(const std::string& model_path, RacketParameters::SharedPtr params) {
  return impl_->InitializeModel(model_path, params);
}
bool RacketInference::ParseResults(const float* output, RacketPoseFeatures::SharedPtr& results) {
  /////generate proposals
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<int> class_ids;
  std::vector<std::vector<cv::Point3f>> keypoints;
  float ratioh = 1;  //(float)height / (float)impl_->input_size.y();
  float ratiow = 1;  // (float)width / (float)impl_->input_size.x();

  bool transposed = false;
  auto rows = impl_->output_dim.d[0];
  auto cols = impl_->output_dim.d[1];
  int kpt_offset = 6;  // v7
  if (impl_->output_dim.d[0] < impl_->output_dim.d[1]) {
    rows = impl_->output_dim.d[1];
    cols = impl_->output_dim.d[0];
    kpt_offset = 5;  // v8
    transposed = true;
  }
#define ELEMENT_INDEX(idx) (transposed ? (rows * (idx)) : (idx))
  const float* pdata = output;
  for (int row_ind = 0; row_ind < rows; row_ind++) {
    float box_score = pdata[ELEMENT_INDEX(4)];
    if (box_score > impl_->racket_params->extractor.confidence_threshold) {
      const int class_idx = 0;
      int cx = static_cast<int>(pdata[ELEMENT_INDEX(0)] * ratiow);  /// cx
      int cy = static_cast<int>(pdata[ELEMENT_INDEX(1)] * ratioh);  /// cy
      int w = static_cast<int>(pdata[ELEMENT_INDEX(2)] * ratiow);   /// w
      int h = static_cast<int>(pdata[ELEMENT_INDEX(3)] * ratioh);   /// h
      int left = static_cast<int>(cx - w / 2);
      int top = static_cast<int>(cy - h / 2);

      std::vector<cv::Point3f> kpts;
      // keypoints
      for (int kpi = 0; kpi < 4; ++kpi) {
        float x = pdata[ELEMENT_INDEX(kpt_offset + kpi * 3 + 0)] * ratiow;
        float y = pdata[ELEMENT_INDEX(kpt_offset + kpi * 3 + 1)] * ratioh;
        float c = pdata[ELEMENT_INDEX(kpt_offset + kpi * 3 + 2)];

        kpts.emplace_back(cv::Point3f(x, y, c));
      }
      auto box = cv::Rect(left, top, w, h);
      confidences.push_back(box_score);
      boxes.push_back(box);
      class_ids.push_back(class_idx);
      keypoints.push_back(kpts);
    }
    if (transposed) {
      pdata++;
    } else {
      pdata += cols;
    }
  }

  // Perform non maximum suppression to eliminate redundant overlapping boxes with
  // lower confidences
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, impl_->racket_params->extractor.confidence_threshold,
                    impl_->racket_params->extractor.nms_threshold, indices);

  if (indices.empty()) {
    return false;
  }

  int best_idx = -1;
  float best_conf = 0;

  for (auto idx : indices) {
    auto conf = confidences[idx];
    if (conf > best_conf) {
      best_conf = conf;
      best_idx = idx;
    }
  }

  if (best_idx == -1) {
    return false;
  }

  int idx = best_idx;
  results->confidence = confidences[idx];
  results->bbox(0) = boxes[idx].x;
  results->bbox(1) = boxes[idx].y;
  results->bbox(2) = boxes[idx].width;
  results->bbox(3) = boxes[idx].height;
  results->center = Eigen::Vector3i(boxes[idx].x + boxes[idx].width / 2, boxes[idx].y + boxes[idx].height / 2,
                                    std::max<int>(boxes[idx].width, boxes[idx].height));
  for (int k = 0; k < 4; ++k) {
    results->keypoints[k](0) = keypoints[idx][k].x;
    results->keypoints[k](1) = keypoints[idx][k].y;
    results->keypoints[k](2) = keypoints[idx][k].z * 100;
  }
  return true;
}

bool RacketInference::DetectRacketKeypoints(const std::vector<cv::cuda::GpuMat>& inputs, RacketFrameFeatures& results) {
  if (!impl_->initialized) {
    return false;
  }
  if (!impl_->model->SetInputVectorGpu(0, inputs)) {
    LOG(WARNING) << "Failed to set input vector: " << inputs.size();
    return false;
  }
  if (!impl_->model->RunInference(static_cast<int>(inputs.size()))) {
    LOG(WARNING) << "Failed inferencing: " << inputs.size();
    return false;
  }
  const auto& output_vector = impl_->model->GetOutputVector();

  for (size_t i = 0; i < inputs.size(); ++i) {
    const float* data_ptr = output_vector[i][impl_->output_index].data();
    if (!ParseResults(data_ptr, results.features[i])) {
      results.features[i] = nullptr;
      // LOG(INFO) << "Failed to parse any results";
    }
  }
  return true;
}

cv::cuda::GpuMat RacketInference::PreProcessImage(const cv::Mat& image) {
  cv::cuda::GpuMat frame;
  cv::cuda::GpuMat converted;
  frame = cv::cuda::GpuMat(impl_->input_size.y(), impl_->input_size.x(), image.type());
  frame.setTo(0);
  int height = std::min(image.rows, frame.rows);
  int width = std::min(image.cols, frame.cols);
  cv::Rect roi(0, 0, width, height);
  auto source = image(roi);
  auto target = frame(roi);
  source.copyTo(target);
  if (impl_->input_size.z() == 3) {
    if (image.type() == CV_8UC1) {
      cv::cuda::cvtColor(frame, converted, cv::COLOR_BayerBG2RGB);
    } else {
      cv::cuda::cvtColor(frame, converted, cv::COLOR_RGB2BGR);
    }
    converted.convertTo(frame, CV_32FC3, 1 / 255.0F);
  } else {
    if (image.type() == CV_8UC1) {
      frame.convertTo(frame, CV_32FC1, 1 / 255.0F);
    }  // else we don't know how to convert this yet!
    else {
      throw std::runtime_error("Unable to convert RGB image to bayer");
    }
  }
  return frame;
}

const Eigen::Vector3i& RacketInference::GetInputSize() { return impl_->input_size; }

}  // namespace perception
