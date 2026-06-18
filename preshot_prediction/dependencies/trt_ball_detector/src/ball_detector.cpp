// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include "trt_ball_detector/ball_detector.hpp"

#include <queue>

#include "cuda_common/TRTEngine.hpp"

namespace trt_ball_detector {

const int Width = 1440;
const int Height = 1080;
class BallDetector::BallDetectorImpl {
 public:
  BallDetectorImpl() = default;
  ~BallDetectorImpl() {
    if (encoded_feature_buffer) {
      cudaFree(encoded_feature_buffer);
    }
    if (fixed_embedding_staging_input_) {
      cudaFree(fixed_embedding_staging_input_);
    }
    if (fixed_embedding_staging_output_) {
      cudaFree(fixed_embedding_staging_output_);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }

  void Initialize(BallDetectorParameters::SharedPtr parameters) {
    // initialize the implementation of ball detector, e.g., build/load TRT engines, allocate GPU buffers, etc.
    this->parameters = parameters;

    auto trt_options = cuda_common::TRTEngineOptions();
    trt_options.max_batch_size = parameters->batch_size;
    trt_options.opt_batch_size = parameters->batch_size;
    trt_options.precision = cuda_common::TRTEngineOptions::Precision::kFP16;
    trt_options.input_format = cuda_common::TRTEngineOptions::InputFormat::kArray;
    trt_options.device_index = parameters->device_id;

    model_engine = cuda_common::TRTEngine::SharedPtr(new cuda_common::TRTEngine(trt_options));
    model_engine->BuildAndLoadNetwork(parameters->onnx_engine_path);

    // prepare a CUDA stream for encoder and decoder to share;
    cudaStreamCreate(&this->stream);

    // calculate number of embeddings (window_size)
    const auto& model_in_names = model_engine->GetInputTensorNames();
    this->window_size = 1;
    this->image_index = -1;
    for (int i = 0; i < model_in_names.size(); ++i) {
      const std::string& name = model_in_names[i];
      if (name.rfind("in_embedding_", 0) == 0) {  // starts with "in_embedding_"
        this->window_size++;
      } else if (name == "images") {
        image_index = i;
      } else if (name == "in_sequence_id") {
        in_seq_id_index = i;
      }
    }
    input_embedding_indices.resize(this->window_size - 1, -1);
    for (int i = 0; i < model_in_names.size(); ++i) {
      const std::string& name = model_in_names[i];
      if (name.rfind("in_embedding_", 0) == 0) {  // starts with "in_embedding_"
        int idx = std::stoi(name.substr(std::string("in_embedding_").size()));
        if (idx < this->window_size - 1) {
          input_embedding_indices[idx] = i;
        } else {
          std::cerr << "[WARN] Detected embedding index " << idx << " exceeds expected window size "
                    << this->window_size - 1 << ".\n";
        }
      }
    }

    const auto& out_names = model_engine->GetOutputTensorNames();
    output_embedding_indices.resize(this->window_size - 1, -1);
    int feat_numel = -1;
    for (int i = 0; i < out_names.size(); ++i) {
      if (out_names[i].rfind("out_embedding_", 0) == 0) {  // starts with "out_embedding_"
        int idx = std::stoi(out_names[i].substr(std::string("out_embedding_").size()));
        output_embedding_indices[idx] = i;
        d2h_excluded_indices_.push_back(i);
        const auto& d = model_engine->GetOutputDims()[i];
        feat_numel = 1;
        // std::cout << out_names[i] << " - " << idx << "  Embedding shape: [";
        for (int k = 0; k < d.nbDims; ++k) {
          // std::cout << d.d[k] << (k + 1 < d.nbDims ? "," : "");
          feat_numel *= static_cast<size_t>(std::max(1, static_cast<int>(d.d[k])));
        }
        // std::cout << "]  (" << feat_numel << " elements)\n";
      } else if (out_names[i] == "heatmap") {
        heatmap_out_index = i;
      } else if (out_names[i] == "out_sequence_id") {
        out_seq_id_index = i;
      }
    }

    std::cout << "Determined decoder window size: " << this->window_size << "\n";

    const size_t feat_elem_size = model_engine->GetOutputElementSize(d2h_excluded_indices_[0]);  // bytes per element
    this->feat_stride = feat_numel * feat_elem_size;
    this->batch_feat_stride = feat_stride * parameters->batch_size;
    // decoding_results holds one result per camera view (batch item); sized to batch_size, not window_size.
    decoding_results.reserve(parameters->batch_size);
    for (int i = 0; i < parameters->batch_size; i++) {
      decoding_results.push_back(std::make_shared<DecodingResult>());
    }

    encoded_feature_buffer = nullptr;
    if (cudaMalloc(&encoded_feature_buffer, static_cast<size_t>(this->window_size - 1) * batch_feat_stride) !=
        cudaSuccess) {
      throw std::runtime_error("Failed to allocate GPU staging buffer for features.");
    }
    // Zero-initialise so that padding frames in ResetEncodedImages() see clean data.
    cudaMemset(encoded_feature_buffer, 0, static_cast<size_t>(this->window_size - 1) * batch_feat_stride);

    // Wire TRT context inputs to the fixed staging addresses (done once, permanently).
    for (int t = 0; t < this->window_size - 1; ++t) {
      void* slot = static_cast<char*>(encoded_feature_buffer) + t * batch_feat_stride;
      if (!model_engine->SetExternalInputGpu(input_embedding_indices[t], slot)) {
        throw std::runtime_error("Failed to set fixed external GPU input for in_embedding_" + std::to_string(t));
      }
      if (!model_engine->SetExternalOutputGpu(output_embedding_indices[t], slot)) {
        throw std::runtime_error("Failed to set fixed external GPU output for out_embedding_" + std::to_string(t));
      }
    }
    // skip heatmap if the caller never needs it — avoids wasted PCIe bandwidth.
    if (!parameters->copy_heatmap && heatmap_out_index >= 0) {
      d2h_excluded_indices_.push_back(heatmap_out_index);
    }

    // Warm up the circular buffer with blank frames.
    ResetEncodedImages();

    // Capture the inference CUDA graph now that input shapes and tensor addresses are stable.
    // Sync() drains the last padding frame and idles the stream before capture.
    Sync();
    if (!model_engine->CaptureGraph(this->stream, parameters->batch_size)) {
      std::cerr << "[WARN] BallDetector: CUDA graph capture failed; falling back to direct enqueueV3.\n";
    }
  }

  // Sync the most recently enqueued frame if still in flight: cudaStreamSynchronize,
  // UnpackPinnedOutputs, DecodeBatch. No-op when no frame is in flight.
  void Sync() {
    if (!frame_in_flight_) {
      return;
    }
    cudaStreamSynchronize(stream);
    model_engine->UnpackPinnedOutputs(in_flight_batch_size_, d2h_excluded_indices_);
    DecodeBatch(in_flight_batch_size_);
    frame_in_flight_ = false;
  }

  void ResetEncodedImages() {
    // Drain any GPU work from a previous sequence before wiping state.
    Sync();
    last_encoded_image_buffer = std::queue<uint64_t>();

    // Encode padding images to warm up the embedding circular buffer.
    std::vector<cv::Mat> padding_images(parameters->batch_size, cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0, 0, 0)));
    for (int i = 0; i < this->window_size - 1; ++i) {
      EncodeImages(0, padding_images);
    }
  }

  // Encode a batch of images that belongs to the same timestamp
  bool EncodeImages(uint64_t sequence_id, const std::vector<cv::Mat>& images) {
    const size_t total_buf = static_cast<size_t>(this->window_size) * batch_feat_stride;

    // 1. Async H2D: upload image pixels to the engine's fixed input buffer.
    for (size_t index = 0; index < images.size(); index++) {
      const auto& image = images[index];
      float seq_id_float = static_cast<float>(sequence_id);
      if (!model_engine->CopyToInputDirectly(in_seq_id_index, index, &seq_id_float, sizeof(seq_id_float), 0, stream)) {
        throw std::runtime_error("Failed to copy sequence ID for frame " + std::to_string(index) + " to encoder.");
      }
      if (!model_engine->CopyToInputDirectly(image_index, index, image.data,
                                             static_cast<size_t>(Height * Width) * sizeof(unsigned char), 0, stream)) {
        throw std::runtime_error("Failed to copy frame " + std::to_string(index) + " to encoder.");
      }
    }

    // 2. Async D2D: copy each previous embedding from the circular buffer into the
    //    fixed-address staging inputs. The TRT context's tensor addresses are wired to
    //    these staging slots permanently (set once in Initialize()), so no setTensorAddress
    //    call is needed here.
    // for (int t = 0; t < this->window_size - 1; ++t) {
    //   if (embedding_indices[t] < 0) {
    //     std::cerr << "[WARN] Decoder input 'embedding_" << t << "' not found — skipping.\n";
    //     continue;
    //   }
    //   const size_t offset = (active_offset + static_cast<size_t>(t + 1) * batch_feat_stride) % total_buf;
    //   const void* src = static_cast<const char*>(encoded_feature_buffer) + offset;
    //   void* dst = static_cast<char*>(fixed_embedding_staging_input_) + t * batch_feat_stride;
    //   cudaMemcpyAsync(dst, src, batch_feat_stride, cudaMemcpyDeviceToDevice, stream);
    // }

    // Deferred-sync pipeline:
    //   Sync + decode the PREVIOUS frame here (GPU is likely already done if the caller
    //   spent any time between calls). Then enqueue the CURRENT frame and return immediately.
    //   GetDecodingResults() calls Sync() so results are always valid when read.
    Sync();

    // 3. Run inference.
    //    - If the CUDA graph was captured in Initialize(), replay it (~10µs CPU overhead).
    //    - Otherwise fall back to direct enqueueV3 (~500µs CPU overhead).
    //    TRT writes the new embedding to fixed_embedding_staging_output_.
    if (model_engine->IsGraphCaptured()) {
      if (!model_engine->RunGraphOnStream(stream)) {
        throw std::runtime_error("CUDA graph launch failed");
      }
    } else {
      if (!model_engine->RunInferenceOnStream(static_cast<int>(images.size()), stream)) {
        throw std::runtime_error("Image encoder enqueue failed");
      }
    }

    // 4. Async D2D: move the freshly produced embedding from the fixed staging output
    //    into the current slot of the circular buffer.
    // cudaMemcpyAsync(static_cast<char*>(encoded_feature_buffer) + active_offset,
    //                 fixed_embedding_staging_output_, batch_feat_stride, cudaMemcpyDeviceToDevice, stream);

    // 5. Async D2H for all non-embedding (and optionally non-heatmap) outputs.
    //    d2h_excluded_indices_ is precomputed in Initialize(): no allocation on this hot path.
    if (!model_engine->CopyOutputsOnStream(static_cast<int>(images.size()), stream, d2h_excluded_indices_)) {
      throw std::runtime_error("Failed to copy decoder outputs to CPU.");
    }

    // // 6. Advance the circular buffer pointer (CPU only; stream ordering guarantees the D2D
    // //    write from step 4 is visible to the next frame's D2D read from step 2).
    last_encoded_image_buffer.push(sequence_id);
    // // active_offset = (active_offset + batch_feat_stride) % total_buf;
    if (last_encoded_image_buffer.size() > static_cast<size_t>(this->window_size)) {
      last_encoded_image_buffer.pop();
    }

    // 7. Mark in-flight. GPU is still running; sync/unpack/decode are deferred to
    //    Sync(), which is called at the start of the next EncodeImages() and
    //    inside GetDecodingResults() — whichever comes first.
    in_flight_batch_size_ = static_cast<int>(images.size());
    frame_in_flight_ = true;
    return true;
  }

  bool DecodeBatch(int batch_size) {
    const auto& out_names = model_engine->GetOutputTensorNames();
    const auto& out_dims = model_engine->GetOutputDims();
    const auto& dec_cpu = model_engine->GetOutputVector();  // [batch][output_idx]
    // All images in this batch share the same timestamp: use the most recently pushed sequence_id.
    const uint64_t current_sequence_id = last_encoded_image_buffer.back();
    active_decoding_results.resize(batch_size);
    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
      auto result = decoding_results[batch_index];
      result->reset();
      active_decoding_results[batch_index] = result;
      // result->sequence_id = current_sequence_id;

      // post-process the decoder outputs and fill in the decoding results (ball positions, velocities, heatmap) for
      // each image in the batch here we just copy the raw decoder outputs to the decoding results for demonstration;
      // you can replace this with your actual post-processing logic
      for (size_t oi = 0; oi < out_names.size(); ++oi) {
        const std::string& name = out_names[oi];
        const auto& vals = dec_cpu[batch_index][oi];  // batch batch_index, output oi

        if (name == "coordinates") {
          for (size_t i = 0; i < vals.size(); i += 2) {
            float cx = vals[i];
            float cy = vals[i + 1];

            // undo the padding from cx,cy. Currently there is padding applied on cy only
            const int pad_w = 0;
            const int pad_h = 80;  // 80 pixels of padding applied to make it 640x640
            const int new_w = 640;
            const int new_h = 640;
            const int width = new_w;
            const int height = Height * new_w / Width;
            cx = std::clamp((cx * new_w - pad_w) / std::max(new_w, 1), 0.0f, 1.0f);
            cy = std::clamp((cy * new_h - pad_h) / std::max(height, 1), 0.0f, 1.0f);

            result->ball_positions.push_back(Eigen::Vector2f(cx, cy));
          }
        } else if (name == "confidence") {
          for (size_t i = 0; i < vals.size(); i += 1) {
            result->confidences.push_back(vals[i]);
          }
        } else if (name == "velocity") {
          for (size_t i = 0; i < vals.size(); i += 2) {
            result->ball_velocities.push_back(Eigen::Vector2f(vals[i], vals[i + 1]));
          }
        } else if (name == "radius") {
          for (size_t i = 0; i < vals.size(); i += 1) {
            result->radius.push_back(vals[i]);
          }
        } else if (name == "out_sequence_id") {
          // Optional sanity check: output sequence ID should match the input sequence ID for this batch.
          // std::cout << "Output sequence ID (float): " << vals[0]<<"/"<<current_sequence_id << "\n";
          result->sequence_id = static_cast<uint64_t>(vals[0]);
        } else if (name == "heatmap") {
          const auto& od = out_dims[oi];
          int heatmap_height = (od.nbDims >= 2) ? static_cast<int>(od.d[od.nbDims - 2]) : 1;
          int heatmap_width = (od.nbDims >= 1) ? static_cast<int>(od.d[od.nbDims - 1]) : static_cast<int>(vals.size());
          if (heatmap_height <= 0) heatmap_height = 1;
          if (heatmap_width <= 0) heatmap_width = 1;
          const int num_elems = static_cast<int>(vals.size());
          if (heatmap_height * heatmap_width != num_elems) {
            heatmap_height = 1;
            heatmap_width = num_elems;
          }
          // UnpackPinnedOutputs guarantees vals contains real FP32 values.
          // Use memcpy so result->heatmap owns its data; wrapping vals.data() directly
          // would leave a dangling/aliased pointer after the next inference call.
          if (parameters->copy_heatmap) {
            result->heatmap = cv::Mat(heatmap_height, heatmap_width, CV_32F);
            std::memcpy(result->heatmap.data, vals.data(), static_cast<size_t>(num_elems) * sizeof(float));
          }
        }
      }
    }

    return true;
  }

  BallDetectorParameters::SharedPtr parameters;
  int window_size{1};                         // number of inputs to the decoder
  int image_index{-1};                        // index of the image input to the decoder; -1 if not found
  int in_seq_id_index{-1};                    // index of the sequence ID input to the decoder; -1 if not found
  int out_seq_id_index{-1};                   // index of the sequence ID output from the decoder; -1 if not found
  std::vector<int> input_embedding_indices;   // indices of the embedding inputs to the decoder
  std::vector<int> output_embedding_indices;  // indices of the embedding inputs to the decoder
  // int embedding_out_index{-1}; // index of the embedding output from the encoder; -1 if not found
  int heatmap_out_index{-1};  // index of the heatmap output; -1 if not found

  cuda_common::TRTEngine::SharedPtr model_engine;
  cudaStream_t stream;

  std::vector<DecodingResult::SharedPtr> decoding_results;
  std::vector<DecodingResult::SharedPtr> active_decoding_results;
  void* encoded_feature_buffer{nullptr};  // GPU circular buffer storing embeddings for a window of frames
  // Fixed-address GPU staging buffers used with CUDA graph capture.
  // Addresses are wired into the TRT context once in Initialize() and never change.
  // Hot path: D2D copies move data between the circular buffer and these fixed slots.
  void* fixed_embedding_staging_input_{nullptr};   // (window_size-1) × batch_feat_stride
  void* fixed_embedding_staging_output_{nullptr};  // 1 × batch_feat_stride
  // Pre-computed exclusion list for CopyOutputsOnStream: embedding always excluded;
  // heatmap excluded when copy_heatmap=false. Built once in Initialize().
  std::vector<int> d2h_excluded_indices_;
  // Deferred-sync state: GPU work is left running after EncodeImages() returns.
  bool frame_in_flight_{false};
  int in_flight_batch_size_{0};
  std::queue<uint64_t> last_encoded_image_buffer;
  // size_t active_offset{0};  // circular buffer offset for the current frames
  size_t feat_stride{
    0};  // stride (in bytes) of each encoded feature vector, determined by the encoder's output dims and element size
  size_t batch_feat_stride{0};  // stride (in bytes) of each batch of encoded features, i.e., feat_stride * batch_size
};

BallDetector::BallDetector() : impl_(std::make_shared<BallDetectorImpl>()) {}

bool BallDetector::Initialize(BallDetectorParameters::SharedPtr parameters) {
  impl_->Initialize(parameters);
  return true;
}

void BallDetector::ResetEncodedImages() { impl_->ResetEncodedImages(); }

// Encode a batch of images that belongs to the same timestamp
bool BallDetector::EncodeImages(uint64_t sequence_id, const std::vector<cv::Mat>& images) {
  // encode the input images using image_encoder_engine, and store the encoded features for later decoding
  return impl_->EncodeImages(sequence_id, images);
}

std::vector<DecodingResult::SharedPtr> BallDetector::GetDecodingResults() {
  // Flush any in-flight GPU work so results are valid.
  // In a pipelined caller (other CPU work between EncodeImages and GetDecodingResults),
  // the GPU is already done here — zero sync wait.
  impl_->Sync();
  return impl_->active_decoding_results;
}

}  // namespace trt_ball_detector
