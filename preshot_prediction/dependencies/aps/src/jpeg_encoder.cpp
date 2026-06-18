// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "aps/jpeg_encoder.hpp"

#include <chrono>
#include <fstream>

#include "nvjpeg.h"

namespace aps {
using namespace std::chrono_literals;

JpegEncoder::JpegEncoder(const int cuda_device_id, const bool debayer_only, int quality)
  : cuda_device_id_(cuda_device_id),
    debayer_only_(debayer_only),
    quality_(quality)
// host_mem_(cv::cuda::HostMem::PAGE_LOCKED)
{
  message_available_ = false;

  canceled_.store(false);
  t_ = std::thread(std::bind(&JpegEncoder::WorkerThread, this));
  t_.detach();
}

JpegEncoder::~JpegEncoder() {
  canceled_.store(true);
  if (t_.joinable()) {
    t_.join();
  }
}

void JpegEncoder::SaveImage(const ace_interfaces::msg::ImageData::ConstSharedPtr &message,
                            const std::string &filename) {
  std::unique_lock<std::mutex> lck(m_);
  message_ = message;
  filename_ = filename;
  message_available_ = true;
  lck.unlock();
  cv_.notify_all();
}

void JpegEncoder::WorkerThread() {
  cv::Mat bgr_image_cpu;

  cv::cuda::setDevice(cuda_device_id_);
  cv::cuda::Stream cv_stream;
  cv::cuda::GpuMat raw_image;
  cv::cuda::GpuMat bgr_image;

  raw_image.create(1080, 1440, CV_8UC1);
  bgr_image.create(1080, 1440, CV_8UC3);

  cudaSetDevice(cuda_device_id_);
  nvjpegImage_t nv_image;
  nvjpegHandle_t nv_handle;
  nvjpegEncoderState_t nv_enc_state;
  nvjpegEncoderParams_t nv_enc_params;
  nvjpegInputFormat_t nv_input_format;
  cudaStream_t nv_stream = nullptr;

  memset(&nv_image, 0, sizeof(nv_image));
  nvjpegCreateSimple(&nv_handle);
  nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, nv_stream);
  nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, nv_stream);

  nvjpegEncoderParamsSetEncoding(nv_enc_params, NVJPEG_ENCODING_BASELINE_DCT, nv_stream);
  nvjpegEncoderParamsSetQuality(nv_enc_params, quality_, nv_stream);
  nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 1, nv_stream);
  nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, nv_stream);
  cudaStreamSynchronize(nv_stream);

  nv_input_format = NVJPEG_INPUT_BGRI;

  std::vector<unsigned char> jpeg;
  while (!canceled_.load()) {
    std::unique_lock<std::mutex> lck(m_);
    if (!cv_.wait_for(lck, 1s, [&] { return message_available_; })) {
      continue;
    }

    // copy pointer to image and return lock
    ace_interfaces::msg::ImageData::ConstSharedPtr message = message_;
    message_.reset();
    std::string filename = filename_;
    message_available_ = false;
    lck.unlock();

    // upload image to GPU, convert to BGR, and split into single channels
    void *data_ptr = reinterpret_cast<void *>(
      const_cast<unsigned char *>(message->data.data() + message->step * message->offset_y + message->offset_x));
    const cv::Mat img = cv::Mat(message->height, message->width, CV_8UC1, data_ptr, message->step);
    raw_image.upload(img, cv_stream);
    // raw_image.upload(host_mem_);
    cv::cuda::cvtColor(raw_image, bgr_image, cv::COLOR_BayerBG2BGR, 0, cv_stream);
    cv_stream.waitForCompletion();
    // lck.unlock();
    message.reset();

    if (debayer_only_) {
      bgr_image.download(bgr_image_cpu);
      cv::imwrite(filename_, bgr_image_cpu);
    } else {
      // Fill nv_image.
      nv_image.channel[0] = bgr_image.data;
      nv_image.pitch[0] = static_cast<unsigned int>(bgr_image.step);

      // Encode image.
      nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &nv_image, nv_input_format, bgr_image.cols,
                        bgr_image.rows, nv_stream);

      // Get compressed stream size.
      size_t length;
      nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, nullptr, &length, nv_stream);

      // Get stream itself.
      cudaStreamSynchronize(nv_stream);
      jpeg.reserve(length);
      nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg.data(), &length, nv_stream);

      cudaStreamSynchronize(nv_stream);
      std::ofstream output_file(filename_, std::ios::out | std::ios::binary);
      output_file.write(reinterpret_cast<char *>(jpeg.data()), static_cast<std::streamsize>(length));
      output_file.close();
    }
  }
}

}  // namespace aps
