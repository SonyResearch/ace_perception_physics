// Confidential, Copyright 2025, Sony AI, All rights reserved

#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

// Kernel: Pack 2x2 Bayer blocks into RGBA pixels
__global__ void packBayerToBlocks(const unsigned char* bayer, int bayerPitch, uchar4* blocks, int blockPitch, int width,
                                  int height) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;

  if (bx >= width / 2 || by >= height / 2) return;

  int x = bx * 2;
  int y = by * 2;

  const unsigned char* row0 = bayer + y * bayerPitch;
  const unsigned char* row1 = bayer + (y + 1) * bayerPitch;

  uchar4 block;
  block.x = row0[x];      // top-left
  block.y = row0[x + 1];  // top-right
  block.z = row1[x];      // bottom-left
  block.w = row1[x + 1];  // bottom-right

  ((uchar4*)((char*)blocks + by * blockPitch))[bx] = block;
}

// Kernel: Unpack RGBA blocks back into 2x2 Bayer pixels
__global__ void unpackBlocksToBayer(const uchar4* blocks, int blockPitch, unsigned char* bayer, int bayerPitch,
                                    int width, int height) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;

  if (bx >= width / 2 || by >= height / 2) return;

  int x = bx * 2;
  int y = by * 2;

  uchar4 block = ((const uchar4*)((const char*)blocks + by * blockPitch))[bx];

  unsigned char* row0 = bayer + y * bayerPitch;
  unsigned char* row1 = bayer + (y + 1) * bayerPitch;

  row0[x] = block.x;
  row0[x + 1] = block.y;
  row1[x] = block.z;
  row1[x + 1] = block.w;
}

// Main resize function
void resizeBayerCuda(const cv::cuda::GpuMat& bayerImage, cv::cuda::GpuMat& resizedBayer) {
  int h = bayerImage.rows;
  int w = bayerImage.cols;

  int newWidth = resizedBayer.cols;
  int newHeight = resizedBayer.rows;

  if (h % 2 != 0 || w % 2 != 0) throw std::runtime_error("Input dimensions must be even.");
  if (newWidth % 2 != 0 || newHeight % 2 != 0) throw std::runtime_error("Output dimensions must be even.");

  // Allocate GPU memory for packed blocks
  cv::cuda::GpuMat blocks(h / 2, w / 2, CV_8UC4);

  dim3 blockDim(16, 16);
  dim3 gridDim((w / 2 + blockDim.x - 1) / blockDim.x, (h / 2 + blockDim.y - 1) / blockDim.y);
  // Pack Bayer -> blocks
  packBayerToBlocks<<<gridDim, blockDim>>>(bayerImage.ptr<unsigned char>(), bayerImage.step,
                                           (uchar4*)blocks.ptr<uchar4>(), blocks.step, w, h);
  cudaDeviceSynchronize();

  // Resize block grid on GPU
  cv::cuda::GpuMat resizedBlocks;
  cv::cuda::resize(blocks, resizedBlocks, cv::Size(newWidth / 2, newHeight / 2), 0, 0, cv::INTER_NEAREST);

  dim3 gridDim2((newWidth / 2 + blockDim.x - 1) / blockDim.x, (newHeight / 2 + blockDim.y - 1) / blockDim.y);

  // Unpack blocks -> Bayer
  unpackBlocksToBayer<<<gridDim2, blockDim>>>((const uchar4*)resizedBlocks.ptr<uchar4>(), resizedBlocks.step,
                                              resizedBayer.ptr<unsigned char>(), resizedBayer.step, newWidth,
                                              newHeight);
  cudaDeviceSynchronize();
}
