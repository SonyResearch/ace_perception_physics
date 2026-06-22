// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "evs_ball_cpp/imageCHW.hpp"

ImageCHW::ImageCHW() = default;

ImageCHW::ImageCHW(int channels, int height, int width) : channels_(channels), height_(height), width_(width) {
  pixels_.resize(channels_ * height_ * width_);
}

half_float::half& ImageCHW::At(int c, int h, int w) { return pixels_[c * height_ * width_ + h * width_ + w]; }

void ImageCHW::ResetToZero() { std::fill(pixels_.begin(), pixels_.end(), 0); }

half_float::half* ImageCHW::GetDataPointer() {
  return pixels_.data();  // Access the underlying data array using std::vector's data() method
}

std::vector<half_float::half> ImageCHW::GetVals() {
  return pixels_;  // Access the underlying data array using std::vector's data() method
}
