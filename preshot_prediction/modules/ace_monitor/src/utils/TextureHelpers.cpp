// Confidential, Copyright 2025, Sony AI, All rights reserved
#include "ace_monitor/utils/TextureHelpers.hpp"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace ace_monitor {

glm::vec4 TextureHelpers::GenerateUniqueColor(int color_seed) {
  // Assign a visually distinct color using HSV to RGB conversion
  auto hue = static_cast<float>(std::fmod(0.23F * static_cast<float>(color_seed), 1.0F));  // step through hues
  auto s = 0.7F;
  auto v = 0.95F;
  auto c = v * s;
  auto x = c * static_cast<float>(1 - std::fabs(std::fmod(hue * 6, 2) - 1));
  auto m = v - c;
  auto r = 0.0F;
  auto g = 0.0F;
  auto b = 0.0F;
  if (hue < 1.0F / 6) {
    r = c;
    g = x;
    b = 0;
  } else if (hue < 2.0F / 6) {
    r = x;
    g = c;
    b = 0;
  } else if (hue < 3.0F / 6) {
    r = 0;
    g = c;
    b = x;
  } else if (hue < 4.0F / 6) {
    r = 0;
    g = x;
    b = c;
  } else if (hue < 5.0F / 6) {
    r = x;
    g = 0;
    b = c;
  } else {
    r = c;
    g = 0;
    b = x;
  }
  return glm::vec4(r + m, g + m, b + m, 1.0F);
}
cv::Mat TextureHelpers::ConvertBayer8ToRGB(const unsigned char* src, int width, int height, int step) {
  auto bayer_img = cv::Mat(height, width, CV_8UC1, const_cast<void*>(static_cast<void const*>(src)), step);
  cv::Mat tmp;
  cv::cvtColor(bayer_img, tmp, cv::COLOR_BayerBG2RGB);
  return tmp;
}
bool TextureHelpers::LoadTexture(const cv::Mat& image, GLuint texture_id) {
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.cols, image.rows, GL_RGB, GL_UNSIGNED_BYTE, image.data);

  glBindTexture(GL_TEXTURE_2D, 0);
  return true;
}
GLuint TextureHelpers::CreateTexture(int width, int height, int components, const void* data) {
  GLuint image_texture;
  glGenTextures(1, &image_texture);
  glBindTexture(GL_TEXTURE_2D, image_texture);

  GLenum format = GL_RED;
  if (components == 1) {
    format = GL_RED;
  } else if (components == 3) {
    format = GL_RGB;
  } else if (components == 4) {
    format = GL_RGBA;
  }

  // Setup filtering parameters for display
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                  GL_CLAMP_TO_EDGE);  // This is required on WebGL for non power-of-two textures
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);  // Same

#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
  glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(format), width, height, 0, format, GL_UNSIGNED_BYTE, data);

  glBindTexture(GL_TEXTURE_2D, 0);

  return image_texture;
}
void TextureHelpers::DestroyTexture(GLuint texture_id) { glDeleteTextures(1, &texture_id); }
}  // namespace ace_monitor
