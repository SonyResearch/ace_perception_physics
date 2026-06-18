// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <glm/glm.hpp>
#include <opencv2/core.hpp>

namespace ace_monitor {
class TextureHelpers {
 public:
  static cv::Mat ConvertBayer8ToRGB(const unsigned char* src, int width, int height, int step);
  static bool LoadTexture(const cv::Mat& image, GLuint texture_id);
  static GLuint CreateTexture(int width, int height, int components = 3, const void* data = nullptr);
  static void DestroyTexture(GLuint texture_id);

  static glm::vec4 GenerateUniqueColor(int color_seed);
};
}  // namespace ace_monitor
