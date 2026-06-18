// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <memory>
#include <string>

#include "ace_monitor/scene/Scene3D.hpp"

namespace ace_monitor {

class TextRenderer {
 public:
  using SharedPtr = std::shared_ptr<TextRenderer>;

 private:
  class TextRendererImpl;
  std::unique_ptr<TextRendererImpl> impl_;

 public:
  TextRenderer();
  ~TextRenderer();
  bool Initialize(const std::string& font_path, int pixel_size);

  void RenderText(Scene3D* scene, const std::string& text, float x, float y, float scale, const glm::vec4& color);
};
}  // namespace ace_monitor
