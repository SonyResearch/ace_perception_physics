// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once
#include <memory>
#include <vector>

namespace ace_monitor {
class GLTexture;
class GLShader;

class Material {
 public:
  using SharedPtr = std::shared_ptr<Material>;

 private:
  std::vector<std::shared_ptr<GLTexture>> textures_;

  std::shared_ptr<GLShader> shader_;

 public:
  Material(/* args */);
  ~Material();

  void SetTextures(std::vector<std::shared_ptr<GLTexture>> textures) { textures_ = std::move(textures); }
  [[nodiscard]] const std::vector<std::shared_ptr<GLTexture>>& GetTextures() const { return textures_; }

  void SetShader(std::shared_ptr<GLShader> shader) { shader_ = std::move(shader); }
  [[nodiscard]] std::shared_ptr<GLShader> GetShader() const { return shader_; }

  void Bind();

  void Unbind();
};
}  // namespace ace_monitor
