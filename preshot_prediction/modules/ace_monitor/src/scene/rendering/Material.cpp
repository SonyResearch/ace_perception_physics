// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/scene/rendering/Material.hpp"

#include "ace_monitor/scene/rendering/GLShader.hpp"
#include "ace_monitor/scene/rendering/GLTexture.hpp"

namespace ace_monitor {
Material::Material(/* args */) = default;
Material::~Material() = default;

void Material::Bind() {
  // bind textures
  if (shader_) {
    shader_->Bind();
  }
  for (size_t i = 0; i < textures_.size(); ++i) {
    textures_[i]->Bind(static_cast<int>(i));
  }
}

void Material::Unbind() {
  for (size_t i = 0; i < textures_.size(); ++i) {
    textures_[i]->Unbind(static_cast<int>(i));
  }
  if (shader_) {
    shader_->Unbind();
  }
}
}  // namespace ace_monitor
