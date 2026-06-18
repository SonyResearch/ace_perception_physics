// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/scene/rendering/Model3D.hpp"

#include "ace_monitor/scene/rendering/GLShader.hpp"
#include "ace_monitor/scene/rendering/Material.hpp"
#include "ace_monitor/scene/rendering/Mesh3D.hpp"

namespace ace_monitor {

Model3D::Model3D(std::string name) : name_(std::move(name)) {}

void Model3D::SetShader(std::shared_ptr<GLShader> shader) {
  for (auto& mesh : meshes_) {
    auto mat = mesh->GetMaterial();
    if (mat != nullptr) {
      mat->SetShader(shader);
    }
  }
}
void Model3D::Render() {
  for (auto& mesh : meshes_) {
    mesh->Render();
  }
}
}  // namespace ace_monitor
