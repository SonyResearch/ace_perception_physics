// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ace_monitor {
class Mesh3D;
class GLShader;
class Model3D {
 public:
  using SharedPtr = std::shared_ptr<Model3D>;

 private:
  std::vector<std::shared_ptr<Mesh3D>> meshes_;
  std::string name_;

 public:
  explicit Model3D(std::string name);

  void SetMeshes(std::vector<std::shared_ptr<Mesh3D>> meshes) { meshes_ = std::move(meshes); }
  [[nodiscard]] const std::vector<std::shared_ptr<Mesh3D>>& GetMeshes() const { return meshes_; }

  void SetShader(std::shared_ptr<GLShader> shader);

  [[nodiscard]] const std::string& GetName() const { return name_; }
  void Render();
};
}  // namespace ace_monitor
