// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <map>
#include <memory>

namespace ace_monitor {

class GLTexture;
class GLShader;
class Model3D;

class SceneResources {
 public:
 private:
  std::map<std::string, std::shared_ptr<GLTexture>> textures_;
  std::map<std::string, std::shared_ptr<GLShader>> shaders_;
  std::map<std::string, std::shared_ptr<Model3D>> models_;

 public:
  static SceneResources& GetInstance();

  [[nodiscard]] std::shared_ptr<GLTexture> GetTexture(const std::string& name) const;
  [[nodiscard]] std::shared_ptr<GLShader> GetShader(const std::string& name) const;
  [[nodiscard]] std::shared_ptr<Model3D> GetModel(const std::string& name) const;

  void AddTexture(std::shared_ptr<GLTexture> texture);
  void AddShader(std::shared_ptr<GLShader> shader);
  void AddModel(std::shared_ptr<Model3D> model);

  [[nodiscard]] std::shared_ptr<GLTexture> GetOrCreateTexture(const std::string& path);
  [[nodiscard]] std::shared_ptr<Model3D> GetOrCreateModel(const std::string& path);
  [[nodiscard]] std::shared_ptr<GLShader> GetOrCreateShader(const std::string& path);
};

}  // namespace ace_monitor
