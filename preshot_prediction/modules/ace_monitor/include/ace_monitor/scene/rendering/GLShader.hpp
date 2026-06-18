// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once
#include <map>
#include <memory>
#include <string>

namespace ace_monitor {

class GLShader {
 public:
  using SharedPtr = std::shared_ptr<GLShader>;

  enum class ShaderType { kVertex, kFragment, kGeometry, kProgram, kCount };

  class UniformInfo {
   public:
    std::string name;
    int id;
    int size;           // size of the variable
    unsigned int type;  // type of the variable (float, vec3 or mat4, etc)
  };

 private:
  unsigned int shaders_[static_cast<int>(ShaderType::kCount)];
  std::string name_;

  std::map<std::string, UniformInfo> uniforms_;

  void UpdateUniforms();

  bool CheckForErrors(ShaderType type);  // false if ok

 public:
  explicit GLShader(std::string name);
  ~GLShader();

  [[nodiscard]] const std::string &GetName() const { return name_; }

  [[nodiscard]] bool LoadShaderFromFile(ShaderType type, const std::string &path);
  [[nodiscard]] bool LoadShaderFromString(ShaderType type, const std::string &program);

  [[nodiscard]] bool LinkShaders();

  void Bind() const;
  static void Unbind();

  [[nodiscard]] const std::map<std::string, UniformInfo> &GetUniforms() const { return uniforms_; }
  [[nodiscard]] const UniformInfo *GetUniform(const std::string &name) const;
};

}  // namespace ace_monitor
