// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <glm/glm.hpp>
namespace ace_monitor {
class ShaderUniforms {
 public:
  static void SetBool(int uniform, bool value);
  static void SetInt(int uniform, int value);
  static void SetFloat(int uniform, float value);
  static void SetVec2(int uniform, const glm::vec2 &value);
  static void SetVec2(int uniform, float x, float y);
  static void SetVec3(int uniform, const glm::vec3 &value);
  static void SetVec3(int uniform, float x, float y, float z);
  static void SetVec4(int uniform, const glm::vec4 &value);
  static void SetVec4(int uniform, float x, float y, float z, float w);
  static void SetMat2(int uniform, const glm::mat2 &mat);
  static void SetMat3(int uniform, const glm::mat3 &mat);
  static void SetMat4(int uniform, const glm::mat4 &mat);
};
}  // namespace ace_monitor
