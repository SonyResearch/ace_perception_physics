// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/scene/rendering/ShaderUniforms.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

namespace ace_monitor {

void ShaderUniforms::SetBool(int uniform, bool value) { glUniform1i(uniform, static_cast<int>(value)); }
// ------------------------------------------------------------------------
void ShaderUniforms::SetInt(int uniform, int value) { glUniform1i(uniform, value); }
// ------------------------------------------------------------------------
void ShaderUniforms::SetFloat(int uniform, float value) { glUniform1f(uniform, value); }
// ------------------------------------------------------------------------
void ShaderUniforms::SetVec2(int uniform, const glm::vec2 &value) { glUniform2fv(uniform, 1, &value[0]); }
void ShaderUniforms::SetVec2(int uniform, float x, float y) { glUniform2f(uniform, x, y); }
// ------------------------------------------------------------------------
void ShaderUniforms::SetVec3(int uniform, const glm::vec3 &value) { glUniform3fv(uniform, 1, &value[0]); }
void ShaderUniforms::SetVec3(int uniform, float x, float y, float z) { glUniform3f(uniform, x, y, z); }
// ------------------------------------------------------------------------
void ShaderUniforms::SetVec4(int uniform, const glm::vec4 &value) { glUniform4fv(uniform, 1, &value[0]); }
void ShaderUniforms::SetVec4(int uniform, float x, float y, float z, float w) { glUniform4f(uniform, x, y, z, w); }
// ------------------------------------------------------------------------
void ShaderUniforms::SetMat2(int uniform, const glm::mat2 &mat) {
  glUniformMatrix2fv(uniform, 1, GL_FALSE, &mat[0][0]);
}
// ------------------------------------------------------------------------
void ShaderUniforms::SetMat3(int uniform, const glm::mat3 &mat) {
  glUniformMatrix3fv(uniform, 1, GL_FALSE, &mat[0][0]);
}
// ------------------------------------------------------------------------
void ShaderUniforms::SetMat4(int uniform, const glm::mat4 &mat) {
  glUniformMatrix4fv(uniform, 1, GL_FALSE, &mat[0][0]);
}
}  // namespace ace_monitor
