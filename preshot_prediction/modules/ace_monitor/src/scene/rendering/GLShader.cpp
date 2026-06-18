// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/scene/rendering/GLShader.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

#include <ace_loggers/ace_loggers.hpp>
#include <cstring>
#include <fstream>
#include <iostream>

namespace ace_monitor {

GLShader::GLShader(std::string name) : name_(std::move(name)) {
  memset(shaders_, 0, sizeof(shaders_));
  shaders_[static_cast<int>(ShaderType::kProgram)] = glCreateProgram();
}
GLShader::~GLShader() {
  for (int i = 0; i < static_cast<int>(ShaderType::kCount); ++i) {
    if (i == static_cast<int>(ShaderType::kProgram)) {
      continue;
    }
    if (shaders_[i] != 0) {
      glDeleteShader(shaders_[i]);
      shaders_[i] = 0;
    }
  }
  glDeleteProgram(shaders_[static_cast<int>(ShaderType::kProgram)]);
  shaders_[static_cast<int>(ShaderType::kProgram)] = 0;
}

bool GLShader::CheckForErrors(GLShader::ShaderType type) {
  static const std::string ShadersNames[] = {"Vertex", "Fragment", "Geometry", "Program"};
  GLint success;
  GLchar error_log[1024];
  auto id = shaders_[static_cast<int>(type)];
  if (type != ShaderType::kProgram) {
    glGetShaderiv(id, GL_COMPILE_STATUS, &success);
  } else {
    glGetProgramiv(id, GL_LINK_STATUS, &success);
  }
  if (!success) {
    glGetShaderInfoLog(id, 1024, nullptr, error_log);
    LOG(WARNING) << "Shader[" << ShadersNames[static_cast<int>(type)] << "] Failed to compile:\n" << error_log;
  }
  return success == 0;
}

bool GLShader::LoadShaderFromFile(GLShader::ShaderType type, const std::string& path) {
  std::ifstream shader_file;
  shader_file.open(path);
  if (shader_file.is_open()) {
    std::stringstream program;
    program << shader_file.rdbuf();
    shader_file.close();

    return LoadShaderFromString(type, program.str());
  }
  LOG(WARNING) << "Failed to open shader file: " << path;
  return false;
}
bool GLShader::LoadShaderFromString(GLShader::ShaderType type, const std::string& program) {
  static const unsigned int GLShaderType[] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER};

  auto shader_index = static_cast<int>(type);
  if (shaders_[shader_index] != 0) {
    glDetachShader(shaders_[static_cast<int>(ShaderType::kProgram)], shaders_[shader_index]);
    glDeleteShader(shaders_[shader_index]);
  }
  shaders_[shader_index] = glCreateShader(GLShaderType[shader_index]);
  const auto* program_source = program.c_str();
  glShaderSource(shaders_[shader_index], 1, &program_source, nullptr);
  glCompileShader(shaders_[shader_index]);
  if (!CheckForErrors(type)) {
    glAttachShader(shaders_[static_cast<int>(ShaderType::kProgram)], shaders_[shader_index]);
    return true;
  }
  return false;
}

bool GLShader::LinkShaders() {
  glLinkProgram(shaders_[static_cast<int>(ShaderType::kProgram)]);
  if (!CheckForErrors(ShaderType::kProgram)) {
    UpdateUniforms();
    return true;
  }
  uniforms_.clear();
  return false;
}
void GLShader::UpdateUniforms() {
  GLint count;
  const GLsizei buf_size = 16;  // maximum name length
  GLchar name[buf_size];        // variable name in GLSL
  GLsizei length;               // name length
  UniformInfo info;

  uniforms_.clear();
  glUseProgram(shaders_[static_cast<int>(ShaderType::kProgram)]);
  glGetProgramiv(shaders_[static_cast<int>(ShaderType::kProgram)], GL_ACTIVE_UNIFORMS, &count);
  for (int i = 0; i < count; i++) {
    glGetActiveUniform(shaders_[static_cast<int>(ShaderType::kProgram)], static_cast<GLuint>(i), buf_size, &length,
                       &info.size, &info.type, name);
    info.name = name;
    info.id = i;

    uniforms_[info.name] = info;
  }
  glUseProgram(0);
}
void GLShader::Bind() const { glUseProgram(shaders_[static_cast<int>(ShaderType::kProgram)]); }
void GLShader::Unbind() { glUseProgram(0); }

const GLShader::UniformInfo* GLShader::GetUniform(const std::string& name) const {
  auto it = uniforms_.find(name);
  if (it == uniforms_.end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace ace_monitor
