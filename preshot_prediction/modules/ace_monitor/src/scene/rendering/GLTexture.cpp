// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/scene/rendering/GLTexture.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

namespace ace_monitor {

GLTexture::GLTexture(std::string name) : name_(std::move(name)) {}

GLTexture::~GLTexture() { Destroy(); }

void GLTexture::Destroy() {
  if (texture_id_ != 0) {
    glDeleteTextures(1, &texture_id_);
    texture_id_ = 0;
  }
}

void GLTexture::Bind(int target) const {
  glActiveTexture(GL_TEXTURE0 + target);
  glBindTexture(GL_TEXTURE_2D, texture_id_);
}
void GLTexture::Unbind(int target) {
  glActiveTexture(GL_TEXTURE0 + target);
  glBindTexture(GL_TEXTURE_2D, 0);

  glActiveTexture(GL_TEXTURE0);
}

}  // namespace ace_monitor
