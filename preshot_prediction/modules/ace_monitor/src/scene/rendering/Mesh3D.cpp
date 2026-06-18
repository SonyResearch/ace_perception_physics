// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/scene/rendering/Mesh3D.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

#include <iostream>

#include "ace_monitor/scene/rendering/Material.hpp"

namespace ace_monitor {

Mesh3D::Mesh3D(bool dynamic_verticies, bool dynamic_indicies)
  : dynamic_verticies_(dynamic_verticies), dynamic_indicies_(dynamic_indicies) {
  draw_type_ = GL_TRIANGLES;
  glGenVertexArrays(1, &vao_);
  glGenBuffers(1, &vbo_);
  glGenBuffers(1, &ebo_);

  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);

  // map attributes to Vertex3D members
#define BIND_ATTRIBUTE(index, size, name) \
  glEnableVertexAttribArray(index);       \
  glVertexAttribPointer(index, size, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void *)offsetof(Vertex3D, name));

  // NOLINTBEGIN
  BIND_ATTRIBUTE(0, 3, position);
  BIND_ATTRIBUTE(1, 3, normal);
  BIND_ATTRIBUTE(2, 2, uv0);
  BIND_ATTRIBUTE(3, 3, color);
// NOLINTEND
#undef BIND_ATTRIBUTE
  glBindVertexArray(0);
}
Mesh3D::~Mesh3D() {
  if (vao_ != 0) {
    glDeleteVertexArrays(1, &vao_);
    vao_ = 0;
  }
  if (vbo_ != 0) {
    glDeleteBuffers(1, &vbo_);
    vbo_ = 0;
  }
  if (ebo_ != 0) {
    glDeleteBuffers(1, &ebo_);
    ebo_ = 0;
  }
}

void Mesh3D::UpdateVerticies() {
  bool force = last_verticies_size_ < verticies_.size();
  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  if (force) {
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verticies_.size() * sizeof(Vertex3D)), verticies_.data(),
                 dynamic_verticies_ ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
  } else {
    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(verticies_.size() * sizeof(Vertex3D)),
                    verticies_.data());
  }
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  last_verticies_size_ = verticies_.size();
}
void Mesh3D::UpdateIndicies() {
  bool force = last_indicies_size_ < indicies_.size();
  glBindVertexArray(vao_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
  if (force) {
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indicies_.size() * sizeof(IndexType)),
                 indicies_.data(), dynamic_indicies_ ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
  } else {
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(indicies_.size() * sizeof(IndexType)),
                    indicies_.data());
  }
  glBindVertexArray(0);
  last_indicies_size_ = indicies_.size();
}

void Mesh3D::Render() {
  auto length = static_cast<GLsizei>(draw_length_ < 0 ? last_indicies_size_ : draw_length_);
  if (length == 0) {
    return;
  }
  if (material_ != nullptr) {
    material_->Bind();
  }

  glBindVertexArray(vao_);
  glDrawElements(draw_type_, length, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);

  if (material_ != nullptr) {
    material_->Unbind();
  }
}
}  // namespace ace_monitor
