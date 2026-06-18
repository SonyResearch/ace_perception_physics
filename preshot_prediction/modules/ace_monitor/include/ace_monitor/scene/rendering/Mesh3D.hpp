// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ace_monitor/scene/rendering/Vertex3D.hpp"

namespace ace_monitor {

class Material;

class Mesh3D {
 public:
  using SharedPtr = std::shared_ptr<Mesh3D>;
  using IndexType = unsigned int;

 private:
  std::shared_ptr<Material> material_;
  std::vector<IndexType> indicies_;
  std::vector<Vertex3D> verticies_;

  size_t last_verticies_size_{0};
  size_t last_indicies_size_{0};

  unsigned int vao_{0};  // aka vertex array object (main object)
  unsigned int vbo_{0};  // aka vertex buffer object
  unsigned int ebo_{0};  // aka element buffer object (indicies)
  bool dynamic_verticies_;
  bool dynamic_indicies_;

  int draw_length_{-1};
  unsigned int draw_type_{0};

 public:
  Mesh3D(bool dynamic_verticies, bool dynamic_indicies);
  ~Mesh3D();

  void SetMaterial(std::shared_ptr<Material> material) { material_ = std::move(material); }

  [[nodiscard]] std::shared_ptr<Material> GetMaterial() const { return material_; }

  [[nodiscard]] std::vector<unsigned int>& GetIndicies() { return indicies_; }
  [[nodiscard]] std::vector<Vertex3D>& GetVerticies() { return verticies_; }

  // -1 means to draw all
  void SetDrawLength(int draw_length = -1) { draw_length_ = draw_length; }
  // use OpenGL draw type, e.g. GL_LINES
  void SetDrawType(unsigned int draw_type) { draw_type_ = draw_type; }

  void UpdateVerticies();
  void UpdateIndicies();

  void Render();
};
}  // namespace ace_monitor
