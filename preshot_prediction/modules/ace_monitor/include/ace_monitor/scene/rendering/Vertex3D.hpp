// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <glm/glm.hpp>

namespace ace_monitor {
class Vertex3D {
 public:
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec3 color;
  glm::vec2 uv0;
};
}  // namespace ace_monitor
