// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <memory>
namespace ace_monitor {

class Mesh3D;
class MeshGenerator {
 public:
  static std::shared_ptr<Mesh3D> GenerateQuad(float width, float length);
  static std::shared_ptr<Mesh3D> GenerateBox(float width, float length, float height);
  static std::shared_ptr<Mesh3D> GenerateAxis();
  static std::shared_ptr<Mesh3D> GenerateCross();
};
}  // namespace ace_monitor
