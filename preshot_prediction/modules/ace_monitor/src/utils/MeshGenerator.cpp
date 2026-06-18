// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/utils/MeshGenerator.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

#include "ace_monitor/scene/rendering/Mesh3D.hpp"
namespace ace_monitor {
Mesh3D::SharedPtr MeshGenerator::GenerateQuad(float width, float length) {
  auto mesh = std::make_shared<Mesh3D>(false, false);
  auto& vertices = mesh->GetVerticies();
  vertices.resize(4);
  vertices[0].position = glm::vec3(-width, -length, 0);
  vertices[1].position = glm::vec3(-width, +length, 0);
  vertices[2].position = glm::vec3(+width, +length, 0);
  vertices[3].position = glm::vec3(+width, -length, 0);
  for (auto& verex : vertices) {
    verex.color = glm::vec3(1, 1, 1);
  }
  mesh->GetIndicies() = {0, 1, 1, 2, 2, 3, 3, 0};
  mesh->UpdateVerticies();
  mesh->UpdateIndicies();
  return mesh;
}
Mesh3D::SharedPtr MeshGenerator::GenerateBox(float width, float length, float height) {
  auto mesh = std::make_shared<Mesh3D>(false, false);
  auto& vertices = mesh->GetVerticies();
  auto& indices = mesh->GetIndicies();
  vertices.resize(8);
  vertices[0].position = glm::vec3(-width / 2, -length / 2, -height / 2);
  vertices[1].position = glm::vec3(-width / 2, +length / 2, -height / 2);
  vertices[2].position = glm::vec3(+width / 2, +length / 2, -height / 2);
  vertices[3].position = glm::vec3(+width / 2, -length / 2, -height / 2);
  vertices[4].position = glm::vec3(-width / 2, -length / 2, +height / 2);
  vertices[5].position = glm::vec3(-width / 2, +length / 2, +height / 2);
  vertices[6].position = glm::vec3(+width / 2, +length / 2, +height / 2);
  vertices[7].position = glm::vec3(+width / 2, -length / 2, +height / 2);
  for (auto& verex : vertices) {
    verex.color = glm::vec3(1, 1, 1);
  }
  indices.push_back(0);
  indices.push_back(1);
  indices.push_back(1);
  indices.push_back(2);
  indices.push_back(2);
  indices.push_back(3);
  indices.push_back(3);
  indices.push_back(0);

  indices.push_back(4 + 0);
  indices.push_back(4 + 1);
  indices.push_back(4 + 1);
  indices.push_back(4 + 2);
  indices.push_back(4 + 2);
  indices.push_back(4 + 3);
  indices.push_back(4 + 3);
  indices.push_back(4 + 0);

  indices.push_back(0);
  indices.push_back(4 + 0);
  indices.push_back(1);
  indices.push_back(4 + 1);
  indices.push_back(2);
  indices.push_back(4 + 2);
  indices.push_back(3);
  indices.push_back(4 + 3);
  mesh->UpdateVerticies();
  mesh->UpdateIndicies();
  return mesh;
}
Mesh3D::SharedPtr MeshGenerator::GenerateAxis() {
  auto mesh = std::make_shared<Mesh3D>(false, false);
  auto& vertices = mesh->GetVerticies();
  auto& indices = mesh->GetIndicies();
  mesh->SetDrawType(GL_LINES);

  Vertex3D v;
  // x-axis
  v.color = glm::vec3(1, 0, 0);
  v.position = glm::vec3(0, 0, 0);
  vertices.push_back(v);
  v.position = glm::vec3(1, 0, 0);
  vertices.push_back(v);
  // y-axis
  v.color = glm::vec3(0, 1, 0);
  v.position = glm::vec3(0, 0, 0);
  vertices.push_back(v);
  v.position = glm::vec3(0, 1, 0);
  vertices.push_back(v);
  // z-axis
  v.color = glm::vec3(0, 0, 1);
  v.position = glm::vec3(0, 0, 0);
  vertices.push_back(v);
  v.position = glm::vec3(0, 0, 1);
  vertices.push_back(v);

  indices.reserve(6);
  for (int i = 0; i < 6; ++i) {
    indices.push_back(i);
  }

  mesh->UpdateVerticies();
  mesh->UpdateIndicies();
  return mesh;
}
Mesh3D::SharedPtr MeshGenerator::GenerateCross() {
  auto mesh = std::make_shared<Mesh3D>(false, false);
  auto& vertices = mesh->GetVerticies();
  auto& indices = mesh->GetIndicies();

  Vertex3D v;
  // x-axis
  v.color = glm::vec3(1, 1, 1);
  v.position = glm::vec3(-1, 0, 0);
  vertices.push_back(v);
  v.position = glm::vec3(1, 0, 0);
  vertices.push_back(v);
  // y-axis
  v.position = glm::vec3(0, -1, 0);
  vertices.push_back(v);
  v.position = glm::vec3(0, 1, 0);
  vertices.push_back(v);
  // z-axis
  v.position = glm::vec3(0, 0, -1);
  vertices.push_back(v);
  v.position = glm::vec3(0, 0, 1);
  vertices.push_back(v);

  indices.reserve(6);
  for (int i = 0; i < 6; ++i) {
    indices.push_back(i);
  }

  mesh->UpdateVerticies();
  mesh->UpdateIndicies();
  return mesh;
}
}  // namespace ace_monitor
