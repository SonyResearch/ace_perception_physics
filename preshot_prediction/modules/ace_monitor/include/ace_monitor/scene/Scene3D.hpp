// Confidential, Copyright 2024, Sony AI, All rights reserved.

#pragma once

#include <Eigen/Dense>
#include <list>
#include <map>
#include <memory>

// Include glfw3.h after our OpenGL definitions
#include <GL/glew.h>
////
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

namespace ace_monitor {

class Node3D;
class CameraNode;

class Scene3D : public std::enable_shared_from_this<Scene3D> {
 public:
  using SharedPtr = std::shared_ptr<Scene3D>;

 protected:
  glm::ivec2 resolution_;

  GLuint framebuffer_;
  GLuint texture_;
  GLuint depthbuffer_;
  GLenum drawbuffer_;
  std::shared_ptr<Node3D> root_;

  std::shared_ptr<CameraNode> camera_;

 public:
  Scene3D();

  void Initialize(int width, int height);

  const glm::ivec2& GetResolution() { return resolution_; }

  std::shared_ptr<Node3D> GetRoot() { return root_; }

  void Render();

  std::shared_ptr<CameraNode> GetCamera() { return camera_; }
  void SetCamera(std::shared_ptr<CameraNode> cam);

  static void RemoveNode(std::shared_ptr<Node3D> node);
  static void SplitNodes(Node3D* node, std::vector<Node3D*>& opaque_nodes, std::vector<Node3D*>& transparent_nodes);

  std::shared_ptr<Node3D> CreateNode(const std::string& name);
  std::shared_ptr<Scene3D> GetPtr() { return shared_from_this(); }

  std::shared_ptr<Node3D> LoadNode(const std::string& path, const std::string& name, const std::string& shader);
};

}  // namespace ace_monitor
