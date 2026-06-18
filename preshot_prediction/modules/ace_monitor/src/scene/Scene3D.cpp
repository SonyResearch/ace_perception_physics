// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include "ace_monitor/scene/Scene3D.hpp"

#include <imgui.h>

#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/scene/CameraNode.hpp"
#include "ace_monitor/scene/Node3D.hpp"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
// Include GLM

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <iostream>
#include <list>

namespace ace_monitor {
///
Scene3D::Scene3D() = default;

void Scene3D::Initialize(int width, int height) {
  this->resolution_.x = width;
  this->resolution_.y = height;
  // texture
  glGenFramebuffers(1, &this->framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, this->framebuffer_);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glGenTextures(1, &this->texture_);
  glBindTexture(GL_TEXTURE_2D, this->texture_);

  glTexImage2D(GL_TEXTURE_2D,     // target
               0,                 // mipmap level
               GL_RGB,            // internal format
               width,             // scene width
               height,            // scene height
               0,                 // border
               GL_RGB,            // format
               GL_UNSIGNED_BYTE,  // type
               nullptr);          // image buffer

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  // depthbuffer
  glGenRenderbuffers(1, &this->depthbuffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, this->depthbuffer_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, this->depthbuffer_);

  this->drawbuffer_ = GL_COLOR_ATTACHMENT0;

  glFramebufferTexture(GL_FRAMEBUFFER, this->drawbuffer_, this->texture_, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  camera_ = std::make_shared<CameraNode>(this, "main_camera");

  camera_->SetUpAxis(glm::vec3(0, 0, 1));

  camera_->SetViewport(width, height);

  {
    auto* root_ptr = new Node3D(this, "root");
    root_ = Node3D::SharedPtr(root_ptr);
  }
}
void Scene3D::SetCamera(std::shared_ptr<CameraNode> cam) {
  camera_ = cam;
  camera_->SetViewport(resolution_.x, resolution_.y);
}
void Scene3D::SplitNodes(Node3D* node, std::vector<Node3D*>& opaque_nodes, std::vector<Node3D*>& transparent_nodes) {
  if (!node->IsVisible()) {
    return;
  }
  if (node->IsTransparent()) {
    transparent_nodes.push_back(node);
  } else {
    opaque_nodes.push_back(node);
  }

  for (const auto& child : node->GetChilds()) {
    SplitNodes(child.get(), opaque_nodes, transparent_nodes);
  }
}
void Scene3D::Render() {
  glBindFramebuffer(GL_FRAMEBUFFER, this->framebuffer_);
  glDrawBuffers(1, &this->drawbuffer_);

  glClearColor(0.5, 0.5, 0.8, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  glViewport(0, 0, this->resolution_.x, this->resolution_.y);

  root_->Update();

  std::vector<Node3D*> opaque_nodes;
  std::vector<Node3D*> transparent_nodes;

  SplitNodes(root_.get(), opaque_nodes, transparent_nodes);

  for (auto* node : opaque_nodes) {
    node->Render();
  }
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  for (auto* node : transparent_nodes) {
    node->Render();
  }
  glDisable(GL_BLEND);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // NOLINTNEXTLINE
  ImGui::Image(reinterpret_cast<void*>(texture_),
               ImVec2(static_cast<float>(resolution_.x), static_cast<float>(resolution_.y)), ImVec2(0, 1),
               ImVec2(1, 0));
}

Node3D::SharedPtr Scene3D::CreateNode(const std::string& name) {
  auto* ptr = new Node3D(this, name);

  auto node = Node3D::SharedPtr(ptr);
  root_->AddChild(node);
  return node;
}
void Scene3D::RemoveNode(std::shared_ptr<Node3D> node) {
  if (node->GetParent()) {
    node->GetParent()->RemoveChild(node);
  }
}

Node3D::SharedPtr Scene3D::LoadNode(const std::string& path, const std::string& name, const std::string& shader) {
  auto node = CreateNode(name);
  node->LoadModel(path);
  node->LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/" + shader);
  return node;
}
}  // namespace ace_monitor
