// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/scene/Node3D.hpp"

#include "ace_monitor/scene/CameraNode.hpp"
#include "ace_monitor/scene/Scene3D.hpp"
#include "ace_monitor/scene/rendering/GLShader.hpp"
#include "ace_monitor/scene/rendering/Model3D.hpp"
#include "ace_monitor/scene/rendering/SceneResources.hpp"
#include "ace_monitor/scene/rendering/ShaderUniforms.hpp"

namespace ace_monitor {

Node3D::Node3D(Scene3D* scene, std::string name)
  : pos_(0), orientation_(1, 0, 0, 0), scale_(1), tint_color_(1, 1, 1, 1), scene_(scene), name_(std::move(name)) {
  model_ = std::make_shared<Model3D>("");
}
Node3D::~Node3D() = default;

bool Node3D::LoadModel(const std::string& model_path) {
  auto loaded = SceneResources::GetInstance().GetOrCreateModel(model_path);
  if (loaded != nullptr) {
    model_ = loaded;
    return true;
  }
  // Keep the existing empty Model3D so subsequent set/render calls remain safe
  // when the asset is missing (e.g. proprietary mesh packages not installed).
  return false;
}
bool Node3D::LoadShaders(const std::string& path) {
  if (!model_) {
    return false;
  }
  shader_ = SceneResources::GetInstance().GetOrCreateShader(path);
  model_->SetShader(shader_);
  return true;
}

void Node3D::SetPosition(const glm::vec3& pos) {
  pos_ = pos;
  invalide_transform_ = true;
}
void Node3D::SetOrientation(const glm::quat& orientation) {
  orientation_ = orientation;
  invalide_transform_ = true;
}
void Node3D::SetScale(const glm::vec3& scale) {
  scale_ = scale;
  invalide_transform_ = true;
}

void Node3D::UpdateModelMatrix() {
  if (parent_ != nullptr) {
    model_matrix_ = parent_->GetMatrix() * GetLocalMatrix();
  } else {
    model_matrix_ = GetLocalMatrix();
  }
  for (auto& node : childs_) {
    node->UpdateModelMatrix();
  }
}
const glm::mat4& Node3D::GetLocalMatrix() {
  if (invalide_transform_) {
    local_model_matrix_ = glm::translate(glm::mat4(1.0), pos_) * glm::mat4_cast(orientation_);
    local_model_matrix_ = glm::scale(local_model_matrix_, scale_);
    invalide_transform_ = false;
  }
  return local_model_matrix_;
}
glm::vec3 Node3D::GetPosition() {
  const auto& m = GetMatrix();

  return glm::vec3(m[3][0], m[3][1], m[3][2]);
}
const glm::mat4& Node3D::GetMatrix() {
  if (invalide_transform_) {
    UpdateModelMatrix();
  }
  return model_matrix_;
}
void Node3D::AddChild(Node3D::SharedPtr node) {
  if (node->GetParent()) {
    node->GetParent()->RemoveChild(node);
  }
  childs_.push_back(node);
  node->parent_ = GetPtr();
}
void Node3D::RemoveChild(Node3D::SharedPtr node) {
  auto it = std::find(childs_.begin(), childs_.end(), node);
  if (it == childs_.end()) {
    return;
  }
  // LOG(INFO) << GetName() << "  -> Removing: " << node->GetName() ;
  childs_.erase(it);
  node->parent_ = nullptr;
}
Node3D::SharedPtr Node3D::GetChildByName(const std::string& name, bool recursive) const {
  for (const auto& node : childs_) {
    if (node->GetName() == name) {
      return node;
    }
    if (recursive) {
      auto c = node->GetChildByName(name, recursive);
      if (c) {
        return c;
      }
    }
  }
  return nullptr;
}

void Node3D::Update() {
  if (!visible_) {
    return;
  }
  UpdateModelMatrix();
  for (auto& node : childs_) {
    node->Update();
  }
}
void Node3D::SetShaderUniforms() {
  if (shader_) {
    shader_->Bind();

    for (const auto& it : shader_->GetUniforms()) {
      const auto& uniform = it.second;
      if (uniform.name == "M" || uniform.name == "model") {
        ShaderUniforms::SetMat4(uniform.id, GetMatrix());
      } else if (uniform.name == "V" || uniform.name == "view") {
        ShaderUniforms::SetMat4(uniform.id, scene_->GetCamera()->GetViewMatrix());
      } else if (uniform.name == "P" || uniform.name == "projection") {
        ShaderUniforms::SetMat4(uniform.id, scene_->GetCamera()->GetProjectionMatrix());
      } else if (uniform.name == "MVP") {
        auto mvp = scene_->GetCamera()->GetProjectionViewMatrix() * GetMatrix();
        ShaderUniforms::SetMat4(uniform.id, mvp);
      } else if (uniform.name == "tint_color") {
        ShaderUniforms::SetVec4(uniform.id, tint_color_);
      }
    }
  }
}
void Node3D::Render() {
  if (!visible_) {
    return;
  }
  SetShaderUniforms();
  RenderModel();
  if (shader_) {
    shader_->Unbind();
  }
}

void Node3D::RenderModel() { model_->Render(); }
}  // namespace ace_monitor
