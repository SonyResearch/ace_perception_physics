// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ace_monitor {

class Scene3D;
class Model3D;
class GLShader;

class Node3D : public std::enable_shared_from_this<Node3D> {
 public:
  using SharedPtr = std::shared_ptr<Node3D>;

 protected:
  friend class Scene3D;

  std::shared_ptr<Model3D> model_;
  std::shared_ptr<GLShader> shader_;

  glm::mat4 model_matrix_;
  glm::mat4 local_model_matrix_;

  glm::vec3 pos_;
  glm::quat orientation_;
  glm::vec3 scale_;

  glm::vec4 tint_color_;

  bool invalide_transform_{true};
  bool visible_{true};
  std::vector<Node3D::SharedPtr> childs_;
  Node3D::SharedPtr parent_;

  Scene3D* scene_{nullptr};

  std::string name_;

  Node3D(Scene3D* scene, std::string name);
  virtual void Update();
  virtual void Render();
  virtual void RenderModel();

  virtual void UpdateModelMatrix();
  virtual void SetShaderUniforms();

 public:
  virtual ~Node3D();
  bool LoadModel(const std::string& model_path);
  bool LoadShaders(const std::string& path);
  const std::string& GetName() const { return name_; }

  void SetPosition(const glm::vec3& pos);
  void SetOrientation(const glm::quat& orientation);
  void SetScale(const glm::vec3& scale);
  const glm::mat4& GetMatrix();
  virtual const glm::mat4& GetLocalMatrix();

  glm::vec3 GetPosition();

  void AddChild(Node3D::SharedPtr node);
  void RemoveChild(Node3D::SharedPtr node);
  const std::vector<Node3D::SharedPtr>& GetChilds() const { return childs_; }
  Node3D::SharedPtr GetParent() const { return parent_; }

  const std::shared_ptr<Model3D>& GetModel() const { return model_; }

  void Hide() { visible_ = false; }
  void Show() { visible_ = true; }
  bool IsVisible() const { return visible_; }

  void SetTintColor(const glm::vec4& color) { tint_color_ = color; }
  const glm::vec4& GetTintColor() const { return tint_color_; }

  std::shared_ptr<Node3D> GetPtr() { return shared_from_this(); }

  Node3D::SharedPtr GetChildByName(const std::string& name, bool recursive) const;

  virtual bool IsTransparent() const { return tint_color_.a != 1.0F; }
};

}  // namespace ace_monitor
