// Confidential, Copyright 2024, Sony AI, All rights reserved.

#pragma once

#include "Node3D.hpp"

namespace ace_monitor {
class CameraNode : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<CameraNode>;

  enum class CameraProjectionType { kPerspective, kOrtho };

 protected:
  glm::mat4 projection_matrix_;
  glm::mat4 proj_view_matrix_;

  glm::vec3 up_axis_;
  glm::vec3 target_;

  float fov_{45};
  int view_width_{640};
  int view_height_{480};
  float orth_proj_zoom_{5};

  CameraProjectionType projection_type_{CameraProjectionType::kPerspective};
  void UpdateProjectionMatrix() {
    if (projection_type_ == CameraProjectionType::kPerspective) {
      projection_matrix_ = glm::perspective(
        glm::radians(fov_), static_cast<float>(view_width_) / static_cast<float>(view_height_), 0.1F, 100.0F);
    } else {
      float ratio = static_cast<float>(view_width_) / static_cast<float>(view_height_);
      projection_matrix_ =
        glm::ortho(-orth_proj_zoom_ * ratio, orth_proj_zoom_ * ratio, -orth_proj_zoom_, orth_proj_zoom_, 0.0F, 100.0F);
    }
    proj_view_matrix_ = projection_matrix_ * GetLocalMatrix();
  }

 public:
  CameraNode(Scene3D* scene, std::string name) : Node3D(scene, std::move(name)) {}

  void SetFOV(float fov) {
    fov_ = fov;
    if (projection_type_ == CameraProjectionType::kPerspective) {
      UpdateProjectionMatrix();
    }
  }
  void SetProjectionType(CameraProjectionType type) {
    projection_type_ = type;
    UpdateProjectionMatrix();
  }
  void SetUpAxis(glm::vec3 axis) {
    up_axis_ = std::move(axis);
    invalide_transform_ = true;
    UpdateProjectionMatrix();
  }
  void SetTarget(glm::vec3 target) {
    target_ = std::move(target);
    invalide_transform_ = true;
    UpdateProjectionMatrix();
  }
  void SetOrthSize(float size) {
    orth_proj_zoom_ = size;
    if (projection_type_ == CameraProjectionType::kOrtho) {
      UpdateProjectionMatrix();
    }
  }
  void SetViewport(int width, int height) {
    view_width_ = width;
    view_height_ = height;

    UpdateProjectionMatrix();
  }
  [[nodiscard]] glm::mat4 GetViewMatrix() const { return local_model_matrix_; }
  [[nodiscard]] glm::mat4 GetProjectionMatrix() const { return projection_matrix_; }
  [[nodiscard]] glm::mat4 GetProjectionViewMatrix() const { return proj_view_matrix_; }

  [[nodiscard]] const glm::mat4& GetLocalMatrix() override {
    if (invalide_transform_) {
      local_model_matrix_ = glm::lookAt(pos_, target_, up_axis_);
      invalide_transform_ = false;
    }
    return local_model_matrix_;
  }

  int GetViewWidth() const { return view_width_; }
  int GetViewHeight() const { return view_height_; }
};
}  // namespace ace_monitor
