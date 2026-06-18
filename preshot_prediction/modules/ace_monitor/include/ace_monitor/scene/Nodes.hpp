// Confidential, Copyright 2025, Sony AI, All rights reserved.

#pragma once

#include <utility>

#include "ace_monitor/scene/Node3D.hpp"
#include "ace_monitor/scene/Scene3D.hpp"
#include "ace_monitor/scene/TextRenderer.hpp"
#include "ace_monitor/utils/Geometry.hpp"
#include "ace_monitor/utils/common_values.hpp"
#include "calibration/camera_calibration_parameters.hpp"
namespace ace_monitor {

class GridNode : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<GridNode>;

 protected:
  float spacing_;
  float dash_size_;
  float gap_size_;

  void InitializeMesh();
  void SetShaderUniforms() override;

 public:
  GridNode(Scene3D* scene, std::string name, float spacing, float dash_size, float gap_size);
};

class TextRendererNode3D : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<TextRendererNode3D>;

 protected:
  void Render() override;
  std::string text_;
  TextRenderer::SharedPtr text_renderer_;
  float text_size_{1};

 public:
  TextRendererNode3D(Scene3D* scene, std::string name, TextRenderer::SharedPtr text_renderer)
    : Node3D(scene, std::move(name)), text_renderer_(std::move(text_renderer)) {}

  void SetText(std::string text) { text_ = std::move(text); }
  void SetTextSize(float size) { text_size_ = size; }
  bool IsTransparent() const override { return true; }
};

class PointsRendererNode3D : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<PointsRendererNode3D>;

  class Point {
   public:
    glm::vec3 pos;
    int color_code;
  };

  using ColorCodesMap = std::unordered_map<int, glm::vec4>;

 protected:
  void Render() override;

  std::vector<Point> points_;
  ColorCodesMap color_codes_;

 public:
  PointsRendererNode3D(Scene3D* scene, std::string name) : Node3D(scene, std::move(name)) {}

  void SetPoints(const std::vector<Point>& points) { points_ = points; }
  void SetColorCode(int code, const glm::vec4& color) { color_codes_[code] = color; }

  const ColorCodesMap& GetColorCodes() const { return color_codes_; }
};

class VolumeOfInterestNode3D : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<VolumeOfInterestNode3D>;

 protected:
  void UpdateVerticies();

  void InitializeMesh();

 public:
  VolumeOfInterestNode3D(Scene3D* scene, std::string name);

  void SetPoints(const std::vector<glm::vec3>& verticies, const std::vector<size_t>& indicies);
};

class FrustumNode3D : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<FrustumNode3D>;

 protected:
  void RenderModel() override;

  void UpdateVerticies();

  void InitializeMesh();

  calibration::Camera camera_;
  float zfar_;
  bool invalide_;

  Frustum frustum_;

 public:
  FrustumNode3D(calibration::Camera camera, Scene3D* scene, std::string name);

  void SetZFar(float zfar);
  const Frustum& GetFrustum() { return frustum_; }
};

class TrailListNode3D : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<TrailListNode3D>;
  using PointsList = std::vector<glm::vec3>;

 protected:
  void RenderModel() override;

  void InitializeMesh();
  Node3D::SharedPtr attached_node_;

  std::list<PointsList> lines_;
  size_t max_lines_;

 public:
  TrailListNode3D(size_t max_lines, Scene3D* scene, std::string name);

  void AddTrail(PointsList trail);
  void Reset() { lines_.clear(); }
};
class TrailNode3D : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<TrailNode3D>;

 protected:
  void RenderModel() override;

  void InitializeMesh();
  Node3D::SharedPtr attached_node_;

  std::list<glm::vec3> history_;
  size_t max_history_;
  bool changed_{false};

 public:
  TrailNode3D(size_t history, Scene3D* scene, std::string name);

  void AttachNode(Node3D::SharedPtr node) { attached_node_ = node; }
  void AddPoint(const glm::vec3& p);
  void ClearPoints() {
    history_.clear();
    changed_ = true;
  }
};

class DashedTrailNode3D : public TrailListNode3D {
 public:
  using SharedPtr = std::shared_ptr<DashedTrailNode3D>;

 protected:
  void SetShaderUniforms() override;
  float dash_size_;
  float gap_size_;

 public:
  DashedTrailNode3D(size_t history, Scene3D* scene, std::string name, float dash_size, float gap_size);
};

class PlayerNode3D : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<PlayerNode3D>;

 protected:
  void RenderModel() override;

  common_values::PlayerPoseEstimation3D* player_{nullptr};

  void InitializeMesh(bool use_colors);

 public:
  PlayerNode3D(Scene3D* scene, std::string name, bool use_colors = true);
  void SetPlayer(common_values::PlayerPoseEstimation3D* player) { player_ = player; }
};

class LineNode3D : public Node3D {
 public:
  using SharedPtr = std::shared_ptr<LineNode3D>;

 protected:
  void InitializeMesh();

  Line3D line_;

 public:
  LineNode3D(Scene3D* scene, std::string name);

  void SetLine(const Line3D& line);
  void SetColorGradient(const glm::vec4& start, const glm::vec4& end);
};
}  // namespace ace_monitor
