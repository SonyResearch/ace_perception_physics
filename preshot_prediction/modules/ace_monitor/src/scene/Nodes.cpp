// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/scene/Nodes.hpp"

#include <iostream>
#include <string_view>
#include <unordered_map>

#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/scene/CameraNode.hpp"
#include "ace_monitor/scene/rendering/GLShader.hpp"
#include "ace_monitor/scene/rendering/Mesh3D.hpp"
#include "ace_monitor/scene/rendering/Model3D.hpp"
#include "ace_monitor/scene/rendering/ShaderUniforms.hpp"

namespace ace_monitor {

GridNode::GridNode(Scene3D* scene, std::string name, float spacing, float dash_size, float gap_size)
  : Node3D(scene, std::move(name)), spacing_(spacing), dash_size_(dash_size), gap_size_(gap_size) {
  InitializeMesh();

  LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/dashed_line");
}

void GridNode::SetShaderUniforms() {
  const auto* u_res = shader_->GetUniform("view_resolution");
  const auto* u_dash_size = shader_->GetUniform("dash_size");
  const auto* u_gap_size = shader_->GetUniform("gap_size");

  shader_->Bind();
  ShaderUniforms::SetVec2(u_res->id, static_cast<float>(scene_->GetResolution().x),
                          static_cast<float>(scene_->GetResolution().y));
  ShaderUniforms::SetFloat(u_dash_size->id, dash_size_);
  ShaderUniforms::SetFloat(u_gap_size->id, gap_size_);
  shader_->Unbind();

  Node3D::SetShaderUniforms();
}
void GridNode::InitializeMesh() {
  auto mesh = std::make_shared<Mesh3D>(false, false);
  auto& vertices = mesh->GetVerticies();
  auto& indices = mesh->GetIndicies();
  mesh->SetDrawType(GL_LINES);

  float length = 15;
  int samples_count = static_cast<int>(length / spacing_);

  Vertex3D v;

  v.color = glm::vec3(1, 1, 1);

  float height = -0.7F;

  int index = 0;
#define ADD_VERTEX(pos, uv) \
  v.position = pos;         \
  v.uv0.x = uv;             \
  vertices.push_back(v);    \
  indices.push_back(index++);

  for (int i = -samples_count; i < samples_count; ++i) {
    auto point = static_cast<float>(i) * spacing_;

    ADD_VERTEX(glm::vec3(point, -length, height), 0)
    ADD_VERTEX(glm::vec3(point, length, height), 1)

    ADD_VERTEX(glm::vec3(-length, point, height), 0)
    ADD_VERTEX(glm::vec3(length, point, height), 1)

    ADD_VERTEX(glm::vec3(point, -length, height), 0)
    ADD_VERTEX(glm::vec3(point, -length, length), 1)

    if (point > height) {
      ADD_VERTEX(glm::vec3(-length, -length, point), 0)
      ADD_VERTEX(glm::vec3(length, -length, point), 1)
    }
    ADD_VERTEX(glm::vec3(-length, point, height), 0)
    ADD_VERTEX(glm::vec3(-length, point, length), 1)

    if (point > height) {
      ADD_VERTEX(glm::vec3(-length, -length, point), 0)
      ADD_VERTEX(glm::vec3(-length, length, point), 1)
    }
  }
#undef ADD_VERTEX

  mesh->UpdateVerticies();
  mesh->UpdateIndicies();
  model_->SetMeshes({mesh});
}

////

void TextRendererNode3D::Render() {
  if (!visible_ || text_.empty() || !text_renderer_) {
    return;
  }

  auto pos = GetPosition();

  auto projected_pos = scene_->GetCamera()->GetProjectionViewMatrix() * glm::vec4(pos.x, pos.y, pos.z, 1);

  projected_pos /= projected_pos.w;
  if (projected_pos.x < -1 || projected_pos.x > 1 ||  //
      projected_pos.y < -1 || projected_pos.y > 1 ||  //
      projected_pos.z < -1 || projected_pos.z > 1) {
    // behind camera
    return;
  }
  glDisable(GL_DEPTH_TEST);

  text_renderer_->RenderText(scene_, text_, projected_pos.x, projected_pos.y, text_size_, tint_color_);
  glEnable(GL_DEPTH_TEST);
}

void PointsRendererNode3D::Render() {
  if (!visible_ || points_.empty()) {
    return;
  }

  for (auto& pt : points_) {
    if (color_codes_.find(pt.color_code) != color_codes_.end()) {
      SetTintColor(color_codes_[pt.color_code]);
    }
    SetPosition(pt.pos);
    Node3D::Render();
  }
}

VolumeOfInterestNode3D::VolumeOfInterestNode3D(Scene3D* scene, std::string name) : Node3D(scene, std::move(name)) {
  InitializeMesh();

  LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/color_shader");
}

void VolumeOfInterestNode3D::InitializeMesh() {
  std::vector<Vertex3D> vertices;
  std::vector<Mesh3D::IndexType> indices;
  vertices.resize(1000);
  indices.resize(1000);
  auto mesh = std::make_shared<Mesh3D>(true, true);
  mesh->SetDrawType(GL_TRIANGLES);
  mesh->SetDrawLength(0);

  model_->SetMeshes({mesh});
}

void VolumeOfInterestNode3D::SetPoints(const std::vector<glm::vec3>& points, const std::vector<size_t>& idx) {
  const auto& mesh = model_->GetMeshes()[0];
  auto& vertices = mesh->GetVerticies();
  auto& indices = mesh->GetIndicies();
  vertices.resize(points.size());
  indices.resize(idx.size());
  for (size_t i = 0; i < points.size(); ++i) {
    vertices[i].position = points[i];
    vertices[i].color = glm::vec3(1, 1, 1);
  }
  for (size_t i = 0; i < idx.size(); ++i) {
    indices[i] = idx[i];
  }
  mesh->SetDrawLength(static_cast<int>(idx.size()));

  mesh->UpdateVerticies();
  mesh->UpdateIndicies();
}
///////

FrustumNode3D::FrustumNode3D(calibration::Camera camera, Scene3D* scene, std::string name)
  : Node3D(scene, std::move(name)), camera_(std::move(camera)) {
  InitializeMesh();

  LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/lines");
}

void FrustumNode3D::RenderModel() {
  if (invalide_) {
    UpdateVerticies();
    invalide_ = false;
  }
  Node3D::RenderModel();
}

void FrustumNode3D::UpdateVerticies() {
  const auto& mesh = model_->GetMeshes()[0];
  auto& vertices = mesh->GetVerticies();
  vertices.resize(8);

  auto hfov = 2 * atan2(camera_.resolution[0], 2 * camera_.camera_matrix.coeff(0, 0));
  auto vfov = 2 * atan2(camera_.resolution[1], 2 * camera_.camera_matrix.coeff(1, 1));
  auto near = ace_yaml::SafeGetValue<double>(ACEMonitor::GetInstance().GetConfigurations()["frustum_node"], 0.1);
  auto far = zfar_;

  std::vector<Eigen::Vector3f> corners;
  Frustum::FillCorners(hfov, vfov, near, far, corners);
  for (int i = 0; i < 8; ++i) {
    vertices[i].position = eigen_to_glm(corners[i]);
  }

  mesh->UpdateVerticies();

  const auto& mat = GetMatrix();
  Eigen::Matrix4f pose = Eigen::Matrix4f(&mat[0][0]);  //.transpose();

  frustum_ = Frustum(pose, hfov, vfov, 0.1, 15);
}

void FrustumNode3D::InitializeMesh() {
  auto mesh = std::make_shared<Mesh3D>(true, true);
  auto& vertices = mesh->GetVerticies();
  auto& indices = mesh->GetIndicies();

  indices = {
    0, 1, 1, 3, 3, 2, 2, 0,  ////
    6, 7, 7, 5, 5, 4, 4, 6,  ////
    0, 6, 6, 4, 4, 2, 2, 0,  ////
    1, 3, 3, 5, 5, 7, 7, 1   ////
  };
  vertices.resize(8);

  for (int i = 0; i < 8; ++i) {
    vertices[i].color = glm::vec3(1, 1, 1);
  }

  mesh->SetDrawType(GL_LINE_STRIP);
  model_->SetMeshes({mesh});

  mesh->UpdateIndicies();
  UpdateVerticies();
}

void FrustumNode3D::SetZFar(float zfar) {
  zfar_ = zfar;
  invalide_ = true;
}
///////

void TrailListNode3D::RenderModel() { Node3D::RenderModel(); }

void TrailListNode3D::InitializeMesh() {
  std::vector<Mesh3D::SharedPtr> meshes;
  std::cout << max_lines_ << std::endl;
  for (size_t i = 0; i < max_lines_; ++i) {
    auto mesh = std::make_shared<Mesh3D>(true, true);
    mesh->SetDrawType(GL_LINE_STRIP);
    meshes.emplace_back(mesh);
  }
  model_->SetMeshes(meshes);
}
TrailListNode3D::TrailListNode3D(size_t max_lines, Scene3D* scene, std::string name)
  : Node3D(scene, std::move(name)), max_lines_(max_lines) {
  InitializeMesh();

  LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/lines");
}

void TrailListNode3D::AddTrail(TrailListNode3D::PointsList trail) {
  lines_.emplace_back(std::move(trail));
  if (lines_.size() > max_lines_) {
    lines_.pop_front();
  }
  float step = 1.0F / static_cast<float>(lines_.size());
  float alpha = step;
  auto line_it = lines_.begin();
  for (const auto& mesh : model_->GetMeshes()) {
    std::vector<Vertex3D>& vertices = mesh->GetVerticies();
    std::vector<Mesh3D::IndexType>& indices = mesh->GetIndicies();
    auto& line = *line_it;
    indices.resize(line.size());
    vertices.resize(line.size());
    for (size_t i = 0; i < line.size(); ++i) {
      indices[i] = i;
      vertices[i].position = line[i];
      vertices[i].uv0.x = static_cast<float>(i) * 1e-2F;
      vertices[i].color = glm::vec3(alpha, alpha, alpha);
    }
    mesh->UpdateVerticies();
    mesh->UpdateIndicies();
    alpha += step;
    ++line_it;
    if (line_it == lines_.end()) {
      break;
    }
  }
}
///////

void TrailNode3D::RenderModel() {
  if (attached_node_) {
    auto pos = attached_node_->GetPosition();
    AddPoint(pos);
  }
  if (changed_) {
    const auto& mesh = model_->GetMeshes()[0];
    auto& verticies = mesh->GetVerticies();
    int index = 0;
    glm::vec3 last_p;
    if (!history_.empty()) {
      last_p = *history_.begin();
    }
    for (const auto& p : history_) {
      verticies[index].position = p;
      verticies[index].uv0.x = static_cast<float>(index) * 1e-2F;
      last_p = p;
      ++index;
    }
    mesh->SetDrawLength(index);
    mesh->UpdateVerticies();
    changed_ = false;
  }
  Node3D::RenderModel();
}

void TrailNode3D::InitializeMesh() {
  auto mesh = std::make_shared<Mesh3D>(true, false);
  std::vector<Vertex3D>& vertices = mesh->GetVerticies();
  std::vector<Mesh3D::IndexType>& indices = mesh->GetIndicies();
  Vertex3D v;
  v.position = glm::vec3(0);
  v.color = glm::vec3(1, 1, 1);
  for (size_t i = 0; i < max_history_; ++i) {
    float c = static_cast<float>(i) / static_cast<float>(max_history_);
    v.color = glm::vec3(c, c, c);
    vertices.push_back(v);
    indices.push_back(i);
  }
  mesh->UpdateVerticies();
  mesh->UpdateIndicies();
  mesh->SetDrawType(GL_LINE_STRIP);
  model_->SetMeshes({mesh});
}
TrailNode3D::TrailNode3D(size_t history, Scene3D* scene, std::string name)
  : Node3D(scene, std::move(name)), max_history_(history) {
  InitializeMesh();

  LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/lines");
}

void TrailNode3D::AddPoint(const glm::vec3& pos) {
  history_.push_back(pos);
  if (history_.size() > max_history_) {
    history_.pop_front();
  }
  changed_ = true;
}

////////////////////

DashedTrailNode3D::DashedTrailNode3D(size_t history, Scene3D* scene, std::string name, float dash_size, float gap_size)
  : TrailListNode3D(history, scene, std::move(name)), dash_size_(dash_size), gap_size_(gap_size) {
  LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/dashed_line");
}

void DashedTrailNode3D::SetShaderUniforms() {
  const auto* u_res = shader_->GetUniform("view_resolution");
  const auto* u_dash_size = shader_->GetUniform("dash_size");
  const auto* u_gap_size = shader_->GetUniform("gap_size");

  shader_->Bind();
  ShaderUniforms::SetVec2(u_res->id, static_cast<float>(scene_->GetResolution().x),
                          static_cast<float>(scene_->GetResolution().y));
  ShaderUniforms::SetFloat(u_dash_size->id, dash_size_);
  ShaderUniforms::SetFloat(u_gap_size->id, gap_size_);
  shader_->Unbind();

  Node3D::SetShaderUniforms();
}
////////////////////

struct PlayerLinkInfo {
 public:
  PlayerLinkInfo() = default;
  PlayerLinkInfo(std::string_view kp1, std::string_view kp2, std::string_view c)
    : keypoint1(kp1), keypoint2(kp2), color(c) {}
  std::string_view keypoint1;
  std::string_view keypoint2;
  std::string_view color;
};

class PlayerNodeHelpers {
  static std::unordered_map<std::string_view, int> CreateKeypointsMap() {
    std::unordered_map<std::string_view, int> keypoint_map;
    int index = 0;
    for (const auto& name : {"nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_hand",
                             "r_hand", "l_hip", "r_hip", "l_knee", "r_knee", "l_foot", "r_foot"}) {
      keypoint_map[name] = index++;
    }
    return keypoint_map;
  }

  static std::unordered_map<std::string_view, glm::vec3> CreateColorsMap() {
    std::unordered_map<std::string_view, glm::vec3> colors_map;
    colors_map["purple"] = glm::vec3(0.5, 0, 0.5);
    colors_map["yellow"] = glm::vec3(0.5, 0.5, 0);
    colors_map["blue"] = glm::vec3(1, 0, 0);
    colors_map["green"] = glm::vec3(0, 1, 0);
    colors_map["red"] = glm::vec3(0, 0, 1);
    colors_map["skyblue"] = glm::vec3(0, 0.5, 0.5);
    return colors_map;
  }
  static std::vector<PlayerLinkInfo> CreatePlayerLinks() {
    std::vector<PlayerLinkInfo> player_links;
    player_links.emplace_back("nose", "r_eye", "purple");
    player_links.emplace_back("nose", "l_eye", "purple");
    player_links.emplace_back("nose", "r_sho", "yellow");
    player_links.emplace_back("nose", "l_sho", "yellow");
    player_links.emplace_back("r_sho", "l_sho", "blue");
    player_links.emplace_back("r_sho", "r_elb", "blue");
    player_links.emplace_back("r_elb", "r_hand", "green");
    player_links.emplace_back("l_sho", "l_elb", "blue");
    player_links.emplace_back("l_elb", "l_hand", "green");
    player_links.emplace_back("r_sho", "r_hip", "yellow");
    player_links.emplace_back("l_sho", "l_hip", "yellow");
    player_links.emplace_back("r_hip", "l_hip", "blue");
    player_links.emplace_back("r_hip", "r_knee", "red");
    player_links.emplace_back("r_knee", "r_foot", "skyblue");
    player_links.emplace_back("l_hip", "l_knee", "red");
    player_links.emplace_back("l_knee", "l_foot", "skyblue");
    return player_links;
  }

 public:
  static int GetKeypointIndex(std::string_view name) {
    static const auto Values = CreateKeypointsMap();
    auto it = Values.find(name);
    if (it == Values.end()) {
      throw std::runtime_error(std::string("Key: ") + std::string(name) + " not found in player's keypoints");
    }
    return it->second;
  }

  static const std::vector<PlayerLinkInfo>& GetPlayerLinks() {
    static const auto Values = CreatePlayerLinks();
    return Values;
  }
  static glm::vec3 GetLinkColor(std::string_view name) {
    static const auto Values = CreateColorsMap();
    auto it = Values.find(name);
    if (it == Values.end()) {
      throw std::runtime_error(std::string("Color: ") + std::string(name) + " not found in player's color map");
    }
    return it->second;
  }
};
void PlayerNode3D::RenderModel() {
  if (!player_) {
    return;
  }

  const auto& mesh = model_->GetMeshes()[0];
  auto& vertices = mesh->GetVerticies();
  int index = 0;
  for (const auto& link : PlayerNodeHelpers::GetPlayerLinks()) {
    auto kp1_idx = PlayerNodeHelpers::GetKeypointIndex(link.keypoint1);
    auto kp2_idx = PlayerNodeHelpers::GetKeypointIndex(link.keypoint2);
    if (!player_->valid[kp1_idx] || !player_->valid[kp2_idx]) {
      vertices[index].position = glm::vec3(0, 0, -100);
      vertices[index + 1].position = glm::vec3(0, 0, -100);
    } else {
      const auto& kp1 = player_->keypoints[kp1_idx];
      const auto& kp2 = player_->keypoints[kp2_idx];
      vertices[index].position = eigen_to_glm(kp1);
      vertices[index + 1].position = eigen_to_glm(kp2);
    }
    index += 2;
  }
  mesh->UpdateVerticies();

  Node3D::RenderModel();
}

void PlayerNode3D::InitializeMesh(bool use_colors) {
  auto mesh = std::make_shared<Mesh3D>(true, false);
  auto& vertices = mesh->GetVerticies();
  auto& indices = mesh->GetIndicies();
  int index = 0;
  for (const auto& link : PlayerNodeHelpers::GetPlayerLinks()) {
    Vertex3D v;
    if (use_colors) {
      v.color = PlayerNodeHelpers::GetLinkColor(link.color);
    } else {
      v.color = glm::vec4(1, 1, 1, 1);
    }
    vertices.push_back(v);
    vertices.push_back(v);
    indices.push_back(index++);
    indices.push_back(index++);
  }

  mesh->UpdateIndicies();
  mesh->SetDrawType(GL_LINES);
  model_->SetMeshes({mesh});
}
PlayerNode3D::PlayerNode3D(Scene3D* scene, std::string name, bool use_colors) : Node3D(scene, std::move(name)) {
  InitializeMesh(use_colors);
  LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/lines");
}

//////

void LineNode3D::InitializeMesh() {
  auto mesh = std::make_shared<Mesh3D>(true, false);
  auto& vertices = mesh->GetVerticies();
  auto& indices = mesh->GetIndicies();
  Vertex3D v;
  v.color = glm::vec4(1, 1, 1, 1);
  vertices.push_back(v);
  vertices.push_back(v);
  indices.push_back(0);
  indices.push_back(1);

  mesh->UpdateIndicies();
  mesh->SetDrawType(GL_LINES);
  model_->SetMeshes({mesh});
}

LineNode3D::LineNode3D(Scene3D* scene, std::string name) : Node3D(scene, std::move(name)) {
  InitializeMesh();
  LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/lines");
}

void LineNode3D::SetLine(const Line3D& line) {
  line_ = line;

  const auto& mesh = model_->GetMeshes()[0];
  auto& vertices = mesh->GetVerticies();
  vertices[0].position = eigen_to_glm(line.start);
  vertices[1].position = eigen_to_glm(line.end);

  mesh->UpdateVerticies();
}
void LineNode3D::SetColorGradient(const glm::vec4& start, const glm::vec4& end) {
  const auto& mesh = model_->GetMeshes()[0];
  auto& vertices = mesh->GetVerticies();
  vertices[0].color = start;
  vertices[1].color = end;

  mesh->UpdateVerticies();
}

}  // namespace ace_monitor
