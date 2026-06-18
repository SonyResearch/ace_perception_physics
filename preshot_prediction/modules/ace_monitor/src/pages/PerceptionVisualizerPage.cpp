// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"

#include <glm/gtx/color_space.hpp>

#include "ace_interfaces/msg/image_data.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/scene/CameraNode.hpp"
#include "ace_monitor/scene/Nodes.hpp"
#include "ace_monitor/scene/TextRenderer.hpp"
#include "ace_monitor/scene/rendering/Mesh3D.hpp"
#include "ace_monitor/scene/rendering/Model3D.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/Geometry.hpp"
#include "ace_monitor/utils/MeshGenerator.hpp"
#include "ace_monitor/utils/TextureHelpers.hpp"
#include "ace_monitor/utils/common_values.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

////
#include "quickhull/Structs/Plane.hpp"
////
#include <unordered_map>

#include "quickhull/MathUtils.hpp"
#include "quickhull/QuickHull.hpp"

namespace ace_monitor {

class PerceptionVisualizerPage::VisualizerPageImpl {
 public:
  void Update() { UpdateCameras(); }
  void UpdateCameras() {
    auto* camera_list = static_cast<common_values::CameraDetectionList*>(
      ACEMonitor::GetInstance().GetValue(common_values::CategoryNames::kCamera, "camera_list"));
    if (!camera_list) {
      for (const auto& cam : camera_model_nodes) {
        cam.second->SetTintColor(glm::vec4(0.5, 0.5, 0.5, 1));
        frustum_nodes[cam.first]->SetTintColor(glm::vec4(0.5, 0.5, 0.5, 1));
      }
      return;
    }

    for (const auto& cam : camera_list->names) {
      auto* camera_info = static_cast<common_values::CameraDetection*>(
        ACEMonitor::GetInstance().GetValue(common_values::CategoryNames::kCamera, cam));
      if (!camera_info) {
        continue;
      }
      auto it = camera_model_nodes.find(cam);
      if (it == camera_model_nodes.end()) {
        continue;
      }
      if (camera_info->all_fps == 0) {
        it->second->SetTintColor(glm::vec4(Colors::kError.x, Colors::kError.y, Colors::kError.z, 1));
      } else if (camera_info->ball_visible) {
        it->second->SetTintColor(glm::vec4(Colors::kNormal.x, Colors::kNormal.y, Colors::kNormal.z, 1));
      } else {
        it->second->SetTintColor(glm::vec4(Colors::kWarning.x, Colors::kWarning.y, Colors::kWarning.z, 1));
      }
      frustum_nodes[it->first]->SetTintColor(it->second->GetTintColor());
    }
  }

  void Reset() {}

  Node3D::SharedPtr Create3DAxis(const std::string& name) {
    auto mesh = MeshGenerator::GenerateAxis();

    auto node = scene->CreateNode(name);
    node->GetModel()->SetMeshes({mesh});
    node->LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/lines");
    return node;
  }

  void DrawNodeTree(const Node3D::SharedPtr& node) {
    bool visible = node->IsVisible();
    ImGui::PushID(node->GetName().c_str());
    if (ImGui::Checkbox("", &visible)) {
      if (visible) {
        node->Show();
      } else {
        node->Hide();
      }
    }
    ImGui::PopID();
    ImGui::SameLine();
    if (node->GetChilds().empty()) {
      ImGui::Text("%s", node->GetName().c_str());
    } else {
      if (ImGui::TreeNodeEx(node->GetName().c_str())) {
        for (const auto& child : node->GetChilds()) {
          DrawNodeTree(child);
        }
        ImGui::TreePop();
      }
    }
  }

  void SetViewingCamera(const std::string& name) {
    auto& camnera = cameras["cam"];
    auto& frustum_node = frustum_nodes[name];
    auto& cam_node = camera_nodes[name];
    const auto& frustum = frustum_node->GetFrustum();
    auto pos = cam_node->GetPosition();
    auto target = cam_node->GetMatrix() * glm::vec4(0, 0, 1, 1);
    auto up = cam_node->GetMatrix() * glm::vec4(0, -1, 0, 0);
    camnera->SetPosition(pos);
    camnera->SetUpAxis(glm::vec3(up.x, up.y, up.z));
    camnera->SetTarget(glm::vec3(target.x, target.y, target.z));
    camnera->SetFOV(static_cast<float>(frustum.GetVFov() / M_PI) * 180.0F);
    {
      auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
      auto topics_ros = node->get_topic_names_and_types();
      auto topic = "/sensors/" + name + "/image";
      auto cb = [&](ace_interfaces::msg::ImageData::SharedPtr msg) { OnCameraImageMessage(msg); };
      LOG(INFO) << "Subscribing to camera: " << topic;
      camera_image_subscriber = node->create_subscription<ace_interfaces::msg::ImageData>(topic, 1, cb);
      enable_camera_texture = true;
    }
  }

  void UpdateVOI() {
    Plane3D plane(-voi_z_height, Eigen::Vector3f(0, 0, 1));
    float width = 6;
    float length = 14;
    float height = 4;
    auto samples_x = static_cast<int>(length / voi_precision);
    auto samples_y = static_cast<int>(width / voi_precision);
    auto samples_z = static_cast<int>(voi_3d ? (height / voi_precision) : 1);

    std::vector<quickhull::Vector3<float>> point_cloud;

    voi_points.clear();

    if (!show_cameras) {
      for (auto& cam : selected_cameras_voi) {
        if (cam.second) {
          camera_nodes[cam.first]->Show();
        } else {
          camera_nodes[cam.first]->Hide();
        }
      }
    }
    voi_volume = 0;
    for (int x_idx = 0; x_idx < samples_x; ++x_idx) {
      auto x = static_cast<float>(x_idx) * voi_precision - length / 2;
      for (int y_idx = 0; y_idx < samples_y; ++y_idx) {
        auto y = static_cast<float>(y_idx) * voi_precision - width / 2;
        for (int z_idx = 0; z_idx < samples_z; ++z_idx) {
          auto z = voi_3d ? (static_cast<float>(z_idx) * voi_precision - 0.7F) : voi_z_height;

          Eigen::Vector3f point(x, y, z);
          bool valid = true;
          int count = 0;

          for (auto& cam : selected_cameras_voi) {
            if (cam.second) {
              auto dist =
                glm::distance(frustum_nodes[cam.first]->GetPosition(), glm::vec3(point.x(), point.y(), point.z()));
              if (!frustum_nodes[cam.first]->GetFrustum().IsPointInside(point) || dist > 9) {
                if (voi_min_cameras == 0) {
                  valid = false;
                  break;
                }
                continue;
              }
              ++count;
            }
          }

          if (valid && count >= voi_min_cameras && count > 0) {
            voi_volume++;
            PointsRendererNode3D::Point pt;
            pt.pos = glm::vec3(point.x(), point.y(), point.z());
            pt.color_code = count;
            voi_points.push_back(pt);
            point_cloud.emplace_back(point.x(), point.y(), point.z());
          }
        }
      }
    }
    voi_volume *= std::pow<float>(voi_precision, 3.0F);

    if (point_cloud.empty()) {
      voi_convex_node->Hide();
      voi_points_node->Hide();
      return;
    }
    auto hull = voi_hull_calculator.getConvexHull(point_cloud, false, false);
    auto vb = hull.getVertexBuffer();
    auto ib = hull.getIndexBuffer();
    std::vector<glm::vec3> points;
    points.reserve(vb.size());
    for (const auto& v : vb) {
      points.emplace_back(v.x, v.y, v.z);
    }

    if (voi_point_cloud) {
      voi_points_node->SetScale(glm::vec3(voi_precision));
      voi_points_node->SetPoints(voi_points);
      voi_points_node->Show();
      voi_convex_node->Hide();
    } else {
      voi_convex_node->Show();
      voi_points_node->Hide();
      voi_convex_node->SetPoints(points, ib);
    }
  }

  void OnCameraImageMessage(ace_interfaces::msg::ImageData::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(image_mutex);
    last_image_message = msg;
  }
  void UpdateCameraTexture() {
    if (enable_camera_texture) {
      std::scoped_lock<std::mutex> lock(image_mutex);
      if (last_image_message != nullptr) {
        if (camera_texture_id == 0) {
          camera_texture_id = TextureHelpers::CreateTexture(last_image_message->width, last_image_message->height);
        }
        auto img = TextureHelpers::ConvertBayer8ToRGB(last_image_message->data.data(), last_image_message->width,
                                                      last_image_message->height, last_image_message->step);
        TextureHelpers::LoadTexture(img, camera_texture_id);
        last_image_message = nullptr;
      }
    }
  }
  void DisableCameraTexture() {
    enable_camera_texture = false;
    std::lock_guard<std::mutex> lock(image_mutex);
    last_image_message = nullptr;
    camera_image_subscriber = nullptr;
    if (camera_texture_id != 0) {
      TextureHelpers::DestroyTexture(camera_texture_id);
      camera_texture_id = 0;
    }
  }

  void Render() {
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
    auto curr_pos = ImGui::GetCursorPos();
    scene->Render();
    if (enable_camera_texture && camera_texture_id != 0) {
      const auto& resolution = scene->GetResolution();
      ImGui::SetCursorPos(curr_pos);
      // NOLINTNEXTLINE
      ImGui::Image(reinterpret_cast<void*>(camera_texture_id),
                   ImVec2(static_cast<float>(resolution.x), static_cast<float>(resolution.y)), ImVec2(0, 0),
                   ImVec2(1, 1), ImVec4(1, 1, 1, camera_overlay_opacity));
    }
  }

  Scene3D::SharedPtr scene;
  std::unordered_map<std::string, CameraNode::SharedPtr> cameras;
  std::unordered_map<std::string, Node3D::SharedPtr> camera_nodes;
  std::unordered_map<std::string, Node3D::SharedPtr> camera_model_nodes;
  std::vector<Node3D::SharedPtr> axis_nodes;
  std::unordered_map<std::string, FrustumNode3D::SharedPtr> frustum_nodes;

  VolumeOfInterestNode3D::SharedPtr voi_convex_node;
  PointsRendererNode3D::SharedPtr voi_points_node;
  GLuint voi_colorcodes_texture_id{0};

  Node3D::SharedPtr grid_node;

  std::unordered_map<std::string, bool> selected_cameras_voi;

  std::unordered_map<std::string, TextRendererNode3D::SharedPtr> camera_names;

  std::vector<PointsRendererNode3D::Point> voi_points;

  TextRenderer::SharedPtr text_renderer;
  quickhull::QuickHull<float> voi_hull_calculator;

  std::mutex image_mutex;
  ace_interfaces::msg::ImageData::SharedPtr last_image_message;
  GLuint camera_texture_id{0};
  rclzmq::Subscription<ace_interfaces::msg::ImageData>::SharedPtr camera_image_subscriber;
  bool enable_camera_texture{false};
  float camera_overlay_opacity{0.5F};

  bool wireframe{false};
  float voi_precision{0.1F};

  std::string selected_camera;
  bool show_cameras{true};
  bool show_grid{true};
  bool show_axis{true};
  bool show_frustum{true};

  float frustum_zfar{0.5F};
  float voi_z_height{0};
  bool voi_3d{false};
  int voi_min_cameras{0};
  bool voi_point_cloud{false};

  float ortho_zoom{5};
  float voi_volume{0};

  std::string config_name;
  calibration::CameraCalibrationParameters::SharedPtr camera_calib_params_ptr;
};

PerceptionVisualizerPage::PerceptionVisualizerPage() : IMonitorPage("Perception Visualizer", "Visualizer") {
  impl_ = std::make_unique<VisualizerPageImpl>();
  impl_->camera_calib_params_ptr = std::make_shared<calibration::CameraCalibrationParameters>();
}
PerceptionVisualizerPage::~PerceptionVisualizerPage() = default;

void PerceptionVisualizerPage::Initialize(PerceptionVisualizerPage*) {
  impl_->scene = std::make_shared<Scene3D>();
  auto width = ace_yaml::SafeGetValue<int>(ACEMonitor::GetInstance().GetConfigurations()["visualizer"]["width"], 1280);
  auto height = ace_yaml::SafeGetValue<int>(ACEMonitor::GetInstance().GetConfigurations()["visualizer"]["height"], 720);
  auto ortho_size =
    ace_yaml::SafeGetValue<float>(ACEMonitor::GetInstance().GetConfigurations()["visualizer"]["ortho_size"], 5);
  impl_->scene->Initialize(width, height);

  // side view camera
  for (const auto& view : {"side", "front", "back", "top", "perspective", "cam"}) {
    impl_->cameras[view] = std::make_shared<CameraNode>(impl_->scene.get(), view);
    impl_->cameras[view]->SetUpAxis(glm::vec3(0, 0, 1));
    impl_->cameras[view]->SetTarget(glm::vec3(0, 0, 2));
    impl_->cameras[view]->SetProjectionType(CameraNode::CameraProjectionType::kOrtho);
    impl_->cameras[view]->SetOrthSize(ortho_size);
  }

  impl_->cameras["side"]->SetPosition(glm::vec3(0, 3, 2));
  impl_->cameras["side"]->SetTarget(glm::vec3(0, 0, 2));

  // Front view camera
  impl_->cameras["front"]->SetPosition(glm::vec3(3, 0, 2));
  impl_->cameras["front"]->SetTarget(glm::vec3(0, 0, 2));

  // Back view camera
  impl_->cameras["back"]->SetPosition(glm::vec3(-3, 0, 2));
  impl_->cameras["back"]->SetTarget(glm::vec3(0, 0, 1));

  // top view camera
  impl_->cameras["top"]->SetPosition(glm::vec3(0, 0, 5));
  impl_->cameras["top"]->SetTarget(glm::vec3(0, 0, 0));
  impl_->cameras["top"]->SetUpAxis(glm::vec3(0, -1, 0));

  // Perspective view camera
  impl_->cameras["perspective"]->SetPosition(glm::vec3(4, 4, 2));
  impl_->cameras["perspective"]->SetTarget(glm::vec3(0, 0, 0));
  impl_->cameras["perspective"]->SetUpAxis(glm::vec3(0, 0, 1));
  impl_->cameras["perspective"]->SetProjectionType(CameraNode::CameraProjectionType::kPerspective);

  impl_->cameras["cam"]->SetPosition(glm::vec3(4, 4, 2));
  impl_->cameras["cam"]->SetTarget(glm::vec3(0, 0, 0));
  impl_->cameras["cam"]->SetUpAxis(glm::vec3(0, 0, 1));
  impl_->cameras["cam"]->SetProjectionType(CameraNode::CameraProjectionType::kPerspective);

  impl_->selected_camera = "perspective";
  impl_->scene->SetCamera(impl_->cameras[impl_->selected_camera]);

  std::string tt_path;
  std::string ground_path;
  std::string ball_path;
  {
    auto base = ACEMonitor::GetInstance().GetDataPath();
    if (!base.empty()) {
      tt_path = base + "/models/table_tennis_table/table.obj";
      ground_path = base + "/models/plane/plane.obj";
      ball_path = base + "/models/ping_pong_ball/textured_sphere.obj";
    }
  }

  auto table = impl_->scene->LoadNode(tt_path, "table", "simple_shader");
  table->SetPosition(glm::vec3(0, 0, -0.76 - 0.039872));
  table->SetOrientation(glm::quat(glm::vec3(0, 0, M_PI / 2)));
  auto floor = impl_->scene->LoadNode(ground_path, "floor", "simple_shader");
  floor->SetPosition(glm::vec3(0, 0, -0.76));

  auto axis = impl_->Create3DAxis("scene_axis");
  axis->SetPosition(glm::vec3(0, 0, 0.15));
  axis->SetScale(glm::vec3(0.3));
  impl_->axis_nodes.push_back(axis);

  impl_->voi_points_node = std::make_shared<PointsRendererNode3D>(impl_->scene.get(), "voi_renderer");
  impl_->scene->GetRoot()->AddChild(impl_->voi_points_node);
  {
    auto mesh = MeshGenerator::GenerateBox(1, 1, 1);
    mesh->SetDrawType(GL_LINES);

    impl_->voi_points_node->GetModel()->SetMeshes({mesh});
    impl_->voi_points_node->SetTintColor(glm::vec4(1, 0, 0, 1));
    impl_->voi_points_node->LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/lines");
  }
  {
    impl_->grid_node = impl_->scene->CreateNode("grid_node");
    auto grid_fine = std::make_shared<GridNode>(impl_->scene.get(), "grid_fine", 0.5F, 5.0F, 5.0F);
    grid_fine->SetTintColor(glm::vec4(0.5, 0.5, 0.5, 0.5));
    impl_->grid_node->AddChild(grid_fine);
    auto grid_coarse = std::make_shared<GridNode>(impl_->scene.get(), "grid_coarse", 1.0F, 5.0F, 0.0F);
    grid_coarse->SetTintColor(glm::vec4(0.5, 0.5, 0.5, 1));
    impl_->grid_node->AddChild(grid_coarse);
  }

  impl_->text_renderer = std::make_shared<TextRenderer>();
  int camera_font_size =
    ace_yaml::SafeGetValue<int>(ACEMonitor::GetInstance().GetConfigurations()["visualizer"]["camera_font_size"], 24);
  impl_->text_renderer->Initialize(ACEMonitor::GetInstance().GetDataPath() + "fonts/Antonio-Regular.ttf",
                                   camera_font_size);
  // initialize calibration

  impl_->config_name = ACEMonitor::GetInstance().GetAppArg("--config", "");
  if (!impl_->config_name.empty()) {
    std::string calibration_share = ament_index_cpp::get_package_share_directory("calibration");
    std::string camera_calib_params_path =
      calibration_share.empty() ? std::string{}
                                : calibration_share + "/parameters/camera_calibration/" + impl_->config_name + ".yaml";
    if (!impl_->camera_calib_params_ptr->Initialize(camera_calib_params_path)) {
      LOG(WARNING) << "Error: Failed to load camera calibration file: " << camera_calib_params_path;
    } else {
      // Initialize camera nodes
      std::string camera_model_path;
      impl_->voi_convex_node = std::make_shared<VolumeOfInterestNode3D>(impl_->scene.get(), "volume_of_interest");
      impl_->scene->GetRoot()->AddChild(impl_->voi_convex_node);
      impl_->voi_convex_node->SetTintColor(glm::vec4(1, 0, 0, 0.5));

      camera_model_path = ACEMonitor::GetInstance().GetDataPath() + "models/camera/tinker.obj";
      for (const auto& camera_name : impl_->camera_calib_params_ptr->camera_names) {
        auto camera = impl_->camera_calib_params_ptr->cameras[camera_name];
        auto camera_node = impl_->scene->CreateNode("camera_" + camera_name);
        auto model_node = impl_->scene->LoadNode(camera_model_path, "camera_" + camera_name + "_mesh", "color_shader");
        auto position = camera.T_origin_camera.block<3, 1>(0, 3);
        glm::mat4 matrix = glm::mat4(1);
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            matrix[i][j] = camera.T_origin_camera.coeff(j, i);
          }
        }
        camera_node->SetPosition(glm::vec3(position.x(), position.y(), position.z()));
        camera_node->SetOrientation(glm::quat_cast(matrix));
        model_node->SetTintColor(glm::vec4(0.5, 0.5, 0.5, 1));
        model_node->SetScale(glm::vec3(0.005));
        model_node->SetOrientation(glm::quat(glm::vec3(-M_PI_2, 0, 0)));

        auto cam_axis = impl_->Create3DAxis("camera_" + camera_name + "_axis");
        auto cam_frustum =
          std::make_shared<FrustumNode3D>(camera, impl_->scene.get(), "camera_" + camera_name + "_frustum");
        cam_frustum->SetZFar(impl_->frustum_zfar);
        cam_axis->SetScale(glm::vec3(0.3));
        camera_node->AddChild(cam_axis);
        camera_node->AddChild(model_node);
        camera_node->AddChild(cam_frustum);
        impl_->camera_nodes[camera_name] = camera_node;
        impl_->camera_model_nodes[camera_name] = model_node;
        impl_->axis_nodes.push_back(cam_axis);
        impl_->frustum_nodes[camera_name] = cam_frustum;
        impl_->selected_cameras_voi[camera_name] = false;

        auto camera_name_node =
          std::make_shared<TextRendererNode3D>(impl_->scene.get(), camera_name + "_name", impl_->text_renderer);
        camera_name_node->SetText(camera_name);
        camera_name_node->SetTextSize(0.5);
        camera_name_node->SetTintColor(glm::vec4(0, 0, 0, 1));
        camera_node->AddChild(camera_name_node);
        impl_->camera_names[camera_name] = camera_name_node;
      }
    }

    {
      // setup color codes
      const int max_camera_count = 20;
      const auto camera_count = impl_->camera_calib_params_ptr->cameras.size();
      std::vector<unsigned char> color_codes_color(camera_count * 3);
      for (size_t i = 0; i < camera_count; ++i) {
        float hue = 360 * (static_cast<float>(i) / static_cast<float>(max_camera_count));
        auto color = glm::rgbColor(glm::vec3(hue, 1, 1));
        impl_->voi_points_node->SetColorCode(static_cast<int>(i + 1), glm::vec4(color.r, color.g, color.b, 1));
        color_codes_color[i * 3 + 0] = static_cast<unsigned char>(color.r * 255);
        color_codes_color[i * 3 + 1] = static_cast<unsigned char>(color.g * 255);
        color_codes_color[i * 3 + 2] = static_cast<unsigned char>(color.b * 255);
      }
      impl_->voi_colorcodes_texture_id =
        TextureHelpers::CreateTexture(static_cast<int>(camera_count), 1, 3, color_codes_color.data());
    }
  }
}
void PerceptionVisualizerPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  impl_->UpdateCameraTexture();
  if (ImGui::BeginCombo("View Type", impl_->selected_camera.c_str())) {
    for (auto& camera : impl_->cameras) {
      bool is_selected = (impl_->selected_camera == camera.first);
      if (ImGui::Selectable(camera.first.c_str(), is_selected)) {
        impl_->selected_camera = camera.first;
        impl_->DisableCameraTexture();
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
    impl_->scene->SetCamera(impl_->cameras[impl_->selected_camera]);
  }

  if (ImGui::BeginTable("", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings)) {
    ImGui::TableNextColumn();
    if (ImGui::CollapsingHeader("Settings")) {
      if (!impl_->config_name.empty()) {
        ImGui::Text("Loaded Configuration: %s", impl_->config_name.c_str());
        ImGui::Text("VOI: %.2f m3", impl_->voi_volume);
      }
      ImGui::Checkbox("Wireframe", &impl_->wireframe);
      if (ImGui::Checkbox("Show Axis", &impl_->show_axis)) {
        for (auto& node : impl_->axis_nodes) {
          if (impl_->show_axis) {
            node->Show();
          } else {
            node->Hide();
          }
        }
      }
      if (ImGui::Checkbox("Show Grid", &impl_->show_grid)) {
        if (impl_->show_grid) {
          impl_->grid_node->Show();
        } else {
          impl_->grid_node->Hide();
        }
      }
      if (ImGui::Checkbox("Show Cameras", &impl_->show_cameras)) {
        for (auto& node : impl_->camera_nodes) {
          if (impl_->show_cameras) {
            node.second->Show();
          } else {
            node.second->Hide();
          }
        }
      }
      if (ImGui::Checkbox("Show Frustum", &impl_->show_frustum)) {
        for (auto& node : impl_->frustum_nodes) {
          if (impl_->show_frustum) {
            node.second->Show();
          } else {
            node.second->Hide();
          }
        }
      }

      if (ImGui::SliderFloat("ZFar", &impl_->frustum_zfar, 0.5, 10)) {
        for (auto& node : impl_->frustum_nodes) {
          node.second->SetZFar(impl_->frustum_zfar);
        }
      }
      if (ImGui::SliderFloat("Zoom", &impl_->ortho_zoom, 3, 8)) {
        for (auto& node : impl_->cameras) {
          node.second->SetOrthSize(impl_->ortho_zoom);
        }
      }
    }
    if (ImGui::CollapsingHeader("Scene")) {
      impl_->DrawNodeTree(impl_->scene->GetRoot());
    }
    if (!impl_->config_name.empty()) {
      if (ImGui::CollapsingHeader("Volume of Interest")) {
        if (ImGui::Checkbox("3D Volume", &impl_->voi_3d)) {
          impl_->UpdateVOI();
        }
        if (ImGui::Checkbox("Points Cloud", &impl_->voi_point_cloud)) {
          impl_->UpdateVOI();
        }
        if (ImGui::SliderFloat("Precision", &impl_->voi_precision, 0.1, 0.3)) {
          impl_->UpdateVOI();
        }
        if (!impl_->voi_3d && ImGui::SliderFloat("Z-Height", &impl_->voi_z_height, -0.7, 5)) {
          impl_->UpdateVOI();
        }
        if (ImGui::InputInt("Min Cameras", &impl_->voi_min_cameras)) {
          if (impl_->voi_min_cameras > 0) {
            for (auto& c : impl_->selected_cameras_voi) {
              c.second = true;
            }

          } else {
            for (auto& c : impl_->selected_cameras_voi) {
              c.second = false;
            }
          }
          impl_->UpdateVOI();
        }
        if (impl_->voi_min_cameras > 0 && impl_->voi_point_cloud) {
          auto width = ImGui::GetColumnWidth();
          const auto& colors = impl_->voi_points_node->GetColorCodes();
          // NOLINTNEXTLINE
          ImGui::Image(reinterpret_cast<void*>(impl_->voi_colorcodes_texture_id), ImVec2(width, 10), ImVec2(0, 0),
                       ImVec2(1, 1), ImVec4(1, 1, 1, 1));
          int min = 20;
          int max = 0;
          for (const auto& clr : colors) {
            if (min > clr.first) {
              min = clr.first;
            }
            if (max < clr.first) {
              max = clr.first;
            }
          }
          ImGui::Text("%d", min);
          ImGui::SameLine(width / 2);
          ImGui::Text("%d", (min + max) / 2);
          ImGui::SameLine(width - 5);
          ImGui::Text("%d", max);
        }
        if (impl_->enable_camera_texture) {
          ImGui::SliderFloat("Overlay Opacity", &impl_->camera_overlay_opacity, 0, 1);
        }
        if (impl_->voi_min_cameras == 0) {
          for (auto& cam : impl_->selected_cameras_voi) {
            ImGui::PushID(cam.first.c_str());
            if (ImGui::Checkbox(cam.first.c_str(), &cam.second)) {
              impl_->UpdateVOI();
            }
            if (ImGui::IsItemHovered()) {
              if (ImGui::BeginItemTooltip()) {
                const auto& f = impl_->frustum_nodes[cam.first]->GetFrustum();
                ImGui::Text("HFov: %d", static_cast<int>(f.GetHFov() * 180.0F / M_PI));
                ImGui::Text("VFov: %d", static_cast<int>(f.GetVFov() * 180.0F / M_PI));
                ImGui::EndTooltip();
              }
            }
            ImGui::SameLine();
            if (ImGui::Button("View")) {
              impl_->selected_camera = "cam";
              impl_->scene->SetCamera(impl_->cameras[impl_->selected_camera]);
              impl_->SetViewingCamera(cam.first);
            }
            ImGui::PopID();
          }
        }
      }
    }
    ImGui::TableNextColumn();
    impl_->Render();
    ImGui::EndTable();
  }
}

void PerceptionVisualizerPage::Update(float /*dt*/) {
  if (!is_open_) {
    return;
  }
  impl_->Update();
}

void PerceptionVisualizerPage::OpenPage() {
  IMonitorPage::OpenPage();
  impl_->Reset();
}
void PerceptionVisualizerPage::ClosePage() { IMonitorPage::ClosePage(); }
PageStatus PerceptionVisualizerPage::GetStatus() const { return PageStatus::kNormal; }

Scene3D::SharedPtr PerceptionVisualizerPage::GetScene() { return impl_->scene; }
const std::unordered_map<std::string, CameraNode::SharedPtr>& PerceptionVisualizerPage::GetCameras() {
  return impl_->cameras;
}
const std::unordered_map<std::string, Node3D::SharedPtr>& PerceptionVisualizerPage::GetCalibrationCameras() {
  return impl_->camera_nodes;
}
calibration::CameraCalibrationParameters::SharedPtr PerceptionVisualizerPage::GetCalibration() {
  return impl_->camera_calib_params_ptr;
}
const std::string& PerceptionVisualizerPage::GetConfigName() { return impl_->config_name; }
}  // namespace ace_monitor
