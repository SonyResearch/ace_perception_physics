// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/BallDetectorPage.hpp"

#include "ace_interfaces/msg/balls_with_attributes.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/pages/perception/AlarmsPage.hpp"
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"
#include "ace_monitor/scene/Nodes.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/TextureHelpers.hpp"
#include "ace_monitor/utils/common_values.hpp"

namespace ace_monitor {

class BallDetectorPage::BallDetectorPageImpl {
 public:
  class CameraInfo {
   public:
    explicit CameraInfo(std::string topic, BallDetectorPageImpl* o) : owner(o), topic_name(std::move(topic)) {
      camera_name = topic_name;
      std::string begin("/sensors/");
      auto idx = camera_name.find(begin);
      if (idx != std::string::npos) {
        camera_name = camera_name.substr(begin.size());
      }
      idx = camera_name.find("/ball_detection");
      if (idx != std::string::npos) {
        camera_name = camera_name.substr(0, idx);
      }
      ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kCamera, camera_name, &camera_info);
    }
    ~CameraInfo() { ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kCamera, camera_name, nullptr); }
    void OnMessage(ace_interfaces::msg::BallsWithAttributes::SharedPtr msg) {
      std::lock_guard lock(data_mutex);
      all_fps.AddFrame();
      if (!msg->balls.empty()) {
        camera_info.ball_visible = true;
        valid_fps.AddFrame();
        if (!msg->balls[0].markers.empty()) {
          markers_fps.AddFrame();
        }
      } else {
        camera_info.ball_visible = false;
      }

      std::lock_guard lock2(owner->data_mutex);
      owner->last_sequence_id = msg->header.sequence_number;
      last_msg = msg;
      msg_invalidate = true;
    }

    void Update() {
      all_fps.Update();
      valid_fps.Update();
      markers_fps.Update();

      camera_info.all_fps = all_fps.CurrentFPS();
      camera_info.valid_fps = valid_fps.CurrentFPS();
      camera_info.markers_fps = markers_fps.CurrentFPS();

      if (camera_info.all_fps == 0) {
        ACEMonitor::GetInstance().GetAlarmsPage()->SetAlarm(camera_name, "No ball detections",
                                                            AlarmsPage::AlarmLevel::kCritical, 10);
      }
    }
    rclzmq::Subscription<ace_interfaces::msg::BallsWithAttributes>::SharedPtr sub;
    FPSCalculator all_fps;
    FPSCalculator valid_fps;
    FPSCalculator markers_fps;

    BallDetectorPageImpl* owner;

    std::string topic_name;
    std::string camera_name;

    std::mutex data_mutex;

    ace_interfaces::msg::BallsWithAttributes::SharedPtr last_msg;

    bool msg_invalidate{false};

    PageStatus status{PageStatus::kNormal};

    common_values::CameraDetection camera_info;
  };
  std::unordered_map<std::string, std::shared_ptr<CameraInfo>> camera_info;
  std::vector<rclzmq::Subscription<ace_interfaces::msg::BallsWithAttributes>::SharedPtr> subscribers;
  PageStatus curr_status = PageStatus::kNormal;
  std::shared_ptr<BallDetectorPageImpl::CameraInfo> current_active;
  using CameraDetectionsMap = std::map<std::string, std::vector<Eigen::Vector3f>>;
  std::list<CameraDetectionsMap> ball_detections_queue;
  std::mutex data_mutex;
  uint64_t last_sequence_id{0};

  common_values::CameraDetectionList camera_list;

  bool visualize_detection_rays{true};

  bool show_markers{true};
  float markers_scaler{1};

  void DiscoverTopics() {
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    auto topics_ros = node->get_topic_names_and_types();

    for (const auto& topic_ros : topics_ros) {
      const auto& topic = topic_ros.first;
      if (topic.find("gcs") == std::string::npos && camera_info.find(topic) == camera_info.end() &&
          topic_ros.second[0] == "ace_interfaces/msg/BallsWithAttributes") {
        auto ifo = std::make_shared<CameraInfo>(topic, this);
        // Create subscriber and register callback
        auto cb = [&, ifo](ace_interfaces::msg::BallsWithAttributes::SharedPtr msg) { ifo->OnMessage(msg); };
        LOG(INFO) << "adding subscriber: " << topic;
        subscribers.push_back(node->create_subscription<ace_interfaces::msg::BallsWithAttributes>(topic, 1, cb));
        camera_info[topic] = ifo;
        camera_list.names.push_back(ifo->camera_name);
      }
    }
  }

  using DetectionRayVector = std::vector<LineNode3D::SharedPtr>;
  std::map<std::string, DetectionRayVector> camera_detection_rays;
  Node3D::SharedPtr detections_node;
  PerceptionVisualizerPage* visualizer;

  void Initialize3D(PerceptionVisualizerPage* visualizer) {
    this->visualizer = visualizer;
    detections_node = visualizer->GetScene()->CreateNode("detection_node");
    detections_node->SetPosition(glm::vec3(0, 0, 0));
    const auto& cameras = visualizer->GetCalibrationCameras();
    for (const auto& cam : cameras) {
      camera_detection_rays[cam.first] = DetectionRayVector();
    }
  }

  LineNode3D::SharedPtr CreateRay(Node3D::SharedPtr camera, int index) {
    auto node =
      std::make_shared<LineNode3D>(visualizer->GetScene().get(), camera->GetName() + "_" + std::to_string(index));
    node->SetPosition(glm::vec3(0, 0, 0));
    node->SetScale(glm::vec3(1, 1, 1));
    // node->SetColorGradient(glm::vec4(1, 1, 1, 1), glm::vec4(0, 0, 0, 0.5));
    detections_node->AddChild(node);

    return node;
  }
  void UpdateVisuals() {
    if (camera_detection_rays.empty() || (!visualize_detection_rays && !current_active)) {
      return;
    }

    std::vector<std::vector<Eigen::Vector3f>> ball_detections;
    std::map<std::string, int> camera_index_map;

    const auto& calibration_params = visualizer->GetCalibration();
    const auto& camera_nodes = visualizer->GetCalibrationCameras();
    for (auto& ifo : camera_info) {
      if (!visualize_detection_rays && current_active != ifo.second) {
        continue;
      }
      std::lock_guard lock(ifo.second->data_mutex);
      if (!ifo.second->msg_invalidate) {
        continue;
      }
      ifo.second->msg_invalidate = false;
      auto& rays = camera_detection_rays[ifo.second->camera_name];
      if (camera_nodes.find(ifo.second->camera_name) == camera_nodes.end()) {
        continue;
      }
      auto camera = camera_nodes.at(ifo.second->camera_name);

      const auto& cam_calib = calibration_params->cameras[ifo.second->camera_name];

      if (!ifo.second->last_msg) {
        for (auto& r : rays) {
          r->Hide();
        }
        continue;
      }

      if (rays.size() < ifo.second->last_msg->balls.size()) {
        size_t current_count = rays.size();
        rays.resize(ifo.second->last_msg->balls.size());
        for (size_t i = current_count; i < rays.size(); ++i) {
          rays[i] = CreateRay(camera, static_cast<int>(i));
        }
      }

      size_t ray_idx = 0;
      Plane3D floor(Eigen::Vector3f(0, 0, -0.76), Eigen::Vector3f(0, 0, 1));
      Ray3D ray;
      bool intersected;
      Line3D line;
      for (auto& detection : ifo.second->last_msg->balls) {
        auto detection_p2d =
          cam_calib.UndistortPoint(Eigen::Vector2d(detection.ball.center.x, detection.ball.center.y).cast<float>());
        ray.start = glm_to_eign(camera->GetPosition());
        ray.normal = (cam_calib.projection_matrix_inv.block<3, 3>(0, 0) * detection_p2d.homogeneous()).normalized();
        line.start = ray.start;
        line.end = floor.IntersectRay(ray, intersected);
        rays[ray_idx]->SetLine(line);
        rays[ray_idx]->Show();
        
        ray_idx++;
      }
      for (; ray_idx < rays.size(); ++ray_idx) {
        rays[ray_idx]->Hide();
      }
    }
  }

  void ToggleDetectionRays() {
    if (visualize_detection_rays) {
      return;
    }

    for (const auto& rays : camera_detection_rays) {
      for (const auto& ray : rays.second) {
        ray->Hide();
      }
    }
  }
};

BallDetectorPage::BallDetectorPage() : IMonitorPage("Ball Detector", "Perception") {
  impl_ = std::make_unique<BallDetectorPageImpl>();
}
BallDetectorPage::~BallDetectorPage() = default;
void BallDetectorPage::Initialize(PerceptionVisualizerPage* visualizer) { impl_->Initialize3D(visualizer); }
void BallDetectorPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();

  int total_active = 0;
  int total_detections = 0;

  for (auto& sub : impl_->camera_info) {
    auto& ifo = sub.second;
    if (ifo->all_fps.CurrentFPS() > 0) {
      total_active++;
    }
    if (ifo->valid_fps.CurrentFPS() > 0) {
      total_detections++;
    }
  }
  ImVec4 color;
  if (total_active == 0) {
    color = Colors::kError;
  } else if (total_detections < total_active / 2) {
    color = Colors::kWarning;
  } else {
    color = Colors::kNormal;
  }
  ImGui::TextColored(color, "Active: %d/%d | Detections: %d/%d", total_active,
                     static_cast<int>(impl_->camera_info.size()), total_detections, total_active);
  if (ImGui::Checkbox("Visualize Detection Rays", &impl_->visualize_detection_rays)) {
    impl_->ToggleDetectionRays();
  }

  static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
  if (ImGui::BeginTable("table1", 4, flags)) {
    ImGui::TableSetupColumn("Camera");
    ImGui::TableSetupColumn("All");
    ImGui::TableSetupColumn("Valid");
    ImGui::TableSetupColumn("Markers");
    ImGui::TableHeadersRow();

    for (auto& sub : impl_->camera_info) {
      auto& ifo = sub.second;
      ImGui::TableNextRow();
      int fps[3];
      PageStatus status[3];
      fps[0] = ifo->all_fps.CurrentFPS();
      fps[1] = ifo->valid_fps.CurrentFPS();
      fps[2] = ifo->markers_fps.CurrentFPS();

      status[0] = ifo->status;
      status[1] = fps[1] < fps[0] * 0.5 ? PageStatus::kWarning : PageStatus::kNormal;
      status[2] = fps[2] < fps[0] * 0.5 ? PageStatus::kWarning : PageStatus::kNormal;
      bool color_changed = false;
      if (impl_->current_active == ifo) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0.5, 0, 1));
        color_changed = true;
      }
      ImGui::TableSetColumnIndex(0);
      if (ImGui::Button(ifo->camera_name.c_str())) {
        impl_->ToggleDetectionRays();
        if (impl_->current_active == ifo) {
          impl_->current_active = nullptr;
        } else {
          impl_->current_active = ifo;
        }
      }
      if (color_changed) {
        ImGui::PopStyleColor();
      }
      for (int i = 0; i < 3; ++i) {
        if (status[i] == PageStatus::kError) {
          color = Colors::kError;
        } else if (status[i] == PageStatus::kWarning) {
          color = Colors::kWarning;
        } else {
          color = Colors::kNormal;
        }
        ImGui::TableSetColumnIndex(i + 1);
        ImGui::TextColored(color, "%s", std::to_string(fps[i]).c_str());
      }
    }
    ImGui::EndTable();
    if (impl_->current_active) {
      float scaler = 1 / 4.0F;
      float width = 1440;
      float height = 1080;
      ImGui::Checkbox("", &impl_->show_markers);
      ImGui::SameLine();
      ImGui::SliderFloat("Markers", &impl_->markers_scaler, 1, 4);
      const ImVec2 p = ImGui::GetCursorScreenPos();
      ImDrawList* draw_list = ImGui::GetWindowDrawList();
      draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), ImColor(0, 0, 0, 255), 0, 0);

      const auto& calibration_params = impl_->visualizer->GetCalibration();
      if (calibration_params) {
        const auto& camera_calib = calibration_params->cameras[impl_->current_active->camera_name];
        static constexpr float kTableLength = 2.74;
        static constexpr float kTableWidth = 1.525;
        static const std::vector<Eigen::Vector3f> TablePoints = {
          Eigen::Vector3f(-kTableLength * 0.5, -kTableWidth * 0.5, 0),
          Eigen::Vector3f(+kTableLength * 0.5, -kTableWidth * 0.5, 0),
          Eigen::Vector3f(+kTableLength * 0.5, +kTableWidth * 0.5, 0),
          Eigen::Vector3f(-kTableLength * 0.5, +kTableWidth * 0.5, 0),
          Eigen::Vector3f(0, -kTableWidth * 0.5, 0),
          Eigen::Vector3f(0, +kTableWidth * 0.5, 0),
        };
        static const std::vector<Eigen::Vector2i> TableIndicies = {Eigen::Vector2i(0, 1), Eigen::Vector2i(1, 2),
                                                                   Eigen::Vector2i(2, 3), Eigen::Vector2i(3, 0),
                                                                   Eigen::Vector2i(4, 5)

        };

        // draw table
        for (const auto& index : TableIndicies) {
          Eigen::Vector2f p1;
          Eigen::Vector2f p2;
          camera_calib.ProjectPoint(TablePoints[index.x()], &p1);
          camera_calib.ProjectPoint(TablePoints[index.y()], &p2);
          p1.x() = p.x + p1.x() * scaler;
          p1.y() = p.y + p1.y() * scaler;
          p2.x() = p.x + p2.x() * scaler;
          p2.y() = p.y + p2.y() * scaler;
          draw_list->AddLine(ImVec2(p1.x(), p1.y()), ImVec2(p2.x(), p2.y()), ImColor(128, 128, 128, 255));
        }

        // add X/Y axis
        Eigen::Vector2f p_origin;
        Eigen::Vector2f p_x;
        Eigen::Vector2f p_y;
        camera_calib.ProjectPoint(Eigen::Vector3f(0, 0, 0), &p_origin);
        camera_calib.ProjectPoint(Eigen::Vector3f(0.3, 0, 0), &p_x);
        camera_calib.ProjectPoint(Eigen::Vector3f(0, 0.3, 0), &p_y);
        p_origin.x() = p.x + p_origin.x() * scaler;
        p_origin.y() = p.y + p_origin.y() * scaler;
        p_x.x() = p.x + p_x.x() * scaler;
        p_x.y() = p.y + p_x.y() * scaler;
        p_y.x() = p.x + p_y.x() * scaler;
        p_y.y() = p.y + p_y.y() * scaler;
        draw_list->AddLine(ImVec2(p_origin.x(), p_origin.y()), ImVec2(p_x.x(), p_x.y()), ImColor(255, 0, 0, 255), 1);
        draw_list->AddLine(ImVec2(p_origin.x(), p_origin.y()), ImVec2(p_y.x(), p_y.y()), ImColor(0, 255, 0, 255), 1);
      }
      if (impl_->current_active->last_msg != nullptr) {
        for (auto& ball : impl_->current_active->last_msg->balls) {
          draw_list->AddCircleFilled(ImVec2(p.x + static_cast<float>(ball.ball.center.x) * scaler,
                                            p.y + static_cast<float>(ball.ball.center.y) * scaler),
                                     static_cast<float>(ball.ball.radius) * scaler, ImColor(255, 128, 128, 255), 0);
          if (impl_->show_markers) {
            for (auto& marker : ball.markers) {
              draw_list->AddCircleFilled(ImVec2(p.x + static_cast<float>(marker.center.x) * scaler,
                                                p.y + static_cast<float>(marker.center.y) * scaler),
                                         static_cast<float>(marker.radius * scaler * impl_->markers_scaler),
                                         ImColor(0, 0, 0, 255), 0);
            }
          }
        }
      }
      ImGui::Dummy(ImVec2(static_cast<float>(width) * scaler, static_cast<float>(height) * scaler));
    }
  }
}

void BallDetectorPage::Update(float /*dt*/) {
  if (!is_open_) {
    return;
  }
  impl_->DiscoverTopics();
  impl_->UpdateVisuals();
  int status = 0;
  for (auto& sub : impl_->camera_info) {
    auto& ifo = sub.second;
    ifo->Update();
    auto last_fps = ifo->all_fps.LastFPS();
    auto curr_fps = ifo->all_fps.CurrentFPS();
    if (curr_fps == 0) {
      ifo->status = PageStatus::kError;
      status |= 0x1;
    } else if (curr_fps < last_fps * 0.9) {
      ifo->status = PageStatus::kWarning;
      status |= 0x2;
    } else {
      ifo->status = PageStatus::kNormal;
    }
  }
  if (status == 0) {
    impl_->curr_status = PageStatus::kNormal;
  } else if (status & 0x1) {
    impl_->curr_status = PageStatus::kError;
  } else if (status & 0x2) {
    impl_->curr_status = PageStatus::kWarning;
  }
}

void BallDetectorPage::OpenPage() {
  IMonitorPage::OpenPage();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kCamera, "camera_list", &impl_->camera_list);
}
void BallDetectorPage::ClosePage() {
  IMonitorPage::ClosePage();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kCamera, "camera_list", nullptr);
  impl_->current_active = nullptr;
  impl_->subscribers.clear();
  impl_->camera_info.clear();
  impl_->camera_list.names.clear();
  impl_->last_sequence_id = -1;
  std::lock_guard lock(impl_->data_mutex);
  impl_->ball_detections_queue.clear();
}
PageStatus BallDetectorPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
