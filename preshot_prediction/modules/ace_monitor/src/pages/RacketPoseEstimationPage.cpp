// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/RacketPoseEstimationPage.hpp"

#include "ace_interfaces/msg/racket_pose.hpp"
#include "ace_interfaces/msg/racket_pose_estimate.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"
#include "ace_monitor/scene/Nodes.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/TextureHelpers.hpp"
#include "ace_monitor/utils/common_values.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "implot.h"

namespace ace_monitor {

class RacketPoseEstimationPage::RacketPoseEstimationPageImpl {
 public:
  class SharedInfo {
   public:
    static constexpr float kDefaultError = 10.0F;
    float reprojection_err{kDefaultError};
  };

  class RacketInfo {
   public:
    RacketInfo(const std::string& topic, SharedInfo* ifo) : topic_name(topic), shared_info(ifo) {
      racket_name = topic;
      if (topic_name.find("vive") != std::string::npos) {
        racket_pose.type = common_values::RacketPoseEstimation3D::RacketType::kVive;
      } else if (topic_name.find("pose") != std::string::npos) {
        racket_pose.type = common_values::RacketPoseEstimation3D::RacketType::kEstimated;
      } else {
        racket_pose.type = common_values::RacketPoseEstimation3D::RacketType::kOther;
      }
      ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kRacket, racket_name, &racket_pose);
    }
    ~RacketInfo() { ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kRacket, racket_name, nullptr); }
    void OnPose(const ace_interfaces::msg::RacketPose& msg) {
      all_fps.AddFrame();
      racket_pose.position = Eigen::Vector3d(msg.position.x, msg.position.y, msg.position.z).cast<float>();
      racket_pose.orientation =
        Eigen::Quaterniond(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z).cast<float>();
      racket_pose.visible = msg.tracked;
      racket_pose.projection_error = 0;
      // racket_pose.projection_error = msg->projection_error;
      if (msg.tracked && racket_pose.projection_error < shared_info->reprojection_err) {
        valid_fps.AddFrame();
      }
    }
    void OnMessage(ace_interfaces::msg::RacketPose::SharedPtr msg) { OnPose(*msg.get()); }
    void OnMessageEst(ace_interfaces::msg::RacketPoseEstimate::SharedPtr msg) { OnPose(msg->pose); }

    void Update() {
      all_fps.Update();
      valid_fps.Update();

      FPSCalculator::Timepoint curr_time = FPSCalculator::Clock::now();
      auto delta_time = std::chrono::duration_cast<FPSCalculator::Seconds>(curr_time - last_time);
      if (delta_time >= FPSCalculator::Seconds(1)) {
        fps_counters[0].push_back(all_fps.CurrentFPS());
        fps_counters[1].push_back(valid_fps.CurrentFPS());
        timestamp.push_back(time_counter++);
        last_time = curr_time;
        if (timestamp.size() > 10) {
          timestamp.erase(timestamp.begin());
          fps_counters[0].erase(fps_counters[0].begin());
          fps_counters[1].erase(fps_counters[1].begin());
        }
      }
    }
    void Reset() {
      all_fps.Reset();
      valid_fps.Reset();

      for (auto& fps_counter : fps_counters) {
        fps_counter.clear();
      }
      timestamp.clear();
      time_counter = 0;
      last_time = FPSCalculator::Clock::now();
    }
    FPSCalculator all_fps;
    FPSCalculator valid_fps;
    std::vector<int> fps_counters[2];
    std::vector<int> timestamp;
    int time_counter{0};

    std::string topic_name;
    std::string racket_name;
    SharedInfo* shared_info;

    FPSCalculator::Timepoint last_time;

    common_values::RacketPoseEstimation3D racket_pose;
  };

  void Update() {
    DiscoverTopics();
    for (auto& racket : racket_info) {
      racket.second->Update();
    }
  }

  void Reset() {
    racket_list.names.clear();
    subscribers.clear();
    subscribers_est.clear();
    racket_info.clear();
    for (auto& racket : rackets_nodes) {
      racket.second->Hide();
    }
  }
  void Render() {
    auto color = Colors::kNormal;
    ImGui::SliderFloat("Reprojection Error", &shared_info.reprojection_err, 1, 30);
    for (auto& it : racket_info) {
      auto& racket = it.second;
      auto current = racket->all_fps.CurrentFPS();
      auto valid = racket->valid_fps.CurrentFPS();
      ImGui::Separator();
      ImGui::SameLine();
      ImVec4 color_box = ImVec4(racket_colors[racket->racket_name].r, racket_colors[racket->racket_name].g,
                                racket_colors[racket->racket_name].b, 1.0f);
      ImGui::ColorButton("", color_box, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop, ImVec2(16, 16));
      ImGui::SameLine();
      ImGui::Checkbox(racket->racket_name.c_str(), &it.second->racket_pose.enabled);
      ImGui::TextColored(color, "All= %d | Valid=%d ", current, valid);
    }
  }

  std::map<std::string, std::shared_ptr<RacketInfo>> racket_info;
  std::vector<rclzmq::Subscription<ace_interfaces::msg::RacketPose>::SharedPtr> subscribers;
  std::vector<rclzmq::Subscription<ace_interfaces::msg::RacketPoseEstimate>::SharedPtr> subscribers_est;
  SharedInfo shared_info;
  common_values::RacketPoseEstimation3DList racket_list;

  PageStatus curr_status = PageStatus::kNormal;

  void DiscoverTopics() {
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    auto topics_ros = node->get_topic_names_and_types();
    std::vector<std::string> topics;

    for (const auto& topic_ros : topics_ros) {
      const auto& topic = topic_ros.first;
      if (racket_info.find(topic) == racket_info.end()) {
        if (topic_ros.second[0] == "ace_interfaces/msg/RacketPose") {
          auto ifo = std::make_shared<RacketInfo>(topic, &shared_info);
          // Create subscriber and register callback
          auto cb = [&, ifo](ace_interfaces::msg::RacketPose::SharedPtr msg) { ifo->OnMessage(msg); };
          LOG(INFO) << "adding subscriber: " << topic;
          subscribers.push_back(node->create_subscription<ace_interfaces::msg::RacketPose>(topic, 1, cb));
          racket_info[topic] = ifo;
          racket_list.names.push_back(ifo->racket_name);
        } else if (topic_ros.second[0] == "ace_interfaces/msg/RacketPoseEstimate") {
          auto ifo = std::make_shared<RacketInfo>(topic, &shared_info);
          // Create subscriber and register callback
          auto cb = [&, ifo](ace_interfaces::msg::RacketPoseEstimate::SharedPtr msg) { ifo->OnMessageEst(msg); };
          LOG(INFO) << "adding subscriber: " << topic;
          subscribers_est.push_back(node->create_subscription<ace_interfaces::msg::RacketPoseEstimate>(topic, 1, cb));
          racket_info[topic] = ifo;
          racket_list.names.push_back(ifo->racket_name);
        }
      }
    }
  }

  std::unordered_map<std::string, Node3D::SharedPtr> rackets_nodes;
  std::unordered_map<std::string, glm::vec4> racket_colors;
  Scene3D::SharedPtr scene;
  std::string racket_path;

  void Initialize3D(PerceptionVisualizerPage* visualizer) {
    scene = visualizer->GetScene();
    racket_path = ACEMonitor::GetInstance().GetDataPath() + "models/racket/racket.obj";
  }

  void Update3D() {
    for (const auto& it : racket_info) {
      const auto& racket_name = it.first;
      Node3D::SharedPtr racket_node;
      auto* racket_ptr = static_cast<common_values::RacketPoseEstimation3D*>(
        ACEMonitor::GetInstance().GetValue(common_values::CategoryNames::kRacket, racket_name));
      if (rackets_nodes.find(racket_name) == rackets_nodes.end()) {
        auto model = scene->LoadNode(racket_path, racket_name + "_mesh", "color_shader");
        model->SetOrientation(glm::quat(glm::vec3(0, M_PI_2, 0)));
        model->SetPosition(glm::vec3(0, 0, -0.15));

        static int color_seed = 0;
        auto color = TextureHelpers::GenerateUniqueColor(color_seed);
        model->SetTintColor(color);
        racket_colors[racket_name] = color;
        color_seed += 1;

        racket_node = scene->CreateNode(racket_name);
        racket_node->AddChild(model);
        rackets_nodes[racket_name] = racket_node;
      } else {
        racket_node = rackets_nodes[racket_name];
      }

      if (!racket_ptr || !racket_ptr->visible || !racket_ptr->enabled) {
        racket_node->Hide();
      } else {
        racket_node->Show();
        racket_node->SetPosition(
          glm::vec3(racket_ptr->position.x(), racket_ptr->position.y(), racket_ptr->position.z()));
        racket_node->SetOrientation(glm::quat(racket_ptr->orientation.w(), racket_ptr->orientation.x(),
                                              racket_ptr->orientation.y(), racket_ptr->orientation.z()));
      }
    }
  }
};

RacketPoseEstimationPage::RacketPoseEstimationPage() : IMonitorPage("Racket Pose Estimation", "Perception") {
  impl_ = std::make_unique<RacketPoseEstimationPageImpl>();
}
RacketPoseEstimationPage::~RacketPoseEstimationPage() = default;

void RacketPoseEstimationPage::Initialize(PerceptionVisualizerPage* visualizer) { impl_->Initialize3D(visualizer); }
void RacketPoseEstimationPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  impl_->Render();
}

void RacketPoseEstimationPage::Update(float /*dt*/) {
  if (!is_open_) {
    return;
  }
  impl_->Update();
  impl_->Update3D();
}

void RacketPoseEstimationPage::OpenPage() {
  IMonitorPage::OpenPage();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kRacket, "rackets_list", &impl_->racket_list);
}
void RacketPoseEstimationPage::ClosePage() {
  IMonitorPage::ClosePage();
  impl_->Reset();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kRacket, "rackets_list", nullptr);
}
PageStatus RacketPoseEstimationPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
