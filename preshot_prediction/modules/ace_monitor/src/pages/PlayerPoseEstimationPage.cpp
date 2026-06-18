// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/PlayerPoseEstimationPage.hpp"

#include "ace_interfaces/msg/player_pose.hpp"
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

class PlayerPoseEstimationPage::PlayerPoseEstimationPageImpl {
 public:
  class SharedInfo {
   public:
    static constexpr float kDefaultError = 10.0F;
    float reprojection_err{kDefaultError};
    float confidence{0.5};
  };

  static constexpr size_t kMinValidKeypoints = 5;
  static constexpr size_t kMaxHistoryTimestamps = 10;

  class PlayerInfo {
   public:
    PlayerInfo(const std::string& topic, SharedInfo* ifo) : topic_name(topic), shared_info(ifo) {
      this->player_name = topic;
      ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kPlayer, player_name, &player_pose);
    }
    ~PlayerInfo() { ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kPlayer, player_name, nullptr); }
    void OnMessage(ace_interfaces::msg::PlayerPose::SharedPtr msg) {
      all_fps.AddFrame();
      size_t total_valid = 0;
      for (size_t i = 0; i < std::size(player_pose.keypoints); ++i) {
        player_pose.keypoints[i] =
          Eigen::Vector3d(msg->keypoints[i].x, msg->keypoints[i].y, msg->keypoints[i].z).cast<float>();
        player_pose.confidence[i] = msg->confidences[i];
        player_pose.projection_err[i] = msg->projection_error[i];
        if (msg->projection_error[i] < shared_info->reprojection_err &&
            msg->projection_error[i] > shared_info->confidence) {
          total_valid++;
          player_pose.valid[i] = true;
        } else {
          player_pose.valid[i] = false;
        }
      }
      if (total_valid > kMinValidKeypoints) {
        valid_fps.AddFrame();
      }

      last_msg = msg;
    }

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
        if (timestamp.size() > kMaxHistoryTimestamps) {
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
      last_msg.reset();
    }

    bool enabled{true};
    FPSCalculator all_fps;
    FPSCalculator valid_fps;
    std::vector<int> fps_counters[2];
    std::vector<int> timestamp;
    int time_counter{0};

    std::string topic_name;
    std::string player_name;
    SharedInfo* shared_info;

    FPSCalculator::Timepoint last_time;
    ace_interfaces::msg::PlayerPose::SharedPtr last_msg;

    common_values::PlayerPoseEstimation3D player_pose;
  };

  void Update() {
    DiscoverTopics();
    for (auto& player : player_info) {
      player.second->Update();
    }
  }

  void Reset() {
    player_list.names.clear();
    subscribers.clear();
    player_info.clear();
    for (auto& player : players_nodes) {
      player.second->Hide();
    }
  }
  void Render() {
    auto color = Colors::kNormal;
    ImGui::SliderFloat("Reprojection Error", &shared_info.reprojection_err, 1, 30);
    ImGui::SliderFloat("Confidence", &shared_info.confidence, 0, 1);
    for (auto& it : player_info) {
      auto& player = it.second;
      auto current = player->all_fps.CurrentFPS();
      auto valid = player->valid_fps.CurrentFPS();
      ImGui::Separator();
      ImVec4 color_box = ImVec4(player_colors[player->player_name].r, player_colors[player->player_name].g,
                                player_colors[player->player_name].b, 1.0f);
      ImGui::ColorButton("", color_box, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop, ImVec2(16, 16));
      ImGui::SameLine();
      ImGui::Checkbox(player->player_name.c_str(), &player->enabled);
      ImGui::TextColored(color, "%s", player->player_name.c_str());
      ImGui::TextColored(color, "All= %d | Valid=%d ", current, valid);
    }
  }

  std::unordered_map<std::string, PlayerNode3D::SharedPtr> players_nodes;
  std::unordered_map<std::string, glm::vec4> player_colors;
  Scene3D::SharedPtr scene;

  void Initialize3D(PerceptionVisualizerPage* visualizer) { scene = visualizer->GetScene(); }
  void Update3D() {
    for (const auto& it : player_info) {
      const auto& player_name = it.first;
      PlayerNode3D::SharedPtr player_node;
      if (players_nodes.find(player_name) == players_nodes.end()) {
        player_node = std::make_shared<PlayerNode3D>(scene.get(), player_name, false);

        static int color_seed = 0;
        auto color = TextureHelpers::GenerateUniqueColor(color_seed);
        player_node->SetTintColor(color);
        player_colors[player_name] = color;
        color_seed += 1;

        scene->GetRoot()->AddChild(player_node);
        players_nodes[player_name] = player_node;
      } else {
        player_node = players_nodes[player_name];
      }
      auto* player = static_cast<common_values::PlayerPoseEstimation3D*>(
        ACEMonitor::GetInstance().GetValue(common_values::CategoryNames::kPlayer, player_name));
      if (!player || !it.second->enabled) {
        player_node->Hide();
        continue;
      }
      player_node->Show();

      player_node->SetPlayer(player);
    }
  }
  std::map<std::string, std::shared_ptr<PlayerInfo>> player_info;
  std::vector<rclzmq::Subscription<ace_interfaces::msg::PlayerPose>::SharedPtr> subscribers;
  SharedInfo shared_info;
  common_values::PlayerPoseEstimation3DList player_list;

  PageStatus curr_status = PageStatus::kNormal;

  void DiscoverTopics() {
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    auto topics_ros = node->get_topic_names_and_types();
    std::vector<std::string> topics;

    for (const auto& topic_ros : topics_ros) {
      const auto& topic = topic_ros.first;
      if (player_info.find(topic) == player_info.end() && topic_ros.second[0] == "ace_interfaces/msg/PlayerPose") {
        auto ifo = std::make_shared<PlayerInfo>(topic, &shared_info);
        // Create subscriber and register callback
        auto cb = [&, ifo](ace_interfaces::msg::PlayerPose::SharedPtr msg) { ifo->OnMessage(msg); };
        LOG(INFO) << "adding subscriber: " << topic;
        subscribers.push_back(node->create_subscription<ace_interfaces::msg::PlayerPose>(topic, 1, cb));
        player_info[topic] = ifo;
        player_list.names.push_back(ifo->player_name);
      }
    }
  }

};  // namespace ace_monitor

PlayerPoseEstimationPage::PlayerPoseEstimationPage() : IMonitorPage("Player Pose Estimation", "Perception") {
  impl_ = std::make_unique<PlayerPoseEstimationPageImpl>();
}
PlayerPoseEstimationPage::~PlayerPoseEstimationPage() = default;

void PlayerPoseEstimationPage::Initialize(PerceptionVisualizerPage* visualizer) { impl_->Initialize3D(visualizer); }
void PlayerPoseEstimationPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  impl_->Render();
}

void PlayerPoseEstimationPage::Update(float /*dt*/) {
  if (!is_open_) {
    return;
  }
  impl_->Update();
  impl_->Update3D();
}

void PlayerPoseEstimationPage::OpenPage() {
  IMonitorPage::OpenPage();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kPlayer, "players_list", &impl_->player_list);
}
void PlayerPoseEstimationPage::ClosePage() {
  IMonitorPage::ClosePage();
  impl_->Reset();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kPlayer, "players_list", nullptr);
}
PageStatus PlayerPoseEstimationPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
