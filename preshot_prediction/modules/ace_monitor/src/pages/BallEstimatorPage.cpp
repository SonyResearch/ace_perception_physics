// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/BallEstimatorPage.hpp"

#include "ace_interfaces/msg/ball_state.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"
#include "ace_monitor/scene/Nodes.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/common_values.hpp"
#include "physics_layer/PhysicsLayerAPI.hpp"

namespace ace_monitor {

class BallEstimatorPage::BallEstimatorPageImpl {
 public:
  class EstimatorTopicInfo {
   public:
    explicit EstimatorTopicInfo(std::string topic) : topic_name(std::move(topic)) {}
    void OnMessage(ace_interfaces::msg::BallState::SharedPtr msg) {
      all_fps.AddFrame();

      std::scoped_lock<std::mutex> lock(ball_pose.history_mutex);
      ball_pose.all_fps = all_fps.CurrentFPS();

      auto msg_timestamp = static_cast<double>(msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec);
      ball_pose.latest.clear();
      if (msg->state_valid) {
        valid_fps.AddFrame();
        ball_pose.valid_fps = valid_fps.CurrentFPS();
        common_values::BallPoseEstimation::Estimation est;
        est.timestamp = msg_timestamp;
        est.position = Eigen::Vector3d(msg->position.x, msg->position.y, msg->position.z).cast<float>();
        est.velocity = Eigen::Vector3d(msg->velocity.x, msg->velocity.y, msg->velocity.z).cast<float>();
        est.spin = Eigen::Vector3d(msg->spin.x, msg->spin.y, msg->spin.z).cast<float>();
        ball_pose.latest.push_back(est);
      }
    }

    void Update() {
      all_fps.Update();
      valid_fps.Update();
    }
    void Reset() {
      all_fps.Reset();
      valid_fps.Reset();
    }
    std::string topic_name;
    FPSCalculator all_fps;
    FPSCalculator valid_fps;

    common_values::BallPoseEstimation ball_pose;
  };

  void Update() {
    for (auto &sub : estimator_info) {
      auto &ifo = sub.second;
      ifo->Update();
    }
  }

  void Reset() {}

  std::vector<rclzmq::Subscription<ace_interfaces::msg::BallState>::SharedPtr> subscribers;
  std::unordered_map<std::string, std::shared_ptr<EstimatorTopicInfo>> estimator_info;

  PageStatus curr_status = PageStatus::kNormal;

  void DiscoverTopics() {
    // Create subscriber and register callback
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    auto topics_ros = node->get_topic_names_and_types();
    for (const auto &topic_ros : topics_ros) {
      const auto &topic = topic_ros.first;
      if (estimator_info.find(topic) == estimator_info.end() && topic_ros.second[0] == "ace_interfaces/msg/BallState") {
        auto ifo = std::make_shared<EstimatorTopicInfo>(topic);
        // Create subscriber and register callback
        auto cb = [&, ifo](ace_interfaces::msg::BallState::SharedPtr msg) { ifo->OnMessage(msg); };
        LOG(INFO) << "adding subscriber: " << topic;
        subscribers.push_back(node->create_subscription<ace_interfaces::msg::BallState>(topic, 1, cb));
        estimator_info[topic] = ifo;
        // estimator_list.names.push_back(ifo->topic_name);
      }
    }
  }

  ////
  class EstimatedBallNode {
   public:
    Node3D::SharedPtr node;
    Node3D::SharedPtr ball;
    DashedTrailNode3D::SharedPtr prediction;
  };
  std::unordered_map<std::string, EstimatedBallNode> estimator_topic_est_nodes;
  Node3D::SharedPtr estimator_topics_node;
  Scene3D::SharedPtr scene;

  physics::PhysicsParameters::SharedPtr physics_params;

  static constexpr int kMaxHistory = 10;

  void Initialize3D(PerceptionVisualizerPage *visualizer) {
    scene = visualizer->GetScene();
    estimator_topics_node = scene->CreateNode("ball_estimator_topics");

    std::string parameters_path =
      ACEMonitor::GetInstance().GetDataPath() + "/config/parameters_default.yaml";
    physics_params = physics::PhysicsParameters::LoadParametersFromYamlFile(parameters_path, "");
  }
  void Update3D() {
    for (const auto &it : estimator_info) {
      const auto &name = it.first;
      if (estimator_topic_est_nodes.find(name) == estimator_topic_est_nodes.end()) {
        std::string ball_path =
          ACEMonitor::GetInstance().GetDataPath() + "/models/ping_pong_ball/textured_sphere.obj";
        EstimatedBallNode node;
        node.node = scene->CreateNode(name + "_node");
        estimator_topics_node->AddChild(node.node);
        node.prediction = std::make_shared<DashedTrailNode3D>(kMaxHistory, scene.get(), name + "_prediction", 10, 10);
        node.ball = scene->LoadNode(ball_path, name + "_ball", "simple_shader");
        node.ball->SetScale(glm::vec3(0.04));
        node.ball->SetPosition(glm::vec3(0, 0, 0.4));
        node.ball->SetTintColor(glm::vec4(1, 0.3F, 0.3F, 1.0F));
        node.prediction->SetTintColor(glm::vec4(1, 0.3F, 0.3F, 1.0F));
        node.node->AddChild(node.ball);
        node.node->AddChild(node.prediction);

        estimator_topic_est_nodes[name] = node;
      }

      auto &node = estimator_topic_est_nodes[name];

      auto &ball_pose = it.second->ball_pose;
      if (ball_pose.enabled) {
        node.ball->Show();
        node.prediction->Show();

      } else {
        node.ball->Hide();
        node.prediction->Hide();
      }
      if (ball_pose.enabled && !ball_pose.latest.empty()) {
        std::scoped_lock<std::mutex> lock(ball_pose.history_mutex);
        const auto &pose = ball_pose.latest.front();
        node.ball->Show();
        node.ball->SetTintColor(glm::vec4(1, 0, 0, 1));
        node.ball->SetPosition(glm::vec3(pose.position.x(), pose.position.y(), pose.position.z()));

        physics::BallState ball_state;
        ball_state.position = pose.position;
        ball_state.linear_velocity = pose.velocity;
        ball_state.angular_velocity = pose.spin;

        TrailListNode3D::PointsList points;
        points.push_back(glm::vec3(ball_state.position.x(), ball_state.position.y(), ball_state.position.z()));

        for (int i = 0; i < 200; ++i) {
          ball_state = physics::PhysicsAPI::PredictBallState(ball_state, 0.005, physics_params, true);

          points.push_back(glm::vec3(ball_state.position.x(), ball_state.position.y(), ball_state.position.z()));
        }
        node.prediction->AddTrail(points);
        ball_pose.latest.clear();
      }
    }
  }
};

BallEstimatorPage::BallEstimatorPage() : IMonitorPage("Ball State Est.", "Prediction") {
  impl_ = std::make_unique<BallEstimatorPageImpl>();
}
BallEstimatorPage::~BallEstimatorPage() = default;

void BallEstimatorPage::Initialize(PerceptionVisualizerPage *visualizer) { impl_->Initialize3D(visualizer); }
void BallEstimatorPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  ImVec4 color;
  if (impl_->curr_status == PageStatus::kError) {
    color = Colors::kError;
  } else if (impl_->curr_status == PageStatus::kWarning) {
    color = Colors::kWarning;
  } else {
    color = Colors::kNormal;
  }

  static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
  if (ImGui::BeginTable("table1", 4, flags)) {
    ImGui::TableSetupColumn("Enable");
    ImGui::TableSetupColumn("Topic");
    ImGui::TableSetupColumn("All");
    ImGui::TableSetupColumn("Valid");
    ImGui::TableHeadersRow();
    for (auto &info : impl_->estimator_info) {
      ImGui::TableNextRow();
      auto all = info.second->all_fps.CurrentFPS();
      auto valid = info.second->valid_fps.CurrentFPS();
      ImGui::PushID(info.first.c_str());

      ImGui::TableSetColumnIndex(0);
      ImGui::Checkbox("", &info.second->ball_pose.enabled);
      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(color, "%s", info.first.c_str());
      ImGui::TableSetColumnIndex(2);
      ImGui::TextColored(color, "%d", all);
      ImGui::TableSetColumnIndex(3);
      ImGui::TextColored(color, "%d", valid);

      ImGui::PopID();
    }
    ImGui::EndTable();
  }
}

void BallEstimatorPage::Update(float /*dt*/) {
  if (!is_open_) {
    return;
  }
  impl_->DiscoverTopics();
  impl_->Update();
  impl_->Update3D();
}

void BallEstimatorPage::OpenPage() {
  IMonitorPage::OpenPage();

  impl_->Reset();
}
void BallEstimatorPage::ClosePage() {
  IMonitorPage::ClosePage();
  impl_->subscribers.clear();
  impl_->estimator_info.clear();
}
PageStatus BallEstimatorPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
