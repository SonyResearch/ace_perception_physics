// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/EstimatorTopicsPage.hpp"

#include <nav_msgs/msg/path.hpp>
#include <unordered_map>

#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"
#include "ace_monitor/scene/Nodes.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/common_values.hpp"
#include "implot.h"

namespace ace_monitor {

class EstimatorTopicsPage::EstimatorTopicsPageImpl {
 public:
  class EstimatorTopicInfo {
   public:
    explicit EstimatorTopicInfo(std::string topic) : topic_name(std::move(topic)) {
      ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kEstimatorTopic, topic_name, &topic_details);
    }
    ~EstimatorTopicInfo() {
      ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kEstimatorTopic, topic_name, nullptr);
    }
    void OnMessage(nav_msgs::msg::Path::SharedPtr msg) {
      all_fps.AddFrame();

      std::lock_guard<std::mutex> lock(topic_details.data_mutex);
      topic_details.all_fps = all_fps.CurrentFPS();
      last_msg = msg;

      if (!topic_details.enabled) {
        return;
      }
      common_values::EstimatorTopic::PredictionVector prediction;
      prediction.first = static_cast<double>(msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec);
      prediction.second.reserve(msg->poses.size());
      for (const auto &pos : msg->poses) {
        prediction.second.emplace_back(pos.pose.position.x, pos.pose.position.y, pos.pose.position.z);
      }
      topic_details.predictions = std::move(prediction);
    }

    void Update() { all_fps.Update(); }
    void Reset() {
      all_fps.Reset();
      last_msg.reset();
    }
    std::string topic_name;
    FPSCalculator all_fps;
    nav_msgs::msg::Path::SharedPtr last_msg;

    common_values::EstimatorTopic topic_details;
  };
  std::unordered_map<std::string, std::shared_ptr<EstimatorTopicInfo>> estimator_info;
  std::vector<rclzmq::Subscription<nav_msgs::msg::Path>::SharedPtr> subscribers;

  PageStatus curr_status = PageStatus::kNormal;
  common_values::EstimatorTopicsList estimator_list;

  void DiscoverTopics() {
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    auto topics_ros = node->get_topic_names_and_types();

    for (const auto &topic_ros : topics_ros) {
      const auto &topic = topic_ros.first;
      if (topic.find("estimator") != std::string::npos && estimator_info.find(topic) == estimator_info.end() &&
          topic_ros.second[0] == "nav_msgs/msg/Path") {
        auto ifo = std::make_shared<EstimatorTopicInfo>(topic);
        // Create subscriber and register callback
        auto cb = [&, ifo](nav_msgs::msg::Path::SharedPtr msg) { ifo->OnMessage(msg); };
        LOG(INFO) << "adding subscriber: " << topic;
        subscribers.push_back(node->create_subscription<nav_msgs::msg::Path>(topic, 1, cb));
        estimator_info[topic] = ifo;
        estimator_list.names.push_back(ifo->topic_name);
      }
    }
  }
  void Reset() {
    subscribers.clear();
    estimator_info.clear();
    estimator_list.names.clear();
    for (const auto &estimator : estimator_topic_predictions_nodes) {
      scene->RemoveNode(estimator.second.second);
    }
    estimator_topic_predictions_nodes.clear();
  }
  ////
  std::unordered_map<std::string, std::pair<double, DashedTrailNode3D::SharedPtr>> estimator_topic_predictions_nodes;
  Node3D::SharedPtr estimator_topics_node;
  Scene3D::SharedPtr scene;

  void Initialize3D(PerceptionVisualizerPage *visualizer) {
    scene = visualizer->GetScene();
    estimator_topics_node = scene->CreateNode("estimator_topics");
  }
  void Update3D() {
    for (const auto &it : estimator_info) {
      const auto &name = it.first;
      if (estimator_topic_predictions_nodes.find(name) == estimator_topic_predictions_nodes.end()) {
        int max_history = ace_yaml::SafeGetValue<int>(
          ACEMonitor::GetInstance().GetConfigurations()["visualizer"]["estimator_max_history"], 10);
        auto node = std::make_shared<DashedTrailNode3D>(max_history, scene.get(), name + "_prediction", 10, 10);
        estimator_topics_node->AddChild(node);
        node->SetTintColor(glm::vec4(1, 0.3F, 0.3F, 1.0F));
        estimator_topic_predictions_nodes[name] = std::pair(0, node);
      }
      std::pair<double, DashedTrailNode3D::SharedPtr> &topic_node = estimator_topic_predictions_nodes[name];
      auto *topic = static_cast<common_values::EstimatorTopic *>(
        ACEMonitor::GetInstance().GetValue(common_values::CategoryNames::kEstimatorTopic, name));
      if (!topic || !topic->enabled) {
        topic_node.second->Hide();
        continue;
      }
      topic_node.second->Show();

      if (topic->predictions.first != topic_node.first) {
        topic_node.first = topic->predictions.first;
        std::lock_guard<std::mutex> lock(topic->data_mutex);
        auto &prediction = topic->predictions.second;
        TrailListNode3D::PointsList points;
        for (const auto &p : prediction) {
          points.emplace_back(p.x(), p.y(), p.z());
        }
        topic_node.second->AddTrail(points);
      }
    }
  }
};

EstimatorTopicsPage::EstimatorTopicsPage() : IMonitorPage("Estimator Topics", "Prediction") {
  impl_ = std::make_unique<EstimatorTopicsPageImpl>();
}
EstimatorTopicsPage::~EstimatorTopicsPage() = default;

void EstimatorTopicsPage::Initialize(PerceptionVisualizerPage *visualizer) { impl_->Initialize3D(visualizer); }
void EstimatorTopicsPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  ImVec4 color;
  if (impl_->curr_status == PageStatus::kError) {
    color = Colors::kError;
  } else if (impl_->curr_status == PageStatus::kWarning) {
    color = Colors::kWarning;
  } else {
    color = Colors::kNormal;
  }
  for (auto &info : impl_->estimator_info) {
    auto current = info.second->all_fps.CurrentFPS();
    ImGui::Separator();
    ImGui::TextColored(color, "%s", info.first.c_str());
    ImGui::TextColored(color, "FPS= %d", current);
    ImGui::PushID(info.first.c_str());
    ImGui::Checkbox("Enabled", &info.second->topic_details.enabled);

    ImGui::PopID();
  }
}

void EstimatorTopicsPage::Update(float /*dt*/) {
  if (!is_open_) {
    return;
  }
  impl_->DiscoverTopics();
  for (auto &sub : impl_->estimator_info) {
    auto &ifo = sub.second;
    ifo->Update();
  }
  impl_->Update3D();
}

void EstimatorTopicsPage::OpenPage() {
  IMonitorPage::OpenPage();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kEstimatorTopic, "estimator_list",
                                     &impl_->estimator_list);
}
void EstimatorTopicsPage::ClosePage() {
  IMonitorPage::ClosePage();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kEstimatorTopic, "estimator_list", nullptr);

  impl_->Reset();
}
PageStatus EstimatorTopicsPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
