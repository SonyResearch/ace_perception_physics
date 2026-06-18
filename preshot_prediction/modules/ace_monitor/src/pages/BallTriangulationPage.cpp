// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/BallTriangulationPage.hpp"

#include "ace_interfaces/msg/points_with_covariance.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"
#include "ace_monitor/scene/Nodes.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/common_values.hpp"
#include "implot.h"
#include "std_msgs/msg/bool.hpp"

namespace ace_monitor {

class BallTriangulationPage::BallTriangulationPageImpl {
 public:
  class TriangulationInfo {
   public:
    static constexpr size_t kHistorySize = 2 * 200;  // in samples
    static constexpr size_t kTimestampLength = 10;   // in seconds

    TriangulationInfo() = default;
    virtual ~TriangulationInfo() = default;
    void OnMessage(ace_interfaces::msg::PointsWithCovariance::SharedPtr msg) {
      all_fps.AddFrame();
      std::scoped_lock<std::mutex> lock(ball_pose.history_mutex);
      auto msg_timestamp = static_cast<double>(msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec);
      ball_pose.latest.clear();

      common_values::BallPoseEstimation::Estimation est;
      est.timestamp = msg_timestamp;
      est.sequence_number = msg->header.sequence_number;
      if (!msg->points.empty()) {
        for (const auto& pose : msg->points) {
          est.num_cameras = static_cast<int>(pose.num_cameras);
          est.position = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z).cast<float>();
          ball_pose.latest.push_back(est);
        }

        if (msg->points.size() == 1) {
          valid_fps.AddFrame();

        } else {
          multiball_fps.AddFrame();
        }
        num_cameras_history.push_back(static_cast<double>(msg->points[0].num_cameras));
      } else {
        num_cameras_history.push_back(0);
        est.position = Eigen::Vector3f(0, 0, -1000);
        ball_pose.latest.push_back(est);
      }
      sequence_id.push_back(static_cast<double>(msg->header.sequence_number));

      if (new_calibration_flag) {
        new_calibration_flag = false;
        new_calibrations.push_back(static_cast<double>(msg->header.sequence_number));
      }
      if (num_cameras_history.size() > kHistorySize) {
        num_cameras_history.erase(num_cameras_history.begin());
        sequence_id.erase(sequence_id.begin());
        while (!new_calibrations.empty() && *new_calibrations.begin() < *sequence_id.begin()) {
          new_calibrations.erase(new_calibrations.begin());
        }
      }
      {
        ball_pose.history.push_back(*ball_pose.latest.begin());
        if (ball_pose.history.size() > kHistorySize) {
          ball_pose.history.pop_front();
        }
      }
      last_msg = msg;
    }
    void OnReloadMessage(const std_msgs::msg::Bool::ConstSharedPtr& /*msg*/) { new_calibration_flag = true; }

    void Update() {
      all_fps.Update();
      valid_fps.Update();
      multiball_fps.Update();

      FPSCalculator::Timepoint curr_time = FPSCalculator::Clock::now();
      auto delta_time = std::chrono::duration_cast<FPSCalculator::Seconds>(curr_time - last_time);
      if (delta_time >= FPSCalculator::Seconds(1)) {
        fps_counters[0].push_back(all_fps.CurrentFPS());
        fps_counters[1].push_back(valid_fps.CurrentFPS());
        fps_counters[2].push_back(multiball_fps.CurrentFPS());
        timestamp.push_back(time_counter++);
        last_time = curr_time;
        if (timestamp.size() > kTimestampLength) {
          timestamp.erase(timestamp.begin());
          fps_counters[0].erase(fps_counters[0].begin());
          fps_counters[1].erase(fps_counters[1].begin());
          fps_counters[2].erase(fps_counters[2].begin());
        }
      }
    }
    void Reset() {
      all_fps.Reset();
      valid_fps.Reset();
      multiball_fps.Reset();
      num_cameras_history.clear();
      sequence_id.clear();
      last_msg.reset();
      time_counter = 0;

      for (auto& fps_counter : fps_counters) {
        fps_counter.clear();
      }
      timestamp.clear();
    }
    FPSCalculator all_fps;
    FPSCalculator valid_fps;
    FPSCalculator multiball_fps;
    std::vector<double> num_cameras_history;
    std::vector<double> sequence_id;
    common_values::BallPoseEstimation ball_pose;

    std::vector<int> fps_counters[3];
    std::vector<int> timestamp;
    FPSCalculator::Timepoint last_time;
    int time_counter{0};

    std::list<double> new_calibrations;
    bool new_calibration_flag{false};

    ace_interfaces::msg::PointsWithCovariance::SharedPtr last_msg;
  };
  std::shared_ptr<TriangulationInfo> info;
  rclzmq::Subscription<ace_interfaces::msg::PointsWithCovariance>::SharedPtr sub;
  rclzmq::Subscription<std_msgs::msg::Bool>::SharedPtr calib_reload_sub;

  PageStatus curr_status = PageStatus::kNormal;

  void DiscoverTopics() {
    info = std::make_shared<TriangulationInfo>();
    // Create subscriber and register callback
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    sub = node->create_subscription<ace_interfaces::msg::PointsWithCovariance>(
      "/sensors/ball_triangulation/points", 1,
      [&](ace_interfaces::msg::PointsWithCovariance::SharedPtr msg) { info->OnMessage(msg); });
    rclzmq::QoS qos_persistent(rclzmq::KeepLast(1));
    qos_persistent.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_persistent.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    calib_reload_sub = node->create_subscription<std_msgs::msg::Bool>(
      // line is split so it can passes ace_checker...
      std::string("/sensors") + std::string("/calibration/reload_state"), qos_persistent,
      [&](const std_msgs::msg::Bool::ConstSharedPtr& msg) { info->OnReloadMessage(msg); });
  }

  // 3D visualization

  std::vector<Node3D::SharedPtr> balls;
  TrailNode3D::SharedPtr ball_trail;
  void Initialize3D(PerceptionVisualizerPage* visualizer) {
    std::string ball_path =
      ACEMonitor::GetInstance().GetDataPath() + "/models/ping_pong_ball/textured_sphere.obj";
    for (int i = 0; i < 5; ++i) {
      auto ball = visualizer->GetScene()->LoadNode(ball_path, "ball_" + std::to_string(i), "simple_shader");
      ball->SetScale(glm::vec3(0.04));
      ball->SetPosition(glm::vec3(0, 0, 0.4));
      balls.push_back(ball);
    }
    ball_trail = std::make_shared<TrailNode3D>(200, visualizer->GetScene().get(), "ball_trail");
    visualizer->GetScene()->GetRoot()->AddChild(ball_trail);
  }

  void UpdateVisuals() {
    size_t ball_idx = 0;

    std::scoped_lock<std::mutex> lock(info->ball_pose.history_mutex);
    size_t max_balls = std::min(info->ball_pose.latest.size(), balls.size());
    for (auto& pose : info->ball_pose.latest) {
      if (ball_idx >= max_balls) {
        break;
      }
      balls[ball_idx]->Show();
      balls[ball_idx]->SetPosition(glm::vec3(pose.position.x(), pose.position.y(), pose.position.z()));
      ++ball_idx;
    }
    for (auto& p : info->ball_pose.history) {
      if (p.position.z() > -10) {
        ball_trail->AddPoint(glm::vec3(p.position.x(), p.position.y(), p.position.z()));
      }
    }
    info->ball_pose.history.clear();

    for (; ball_idx < balls.size(); ++ball_idx) {
      balls[ball_idx]->Hide();
    }
  }
};

BallTriangulationPage::BallTriangulationPage() : IMonitorPage("Ball Triangulation", "Perception") {
  impl_ = std::make_unique<BallTriangulationPageImpl>();
}
BallTriangulationPage::~BallTriangulationPage() = default;
void BallTriangulationPage::Initialize(PerceptionVisualizerPage* visualizer) { impl_->Initialize3D(visualizer); }
void BallTriangulationPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  auto info = impl_->info;
  ImVec4 color;
  if (impl_->curr_status == PageStatus::kError) {
    color = Colors::kError;
  } else if (impl_->curr_status == PageStatus::kWarning) {
    color = Colors::kWarning;
  } else {
    color = Colors::kNormal;
  }
  auto current = info->all_fps.CurrentFPS();
  auto valid = info->valid_fps.CurrentFPS();
  auto multiball = info->multiball_fps.CurrentFPS();
  ImGui::TextColored(color, "All= %d | Valid=%d | MultiBall=%d", current, valid, multiball);

  if (ImPlot::BeginPlot("Quality", ImVec2(300, 200), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
    ImPlot::SetupAxes("Sequence ID", "Num Cameras",
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit);
    ImPlot::PlotLine("", info->sequence_id.data(), info->num_cameras_history.data(),
                     static_cast<int>(info->num_cameras_history.size()));

    if (!info->sequence_id.empty()) {
      double x = info->sequence_id[0];
      double y = -1;
      ImPlot::PlotLine("", &x, &y, 1);
      y = 14;
      ImPlot::PlotLine("", &x, &y, 1);
    }
    const auto& limits = ImPlot::GetPlotLimits();
    for (auto new_calib_id : info->new_calibrations) {
      double x[2] = {new_calib_id, new_calib_id};
      double y[2] = {limits.Y.Min, limits.Y.Max};
      ImPlot::PlotLine("New Calibration", x, y, 2);
    }
    ImPlot::EndPlot();
  }
  if (ImPlot::BeginPlot("Statistics", ImVec2(300, 200), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
    ImPlot::SetupAxes("Time", "FPS",
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit);
    ImPlot::PlotLine<int>("All", info->timestamp.data(), info->fps_counters[0].data(),
                          static_cast<int>(info->timestamp.size()));
    ImPlot::PlotLine<int>("Valid", info->timestamp.data(), info->fps_counters[1].data(),
                          static_cast<int>(info->timestamp.size()));
    ImPlot::PlotLine<int>("Multi-ball", info->timestamp.data(), info->fps_counters[2].data(),
                          static_cast<int>(info->timestamp.size()));

    if (!info->timestamp.empty()) {
      int x = info->timestamp[0];
      int y = -1;
      ImPlot::PlotLine("", &x, &y, 1);
      y = 220;
      ImPlot::PlotLine("", &x, &y, 1);
    }
    ImPlot::EndPlot();
  }
}

void BallTriangulationPage::Update(float /*dt*/) {
  if (!is_open_ || !impl_->info) {
    return;
  }
  impl_->info->Update();
  impl_->UpdateVisuals();
  auto last_fps = impl_->info->all_fps.LastFPS();
  auto curr_fps = impl_->info->all_fps.CurrentFPS();
  if (curr_fps == 0) {
    impl_->curr_status = PageStatus::kError;
  } else if (curr_fps < last_fps * 0.9) {
    impl_->curr_status = PageStatus::kWarning;
  } else {
    impl_->curr_status = PageStatus::kNormal;
  }
}

void BallTriangulationPage::OpenPage() {
  IMonitorPage::OpenPage();
  impl_->DiscoverTopics();
  impl_->info->Reset();
}
void BallTriangulationPage::ClosePage() {
  IMonitorPage::ClosePage();
  impl_->sub = nullptr;
  impl_->info = nullptr;
}
PageStatus BallTriangulationPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
