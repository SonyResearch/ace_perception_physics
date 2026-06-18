// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/BallPoseEstimationPage.hpp"

#include "ace_interfaces/msg/poses_with_covariance.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"
#include "ace_monitor/scene/Nodes.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/common_values.hpp"
#include "implot.h"
#include "vision_common/estimators/SpinEstimator.hpp"
#include "vision_common/estimators/VelocityEstimator.hpp"

namespace ace_monitor {

class BallPoseEstimationPage::BallPoseEstimationPageImpl {
 public:
  void OnMessage(ace_interfaces::msg::PosesWithCovariance::SharedPtr msg) {
    all_fps.AddFrame();
    std::scoped_lock<std::mutex> lock(ball_pose.history_mutex);
    auto msg_timestamp = static_cast<double>(msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec);
    ball_pose.latest.clear();
    common_values::BallPoseEstimation::Estimation est;
    est.timestamp = msg_timestamp;
    est.sequence_number = msg->header.sequence_number;
    if (!msg->poses.empty()) {
      valid_fps.AddFrame();

      for (const auto& pose : msg->poses) {
        est.num_cameras = static_cast<int>(pose.num_cameras);
        est.position = Eigen::Vector3d(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z).cast<float>();
        est.orientation = Eigen::Quaternionf(1, 0, 0, 0);
        ball_pose.latest.push_back(est);
      }
      auto& p = *ball_pose.latest.begin();

      state.timestamp.push_back(static_cast<float>(msg_timestamp));
      bool vel_added = false;
      if (msg->poses.size() > 1) {
        multi_ball_fps.AddFrame();
        for (auto& mp : msg->poses) {
          state.multi_ball_fps[0].push_back(static_cast<float>(mp.pose.position.x));
          state.multi_ball_fps[1].push_back(static_cast<float>(mp.pose.position.y));
          state.multi_ball_fps[2].push_back(static_cast<float>(mp.pose.position.z));
          state.multi_ball_fps[3].push_back(static_cast<float>(msg_timestamp));
        }
      }

      if (state.vel_estimator.EstimateVelocity(msg_timestamp, p.position.cast<double>())) {
        const auto& vel = state.vel_estimator.GetLastVelocity().cast<float>();
        state.vel[0].push_back(vel.x());
        state.vel[1].push_back(vel.y());
        state.vel[2].push_back(vel.z());
        state.vel[3].push_back(vel.norm());
        vel_added = true;
      }
      if (!vel_added) {
        state.vel[0].push_back(0);
        state.vel[1].push_back(0);
        state.vel[2].push_back(0);
        state.vel[3].push_back(0);
      }
      state.position[0].push_back(p.position.x());
      state.position[1].push_back(p.position.y());
      state.position[2].push_back(p.position.z());

    } else {
      est.position = Eigen::Vector3f(0, 0, -100);
      ball_pose.latest.push_back(est);
    }

    if (state.timestamp.size() > 400) {
      state.timestamp.erase(state.timestamp.begin());
      for (int i = 0; i < 3; ++i) {
        state.vel[i].erase(state.vel[i].begin());
        state.position[i].erase(state.position[i].begin());
      }
      state.vel[3].erase(state.vel[3].begin());
    }
    while (state.multi_ball_fps[0].size() > 200) {
      for (auto& fps : state.multi_ball_fps) {
        fps.erase(fps.begin());
      }
    }
    {
      ball_pose.history.push_back(*ball_pose.latest.begin());
      if (ball_pose.history.size() > 400) {
        ball_pose.history.pop_front();
      }
    }
    last_msg = msg;
  }

  void Update() {
    all_fps.Update();
    valid_fps.Update();
    multi_ball_fps.Update();

    FPSCalculator::Timepoint curr_time = FPSCalculator::Clock::now();
    auto delta_time = std::chrono::duration_cast<FPSCalculator::Seconds>(curr_time - last_time);
    if (delta_time >= FPSCalculator::Seconds(1)) {
      fps_counters[0].push_back(all_fps.CurrentFPS());
      fps_counters[1].push_back(valid_fps.CurrentFPS());
      fps_counters[2].push_back(multi_ball_fps.CurrentFPS());
      timestamp.push_back(time_counter++);
      last_time = curr_time;
      if (timestamp.size() > 10) {
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
    multi_ball_fps.Reset();
    for (auto& fps_counter : fps_counters) {
      fps_counter.clear();
    }
    timestamp.clear();
    time_counter = 0;
    last_time = FPSCalculator::Clock::now();
    last_msg.reset();
    state.Reset();
  }
  FPSCalculator all_fps;
  FPSCalculator valid_fps;
  FPSCalculator multi_ball_fps;
  std::vector<int> fps_counters[3];
  std::vector<int> timestamp;

  FPSCalculator::Timepoint last_time;

  common_values::BallPoseEstimation ball_pose;

  int time_counter{0};
  class State {
   public:
    vision_common::VelocityEstimator vel_estimator;
    std::vector<float> timestamp;
    std::vector<float> vel[4];
    std::vector<float> position[3];
    std::vector<float> multi_ball_fps[4];

    void Reset() {
      (void)vel_estimator.InitializeYaml(
        ACEMonitor::GetInstance().GetConfigurations()["ball_pos_estimation"]["velocity_estimator"]);
      for (int i = 0; i < 3; ++i) {
        multi_ball_fps[i].clear();
        vel[i].clear();
        position[i].clear();
      }
      vel[3].clear();
      multi_ball_fps[3].clear();
    }
  } state;

  ace_interfaces::msg::PosesWithCovariance::SharedPtr last_msg;
  rclzmq::Subscription<ace_interfaces::msg::PosesWithCovariance>::SharedPtr sub;

  PageStatus curr_status = PageStatus::kNormal;
  bool show_position{false};
  bool show_vel{false};
  bool show_multi_ball{false};

  void DiscoverTopics() {
    // Create subscriber and register callback
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    sub = node->create_subscription<ace_interfaces::msg::PosesWithCovariance>(
      "/sensors/ball_pose_estimation/poses", 1,
      [&](ace_interfaces::msg::PosesWithCovariance::SharedPtr msg) { OnMessage(msg); });
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

    std::scoped_lock<std::mutex> lock(ball_pose.history_mutex);
    size_t max_balls = std::min(ball_pose.latest.size(), balls.size());
    for (auto& pose : ball_pose.latest) {
      if (ball_idx >= max_balls) {
        break;
      }
      balls[ball_idx]->Show();
      balls[ball_idx]->SetPosition(glm::vec3(pose.position.x(), pose.position.y(), pose.position.z()));
      balls[ball_idx]->SetOrientation(
        glm::quat(pose.orientation.w(), pose.orientation.x(), pose.orientation.y(), pose.orientation.z()));
      ++ball_idx;
    }
    for (auto& p : ball_pose.history) {
      ball_trail->AddPoint(glm::vec3(p.position.x(), p.position.y(), p.position.z()));
    }
    ball_pose.history.clear();

    for (; ball_idx < balls.size(); ++ball_idx) {
      balls[ball_idx]->Hide();
    }
  }
};

BallPoseEstimationPage::BallPoseEstimationPage() : IMonitorPage("Ball Pose Estimation", "Perception") {
  impl_ = std::make_unique<BallPoseEstimationPageImpl>();
}
BallPoseEstimationPage::~BallPoseEstimationPage() = default;

void BallPoseEstimationPage::Initialize(PerceptionVisualizerPage* visualizer) { impl_->Initialize3D(visualizer); }
void BallPoseEstimationPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  ImVec4 color;
  if (impl_->curr_status == PageStatus::kError) {
    color = Colors::kError;
  } else if (impl_->curr_status == PageStatus::kWarning) {
    color = Colors::kWarning;
  } else {
    color = Colors::kNormal;
  }
  auto current = impl_->all_fps.CurrentFPS();
  auto valid = impl_->valid_fps.CurrentFPS();
  auto multi_ball_fps = impl_->multi_ball_fps.CurrentFPS();
  ImGui::TextColored(color, "All= %d | Valid=%d | MultiBalls=%d", current, valid, multi_ball_fps);
  ImGui::Checkbox("Position", &impl_->show_position);
  ImGui::Checkbox("Velocity Est.", &impl_->show_vel);
  ImGui::Checkbox("Multi Balls", &impl_->show_multi_ball);

  if (ImPlot::BeginPlot("Statistics", ImVec2(300, 200), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
    ImPlot::SetupAxes("Time", "FPS",
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit);
    ImPlot::PlotLine<int>("All", impl_->timestamp.data(), impl_->fps_counters[0].data(),
                          static_cast<int>(impl_->timestamp.size()));
    ImPlot::PlotLine<int>("Valid", impl_->timestamp.data(), impl_->fps_counters[1].data(),
                          static_cast<int>(impl_->timestamp.size()));
    ImPlot::PlotLine<int>("Multi Balls", impl_->timestamp.data(), impl_->fps_counters[2].data(),
                          static_cast<int>(impl_->timestamp.size()));

    if (!impl_->timestamp.empty()) {
      int x = impl_->timestamp[0];
      int y = -1;
      ImPlot::PlotLine("", &x, &y, 1);
      y = 220;
      ImPlot::PlotLine("", &x, &y, 1);
    }
    ImPlot::EndPlot();
  }

  if (impl_->show_position && ImPlot::BeginPlot("Position", ImVec2(300, 200), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
    ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 0.1F);
    ImPlot::SetupAxes("Time", "Position",
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit);
    ImPlot::PlotScatter("X", impl_->state.timestamp.data(), impl_->state.position[0].data(),
                        static_cast<int>(impl_->state.timestamp.size()));
    ImPlot::PlotScatter("Y", impl_->state.timestamp.data(), impl_->state.position[1].data(),
                        static_cast<int>(impl_->state.timestamp.size()));
    ImPlot::PlotScatter("Z", impl_->state.timestamp.data(), impl_->state.position[2].data(),
                        static_cast<int>(impl_->state.timestamp.size()));
    ImPlot::PopStyleVar();
    ImPlot::EndPlot();
  }
  if (impl_->show_vel &&
      ImPlot::BeginPlot("Velocity Estimation", ImVec2(300, 200), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
    ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 0.1F);
    ImPlot::SetupAxes("Time", "Velocity",
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit);
    ImPlot::PlotScatter("X", impl_->state.timestamp.data(), impl_->state.vel[0].data(),
                        static_cast<int>(impl_->state.timestamp.size()));
    ImPlot::PlotScatter("Y", impl_->state.timestamp.data(), impl_->state.vel[1].data(),
                        static_cast<int>(impl_->state.timestamp.size()));
    ImPlot::PlotScatter("Z", impl_->state.timestamp.data(), impl_->state.vel[2].data(),
                        static_cast<int>(impl_->state.timestamp.size()));
    ImPlot::PlotLine("Norm", impl_->state.timestamp.data(), impl_->state.vel[3].data(),
                     static_cast<int>(impl_->state.timestamp.size()));
    ImPlot::PopStyleVar();
    ImPlot::EndPlot();
  }
  if (impl_->show_multi_ball &&
      ImPlot::BeginPlot("Multi Balls", ImVec2(300, 200), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
    ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 0.1F);
    ImPlot::SetupAxes("Time", "Multi Balls",
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit);
    ImPlot::PlotScatter("X", impl_->state.multi_ball_fps[3].data(), impl_->state.multi_ball_fps[0].data(),
                        static_cast<int>(impl_->state.multi_ball_fps[3].size()));
    ImPlot::PlotScatter("Y", impl_->state.multi_ball_fps[3].data(), impl_->state.multi_ball_fps[1].data(),
                        static_cast<int>(impl_->state.multi_ball_fps[3].size()));
    ImPlot::PlotScatter("Z", impl_->state.multi_ball_fps[3].data(), impl_->state.multi_ball_fps[2].data(),
                        static_cast<int>(impl_->state.multi_ball_fps[3].size()));
    ImPlot::PopStyleVar();
    ImPlot::EndPlot();
  }
}

void BallPoseEstimationPage::Update(float /*dt*/) {
  if (!is_open_) {
    return;
  }
  impl_->Update();
  impl_->UpdateVisuals();
  auto last_fps = impl_->all_fps.LastFPS();
  auto curr_fps = impl_->all_fps.CurrentFPS();
  if (curr_fps == 0) {
    impl_->curr_status = PageStatus::kError;
  } else if (curr_fps < last_fps * 0.9) {
    impl_->curr_status = PageStatus::kWarning;
  } else {
    impl_->curr_status = PageStatus::kNormal;
  }
}

void BallPoseEstimationPage::OpenPage() {
  IMonitorPage::OpenPage();
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kBallPoseEstimation, "ball", &impl_->ball_pose);

  impl_->DiscoverTopics();
  impl_->Reset();
}
void BallPoseEstimationPage::ClosePage() {
  IMonitorPage::ClosePage();
  impl_->sub = nullptr;
  ACEMonitor::GetInstance().SetValue(common_values::CategoryNames::kBallPoseEstimation, "ball", nullptr);
}
PageStatus BallPoseEstimationPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
