// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/BallSpinPage.hpp"

#include "ace_interfaces/msg/points_with_covariance.hpp"
#include "ace_interfaces/msg/spins_with_covariance.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"
#include "ace_monitor/scene/Nodes.hpp"
#include "ace_monitor/scene/rendering/Mesh3D.hpp"
#include "ace_monitor/scene/rendering/Model3D.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/MeshGenerator.hpp"
#include "ace_monitor/utils/common_values.hpp"
#include "implot.h"
#include "vision_common/estimators/SpinEstimator.hpp"
#include "vision_common/estimators/VelocityEstimator.hpp"
namespace ace_monitor {

class BallSpinPage::BallSpinPageImpl {
 public:
  static constexpr int kSamplesCount = 5000;
  static constexpr float kMaxValidSpinRadsPerSec = 700;
  BallSpinPageImpl() {
    velocity_estimator = std::make_shared<vision_common::VelocityEstimator>();
    (void)velocity_estimator->Initialize(200, 4, 6);
  }
  void OnMessage(ace_interfaces::msg::SpinsWithCovariance::SharedPtr msg) {
    all_fps.AddFrame();
    auto msg_timestamp = static_cast<double>(msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec);
    if (!state.timestamp.empty() && msg_timestamp < state.timestamp.back()) {
      state.Reset();
    }
    if (!msg->spins.empty()) {
      Eigen::Vector3f spin =
        Eigen::Vector3d(msg->spins.front().spin.x, msg->spins.front().spin.y, msg->spins.front().spin.z).cast<float>();

      auto norm = spin.norm();  // rads/sec

      if (norm < kMaxValidSpinRadsPerSec && norm > 0) {  // max valid spin (rad/s)
                                                         // Convert spin to desired unit
        switch (current_spin_unit) {
          case SpinUnit::kRevsPerMin:
            spin *= kRadsPerSecToRevsPerMinFactor;
            norm *= kRadsPerSecToRevsPerMinFactor;
            break;
          case SpinUnit::kRevsPerSec:
            spin *= kRadsPerSecToRevsPerSecFactor;
            norm *= kRadsPerSecToRevsPerSecFactor;
            break;
          case SpinUnit::kRadsPerSec:
            break;
          default:
            throw std::runtime_error("Invalid spin unit");
        }
        valid_fps.AddFrame();
        state.timestamp.push_back(static_cast<float>(msg_timestamp));
        state.spin[0].push_back(spin.x());
        state.spin[1].push_back(spin.y());
        state.spin[2].push_back(spin.z());
        state.spin[3].push_back(norm);

        state.current_spin = Eigen::AngleAxisf(norm, spin / norm);
        state.spin_history.push_back(norm);
        state.spin_acc += static_cast<double>(norm);
        if (state.spin_history.size() > kSamplesCount) {
          state.spin_acc -= state.spin_history.front();
          state.spin_history.erase(state.spin_history.begin());
        }

        state.spin_mean = state.spin_acc / static_cast<double>(state.spin_history.size());
        state.spin_std = 0;
        for (auto s : state.spin_history) {
          state.spin_std += (s - state.spin_mean) * (s - state.spin_mean);
        }
        state.spin_std = std::sqrt(state.spin_std / static_cast<double>(state.spin_history.size()));
      }
    }
    while (state.timestamp.size() > kSamplesCount) {
      state.timestamp.erase(state.timestamp.begin());
      for (auto& spin : state.spin) {
        spin.erase(spin.begin());
      }
    }
    last_msg = msg;
  }

  void OnTriangulationMessage(ace_interfaces::msg::PointsWithCovariance::SharedPtr msg) {
    if (!msg->points.empty()) {
      auto msg_timestamp = static_cast<double>(msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec);
      Eigen::Vector3d pos(msg->points[0].position.x, msg->points[0].position.y, msg->points[0].position.z);
      state.position = pos.cast<float>();
      if (velocity_estimator->EstimateVelocity(msg_timestamp, pos)) {
        const auto& vel = velocity_estimator->GetLastVelocity();
        const auto& cov = velocity_estimator->GetLastCovariance();

        auto vel_norm = vel.norm();
        auto cov_norm = cov.norm();
        ball_free_flight = vel_norm > 1.0F &&  // check if ball in free flight (velocity is high)
                           cov_norm < 1.0F;    // and no contact (covariance is low)
        if (!ball_free_flight) {
          velocity_estimator->Reset();
          state.spin_history.clear();
          state.spin_mean = 0;
          state.spin_std = 0;
          state.spin_acc = 0;
        }
      }
    }
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
      while (timestamp.size() > 20) {
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
    state.Reset();
    velocity_estimator->Reset();
  }
  FPSCalculator all_fps;
  FPSCalculator valid_fps;
  std::vector<int> fps_counters[2];
  std::vector<int> timestamp;

  FPSCalculator::Timepoint last_time;

  int time_counter{0};
  class State {
   public:
    std::vector<float> timestamp;
    std::vector<float> spin[4];
    Eigen::AngleAxisf current_spin;
    float current_orientation{0};
    Eigen::Vector3f position;
    std::vector<double> spin_history;

    double spin_acc{0};
    double spin_mean{0};
    double spin_std{0};

    void Reset() {
      timestamp.clear();
      for (auto& s : spin) {
        s.clear();
      }
      spin_acc = 0;
      spin_mean = 0;
      spin_std = 0;
      spin_history.clear();
      current_orientation = 0;
      position = Eigen::Vector3f(0, 0, -100);
      current_spin = Eigen::Quaternionf::Identity();
    }
  } state;

  enum class SpinClassification {
    kUnkown,
    kBackSpin,
    kTopSpin,
    kLeftSpin,
    kRightSpin,
    kCwCorkSpin,
    kCcwCorkSpin,
  };

  ace_interfaces::msg::SpinsWithCovariance::SharedPtr last_msg;
  rclzmq::Subscription<ace_interfaces::msg::SpinsWithCovariance>::SharedPtr sub;
  rclzmq::Subscription<ace_interfaces::msg::PointsWithCovariance>::SharedPtr triang_sub;
  vision_common::VelocityEstimator::SharedPtr velocity_estimator;
  bool ball_free_flight{false};

  PageStatus curr_status = PageStatus::kNormal;
  // unit conversion snippet by Fabian S. (https://github.com/SonyResearch/project_ace/pull/4450)
  enum class SpinUnit { kRadsPerSec, kRevsPerMin, kRevsPerSec };
  static constexpr std::array<std::string_view, 3> kSpinUnitLabels = {"rad/s", "rpm", "rps"};
  SpinUnit current_spin_unit = SpinUnit::kRadsPerSec;
  static constexpr float kRadsPerSecToRevsPerSecFactor = (1.0 / (2.0 * M_PI));
  static constexpr float kRadsPerSecToRevsPerMinFactor = 60.0 * kRadsPerSecToRevsPerSecFactor;

  bool track_ball{false};
  bool use_raw_spin{false};
  void UpdateSubscriber() {
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    sub = node->create_subscription<ace_interfaces::msg::SpinsWithCovariance>(
      use_raw_spin ? "/sensors/spin_estimation/spins" : "/sensors/spin_estimation_filtered/spins", 1,
      [&](ace_interfaces::msg::SpinsWithCovariance::SharedPtr msg) { OnMessage(msg); });
  }
  void DiscoverTopics() {
    // Create subscriber and register callback
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    UpdateSubscriber();
    triang_sub = node->create_subscription<ace_interfaces::msg::PointsWithCovariance>(
      "/sensors/ball_triangulation/points", 1,
      [&](ace_interfaces::msg::PointsWithCovariance::SharedPtr msg) { OnTriangulationMessage(msg); });
  }
  Node3D::SharedPtr root_node;
  Node3D::SharedPtr ball_node;
  void Initialize3D(PerceptionVisualizerPage* visualizer) {
    std::string ball_path =
      ACEMonitor::GetInstance().GetDataPath() + "/models/ping_pong_ball/textured_sphere.obj";
    root_node = visualizer->GetScene()->CreateNode("Ball_Spinner");
    root_node->SetPosition(glm::vec3(0, 0, 0.4));
    root_node->Hide();
    ball_node = visualizer->GetScene()->LoadNode(ball_path, "ball", "simple_shader");
    ball_node->SetScale(glm::vec3(0.05));
    ball_node->SetTintColor(glm::vec4(1, 1, 1, 0.5F));
    root_node->AddChild(ball_node);
    {
      // create ring
      auto node = visualizer->GetScene()->CreateNode("ball_ring");
      root_node->AddChild(node);
      node->LoadShaders(ACEMonitor::GetInstance().GetDataPath() + "shaders/lines");
      node->SetTintColor(glm::vec4(1.0F, 0, 0, 1));
      auto model = node->GetModel();
      auto mesh = std::make_shared<Mesh3D>(false, false);
      auto& vertices = mesh->GetVerticies();
      auto& indices = mesh->GetIndicies();
      mesh->SetDrawType(GL_LINES);
      float radius = 0.1F;

      Vertex3D v;

      v.color = glm::vec3(1, 1, 1);
#define ADD_VERTEX(pos, uv) \
  v.position = pos;         \
  v.uv0.x = uv;             \
  vertices.push_back(v);    \
  indices.push_back(index++);

      int index = 0;
      float max_angle = 270 * M_PI / 180.0F;
      const int samples_count = 29;
      for (int i = 0; i <= samples_count; ++i) {
        glm::vec3 pos;
        pos.x =
          radius * static_cast<float>(std::sin(max_angle * static_cast<float>(i) / static_cast<float>(samples_count)));
        pos.z = 0;
        pos.y =
          radius * static_cast<float>(std::cos(max_angle * static_cast<float>(i) / static_cast<float>(samples_count)));
        ADD_VERTEX(pos, 0);
      }
      // construct arrow head
      glm::vec3 arrow_head;
      glm::vec3 arrow_base;
      arrow_head.x = radius * static_cast<float>(std::sin(0));
      arrow_head.y = radius * static_cast<float>(std::cos(0));
      arrow_head.z = 0;

      arrow_base.x = radius * static_cast<float>(std::sin(max_angle * 5.0F / static_cast<float>(samples_count)));
      arrow_base.y = radius * static_cast<float>(std::cos(max_angle * 5.0F / static_cast<float>(samples_count)));
      arrow_base.z = 0;

      Eigen::Vector3f arrow_normal(arrow_head.x - arrow_base.x, arrow_head.y - arrow_base.y,
                                   arrow_head.z - arrow_base.z);
      arrow_normal.normalize();

      Eigen::Matrix3f rotation_matrix;
      rotation_matrix.block<1, 3>(0, 0) = arrow_normal.cross(Eigen::Vector3f::UnitY()).normalized();
      rotation_matrix.block<1, 3>(1, 0) = rotation_matrix.block<1, 3>(0, 0).cross(arrow_normal).normalized();
      rotation_matrix.block<1, 3>(2, 0) = arrow_normal;

      float arrow_head_radius = 0.03F;
      const int arrow_circle_count = 30;
      for (int i = 0; i < arrow_circle_count; ++i) {
        auto angle = 2 * M_PI * static_cast<double>(i) / static_cast<double>(arrow_circle_count);
        Eigen::Vector3f v_pos;
        v_pos.x() = arrow_head_radius * static_cast<float>(std::sin(angle));
        v_pos.y() = arrow_head_radius * static_cast<float>(std::cos(angle));
        v_pos.z() = 0;

        v_pos = rotation_matrix * v_pos;

        glm::vec3 pos(v_pos.x(), v_pos.y(), v_pos.z());
        pos += arrow_base;

        ADD_VERTEX(arrow_head, 0);
        ADD_VERTEX(pos, 0);
      }

      ADD_VERTEX(glm::vec3(0, 0, 0), 0);
      ADD_VERTEX(glm::vec3(0, 0, 0.1), 0);

      mesh->UpdateVerticies();
      mesh->UpdateIndicies();

      model->SetMeshes({mesh});
    }
  }
};

BallSpinPage::BallSpinPage() : IMonitorPage("Ball Spin", "Perception") { impl_ = std::make_unique<BallSpinPageImpl>(); }
BallSpinPage::~BallSpinPage() = default;
void BallSpinPage::Initialize(PerceptionVisualizerPage* visualizer) { impl_->Initialize3D(visualizer); }
void BallSpinPage::RenderWidgets() {
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
  ImGui::TextColored(color, "All= %d | Valid=%d ", current, valid);
  ImGui::Checkbox("Attach to ball", &impl_->track_ball);

  if (ImGui::Checkbox("Use Raw Data", &impl_->use_raw_spin)) {
    impl_->UpdateSubscriber();
  }
  const char* current_unit = impl_->kSpinUnitLabels[static_cast<int>(impl_->current_spin_unit)].data();
  ImGui::Text("Spin= %.0f±%.0f%s", impl_->state.spin_mean, impl_->state.spin_std, current_unit);
  if (impl_->ball_free_flight) {
    auto spin = -impl_->state.current_spin.axis();
    auto vel = impl_->velocity_estimator->GetLastVelocity().cast<float>().normalized();

    BallSpinPageImpl::SpinClassification spin_class = BallSpinPageImpl::SpinClassification::kUnkown;

    auto spin_vel_dp = spin.dot(vel);
    auto spin_vel_cross = spin.cross(vel);
    auto spin_vel_cross_len = spin_vel_cross.norm();
    if (spin_vel_cross_len > 0) {
      spin_vel_cross /= spin_vel_cross_len;
    }

    if (std::abs(spin_vel_dp) > 0.5F) {
      // cork spin
      if (spin_vel_dp < 0) {
        spin_class = BallSpinPageImpl::SpinClassification::kCwCorkSpin;
      } else {
        spin_class = BallSpinPageImpl::SpinClassification::kCcwCorkSpin;
      }
    } else {
      if (std::abs(spin.y()) > 0.4) {
        if (spin_vel_cross.z() < 0) {
          spin_class = BallSpinPageImpl::SpinClassification::kBackSpin;
        } else {
          spin_class = BallSpinPageImpl::SpinClassification::kTopSpin;
        }
      } else {
        if (spin_vel_cross.y() < 0) {
          spin_class = BallSpinPageImpl::SpinClassification::kLeftSpin;
        } else {
          spin_class = BallSpinPageImpl::SpinClassification::kRightSpin;
        }
      }
    }

    std::map<BallSpinPageImpl::SpinClassification, std::string> mapping_str;
    mapping_str[BallSpinPageImpl::SpinClassification::kUnkown] = "Unkown";
    mapping_str[BallSpinPageImpl::SpinClassification::kLeftSpin] = "LeftSpin";
    mapping_str[BallSpinPageImpl::SpinClassification::kRightSpin] = "RightSpin";
    mapping_str[BallSpinPageImpl::SpinClassification::kTopSpin] = "TopSpin";
    mapping_str[BallSpinPageImpl::SpinClassification::kBackSpin] = "BackSpin";
    mapping_str[BallSpinPageImpl::SpinClassification::kCwCorkSpin] = "CW CorkSpin";
    mapping_str[BallSpinPageImpl::SpinClassification::kCcwCorkSpin] = "CCW CorkSpin";

    ImGui::Text("Classification= %s ", mapping_str[spin_class].c_str());
  }
  // Add dropdown for spin unit selection
  if (ImGui::BeginCombo("Spin Unit", current_unit)) {
    for (size_t i = 0; i < impl_->kSpinUnitLabels.size(); i++) {
      bool is_selected = (current_unit == impl_->kSpinUnitLabels[i]);
      if (ImGui::Selectable(impl_->kSpinUnitLabels[i].data(), is_selected)) {
        auto selected = static_cast<BallSpinPageImpl::SpinUnit>(i);
        if (selected != impl_->current_spin_unit) {
          impl_->state.Reset();
          impl_->current_spin_unit = selected;
          impl_->current_spin_unit = selected;
        }
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  std::ostringstream y_axis_label;
  y_axis_label << "Spin [" << impl_->kSpinUnitLabels[static_cast<int>(impl_->current_spin_unit)].data() << "]";

  if (ImPlot::BeginPlot("Statistics", ImVec2(300, 200), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
    ImPlot::SetupAxes("Time", "FPS",
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit);
    ImPlot::PlotLine<int>("All", impl_->timestamp.data(), impl_->fps_counters[0].data(),
                          static_cast<int>(impl_->timestamp.size()));
    ImPlot::PlotLine<int>("Valid", impl_->timestamp.data(), impl_->fps_counters[1].data(),
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

  if (ImPlot::BeginPlot("Spin", ImVec2(300, 200), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
    ImPlot::SetupAxisLimits(ImAxis_Y1, -600, 600, ImPlotCond_Once);
    ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 0.1F);
    ImPlot::SetupAxes("Time", y_axis_label.str().c_str(),
                      ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                      ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit);
    static const std::string Titles[] = {"X", "Y", "Z", "Norm"};

    for (int i = 0; i < 4; ++i) {
      ImPlot::PlotScatter(Titles[i].c_str(), impl_->state.timestamp.data(), impl_->state.spin[i].data(),
                          static_cast<int>(impl_->state.timestamp.size()));
    }
    ImPlot::PopStyleVar();
    ImPlot::EndPlot();
  }
}

void BallSpinPage::Update(float dt) {
  if (!is_open_) {
    return;
  }
  impl_->Update();
  {
    float spin_direction = 0;
    spin_direction = impl_->state.current_spin.angle() * 1e-2F;
    impl_->state.current_orientation += dt * spin_direction;
    Eigen::Matrix3f rt = Eigen::Matrix3f::Identity();
    Eigen::Vector3f z_axis = -impl_->state.current_spin.axis();
    z_axis.z() = -z_axis.z();
    Eigen::Vector3f x_axis;
    Eigen::Vector3f y_axis;
    if (std::abs(z_axis.dot(Eigen::Vector3f::UnitX())) < 0.9) {
      y_axis = z_axis.cross(Eigen::Vector3f::UnitX()).normalized();
      x_axis = y_axis.cross(z_axis).normalized();
    } else {
      x_axis = Eigen::Vector3f::UnitY().cross(z_axis).normalized();
      y_axis = z_axis.cross(x_axis).normalized();
    }

    rt.block<1, 3>(0, 0) = x_axis;
    rt.block<1, 3>(1, 0) = y_axis;
    rt.block<1, 3>(2, 0) = z_axis;
    auto rotation =
      Eigen::Quaternionf(rt) * Eigen::AngleAxisf(impl_->state.current_orientation, Eigen::Vector3f::UnitZ());
    glm::quat q(rotation.w(), rotation.x(), rotation.y(), rotation.z());
    impl_->root_node->SetOrientation(q);
    if (impl_->track_ball) {
      impl_->ball_node->Hide();
      impl_->root_node->SetPosition(
        glm::vec3(impl_->state.position.x(), impl_->state.position.y(), impl_->state.position.z()));
    } else {
      impl_->ball_node->Show();
      impl_->root_node->SetPosition(glm::vec3(0, 0, 0.5));
    }
  }
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

void BallSpinPage::OpenPage() {
  IMonitorPage::OpenPage();

  impl_->DiscoverTopics();
  impl_->Reset();
  impl_->root_node->Show();
}
void BallSpinPage::ClosePage() {
  IMonitorPage::ClosePage();
  impl_->sub = nullptr;
  impl_->triang_sub = nullptr;
  impl_->root_node->Hide();
}
PageStatus BallSpinPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
