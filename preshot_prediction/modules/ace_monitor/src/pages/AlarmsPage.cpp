// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/AlarmsPage.hpp"

#include <chrono>
#include <unordered_map>

#include "ace_interfaces/msg/balls_with_attributes.hpp"
#include "ace_interfaces/msg/perception_header.hpp"
#include "ace_interfaces/msg/player_pose.hpp"
#include "ace_interfaces/msg/points_with_covariance.hpp"
#include "ace_interfaces/msg/racket_pose_estimate.hpp"
#include "ace_interfaces/msg/spins_with_covariance.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/scene/rendering/GLTexture.hpp"
#include "ace_monitor/scene/rendering/SceneResources.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/common_values.hpp"

using namespace std::chrono_literals;
namespace ace_monitor {

class AlarmsPage::AlarmsPageImpl {
 public:
  class IAlarm {
   public:
    using clock = std::chrono::high_resolution_clock;
    using SharedPtr = std::shared_ptr<IAlarm>;

    using AlarmDesc = std::pair<clock::time_point, std::string>;

    using CallbackFunc = std::function<void((const IAlarm::clock::time_point& tp, const std::string& name,
                                             const std::string& desc, AlarmsPage::AlarmLevel level))>;

   protected:
    std::unordered_map<int, AlarmDesc> alarms_;
    std::unordered_map<int, AlarmsPage::AlarmLevel> alarms_levels_;
    std::unordered_map<int, std::string> alarms_names_;
    CallbackFunc callback_;

    void AddAlarmCode(int code, const std::string& name, AlarmsPage::AlarmLevel level) {
      if (alarms_.find(code) == alarms_.end()) {
        alarms_[code] = AlarmDesc(clock::now(), "");
      }
      alarms_levels_[code] = level;
      alarms_names_[code] = name;
    }

   public:
    virtual ~IAlarm() = default;
    void SetCallback(CallbackFunc cb) { callback_ = cb; }
    virtual std::string GetName() = 0;
    const std::unordered_map<int, AlarmDesc>& GetAlarms() { return alarms_; }
    AlarmsPage::AlarmLevel GetLevel(int code) { return alarms_levels_[code]; }
    const std::string& GetAlarmName(int code) { return alarms_names_[code]; }
    virtual void Update(float dt) = 0;
    virtual void Reset() {
      for (auto& alarm : alarms_) {
        alarm.second.second = "";
      }
    }
    virtual void SetAlarm(int code, const std::string& desc) {
      if (alarms_[code].second != desc) {
        alarms_[code].first = clock::now();
        alarms_[code].second = desc;

        if (!desc.empty() && callback_) {
          callback_(alarms_[code].first, alarms_names_[code], desc, alarms_levels_[code]);
        }
      }
    }
    virtual void Start() { Reset(); }
    virtual void Stop() { Reset(); }
  };

  template <class T>
  class ROSAlarm : public IAlarm {
   protected:
    clock::time_point last_message_time_;
    virtual void OnMessage(typename T::SharedPtr /*msg*/) { last_message_time_ = clock::now(); }
    typename rclcpp::Subscription<T>::SharedPtr sub_;

    std::string topic_name_;

   public:
    explicit ROSAlarm(std::string topic_name) : topic_name_(std::move(topic_name)) {
      last_message_time_ = clock::now();
    }
    void Reset() override {
      IAlarm::Reset();
      last_message_time_ = clock::now();
    }
    void Start() override {
      IAlarm::Start();
      auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
      sub_ = node->create_subscription<T>(topic_name_, 1, [&](typename T::SharedPtr msg) { OnMessage(msg); });
    }
    void Stop() override {
      IAlarm::Stop();
      sub_ = nullptr;
    }
  };

  class BallTriangulationAlarm : public ROSAlarm<ace_interfaces::msg::PointsWithCovariance> {
    enum class AlarmCode { kAlive, kVOI, kFPS, kFPSCritical, kMultiBalls, kGhostBall };

    static constexpr auto kActiveTimeout{1s};
    static constexpr auto kBallAlarmTimeout{10s};
    float voi_z_{-0.4F};
    FPSCalculator fps_;

    void OnMessage(typename ace_interfaces::msg::PointsWithCovariance::SharedPtr msg) override {
      ROSAlarm::OnMessage(msg);
      fps_.AddFrame();
      is_ball_visible_ = !msg->points.empty();
      if (is_ball_visible_) {
        // float timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        auto timestamp = clock::now();
        // check for ghost balls
        int ghosts_count = 0;
        int mb_count = 0;
        for (size_t i = 0; i < msg->points.size(); ++i) {
          Eigen::Vector3f p1 =
            Eigen::Vector3d(msg->points[i].position.x, msg->points[i].position.y, msg->points[i].position.z)
              .cast<float>();
          if (p1.z() < -0.1) {
            last_voi_pos_.first = timestamp;
            last_voi_pos_.second = p1;
          }
          for (size_t j = i + 1; j < msg->points.size(); ++j) {
            Eigen::Vector3f p2 =
              Eigen::Vector3d(msg->points[j].position.x, msg->points[j].position.y, msg->points[j].position.z)
                .cast<float>();
            float dist = (p1 - p2).squaredNorm();
            if (dist < 0.04 * 0.04) {
              ghosts_count++;
            } else {
              mb_count++;
            }
          }
        }
        if (ghosts_count > 0) {
          last_ghost_balls_.first = timestamp;
          last_ghost_balls_.second = ghosts_count;
        }
        if (mb_count > 0) {
          last_multi_balls_.first = timestamp;
          last_multi_balls_.second = 1 + mb_count;
        }
      }
    }

    std::pair<clock::time_point, Eigen::Vector3f> last_voi_pos_;
    std::pair<clock::time_point, int> last_multi_balls_;
    std::pair<clock::time_point, int> last_ghost_balls_;
    std::pair<clock::time_point, int> last_fps_dropped_;
    std::pair<clock::time_point, int> last_fps_critical_dropped_;

    bool is_ball_visible_{false};

   public:
    BallTriangulationAlarm() : ROSAlarm("/sensors/ball_triangulation/points") {
      AddAlarmCode(static_cast<int>(AlarmCode::kAlive), "Alive", AlarmsPage::AlarmLevel::kCritical);
      AddAlarmCode(static_cast<int>(AlarmCode::kVOI), "VOI", AlarmsPage::AlarmLevel::kCritical);
      AddAlarmCode(static_cast<int>(AlarmCode::kFPSCritical), "FPS Critical", AlarmsPage::AlarmLevel::kCritical);
      AddAlarmCode(static_cast<int>(AlarmCode::kFPS), "FPS", AlarmsPage::AlarmLevel::kWarning);
      AddAlarmCode(static_cast<int>(AlarmCode::kMultiBalls), "Multi Balls", AlarmsPage::AlarmLevel::kCritical);
      AddAlarmCode(static_cast<int>(AlarmCode::kGhostBall), "Ghosts", AlarmsPage::AlarmLevel::kWarning);

      auto now = clock::now();
      last_voi_pos_.first = now;
      last_multi_balls_.first = now;
      last_ghost_balls_.first = now;
      last_fps_dropped_.first = now;

      voi_z_ = ACEMonitor::GetInstance().GetConfigurations()["alarms"]["voi_z"].as<float>();
    }
    std::string GetName() override { return "Triangulation"; }
    void Update(float /*dt*/) override {
      auto now = clock::now();
      fps_.Update();

      auto current_fps = static_cast<int>(fps_.CurrentFPS());
      if (current_fps > 0 && current_fps < 199) {
        if (current_fps < 190) {
          last_fps_critical_dropped_.first = now;
          last_fps_critical_dropped_.second = current_fps;
        } else {
          last_fps_dropped_.first = now;
          last_fps_dropped_.second = current_fps;
        }
      }

      bool trigger_alive = (now - last_message_time_ > kActiveTimeout);
      bool trigger_voi = (last_voi_pos_.second.z() < voi_z_ && (now - last_voi_pos_.first) < kBallAlarmTimeout);
      bool trigger_ghosts = (last_ghost_balls_.second > 0 && (now - last_ghost_balls_.first) < kBallAlarmTimeout);
      bool trigger_mb = (last_multi_balls_.second > 0 && (now - last_multi_balls_.first) < kBallAlarmTimeout);
      bool trigger_fps_critical = !trigger_alive && ((now - last_fps_critical_dropped_.first) < kBallAlarmTimeout) &&
                                  last_fps_critical_dropped_.second > 0;
      bool trigger_fps = !trigger_alive && !trigger_fps_critical &&
                         ((now - last_fps_dropped_.first) < kBallAlarmTimeout) && last_fps_dropped_.second > 0;

      SetAlarm(static_cast<int>(AlarmCode::kAlive), trigger_alive ? "Triangulation is not active!" : "");

      SetAlarm(static_cast<int>(AlarmCode::kVOI),
               trigger_voi ? ("Ball out of VOI at Z: " + std::to_string(last_voi_pos_.second.z())) : "");
      SetAlarm(static_cast<int>(AlarmCode::kGhostBall),
               trigger_ghosts ? ("Ghost balls detected: " + std::to_string(last_ghost_balls_.second)) : "");

      SetAlarm(static_cast<int>(AlarmCode::kMultiBalls),
               trigger_mb ? ("Multi balls detected: " + std::to_string(last_multi_balls_.second)) : "");
      SetAlarm(static_cast<int>(AlarmCode::kFPS),
               trigger_fps ? ("Framerate dropped: " + std::to_string(last_fps_dropped_.second)) : "");
      SetAlarm(static_cast<int>(AlarmCode::kFPSCritical),
               trigger_fps_critical
                 ? ("Framerate significantly dropped: " + std::to_string(last_fps_critical_dropped_.second))
                 : "");
    }

    bool BallVisible() const { return is_ball_visible_; }

    void Reset() override {
      ROSAlarm::Reset();
      fps_.Reset();
      last_voi_pos_.second = Eigen::Vector3f(0, 0, 0);
      last_multi_balls_.second = 0;
      last_ghost_balls_.second = 0;
      last_fps_dropped_.second = 0;
    }
  };
  class GCSAlarm : public ROSAlarm<ace_interfaces::msg::SpinsWithCovariance> {
    enum class AlarmCode { kAlive };

    static constexpr auto kActiveTimeout{0.05s};

    std::shared_ptr<BallTriangulationAlarm> ball_alarm_;

   public:
    explicit GCSAlarm(std::shared_ptr<BallTriangulationAlarm> ball_alarm)
      : ROSAlarm("/sensors/gcs/spin_estimation_filtered/spins"), ball_alarm_(std::move(ball_alarm)) {
      AddAlarmCode(static_cast<int>(AlarmCode::kAlive), "Alive", AlarmsPage::AlarmLevel::kCritical);
    }
    std::string GetName() override { return "GCS"; }
    void Update(float /*dt*/) override {
      auto now = clock::now();
      auto message_dt = now - last_message_time_;
      bool trigger_alive = (message_dt > kActiveTimeout);
      if (ball_alarm_->BallVisible()) {
        SetAlarm(static_cast<int>(AlarmCode::kAlive), trigger_alive ? "No GCS Data!" : "");
      }
    }
  };

  class GenericAlarm : public IAlarm {
    std::unordered_map<std::string, int> code_map_;
    std::unordered_map<int, std::chrono::seconds> timeout_map_;
    int last_code_{0};

   public:
    GenericAlarm() = default;
    std::string GetName() override { return "Other"; }
    void Update(float /*dt*/) override {
      auto now = clock::now();
      for (auto& alarm : alarms_) {
        if (!alarm.second.second.empty() && (now - alarm.second.first) > timeout_map_[alarm.first]) {
          alarm.second.second = "";
        }
      }
    }
    void SetAlarmDetailed(const std::string& name, const std::string& desc, AlarmLevel level, int timeout) {
      auto it = code_map_.find(name);
      int code;
      if (it == code_map_.end()) {
        code_map_[name] = ++last_code_;
        code = last_code_;
      } else {
        code = it->second;
      }
      timeout_map_[code] = std::chrono::seconds(timeout);
      IAlarm::AddAlarmCode(code, name, level);
      IAlarm::SetAlarm(code, desc);
    }
  };
  class AlarmLog {
   public:
    IAlarm::clock::time_point timepoint;
    std::string name;
    std::string desc;
    AlarmsPage::AlarmLevel level;

    std::string log_str;
    void UpdateString() {
      std::stringstream ss;
      std::time_t tp = AlarmsPageImpl::IAlarm::clock::to_time_t(timepoint);
      auto* local_t = std::localtime(&tp);
      ss << std::put_time(local_t, "%H:%M:%S") << " - " << name << ": " << desc;

      log_str = ss.str();
    }
  };

  std::vector<IAlarm::SharedPtr> alarms;
  std::shared_ptr<GenericAlarm> generic_alarm;

  bool show_log{false};
  std::list<AlarmLog> alarms_log;

  PageStatus curr_status = PageStatus::kNormal;

  IAlarm::CallbackFunc callback_func;

  std::unordered_map<AlarmsPage::AlarmLevel, GLTexture::SharedPtr> alarm_levels_images;

  std::string alarm_audio_command;
  IAlarm::clock::time_point last_audio_played;
  std::thread alarm_subprocess;
  bool enable_audio{true};

  int total_warnings{0};
  int total_critical{0};
  void AddAlarm(IAlarm::SharedPtr alarm) {
    alarms.emplace_back(alarm);
    alarm->SetCallback(callback_func);
  }

  void PlayAlarmAudio() {
    if (!enable_audio) {
      return;
    }
    auto now = IAlarm::clock::now();
    if (now - last_audio_played > 5s) {
      if (alarm_subprocess.joinable()) {
        alarm_subprocess.join();
      }
      alarm_subprocess = std::thread([this]() { (void)system((alarm_audio_command + " > /dev/null 2>&1").c_str()); });
      last_audio_played = now;
    }
  }
  void Reset() {
    alarms_log.clear();
    for (const auto& alarm : alarms) {
      alarm->Reset();
    }
    last_audio_played = IAlarm::clock::now() - 10s;
  }

  AlarmsPageImpl() {
    callback_func = [this](const IAlarm::clock::time_point& tp, const std::string& name, const std::string& desc,
                           AlarmsPage::AlarmLevel level) {
      AlarmLog log;
      log.timepoint = tp;
      log.name = name;
      log.desc = desc;
      log.level = level;
      log.UpdateString();
      if (level == AlarmsPage::AlarmLevel::kCritical) {
        PlayAlarmAudio();
      }
      alarms_log.emplace_front(log);
    };
    generic_alarm = std::make_shared<GenericAlarm>();
    auto ball_tri = std::make_shared<BallTriangulationAlarm>();
    AddAlarm(ball_tri);
    AddAlarm(std::make_shared<GCSAlarm>(ball_tri));
    AddAlarm(generic_alarm);

    std::string images_path = ACEMonitor::GetInstance().GetDataPath() + "/images/";
    std::string alarm_audio_path = ACEMonitor::GetInstance().GetDataPath() + "/audio/alarm.mp3";
    alarm_audio_command = "gst-launch-1.0 filesrc location='" + alarm_audio_path +
                          "' ! decodebin ! audioresample ! audioconvert ! autoaudiosink > /dev/null 2>&1";

    alarm_levels_images[AlarmsPage::AlarmLevel::kNone] =
      SceneResources::GetInstance().GetOrCreateTexture(images_path + "none.png");
    alarm_levels_images[AlarmsPage::AlarmLevel::kWarning] =
      SceneResources::GetInstance().GetOrCreateTexture(images_path + "warning.png");
    alarm_levels_images[AlarmsPage::AlarmLevel::kCritical] =
      SceneResources::GetInstance().GetOrCreateTexture(images_path + "error.png");

    last_audio_played = IAlarm::clock::now();
  }

  void Update(float dt) {
    for (const auto& alarm : alarms) {
      alarm->Update(dt);
    }
  }
};  // namespace ace_monitor

AlarmsPage::AlarmsPage() : IMonitorPage("Alarms", "Perception") { impl_ = std::make_unique<AlarmsPageImpl>(); }
AlarmsPage::~AlarmsPage() = default;
void AlarmsPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();

  {
    auto icon = impl_->alarm_levels_images[AlarmLevel::kNone];
    if (impl_->total_critical > 0) {
      icon = impl_->alarm_levels_images[AlarmLevel::kCritical];
    } else if (impl_->total_warnings) {
      icon = impl_->alarm_levels_images[AlarmLevel::kWarning];
    }
    ImGui::SameLine((ImGui::GetWindowWidth() - 64) / 2);
    // NOLINTNEXTLINE
    ImGui::Image(reinterpret_cast<void*>(icon->GetTextureID()), ImVec2(64, 64));
  }

  ImVec4 color;
  std::string tp_str(30, '\0');
  {
    std::time_t tp = AlarmsPageImpl::IAlarm::clock::to_time_t(AlarmsPageImpl::IAlarm::clock::now());
    std::strftime(&tp_str[0], tp_str.size(), "%H:%M:%S", std::localtime(&tp));
    ImGui::Text("Time now: %s", tp_str.c_str());
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset")) {
    impl_->Reset();
  }
  ImGui::Checkbox("Enable Audio Alarm", &impl_->enable_audio);
  static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
  impl_->total_warnings = 0;
  impl_->total_critical = 0;

  if (ImGui::BeginTable("table1", 4, flags)) {
    ImGui::TableSetupColumn("Source");
    ImGui::TableSetupColumn("Time");
    ImGui::TableSetupColumn("Alarm");
    ImGui::TableSetupColumn("Description");
    ImGui::TableHeadersRow();
    for (const auto& alarm : impl_->alarms) {
      const auto& alarm_codes = alarm->GetAlarms();
      for (const auto& code : alarm_codes) {
        const auto& alarm_code = code.second;
        if (alarm_code.second.empty()) {
          continue;
        }
        auto level = alarm->GetLevel(code.first);
        const auto& desc = alarm->GetAlarmName(code.first);
        if (level == AlarmLevel::kCritical) {
          color = Colors::kError;
          impl_->total_critical++;
        } else if (level == AlarmLevel::kWarning) {
          color = Colors::kWarning;
          impl_->total_warnings++;
        } else {
          color = Colors::kNormal;
        }
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", alarm->GetName().c_str());
        ImGui::TableSetColumnIndex(1);
        std::time_t tp = AlarmsPageImpl::IAlarm::clock::to_time_t(alarm_code.first);
        std::strftime(&tp_str[0], tp_str.size(), "%H:%M:%S", std::localtime(&tp));
        ImGui::TextColored(color, "%s", tp_str.c_str());
        ImGui::TableSetColumnIndex(2);
        ImGui::TextColored(color, "%s", desc.c_str());
        ImGui::TableSetColumnIndex(3);
        ImGui::TextColored(color, "%s", alarm_code.second.c_str());
      }
    }
    ImGui::EndTable();
  }

  ImGui::Checkbox("Show Log", &impl_->show_log);
  if (impl_->show_log) {
    ImGui::PushItemWidth(-1);
    if (ImGui::BeginListBox("", ImVec2(0, 300))) {
      for (const auto& log : impl_->alarms_log) {
        if (log.level == AlarmLevel::kCritical) {
          color = Colors::kError;
        } else if (log.level == AlarmLevel::kWarning) {
          color = Colors::kWarning;
        } else {
          color = Colors::kNormal;
        }
        ImGui::TextColored(color, "%s", log.log_str.c_str());
      }
      ImGui::EndListBox();
    }
    ImGui::PopItemWidth();
  }
}

void AlarmsPage::Update(float dt) {
  if (!is_open_) {
    return;
  }
  impl_->Update(dt);
}
void AlarmsPage::OpenPage() {
  IMonitorPage::OpenPage();
  impl_->Reset();
  for (const auto& alarm : impl_->alarms) {
    alarm->Start();
  }
}
void AlarmsPage::ClosePage() {
  IMonitorPage::ClosePage();
  for (const auto& alarm : impl_->alarms) {
    alarm->Stop();
  }
}
PageStatus AlarmsPage::GetStatus() const { return impl_->curr_status; }
void AlarmsPage::SetAlarm(const std::string& name, const std::string& desc, AlarmsPage::AlarmLevel level, int timeout) {
  impl_->generic_alarm->SetAlarmDetailed(name, desc, level, timeout);
}

}  // namespace ace_monitor
