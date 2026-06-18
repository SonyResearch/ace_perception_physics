// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/PerceptionDelayPage.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

#include "ace_interfaces/msg/ball_state.hpp"
#include "ace_interfaces/msg/balls_with_attributes.hpp"
#include "ace_interfaces/msg/logger_control.hpp"
#include "ace_interfaces/msg/perception_header.hpp"
#include "ace_interfaces/msg/player_pose.hpp"
#include "ace_interfaces/msg/points_with_covariance.hpp"
#include "ace_interfaces/msg/poses_with_covariance.hpp"
#include "ace_interfaces/msg/racket_pose_estimate.hpp"
#include "ace_interfaces/msg/spins_with_covariance.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "datalogger/datalogger.hpp"
#include "implot.h"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"

namespace ace_monitor {

class PerceptionDelayPage::PerceptionDelayPageImpl {
 public:
  using clock = std::chrono::high_resolution_clock;
  enum class AvailableMessages {
    kEtherCat = 0,
    kBallTriangulation,
    kBallPoses,
    kGCSSpin,
    kPlayer,
    kRacket,
    kPreshot,
    kBallDetection,
    kCount = 100
  };
  static const int kSamplingFreq = 200;
  static constexpr int kHistorySize = kSamplingFreq * 10;

  std::map<int, std::string> message_type_str;
  bool record_delays{true};
  std::string delays_file_path;

  std::ofstream delays_csv_file;
  int latest_message_index{static_cast<int>(AvailableMessages::kBallDetection)};
  rclzmq::Subscription<ace_interfaces::msg::LoggerControl>::SharedPtr loggercontrol_sub;

  PerceptionDelayPageImpl() {
    message_type_str[static_cast<int>(AvailableMessages::kEtherCat)] = "Reference";
    message_type_str[static_cast<int>(AvailableMessages::kBallTriangulation)] = "BallTriangulation";
    message_type_str[static_cast<int>(AvailableMessages::kBallPoses)] = "BallPoses";
    message_type_str[static_cast<int>(AvailableMessages::kGCSSpin)] = "GCSSpin";
    message_type_str[static_cast<int>(AvailableMessages::kPlayer)] = "Player";
    message_type_str[static_cast<int>(AvailableMessages::kRacket)] = "Racket";
    message_type_str[static_cast<int>(AvailableMessages::kPreshot)] = "Preshot";
    // message_type_str[static_cast<int>(AvailableMessages::kBallDetection)] = "BallDetection";
  }

  void RecordDelaysToCSV(bool record) {
    record_delays = record;
    if (!record) {
      if (delays_csv_file.is_open()) {
        delays_csv_file.close();
      }
      info->on_message_arrived = nullptr;
      loggercontrol_sub = nullptr;
      return;
    }

    rclzmq::QoS qos_persistent(rclzmq::KeepLast(1));
    qos_persistent.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_persistent.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    // Set up LoggerControl callback
    loggercontrol_sub =
      ACEMonitor::GetInstance().GetRosNode()->Node()->create_subscription<ace_interfaces::msg::LoggerControl>(
        "/logger/control", qos_persistent, [&](const ace_interfaces::msg::LoggerControl::ConstSharedPtr& msg_ptr) {
          const auto log_prefix = std::string(reinterpret_cast<const char*>(msg_ptr->log_prefix.data()));
          if (delays_csv_file.is_open()) {
            delays_csv_file.close();
          }
          if (log_prefix.empty()) {
            LOG(INFO) << "Delay logging stopped" << std::endl;
            return;
          }
          delays_file_path = ::datalogger::ConstructSubLogName(log_prefix + ".csv", "perception_delays");
          delays_csv_file = std::ofstream(delays_file_path);
          delays_csv_file << "SequenceID";
          // Skip the reference message since its delay is always 0
          for (int i = 1; i < latest_message_index; ++i) {
            info->last_delays[i] = 0;
            if (message_type_str.find(i) == message_type_str.end()) {
              continue;
            }
            delays_csv_file << "\t" << message_type_str[i];
          }
          delays_csv_file << "\n";
          delays_csv_file.flush();

          info->on_message_arrived = [&](uint64_t seq_id, int meas_type) {
            if (meas_type != static_cast<int>(AvailableMessages::kEtherCat)) {
              return;
            }
            delays_csv_file << seq_id;
            // Skip the reference message since its delay is always 0
            for (int i = 1; i < latest_message_index; ++i) {
              if (message_type_str.find(i) == message_type_str.end()) {
                continue;
              }
              delays_csv_file << "\t" << std::setprecision(2) << std::fixed << info->last_delays[i];
            }
            delays_csv_file << "\n";
            delays_csv_file.flush();
          };
          LOG(INFO) << "Delay log filename got externally updated to:\n - \"" << delays_file_path << "\"" << std::endl;
        });
  }
  class DelayInfo {
   public:
    DelayInfo() = default;
    virtual ~DelayInfo() = default;
    std::map<int, float> last_delays;

    std::function<void(uint64_t, int)> on_message_arrived;

    void OnMessageUpdated(int index, int meas_type) {
      if (on_message_arrived) {
        on_message_arrived(local_times[index].seq_id, meas_type);
      }
      if (meas_type == static_cast<int>(AvailableMessages::kEtherCat)) {
        ++delay_counter;
        auto curr = local_times[index].seq_id;
        if (curr < last_seq_id) {
          Reset();
        }
        last_seq_id = curr;
        return;
      }
      std::lock_guard lock(update_mutex);
      auto& prev_time = local_times[index];
      auto min_index = static_cast<double>(static_cast<int64_t>(prev_time.seq_id) - kHistorySize);
      if (prev_time.IsValidReference() && prev_time.messages[meas_type].IsValid()) {
        auto dt = prev_time.Duration(static_cast<AvailableMessages>(meas_type));
        last_delays[meas_type] = static_cast<float>(dt);
        delay_list[meas_type].delay_sequence_id.push_back(static_cast<double>(prev_time.seq_id));
        delay_list[meas_type].history.push_back(std::max<double>(dt, 0.0F));

        while (delay_list[meas_type].delay_sequence_id.front() < min_index) {
          delay_list[meas_type].delay_sequence_id.erase(delay_list[meas_type].delay_sequence_id.begin());
          delay_list[meas_type].history.erase(delay_list[meas_type].history.begin());
        }
      }
    }

    void Update() {
      std::lock_guard lock(update_mutex);
      // calculate mean per message
      for (size_t i = 0; i < delay_list.size(); ++i) {
        if (delay_list[i].samples_counter >= kSamplingFreq - 10) {
          CalculateMeans(i);
          delay_list[i].samples_counter = 0;
        }
      }

      if (delay_counter >= kSamplingFreq - 10) {
        CalculateDelays();
        delay_counter = 0;
      }
    }
    void Reset() {
      std::lock_guard lock(update_mutex);
      last_seq_id = 0;
      delay_counter = 0;
      for (auto& delay : delay_list) {
        delay.Reset();
      }

      for (auto& times : local_times) {
        times.Reset();
      }
    }

    struct PairedTimes {
      struct MessagePoint {
        clock::time_point arrival_time;
        double message_time{-1};
        bool delay_calculated{false};
        [[nodiscard]] bool IsValid() const { return message_time != -1; }
        void Invalidate() {
          message_time = -1;
          delay_calculated = false;
        }
        [[nodiscard]] double Duration(const clock::time_point& reference) const {
          return static_cast<double>(
                   std::chrono::duration_cast<std::chrono::nanoseconds>(arrival_time - reference).count()) *
                 1e-6;
        }
        [[nodiscard]] double DeltaTime(const MessagePoint& prev, bool normalize) const {
          auto dt = static_cast<double>(
                      std::chrono::duration_cast<std::chrono::nanoseconds>(arrival_time - prev.arrival_time).count()) *
                    1e-6F;

          if (normalize) {
            dt *= 5e-3F / (message_time - prev.message_time);
          }
          return dt;
        }
      };
      uint64_t seq_id;
      std::array<MessagePoint, static_cast<int>(AvailableMessages::kCount)> messages;
      void Reset() {
        seq_id = -1;
        for (auto& m : messages) {
          m.Invalidate();
        }
      }
      [[nodiscard]] bool IsValidReference() const {
        return messages[static_cast<int>(AvailableMessages::kEtherCat)].IsValid();
      }

      [[nodiscard]] double Duration(AvailableMessages type) const {
        if (!messages[static_cast<int>(type)].IsValid() || !IsValidReference()) {
          return -1;
        }
        return messages[static_cast<int>(type)].Duration(
          messages[static_cast<int>(AvailableMessages::kEtherCat)].arrival_time);
      }

      [[nodiscard]] MessagePoint& GetMessage(AvailableMessages type) { return messages[static_cast<int>(type)]; }
      [[nodiscard]] MessagePoint& GetReference() { return messages[static_cast<int>(AvailableMessages::kEtherCat)]; }
    };

    void CalculateDelays() {
      for (auto& delay : delay_list) {
        delay.last_delay_mean = delay.delay_mean;
        delay.delay_mean = 0;
        delay.delay_stdev = 0;
        std::vector<double> valid_delay;
        for (auto& d : delay.history) {
          if (d > 0) {
            valid_delay.push_back(d);
          }
          delay.delay_mean += d;
        }
        if (!valid_delay.empty()) {
          delay.delay_mean /= static_cast<float>(valid_delay.size());
          for (auto& v : valid_delay) {
            delay.delay_stdev += std::pow(v - delay.delay_mean, 2);
          }
          delay.delay_stdev = std::sqrt(delay.delay_stdev / static_cast<double>(valid_delay.size()));
        }
      }
    }

    void CalculateMeans(size_t index) {
      PairedTimes prev_valid_time;
      std::vector<double> messages_dt;

      delay_list[index].last_dt_mean = delay_list[index].dt_mean;
      delay_list[index].dt_mean = 0;
      delay_list[index].dt_stdev = 0;
      for (auto& times : local_times) {
        if (times.messages[index].IsValid() &&
            times.messages[index].message_time > prev_valid_time.messages[index].message_time) {
          if (prev_valid_time.messages[index].IsValid()) {
            auto dt = times.messages[index].DeltaTime(prev_valid_time.messages[index], normalized_means);
            delay_list[index].dt_mean += dt;
            messages_dt.push_back(dt);
          }

          prev_valid_time.messages[index].message_time = times.messages[index].message_time;
          prev_valid_time.messages[index].arrival_time = times.messages[index].arrival_time;
        }
        times.messages[index].Invalidate();
      }

      // calculate stdev per message
      if (!messages_dt.empty()) {
        delay_list[index].dt_mean /= static_cast<double>(messages_dt.size());

        for (auto& v : messages_dt) {
          delay_list[index].dt_stdev += std::pow(v - delay_list[index].dt_mean, 2);
        }
        delay_list[index].dt_stdev = std::sqrt(delay_list[index].dt_stdev / static_cast<double>(messages_dt.size()));
      }
    }
    std::mutex update_mutex;
    PairedTimes local_times[kHistorySize];
    int delay_counter{0};
    uint64_t last_seq_id{0};

    class DelayHistory {
     public:
      std::vector<double> history;
      std::vector<double> delay_sequence_id;
      double dt_mean{0};
      double dt_stdev{0};
      double delay_mean{0};
      double delay_stdev{0};

      double last_delay_mean{0};
      double last_dt_mean{0};

      int samples_counter{0};

      void Reset() {
        history.clear();
        delay_sequence_id.clear();
        delay_mean = 0;
        delay_stdev = 0;
        dt_mean = 0;
        dt_stdev = 0;
        samples_counter = 0;
        last_delay_mean = 0;
        last_dt_mean = 0;
      }
    };

    std::array<DelayHistory, static_cast<int>(AvailableMessages::kCount)> delay_list;

    bool normalized_means{false};
    bool show_delays_graph{true};
  };
  class IMessageSubscriber {
   public:
    virtual void Subscribe(const std::string& topic, std::shared_ptr<DelayInfo> ifo) = 0;
  };
  template <class T>
  class MessageSubscriber : public IMessageSubscriber {
    typename rclcpp::Subscription<T>::SharedPtr sub_;
    std::shared_ptr<DelayInfo> info_;
    std::function<ace_interfaces::msg::PerceptionHeader(typename T::SharedPtr)> header_func_;
    std::string topic_name_;
    int type_index_;

   public:
    MessageSubscriber(int index, std::function<ace_interfaces::msg::PerceptionHeader(typename T::SharedPtr)> f)
      : header_func_(std::move(f)), type_index_(index) {}
    void OnMessage(typename T::SharedPtr msg) {
      auto header = header_func_(msg);
      int seq_id = header.sequence_number;
      {
        if (seq_id % 5 == 0) {
          double timestamp = header.stamp.sec + header.stamp.nanosec * 1e-9;
          int time_index = static_cast<int>(std::round(seq_id / 5)) % kHistorySize;
          {
            std::lock_guard lock(info_->update_mutex);
            if (type_index_ == static_cast<int>(AvailableMessages::kEtherCat)) {
              info_->local_times[time_index].seq_id = header.sequence_number;
            }
            auto& time = info_->local_times[time_index].messages[type_index_];
            time.arrival_time = clock::now();
            time.message_time = timestamp;
            ++info_->delay_list[type_index_].samples_counter;
          }

          info_->OnMessageUpdated(time_index, type_index_);
        }
      }
    }

    void Subscribe(const std::string& topic, std::shared_ptr<DelayInfo> ifo) override {
      info_ = ifo;
      auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
      topic_name_ = topic;
      sub_ = node->create_subscription<T>(topic, 1, [&](typename T::SharedPtr msg) { OnMessage(msg); });
    }
  };
  std::shared_ptr<DelayInfo> info;
  std::array<std::pair<std::string, std::string>, static_cast<int>(AvailableMessages::kCount)> topics_names = {
    std::pair<std::string, std::string>("Reference", "/_ethercat_clock"),
    std::pair<std::string, std::string>("Ball Triangulation", "/sensors/ball_triangulation/points"),
    std::pair<std::string, std::string>("Ball Poses", "/sensors/ball_pose_estimation/poses"),
    std::pair<std::string, std::string>("GCS Spin", "/sensors/gcs/spin_estimation_filtered/spins"),
    std::pair<std::string, std::string>("Player Pose", "/sensors/player0/pose"),
    std::pair<std::string, std::string>("Racket Pose", "/sensors/racket0/pose"),
    std::pair<std::string, std::string>("Preshot", "/estimator/preshot"),
  };
  std::array<std::shared_ptr<IMessageSubscriber>, static_cast<int>(AvailableMessages::kCount)> subscribers;

  PageStatus curr_status = PageStatus::kNormal;

  void Startup() {
    info = std::make_shared<DelayInfo>();
    DiscoverTopics();
    RecordDelaysToCSV(false);
    for (size_t i = 0; i < subscribers.size(); ++i) {
      if (!topics_names[i].second.empty()) {
        subscribers[i]->Subscribe(topics_names[i].second, info);
      }
    }
  }

  void Reset() {
    subscribers[static_cast<int>(AvailableMessages::kEtherCat)] =
      std::make_shared<MessageSubscriber<ace_interfaces::msg::PerceptionHeader>>(
        static_cast<int>(AvailableMessages::kEtherCat),
        [](ace_interfaces::msg::PerceptionHeader::SharedPtr msg) { return *msg; });
    subscribers[static_cast<int>(AvailableMessages::kBallTriangulation)] =
      std::make_shared<MessageSubscriber<ace_interfaces::msg::PointsWithCovariance>>(
        static_cast<int>(AvailableMessages::kBallTriangulation),
        [](ace_interfaces::msg::PointsWithCovariance::SharedPtr msg) { return msg->header; });
    subscribers[static_cast<int>(AvailableMessages::kBallPoses)] =
      std::make_shared<MessageSubscriber<ace_interfaces::msg::PosesWithCovariance>>(
        static_cast<int>(AvailableMessages::kBallPoses),
        [](ace_interfaces::msg::PosesWithCovariance::SharedPtr msg) { return msg->header; });
    subscribers[static_cast<int>(AvailableMessages::kGCSSpin)] =
      std::make_shared<MessageSubscriber<ace_interfaces::msg::SpinsWithCovariance>>(
        static_cast<int>(AvailableMessages::kGCSSpin),
        [](ace_interfaces::msg::SpinsWithCovariance::SharedPtr msg) { return msg->header; });
    subscribers[static_cast<int>(AvailableMessages::kPlayer)] =
      std::make_shared<MessageSubscriber<ace_interfaces::msg::PlayerPose>>(
        static_cast<int>(AvailableMessages::kPlayer),
        [](ace_interfaces::msg::PlayerPose::SharedPtr msg) { return msg->header; });
    subscribers[static_cast<int>(AvailableMessages::kRacket)] =
      std::make_shared<MessageSubscriber<ace_interfaces::msg::RacketPoseEstimate>>(
        static_cast<int>(AvailableMessages::kRacket),
        [](ace_interfaces::msg::RacketPoseEstimate::SharedPtr msg) { return msg->pose.header; });
    subscribers[static_cast<int>(AvailableMessages::kPreshot)] =
      std::make_shared<MessageSubscriber<ace_interfaces::msg::BallState>>(
        static_cast<int>(AvailableMessages::kPreshot),
        [](ace_interfaces::msg::BallState::SharedPtr msg) { return msg->header; });

    for (size_t i = static_cast<int>(AvailableMessages::kBallDetection); i < subscribers.size(); ++i) {
      subscribers[i] = nullptr;
    }
  }

  void DiscoverTopics() {
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    auto topics_ros = node->get_topic_names_and_types();

    int index = 0;

    for (const auto& topic_ros : topics_ros) {
      const auto& topic = topic_ros.first;
      if (topic.find("gcs") == std::string::npos && topic_ros.second[0] == "ace_interfaces/msg/BallsWithAttributes") {
        int msg_index = static_cast<int>(AvailableMessages::kBallDetection) + index;
        std::stringstream topic_stream(topic);
        std::string cam_name;
        std::getline(topic_stream, cam_name, '/');
        std::getline(topic_stream, cam_name, '/');
        std::getline(topic_stream, cam_name, '/');
        topics_names[msg_index] = std::pair<std::string, std::string>(cam_name, topic);
        subscribers[msg_index] = std::make_shared<MessageSubscriber<ace_interfaces::msg::BallsWithAttributes>>(
          msg_index, [](ace_interfaces::msg::BallsWithAttributes::SharedPtr msg) { return msg->header; });
        message_type_str[msg_index] = cam_name;
        ++index;
      }
    }
    latest_message_index = static_cast<int>(AvailableMessages::kBallDetection) + index;
  }
};

PerceptionDelayPage::PerceptionDelayPage() : IMonitorPage("Perception Delay", "Perception") {
  impl_ = std::make_unique<PerceptionDelayPageImpl>();
  impl_->Reset();
}
PerceptionDelayPage::~PerceptionDelayPage() = default;
void PerceptionDelayPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  auto info = impl_->info;
  ImVec4 color;
  if (ImGui::Button("Reset")) {
    info->Reset();
  }
  ImGui::SameLine();
  if (ImGui::Button(impl_->record_delays ? "Stop Recording" : "Record to CSV")) {
    impl_->RecordDelaysToCSV(!impl_->record_delays);
  }
  if (impl_->record_delays && impl_->delays_csv_file.is_open()) {
    ImGui::SameLine();
    ImGui::Text("Recording started");
  }
  ImGui::Checkbox("Normalize Means", &info->normalized_means);
  ImGui::SameLine();
  ImGui::Checkbox("Show Graph", &info->show_delays_graph);
  static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
  if (ImGui::BeginTable("means", 3, flags)) {
    ImGui::TableSetupColumn("Name");
    ImGui::TableSetupColumn("Delay");
    ImGui::TableSetupColumn(info->normalized_means ? "Mean (Normalized)" : "Mean");
    ImGui::TableHeadersRow();
    for (int i = 0; i < static_cast<int>(PerceptionDelayPageImpl::AvailableMessages::kCount); ++i) {
      if (impl_->topics_names[i].first.empty()) {
        continue;
      }
      auto& delays = info->delay_list[i];
      if (delays.dt_mean < delays.last_dt_mean * 0.8 || delays.dt_stdev > 1) {
        color = Colors::kError;
      } else if (delays.delay_mean < delays.last_delay_mean * 0.8 || delays.delay_stdev > 1) {
        color = Colors::kWarning;
      } else {
        color = Colors::kNormal;
      }
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextColored(color, "%s", impl_->topics_names[i].first.c_str());
      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(color, "%0.2f±%0.2fms", delays.delay_mean, delays.delay_stdev);
      ImGui::TableSetColumnIndex(2);
      ImGui::TextColored(color, "%0.2f±%0.2fms", delays.dt_mean, delays.dt_stdev);
    }
    ImGui::EndTable();
  }

  if (info->show_delays_graph) {
    bool first_plot = true;
    if (ImPlot::BeginPlot("Delays", ImVec2(400, 300), ImPlotFlags_::ImPlotFlags_NoBoxSelect)) {
      ImPlot::SetupAxes("Seq ID", "Delay (ms)",
                        ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_::ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_::ImPlotAxisFlags_AutoFit);
      for (int i = 1; i < static_cast<int>(PerceptionDelayPageImpl::AvailableMessages::kCount); ++i) {
        if (impl_->topics_names[i].first.empty()) {
          continue;
        }
        auto& delays = info->delay_list[i];
        ImPlot::PlotLine(impl_->topics_names[i].first.c_str(), delays.delay_sequence_id.data(), delays.history.data(),
                         static_cast<int>(delays.history.size()));
        if (first_plot && !delays.delay_sequence_id.empty()) {
          first_plot = false;
          double x = delays.delay_sequence_id[0];
          double y = -1;
          ImPlot::PlotLine("", &x, &y, 1);
        }
      }
      ImPlot::EndPlot();
    }
  }
}

void PerceptionDelayPage::Update(float /*dt*/) {
  if (!is_open_ || !impl_->info) {
    return;
  }
  impl_->info->Update();
  impl_->curr_status = PageStatus::kNormal;
}

void PerceptionDelayPage::OpenPage() {
  IMonitorPage::OpenPage();
  impl_->Startup();
}
void PerceptionDelayPage::ClosePage() {
  IMonitorPage::ClosePage();
  impl_->info = nullptr;
}
PageStatus PerceptionDelayPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
