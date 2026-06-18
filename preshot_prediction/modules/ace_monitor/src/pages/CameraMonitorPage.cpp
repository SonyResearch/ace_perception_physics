// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ace_monitor/pages/perception/CameraMonitorPage.hpp"

#include <mutex>

#include "ace_interfaces/msg/image_data.hpp"
#include "ace_interfaces/msg/statistics.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/utils/FPSCalculator.hpp"
#include "ace_monitor/utils/TextureHelpers.hpp"

namespace ace_monitor {

class CameraMonitorPage::CameraMonitorPageImpl {
 public:
  class CameraInfo {
   public:
    explicit CameraInfo(const std::string& topic) : topic_name(topic) {
      camera_name = topic;
      std::string begin("/sensors/");
      auto idx = camera_name.find(begin);
      if (idx != std::string::npos) {
        camera_name = camera_name.substr(begin.size());
      }
      idx = camera_name.find("/image");
      if (idx != std::string::npos) {
        camera_name = camera_name.substr(0, idx);
      }
    }
    ~CameraInfo() = default;
    void OnMessage(ace_interfaces::msg::Statistics::SharedPtr msg) {
      last_fps = curr_fps;
      curr_fps = msg->framerate;
      last_msg_time = Clock::now();
    }

    void Update() {
      auto now = Clock::now();
      auto dt =
        static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_msg_time).count()) / 1e3;
      if (dt > 1) {
        // no new messages for over 1 second
        last_fps = 0;
      }
    }

    using Clock = std::chrono::high_resolution_clock;
    int last_fps{0};
    int curr_fps{0};
    Clock::time_point last_msg_time;

    std::string topic_name;
    std::string camera_name;

    PageStatus status{PageStatus::kNormal};
  };
  std::unordered_map<std::string, std::shared_ptr<CameraInfo>> camera_info;
  std::vector<rclzmq::Subscription<ace_interfaces::msg::Statistics>::SharedPtr> subscribers;
  PageStatus curr_status = PageStatus::kNormal;
  std::mutex image_mutex;
  ace_interfaces::msg::ImageData::SharedPtr last_image_msg;
  GLuint texture_id{0};
  int width{0};
  int height{0};
  rclzmq::Subscription<ace_interfaces::msg::ImageData>::SharedPtr image_subscriber;
  std::string active_camera;

  void OnCameraMessage(ace_interfaces::msg::ImageData::SharedPtr msg) {
    {
      std::scoped_lock<std::mutex> lock(image_mutex);
      last_image_msg = std::make_shared<ace_interfaces::msg::ImageData>(*msg);
      width = msg->width;
      height = msg->height;
    }
    Deactivate(true);
  }

  void Activate(const std::string& camera) {
    if (active_camera == camera) {
      return;
    }
    if (camera.empty()) {
      Deactivate(false);
      return;
    }
    active_camera = camera;

    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    auto topics_ros = node->get_topic_names_and_types();
    auto topic = "/sensors/" + camera + "/image";
    auto cb = [&](ace_interfaces::msg::ImageData::SharedPtr msg) { OnCameraMessage(msg); };
    LOG(INFO) << "Subscribing to camera: " << topic;
    image_subscriber = node->create_subscription<ace_interfaces::msg::ImageData>(topic, 1, cb);
  }
  void Deactivate(bool unsub) {
    if (active_camera.empty()) {
      return;
    }
    image_subscriber.reset();
    LOG(INFO) << "Unsubscribing from camera: " << active_camera;
    if (!unsub) {
      active_camera = "";
    }
  }

  void Reset() {
    Deactivate(false);
    {
      std::scoped_lock<std::mutex> lock(image_mutex);
      if (texture_id != 0) {
        TextureHelpers::DestroyTexture(texture_id);
        texture_id = 0;
      }
    }
    for (auto& sub : subscribers) {
      sub.reset();
    }
    camera_info.clear();
    subscribers.clear();
  }
  void Update() {
    DiscoverTopics();
    int status = 0;
    for (auto& sub : camera_info) {
      sub.second->Update();
      auto last_fps = sub.second->last_fps;
      auto curr_fps = sub.second->curr_fps;
      if (curr_fps == 0) {
        sub.second->status = PageStatus::kError;
        status |= 0x1;
      } else if (curr_fps < last_fps * 0.75) {
        sub.second->status = PageStatus::kWarning;
        status |= 0x2;
      } else {
        sub.second->status = PageStatus::kNormal;
      }
    }
    if (status == 0) {
      curr_status = PageStatus::kNormal;
    } else if (status & 0x1) {
      curr_status = PageStatus::kError;
    } else if (status & 0x2) {
      curr_status = PageStatus::kWarning;
    }
    {
      std::scoped_lock<std::mutex> lock(image_mutex);
      if (last_image_msg != nullptr) {
        if (texture_id == 0) {
          texture_id = TextureHelpers::CreateTexture(last_image_msg->width, last_image_msg->height);
        }
        auto img = TextureHelpers::ConvertBayer8ToRGB(last_image_msg->data.data(), last_image_msg->width,
                                                      last_image_msg->height, last_image_msg->step);
        TextureHelpers::LoadTexture(img, texture_id);
        last_image_msg = nullptr;
      }
    }
  }
  void DiscoverTopics() {
    auto node = ACEMonitor::GetInstance().GetRosNode()->Node();
    auto topics_ros = node->get_topic_names_and_types();

    for (const auto& topic_ros : topics_ros) {
      const auto& topic = topic_ros.first;
      if (camera_info.find(topic) == camera_info.end() && topic_ros.second[0] == "ace_interfaces/msg/Statistics") {
        auto ifo = std::make_shared<CameraInfo>(topic);
        // Create subscriber and register callback
        auto cb = [&, ifo](ace_interfaces::msg::Statistics::SharedPtr msg) { ifo->OnMessage(msg); };
        LOG(INFO) << "adding subscriber: " << topic;
        subscribers.emplace_back(node->create_subscription<ace_interfaces::msg::Statistics>(topic, 1, cb));

        camera_info[topic] = ifo;
      }
    }
  }
};

CameraMonitorPage::CameraMonitorPage() : IMonitorPage("Cameras", "Perception") {
  impl_ = std::make_unique<CameraMonitorPageImpl>();
}
CameraMonitorPage::~CameraMonitorPage() = default;
void CameraMonitorPage::RenderWidgets() {
  IMonitorPage::RenderWidgets();
  std::shared_ptr<CameraMonitorPageImpl::CameraInfo> active_camera;
  for (auto& sub : impl_->camera_info) {
    ImVec4 color;
    if (sub.second->status == PageStatus::kError) {
      color = Colors::kError;
    } else if (sub.second->status == PageStatus::kWarning) {
      color = Colors::kWarning;
    } else {
      color = Colors::kNormal;
    }
    auto curr_fps = sub.second->curr_fps;
    std::stringstream text;
    text << sub.second->camera_name << "\n" << static_cast<int>(curr_fps) << "FPS";

    ImGui::PushStyleColor(ImGuiCol_Button, color);
    ImGui::Button(text.str().c_str(), ImVec2(100, 50));
    if (ImGui::IsItemHovered()) {
      float scaler = 1.0 / 2.0;
      active_camera = sub.second;
      impl_->Activate(active_camera->camera_name);
      // NOLINTNEXTLINE
      auto* texture_id = reinterpret_cast<ImTextureID>(impl_->texture_id);
      if (ImGui::BeginItemTooltip()) {
        ImGui::Image(texture_id,
                     ImVec2(static_cast<float>(impl_->width) * scaler, static_cast<float>(impl_->height) * scaler));
        ImGui::EndTooltip();
      }
    }
    ImGui::PopStyleColor();
  }
  if (!active_camera) {
    impl_->Deactivate(false);
  }
}

void CameraMonitorPage::Update(float /*dt*/) {
  if (!is_open_) {
    return;
  }
  impl_->Update();
}

void CameraMonitorPage::OpenPage() { IMonitorPage::OpenPage(); }
void CameraMonitorPage::ClosePage() {
  IMonitorPage::ClosePage();
  impl_->Reset();
}
PageStatus CameraMonitorPage::GetStatus() const { return impl_->curr_status; }
}  // namespace ace_monitor
