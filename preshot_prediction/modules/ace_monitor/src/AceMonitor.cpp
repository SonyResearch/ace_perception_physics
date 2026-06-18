// Confidential, Copyright 2025, Sony AI, All rights reserved.

#include "ace_monitor/AceMonitor.hpp"

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitorRosNode.hpp"
#include "ace_monitor/pages/perception/AlarmsPage.hpp"
#include "ace_monitor/pages/perception/PerceptionVisualizerPage.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "std_msgs/msg/string.hpp"
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GL/glew.h>
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers
#include <implot.h>
#include <stb_image.h>

#include <iostream>
#include <rclzmq/rclzmq.hpp>

namespace ace_monitor {

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

class ACEMonitor::ACEMonitorImpl {
 public:
  class Category {
   public:
    std::string name;
    std::vector<IMonitorPage::SharedPtr> pages;
    Category() = default;
    explicit Category(std::string n) : name(std::move(n)) {}
  };
  class HostConfigurations {
   public:
    std::string name;
    std::string hostname;
    std::string ipaddress;
    bool reachable{false};
    bool pingable{false};
  };
  std::map<std::string, Category> categories;

  GLFWwindow* window{nullptr};
  ImGuiIO* io_ptr{nullptr};
  ImVec4 clear_color = ImVec4(0.45F, 0.55F, 0.60F, 1.00F);

  std::shared_ptr<AceMonitorRosNode> ros_node;

  std::shared_ptr<ace_monitor::PerceptionVisualizerPage> visualizer;
  std::shared_ptr<ace_monitor::AlarmsPage> alarms;

  using ValueMap = std::map<std::string, void*>;
  using CategoryValueMap = std::map<std::string, ValueMap>;
  CategoryValueMap values;

  std::string shared_path;
  std::string data_path;

  YAML::Node configurations;
  std::vector<HostConfigurations> hosts_to_ping;

  std::string ace_lab;

  std::map<std::string, std::string> app_args;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr execution_mode_sub;
  std::string execution_mode;

  std::thread ping_thread;

  void PingThread() {
    if (ace_lab.empty()) {
      return;
    }
    while (window) {
      for (auto& config : hosts_to_ping) {
        config.reachable = system(("ping -w1 -W1 -c1 -s1 " + config.hostname + "  > /dev/null 2>&1").c_str()) == 0;
        config.pingable = system(("ping -w1 -W1 -c1 -s1 " + config.ipaddress + "  > /dev/null 2>&1").c_str()) == 0;
      }
      sleep(10);  // ping every 10seconds
    }
  }

  void Initialize(int argc, char** argv) {
    if (window) {
      return;
    }
    for (int i = 0; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg.rfind("--") != std::string::npos) {
        auto idx = arg.find('=');
        if (idx == std::string::npos) {
          continue;
        }
        auto key = arg.substr(0, idx);
        app_args[key] = arg.substr(idx + 1);
        LOG(INFO) << key << ": " << app_args[key];
      }
    }
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize glfw!");
    }
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    // Create window with graphics context
    window = glfwCreateWindow(1280, 720, "ACE Monitor", nullptr, nullptr);
    if (window == nullptr) {
      throw std::runtime_error("Failed to create window!");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io_ptr = &io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    glfwSwapInterval(1);
    bool err = glewInit() != GLEW_OK;
    if (err) {
      throw std::runtime_error("Failed to initialize OpenGL loader!");
    }

    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    // stbi_set_flip_vertically_on_load(true);

    shared_path = ament_index_cpp::get_package_share_directory("ace_monitor");
    data_path = shared_path + "/data/";
    LOG(INFO) << "Shared Path: " << shared_path;

    configurations = ace_yaml::LoadFile(data_path + "config/configurations.yaml");
    {
      const std::string ip_addr_path = data_path + "config/ip_addresses.yaml";
      ::YAML::Node ip_addresses;
      try {
        ip_addresses = ace_yaml::LoadFile(ip_addr_path);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Could not load " << ip_addr_path << ": " << e.what();
      }
      const char* ace_lab_env = std::getenv("ACE_LAB");
      if (ace_lab_env == nullptr) {
        LOG(INFO) << "ACE_LAB env not set; skipping host monitoring.";
      } else if (!ip_addresses || !ip_addresses[ace_lab_env]) {
        LOG(WARNING) << "Couldn't find <" << ace_lab_env << "> in ip_addresses.yaml.";
      } else {
        ace_lab = ace_lab_env;
        for (auto it = ip_addresses[ace_lab].begin(); it != ip_addresses[ace_lab].end(); ++it) {
          HostConfigurations config;
          config.name = it->first.as<std::string>();
          config.hostname = it->second["hostname"].as<std::string>();
          config.ipaddress = it->second["ipaddress"].as<std::string>();
          hosts_to_ping.push_back(config);
        }
        ping_thread = std::thread([this]() { this->PingThread(); });
      }
    }

    ros_node = std::make_shared<AceMonitorRosNode>();
    ros_node->Initialize();

    rclzmq::QoS qos_persistent(rclzmq::KeepLast(1));
    qos_persistent.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_persistent.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

    execution_mode_sub = ros_node->Node()->create_subscription<std_msgs::msg::String>(
      "/execution_mode", qos_persistent,
      [&](std_msgs::msg::String::SharedPtr message) { execution_mode = message->data; });
  }
};

ACEMonitor& ACEMonitor::GetInstance() {
  static ACEMonitor instance;
  return instance;
}

ACEMonitor::ACEMonitor() { impl_ = std::make_unique<ACEMonitorImpl>(); }
ACEMonitor::~ACEMonitor() { Shutdown(); }

void ACEMonitor::Initialize(int argc, char** argv) {
  impl_->Initialize(argc, argv);

  // visualizer
  impl_->visualizer = std::make_shared<ace_monitor::PerceptionVisualizerPage>();
  impl_->alarms = std::make_shared<ace_monitor::AlarmsPage>();
  AddPage(impl_->visualizer);
  AddPage(impl_->alarms);
}
const std::string& ACEMonitor::GetDataPath() const { return impl_->data_path; }
void ACEMonitor::Shutdown() {
  if (!impl_->window) {
    return;
  }
  impl_->alarms = nullptr;
  impl_->visualizer = nullptr;

  impl_->categories.clear();

  impl_->ros_node = nullptr;

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(impl_->window);
  glfwTerminate();
  impl_->window = nullptr;
  impl_->io_ptr = nullptr;
  LOG(INFO) << "ACE Monitor Closed";
}

std::string_view ACEMonitor::GetAppArg(const std::string& name, std::string_view default_value) const {
  auto it = impl_->app_args.find(name);
  if (it != impl_->app_args.end()) {
    return it->second;
  }
  return default_value;
}

void ACEMonitor::AddPage(std::shared_ptr<IMonitorPage> page) {
  if (impl_->categories.find(page->GetCategoryName()) == impl_->categories.end()) {
    impl_->categories[page->GetCategoryName()] = ACEMonitorImpl::Category(page->GetCategoryName());
  }
  impl_->categories[page->GetCategoryName()].pages.push_back(page);
  page->Initialize(impl_->visualizer.get());
}

bool ACEMonitor::Draw() const {
  if (IsDone()) {
    return false;
  }
  glfwPollEvents();

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  auto dt = 1.0F / impl_->io_ptr->Framerate;
  if (ImGui::BeginMainMenuBar()) {
    for (const auto& it : impl_->categories) {
      const auto& cat = it.second;
      if (ImGui::BeginMenu(cat.name.c_str())) {
        for (const auto& page : cat.pages) {
          if (ImGui::MenuItem(page->GetPageName().c_str())) {
            page->OpenPage();
          }
        }
        ImGui::EndMenu();
      }
    }
    ImGui::EndMainMenuBar();
  }

  // update pages
  for (const auto& it : impl_->categories) {
    for (const auto& page : it.second.pages) {
      page->Update(dt);
      page->DrawPage();
    }
  }

  {
    ImGui::Begin("Statistics", nullptr,
                 ImGuiWindowFlags_::ImGuiWindowFlags_NoResize | ImGuiWindowFlags_::ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_::ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::SameLine();
    ImGui::Text("Execution Mode: %s", impl_->execution_mode.c_str());

    if (!impl_->ace_lab.empty()) {
      for (const auto& config : impl_->hosts_to_ping) {
        std::string status;
        auto color = Colors::kNormal;
        if (config.reachable && config.pingable) {
          status = " is reachable";
        } else if (!config.reachable && config.pingable) {
          status = " has incorrect hostname <" + config.hostname + ">";
          color = Colors::kWarning;
        } else if (config.reachable && !config.pingable) {
          status = " has incorrect ipaddress <" + config.ipaddress + ">";
          color = Colors::kWarning;
        } else {
          status = " is not reachable!";
          color = Colors::kError;
        }
        ImGui::TextColored(color, "%s %s", config.name.c_str(), status.c_str());
      }
    }

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0F / impl_->io_ptr->Framerate,
                impl_->io_ptr->Framerate);
    int width;
    int height;
    glfwGetWindowSize(impl_->window, &width, &height);

    ImGui::Text("Wndow Size: %d/%d", width, height);
    ImGui::SetWindowPos(ImVec2(0, static_cast<float>(height) - ImGui::GetWindowHeight()), 0);
    ImGui::End();
  }
  ImGui::Render();

  int display_w;
  int display_h;
  glfwGetFramebufferSize(impl_->window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  glClearColor(impl_->clear_color.x * impl_->clear_color.w, impl_->clear_color.y * impl_->clear_color.w,
               impl_->clear_color.z * impl_->clear_color.w, impl_->clear_color.w);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(impl_->window);
  return true;
}

bool ACEMonitor::IsDone() const {
  return (impl_->window == nullptr) || (glfwWindowShouldClose(impl_->window) != 0) || !rclzmq::ok();
}

void* ACEMonitor::GetValue(const std::string& category, const std::string& name) const {
  auto cat = impl_->values.find(category);
  if (cat == impl_->values.end()) {
    return nullptr;
  }
  auto val = cat->second.find(name);
  if (val == cat->second.end()) {
    return nullptr;
  }
  return val->second;
}
void ACEMonitor::SetValue(const std::string& category, const std::string& name, void* ptr) {
  auto cat = impl_->values.find(category);
  if (cat == impl_->values.end()) {
    impl_->values[category] = ACEMonitorImpl::ValueMap();
  }
  impl_->values[category][name] = ptr;
}
const YAML::Node& ACEMonitor::GetConfigurations() const { return impl_->configurations; }

std::shared_ptr<AceMonitorRosNode> ACEMonitor::GetRosNode() const { return impl_->ros_node; }

AlarmsPage* ACEMonitor::GetAlarmsPage() { return impl_->alarms.get(); }
}  // namespace ace_monitor
