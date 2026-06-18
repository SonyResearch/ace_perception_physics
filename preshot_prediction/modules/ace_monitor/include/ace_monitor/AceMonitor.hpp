// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ace_monitor/IMonitorPage.hpp"
#include "ace_yaml/ace_yaml.hpp"

namespace ace_monitor {

class AceMonitorRosNode;
class AlarmsPage;

class ACEMonitor {
 protected:
  class ACEMonitorImpl;
  std::unique_ptr<ACEMonitorImpl> impl_;

  static std::shared_ptr<ACEMonitor> instance_;

  void AddPage(std::shared_ptr<IMonitorPage> page);
  ACEMonitor();

 public:
  ~ACEMonitor();
  static ACEMonitor& GetInstance();

  void Initialize(int argc, char** argv);
  void Shutdown();

  [[nodiscard]] const std::string& GetDataPath() const;

  template <typename T, typename... Args>
  void AddPage(Args&&... args) {
    static_assert(std::is_base_of_v<IMonitorPage, T>);
    AddPage(std::make_shared<T>(std::forward<Args>(args)...));
  }

  [[nodiscard]] std::string_view GetAppArg(const std::string& name, std::string_view default_value) const;

  [[nodiscard]] bool Draw() const;
  [[nodiscard]] bool IsDone() const;

  [[nodiscard]] void* GetValue(const std::string& category, const std::string& name) const;
  void SetValue(const std::string& category, const std::string& name, void* ptr);

  [[nodiscard]] const YAML::Node& GetConfigurations() const;

  [[nodiscard]] std::shared_ptr<AceMonitorRosNode> GetRosNode() const;

  AlarmsPage* GetAlarmsPage();
};
}  // namespace ace_monitor
