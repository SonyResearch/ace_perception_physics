// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

class AlarmsPage : public IMonitorPage {
 protected:
  class AlarmsPageImpl;
  std::unique_ptr<AlarmsPageImpl> impl_;

  void RenderWidgets() override;

 public:
  enum class AlarmLevel {
    kNone,
    kWarning,
    kCritical,
  };

  AlarmsPage();
  ~AlarmsPage() override;
  void Update(float dt) override;

  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;

  void SetAlarm(const std::string& name, const std::string& desc, AlarmLevel level, int timeout);
};
}  // namespace ace_monitor
