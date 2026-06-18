// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

class CameraMonitorPage : public IMonitorPage {
 protected:
  class CameraMonitorPageImpl;
  std::unique_ptr<CameraMonitorPageImpl> impl_;

  void RenderWidgets() override;

 public:
  CameraMonitorPage();
  ~CameraMonitorPage() override;
  void Update(float dt) override;

  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;
};
}  // namespace ace_monitor
