// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

class PerceptionDelayPage : public IMonitorPage {
 protected:
  class PerceptionDelayPageImpl;
  std::unique_ptr<PerceptionDelayPageImpl> impl_;

  void RenderWidgets() override;

 public:
  PerceptionDelayPage();
  ~PerceptionDelayPage() override;
  void Update(float dt) override;

  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;
};
}  // namespace ace_monitor
