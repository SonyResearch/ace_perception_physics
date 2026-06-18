// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

class BallSpinPage : public IMonitorPage {
 protected:
  class BallSpinPageImpl;
  std::unique_ptr<BallSpinPageImpl> impl_;

  void RenderWidgets() override;

 public:
  BallSpinPage();
  ~BallSpinPage() override;
  void Initialize(PerceptionVisualizerPage* visualizer) override;
  void Update(float dt) override;

  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;
};
}  // namespace ace_monitor
