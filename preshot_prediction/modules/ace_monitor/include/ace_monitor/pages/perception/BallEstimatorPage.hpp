// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

class BallEstimatorPage : public IMonitorPage {
 protected:
  class BallEstimatorPageImpl;
  std::unique_ptr<BallEstimatorPageImpl> impl_;

  void RenderWidgets() override;

 public:
  BallEstimatorPage();
  ~BallEstimatorPage() override;

  void Initialize(PerceptionVisualizerPage* visualizer) override;

  void Update(float dt) override;

  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;
};
}  // namespace ace_monitor
