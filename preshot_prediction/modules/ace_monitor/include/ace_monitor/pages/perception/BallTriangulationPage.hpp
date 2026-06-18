// Confidential, Copyright 2025, Sony AI, All rights reserved
#pragma once

#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

class BallTriangulationPage : public IMonitorPage {
 protected:
  class BallTriangulationPageImpl;
  std::unique_ptr<BallTriangulationPageImpl> impl_;

  void RenderWidgets() override;

 public:
  BallTriangulationPage();
  ~BallTriangulationPage() override;
  void Update(float dt) override;

  void Initialize(PerceptionVisualizerPage* visualizer) override;
  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;
};
}  // namespace ace_monitor
