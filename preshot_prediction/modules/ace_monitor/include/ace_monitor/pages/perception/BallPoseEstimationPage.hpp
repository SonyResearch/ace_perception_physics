// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

class BallPoseEstimationPage : public IMonitorPage {
 protected:
  class BallPoseEstimationPageImpl;
  std::unique_ptr<BallPoseEstimationPageImpl> impl_;

  void RenderWidgets() override;

 public:
  BallPoseEstimationPage();
  ~BallPoseEstimationPage() override;

  void Initialize(PerceptionVisualizerPage* visualizer) override;

  void Update(float dt) override;

  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;
};
}  // namespace ace_monitor
