// Confidential, Copyright 2024, Sony AI, All rights reserved.

#pragma once

#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

class EstimatorTopicsPage : public IMonitorPage {
 protected:
  class EstimatorTopicsPageImpl;
  std::unique_ptr<EstimatorTopicsPageImpl> impl_;

  void RenderWidgets() override;

 public:
  EstimatorTopicsPage();
  ~EstimatorTopicsPage() override;
  void Update(float dt) override;

  void Initialize(PerceptionVisualizerPage* visualizer) override;
  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;
};
}  // namespace ace_monitor
