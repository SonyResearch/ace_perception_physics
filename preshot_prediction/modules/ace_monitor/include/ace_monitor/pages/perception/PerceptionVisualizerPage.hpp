// Confidential, Copyright 2025, Sony AI, All rights reserved.

#pragma once

#include <unordered_map>

#include "ace_monitor/IMonitorPage.hpp"
#include "ace_monitor/scene/CameraNode.hpp"
#include "ace_monitor/scene/Node3D.hpp"
#include "ace_monitor/scene/Scene3D.hpp"
#include "calibration/camera_calibration_parameters.hpp"

namespace ace_monitor {

class PerceptionVisualizerPage : public IMonitorPage {
 protected:
  class VisualizerPageImpl;
  std::unique_ptr<VisualizerPageImpl> impl_;

  void RenderWidgets() override;

 public:
  PerceptionVisualizerPage();
  ~PerceptionVisualizerPage() override;
  void Initialize(PerceptionVisualizerPage*) override;
  void Update(float dt) override;

  void OpenPage() override;
  void ClosePage() override;
  [[nodiscard]] PageStatus GetStatus() const override;

  Scene3D::SharedPtr GetScene();

  const std::unordered_map<std::string, CameraNode::SharedPtr>& GetCameras();
  const std::unordered_map<std::string, Node3D::SharedPtr>& GetCalibrationCameras();
  calibration::CameraCalibrationParameters::SharedPtr GetCalibration();
  const std::string& GetConfigName();
};
}  // namespace ace_monitor
