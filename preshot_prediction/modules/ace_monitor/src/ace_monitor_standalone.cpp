// Standalone fork — no proprietary deps.
#include <cstdio>
#include <exception>
#include <iostream>

#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/AceMonitor.hpp"
#include "ace_monitor/pages/perception/BallDetectorPage.hpp"
#include "ace_monitor/pages/perception/BallEstimatorPage.hpp"
#include "ace_monitor/pages/perception/BallPoseEstimationPage.hpp"
#include "ace_monitor/pages/perception/BallTriangulationPage.hpp"
#include "ace_monitor/pages/perception/CameraMonitorPage.hpp"
#include "ace_monitor/pages/perception/EstimatorTopicsPage.hpp"
#include "ace_monitor/pages/perception/BallSpinPage.hpp"
#include "ace_monitor/pages/perception/PerceptionDelayPage.hpp"
#include "ace_monitor/pages/perception/PlayerPoseEstimationPage.hpp"
#include "ace_monitor/pages/perception/RacketPoseEstimationPage.hpp"

int main(int argc, char **argv) {
  auto &monitor = ace_monitor::ACEMonitor::GetInstance();

  try {
    monitor.Initialize(argc, argv);
    monitor.AddPage<ace_monitor::CameraMonitorPage>();
    monitor.AddPage<ace_monitor::BallDetectorPage>();
    monitor.AddPage<ace_monitor::BallTriangulationPage>();
    monitor.AddPage<ace_monitor::BallPoseEstimationPage>();
    monitor.AddPage<ace_monitor::PlayerPoseEstimationPage>();
    monitor.AddPage<ace_monitor::RacketPoseEstimationPage>();
    monitor.AddPage<ace_monitor::PerceptionDelayPage>();
    monitor.AddPage<ace_monitor::BallSpinPage>();
    monitor.AddPage<ace_monitor::EstimatorTopicsPage>();
    monitor.AddPage<ace_monitor::BallEstimatorPage>();

    while (!monitor.IsDone()) {
      (void)monitor.Draw();
    }
    monitor.Shutdown();
  } catch (const std::exception &e) {
    LOG(FATAL) << "Fatal error: " << e.what();
    return 1;
  }
  return 0;
}
