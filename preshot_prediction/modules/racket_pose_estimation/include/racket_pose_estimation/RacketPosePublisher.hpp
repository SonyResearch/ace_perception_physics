// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "racket_pose_estimation/RacketFeatures.hpp"
#include "racket_pose_estimation/RacketParameters.hpp"

namespace perception {
class RacketPosePublisher {
 public:
  using SharedPtr = std::shared_ptr<RacketPosePublisher>;
  using ConstSharedPtr = std::shared_ptr<const RacketPosePublisher>;

 protected:
  class RacketPosePublisherImpl;
  std::shared_ptr<RacketPosePublisherImpl> impl_;

 public:
  RacketPosePublisher();
  void Initialize(RacketParameters::SharedPtr parameters);
  void PublishMessage(const RacketFrameFeatures& features);
};

}  // namespace perception
