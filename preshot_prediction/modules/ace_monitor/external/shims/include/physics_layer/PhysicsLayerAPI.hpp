// Standalone shim: PhysicsLayer no-op stubs.
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>

namespace physics {

struct BallState {
  Eigen::Vector3f position{Eigen::Vector3f::Zero()};
  Eigen::Vector3f linear_velocity{Eigen::Vector3f::Zero()};
  Eigen::Vector3f angular_velocity{Eigen::Vector3f::Zero()};
};

class PhysicsParameters {
 public:
  using SharedPtr = std::shared_ptr<PhysicsParameters>;

  static SharedPtr LoadParametersFromYamlFile(const std::string& /*path*/, const std::string& /*key*/) {
    return std::make_shared<PhysicsParameters>();
  }
};

class PhysicsAPI {
 public:
  static BallState PredictBallState(const BallState& state, double dt, const PhysicsParameters::SharedPtr& /*params*/,
                                    bool /*apply_air*/) {
    // Trivial integrator without forces (gravity only).
    BallState next = state;
    next.position += state.linear_velocity * static_cast<float>(dt);
    next.linear_velocity.z() -= 9.81F * static_cast<float>(dt);
    return next;
  }
};

}  // namespace physics
