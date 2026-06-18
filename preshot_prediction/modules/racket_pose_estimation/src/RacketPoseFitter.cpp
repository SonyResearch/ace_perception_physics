// Confidential, Copyright 2025, Sony AI, All rights reserved.

#include "racket_pose_estimation/RacketPoseFitter.hpp"

#include <algorithm>

#include "ace_loggers/ace_loggers.hpp"

#define DEBUG false
namespace perception {
RacketPoseFitter::RacketPoseFitter() : last_angles_(1, 0, 0, 0), random_device_(0) {
  model_points_[0] = Eigen::Vector3f(0, 0, 1) * 0.15;
  model_points_[1] = Eigen::Vector3f(0, 1, 0) * 0.15;
  axis_flip_.emplace_back(Eigen::Quaternionf::Identity());
  axis_flip_.emplace_back(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
  axis_flip_.emplace_back(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()));
  axis_flip_.emplace_back(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
}

void RacketPoseFitter::SetCameras(const std::vector<int>& cameras) { cameras_ = cameras; }

float RacketPoseFitter::EvalAt(const Eigen::Vector3f& position, const Eigen::Quaternionf& q,
                               const std::vector<std::vector<Eigen::Vector3f>>& target_axis, bool use_axis_order) {
  float error = 0;
  float total_weight = 0;
  for (size_t i = 0; i < cameras_.size(); ++i) {
    // project points per camera
    for (int j = 0; j < 2; ++j) {
      Eigen::Vector3f pt = q * this->model_points_[j] + position;
      calibration_->cameras[cameras_[i]].ProjectPoint(pt, &projected_points_[i][j], nullptr, false);
    }

    Eigen::Vector2f current_axis[] = {(projected_points_[i][0] - zero_points_[i]).normalized(),
                                      (projected_points_[i][1] - zero_points_[i]).normalized()};

    // calculate the error between current axis angles and target ones
    float e1 = target_axis[i][0].head<2>().dot(current_axis[0]);
    float e2 = target_axis[i][1].head<2>().dot(current_axis[1]);
    if (!use_axis_order) {
      e1 = std::abs(e1);
      e2 = std::abs(e2);
    }

    e1 = std::abs(1 - e1) * target_axis[i][0].z();
    e2 = std::abs(1 - e2) * target_axis[i][1].z();
    // if (use_axis_order) {
    //   std::cout<<e1<<", "<<e2<<std::endl;
    // }

    if (DEBUG && true) {
      LOG(INFO) << "index: " << i;
      LOG(INFO) << target_axis[i][0].transpose() << " x " << current_axis[0].transpose();
      LOG(INFO) << target_axis[i][1].transpose() << " x " << current_axis[1].transpose();
      LOG(INFO) << e1 << " x " << e2;
    }
    projection_errors_[i] = (e1 + e2);
    error += projection_errors_[i] * view_weight_[i];
    total_weight += view_weight_[i];
  }
  return error / total_weight;
}

void RacketPoseFitter::GetGradients(float curr_err, const Eigen::Vector3f& position, const Eigen::Quaternionf& q,
                                    float* gradients, const std::vector<std::vector<Eigen::Vector3f>>& target_axis) {
  static float delta = M_PI / 180.0F;
  static Eigen::AngleAxisf dx = Eigen::AngleAxisf(delta, Eigen::Vector3f::UnitX());
  static Eigen::AngleAxisf dy = Eigen::AngleAxisf(delta, Eigen::Vector3f::UnitY());
  static Eigen::AngleAxisf dz = Eigen::AngleAxisf(delta, Eigen::Vector3f::UnitZ());

  float x_err = EvalAt(position, dx * q, target_axis, false);
  float y_err = EvalAt(position, dy * q, target_axis, false);
  float z_err = EvalAt(position, dz * q, target_axis, false);

  gradients[0] = ((x_err - curr_err) / delta);
  gradients[1] = ((y_err - curr_err) / delta);
  gradients[2] = ((z_err - curr_err) / delta);
  if (DEBUG) {
    LOG(INFO) << "Gradients: " << gradients[0] << " , " << gradients[1] << " , " << gradients[2];
  }
}

Eigen::Quaternionf RacketPoseFitter::FitPoints(const Eigen::Vector3f& position,
                                               const std::vector<std::vector<Eigen::Vector3f>>& keypoints,
                                               int& out_iterations_count, float& out_error) {
  if (calibration_ == nullptr) {
    return last_angles_;
  }
  out_iterations_count = 0;
  std::vector<std::vector<Eigen::Vector3f>> target_axis;
  target_axis.resize(keypoints.size());

  for (size_t i = 0; i < keypoints.size(); ++i) {
    target_axis[i].resize(2);
    target_axis[i][0].head<2>() = ((keypoints[i][1].head<2>()) - (keypoints[i][0].head<2>()));
    target_axis[i][1].head<2>() = ((keypoints[i][3].head<2>()) - (keypoints[i][2].head<2>()));
    float ax0_len = target_axis[i][0].head<2>().norm();
    float ax1_len = target_axis[i][1].head<2>().norm();

    target_axis[i][0].head<2>() /= ax0_len;
    target_axis[i][1].head<2>() /= ax1_len;

    // weight will be 1 if the length of the axis is twice the minimum length, otherwise it will be a gradient 0->1
    // 1e-4 is (1/100*100) to normalize the keypoint confidence to 0->1
    target_axis[i][0].z() =
      keypoints[i][0].z() * keypoints[i][1].z() * 1e-4F *
      std::clamp<float>((ax0_len - params_->fitter.min_axis_length) / params_->fitter.min_axis_length, 0.0F, 1.0F);

    target_axis[i][1].z() =
      keypoints[i][2].z() * keypoints[i][3].z() * 1e-4F *
      std::clamp<float>((ax1_len - params_->fitter.min_axis_length) / params_->fitter.min_axis_length, 0.0F, 1.0F);

    target_axis[i][0].z() = std::pow<float>(target_axis[i][0].z(), 2.0F);
    target_axis[i][1].z() = std::pow<float>(target_axis[i][1].z(), 2.0F);
    // std::cout << i << std::endl;
    // std::cout << "0 : " << ax0_len << ", " << target_axis[i][0].z() << std::endl;
    // std::cout << "1 : " << ax1_len << ", " << target_axis[i][1].z() << std::endl;
  }

  projected_points_.resize(cameras_.size());
  zero_points_.resize(cameras_.size());
  key_points_center_.resize(cameras_.size());
  projection_errors_.resize(cameras_.size());
  view_weight_.resize(cameras_.size());
  for (size_t i = 0; i < cameras_.size(); ++i) {
    projected_points_[i].resize(2);
    view_weight_[i] = 1.0F;
    // zero_points_[i] = (((keypoints[i][1].head<2>()) + (keypoints[i][0].head<2>())) / 2 +
    //                    ((keypoints[i][3].head<2>()) - (keypoints[i][2].head<2>())) / 2) /
    //                   2;

    calibration_->cameras[cameras_[i]].ProjectPoint(position, &zero_points_[i], nullptr, false);
  }

  float current_rate = params_->fitter.initial_rate;
  int current_iter = 0;

  Eigen::Quaternionf current_rotation = last_angles_;
  Eigen::Vector3f home_axis[] = {Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ()};

  current_iter = 0;
  float gradients[3];
  float step[3];
  float total_err_now = 0;
  bool did_skip = false;
  int skipped = 0;
  float delta_err = 1;
  float current_err = EvalAt(position, current_rotation, target_axis, false);

  int next_check = params_->fitter.max_iterations / 4;

  auto max_error = params_->fitter.max_error * params_->fitter.max_error;
  while (current_iter++ < params_->fitter.max_iterations && current_err > max_error) {
    if (!did_skip) {
      GetGradients(current_err, position, current_rotation, gradients, target_axis);
    }

    Eigen::Quaternionf new_rotation = current_rotation;
    for (int axis = 0; axis < 3; ++axis) {
      step[axis] = current_rate * gradients[axis];
      new_rotation = Eigen::AngleAxisf(-step[axis], home_axis[axis]) * new_rotation;
    }

    // calculate projection error using current orientation
    total_err_now = EvalAt(position, new_rotation, target_axis, false);
    if (current_iter % next_check == 0 && projection_errors_.size() > 2) {
      // update weights
      float sum = 0;
      next_check /= 2;
      if (next_check < 100) {
        next_check = 100;
      }
      // std::cout << current_iter << std::endl;
      for (auto projection_error : projection_errors_) {
        sum += projection_error;
      }
      sum /= static_cast<float>(projection_errors_.size());
      if (sum > 0) {
        float max = 0.0F;
        for (size_t i = 0; i < projection_errors_.size(); ++i) {
          auto scaler =
            static_cast<float>(std::pow<float>(sum, projection_errors_[i] * 1e-1F) * 1.00F);  // 0~1 scalar value
          view_weight_[i] = std::max<float>(view_weight_[i] * scaler, 0.0F);
          max = std::max<float>(view_weight_[i], max);
          // std::cout << current_iter << "-> " << cameras_[i] << ": " << projection_errors_[i] << "-->" <<
          // view_weight_[i]
          //           << std::endl;
        }
        if (max > 0) {
          auto imax = 1.0F / max;
          for (size_t i = 0; i < projection_errors_.size(); ++i) {
            view_weight_[i] *= imax;
          }
        }
      }
    }
    delta_err = (total_err_now - current_err);
    if (std::abs(total_err_now) < max_error) {
      break;
    }
    bool perform_random_jump = false;
    bool do_continue = false;

    if (total_err_now > current_err) {
      if (DEBUG) {
        std::cout << "Skipping: " << current_rate << ": " << total_err_now << std::endl;
      }
      current_rate *= 0.5;
      skipped += 1;
      did_skip = true;
      do_continue = true;

      if (skipped > params_->fitter.max_skips_reset && std::abs(delta_err) <= params_->fitter.max_delta_err_reset) {
        perform_random_jump = true;
      }
    }
    if (perform_random_jump) {
      did_skip = false;
      skipped = 0;
      current_rate = params_->fitter.initial_rate;
      new_rotation = current_rotation;
      for (int axis = 0; axis < 3; ++axis) {
        std::uniform_real_distribution<> random_dist(0.0F, 1.0F);
        float rand_angle = M_PI * random_dist(random_device_) * (2 - 1) * projection_errors_[axis];
        new_rotation = Eigen::AngleAxisf(rand_angle, home_axis[axis]) * new_rotation;
      }
      current_rotation = new_rotation;
      if (DEBUG) {
        LOG(INFO) << "Performing random jump";
      }
    }
    if (perform_random_jump || do_continue) {
      continue;
    }
    skipped = 0;
    did_skip = false;
    if (DEBUG) {
      LOG(INFO) << current_iter << " - Error: " << total_err_now;
      for (size_t axis = 0; axis < 3; ++axis) {
        LOG(INFO) << "\t" << axis << " - Step/Derivative: " << step[axis] << "/" << gradients[axis];
      }
    }
    current_rotation = new_rotation;
    current_err = total_err_now;
    out_iterations_count += 1;
  }
  if (DEBUG) {
    LOG(INFO) << "Done at: " << current_iter << " - err : " << total_err_now;
  }

  if (DEBUG) {
    for (size_t i = 0; i < projection_errors_.size(); ++i) {
      LOG(INFO) << "Projection Error[" << i << "]: " << projection_errors_[i];
    }
  }

  {
    // perform axis fitting
    current_err = 99999;
    // std::cout << "------------" << std::endl;
    for (int i = 0; i < 3; ++i) {
      Eigen::Quaternionf best = current_rotation;
      float best_err = std::abs(current_err);
      // int index = 0;
      // std::cout << "------------" << std::endl;
      // std::cout << index << ": " << best_err << std::endl;
      for (auto& axis : axis_flip_) {
        auto rotation = current_rotation * axis;
        auto err = std::abs(EvalAt(position, rotation, target_axis, true));
        // ++index;
        // std::cout << index << ": " << err << std::endl;
        if (err < best_err) {
          best = rotation;
          best_err = err;
        }
      }
      if (current_err == best_err) {
        break;
      }
      current_rotation = best;
      current_err = best_err;
    }
  }
  {
    // choose the best orientation closer to previous observation
    Eigen::AngleAxisf axis_z = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ());

    auto flipped_rotation = current_rotation * axis_z;
    Eigen::Vector3f last_axis = last_angles_ * Eigen::Vector3f::UnitY();
    Eigen::Vector3f current_axis = current_rotation * Eigen::Vector3f::UnitY();
    Eigen::Vector3f flipped_axis = flipped_rotation * Eigen::Vector3f::UnitY();
    auto distance_current = last_axis.dot(current_axis);
    auto distance_flipped = last_axis.dot(flipped_axis);

    if (distance_current < distance_flipped) {
      // use the flipped rotation
      current_rotation = flipped_rotation;
      current_err = std::abs(EvalAt(position, current_rotation, target_axis, true));
    }
  }
  out_error = current_err;
  last_error_ = current_err;
  last_angles_ = current_rotation;
  last_angles_.normalize();
  return last_angles_;
}

}  // namespace perception
