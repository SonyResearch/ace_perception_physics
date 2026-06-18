// Confidential, Copyright 2024, Sony AI, All rights reserved.
// Written by Hamdi Sahloul

#include <cstddef>

#include "eigen3/Eigen/Core"
#include "opencv2/opencv.hpp"

namespace cuda_common {

int GetCudaDeviceID(const std::string& cuda_device_uuid, bool ignore_gpu_checks = false);

}  // namespace cuda_common
