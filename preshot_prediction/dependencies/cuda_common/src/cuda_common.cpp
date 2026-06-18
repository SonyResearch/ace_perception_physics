// Confidential, Copyright 2024, Sony AI, All rights reserved.
// Written by Hamdi Sahloul

#include "cuda_common/cuda_common.hpp"

#include "ace_loggers/ace_loggers.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "nvml.h"

namespace cuda_common {

int GetCudaDeviceID(const std::string& cuda_device_uuid, const bool ignore_gpu_checks) {
  nvmlReturn_t nvml_result;
  nvml_result = nvmlInit();
  if (nvml_result != nvmlReturn_enum::NVML_SUCCESS) {
    LOG(ERROR) << "Failed to initialize NVML (error: " << nvml_result << "). \n";
    return -1;
  }

  nvmlDevice_t device;
  nvml_result = nvmlDeviceGetHandleByUUID(cuda_device_uuid.c_str(), &device);
  if (nvml_result != nvmlReturn_enum::NVML_SUCCESS) {
    LOG(ERROR) << "Failed to get NVML device by UUID (error: " << nvml_result << "). \n";
    return -1;
  }

  if (!ignore_gpu_checks) {
    nvmlEnableState_t display;
    nvml_result = nvmlDeviceGetDisplayMode(device, &display);
    if (nvml_result != nvmlReturn_enum::NVML_SUCCESS) {
      LOG(ERROR) << "Failed to get NVML device display mode (error: " << nvml_result << "). \n";
      return -1;
    }
    if (display == nvmlEnableState_enum::NVML_FEATURE_ENABLED) {
      LOG(ERROR) << "The selected GPU (" << cuda_device_uuid
                 << ") is connected to a display. Please select another GPU.\n";
      return -1;
    }

    nvmlEnableState_t is_active;
    nvml_result = nvmlDeviceGetDisplayActive(device, &is_active);
    if (nvml_result != nvmlReturn_enum::NVML_SUCCESS) {
      LOG(ERROR) << "Failed to get NVML device display active (error: " << nvml_result << "). \n";
      return -1;
    }
    if (is_active == nvmlEnableState_enum::NVML_FEATURE_ENABLED) {
      LOG(ERROR) << "The selected GPU (" << cuda_device_uuid
                 << ") is connected to an active display or has an Xserver running on it. Please select another GPU.\n";
      return -1;
    }
  }

  nvmlPciInfo_t pci_info;
  nvml_result = nvmlDeviceGetPciInfo(device, &pci_info);
  if (nvml_result != nvmlReturn_enum::NVML_SUCCESS) {
    LOG(ERROR) << "Failed to get PCI Info from NVML device (error: " << nvml_result << "). \n";
    return -1;
  }

  nvml_result = nvmlShutdown();
  if (nvml_result != nvmlReturn_enum::NVML_SUCCESS) {
    LOG(ERROR) << "Failed to shutdown NVML (error: " << nvml_result << "). \n";
    return -1;
  }

  int cuda_device_id;
  cudaError_t cuda_result;
  cuda_result = cudaDeviceGetByPCIBusId(&cuda_device_id, pci_info.busIdLegacy);
  if (cuda_result != cudaError::cudaSuccess) {
    LOG(ERROR) << "Failed to get CUDA device id by PCI bus (error: " << cuda_result << "). \n";
    return -1;
  }

  return cuda_device_id;
}

}  // namespace cuda_common
