
// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept {
    // remove this 'if' if you need more logged info
    if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
      std::cout << msg << "n";
    }
  }
};
