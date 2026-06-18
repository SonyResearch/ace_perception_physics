/**
 * \@file logger_pybind.cpp
 * @author Sam Blakeman
 * @date 2021
 * @version 0.0
 * @copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
 *
 * @brief The package provides functions for initializing and setting the logging level in glog from python
 *
 */
#include "ace_loggers/ace_loggers.hpp"
#include "pybind11/pybind11.h"

namespace logger {

void initialize() {
  if (google::IsGoogleLoggingInitialized()) {
    return;
  }
  google::InitGoogleLogging("ace_logging");
  google::SetCommandLineOption("colorlogtostderr", "1");
  google::SetCommandLineOption("logtostderr", "1");
}

void set_level(const char* log_level) { google::SetCommandLineOption("minloglevel", log_level); }

PYBIND11_MODULE(logger_pybind, m) {
  m.def("initialize", &initialize);
  m.def("set_level", &set_level);
}
}  // namespace logger
