/**
 * \@file logger.cpp
 * @author Sam Blakeman
 * @date 2021
 * @version 0.0
 * @copyright Confidential, Copyright 2024, Sony AI, All rights reserved.
 *
 * @brief The package provides functions for initializing and setting the logging level in glog from python
 *
 */
#include "ace_loggers/ace_loggers.hpp"

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

}  // namespace logger
