// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <glog/logging.h>

// NOLINTBEGIN
#define VLOG_INFO 1
#define VLOG_DEBUG 2
#define VLOG_VERBOSE 3

/**
 * Although glog already has internal logic to not write log messages when the severity threshold is not met,
 * the implementation results in the logging expressions (e.g. string allocations, data formatting, etc.) being
 * evaluated even if the severity level does not meet the threshold. The wrapper inserts an additional conditional
 * to avoid evaluating the expressions if the severity threshold is not met.
 */

#define EXTERNAL_LOG(severity) COMPACT_GOOGLE_LOG_##severity.stream()
#define EXTERNAL_LOG_IF(severity, condition) \
  static_cast<void>(0), !(condition) ? static_cast<void>(0) : google::LogMessageVoidify() & EXTERNAL_LOG(severity)

#undef LOG
#define LOG(severity) EXTERNAL_LOG_IF(severity, FLAGS_minloglevel <= google::severity)
#undef LOG_IF
#define LOG_IF(severity, condition) EXTERNAL_LOG_IF(severity, FLAGS_minloglevel <= google::severity && (condition))
#undef LOG_ASSERT
#define LOG_ASSERT(condition) EXTERNAL_LOG_IF(FATAL, !(condition)) << "Assert failed: " #condition

#if !DCHECK_IS_ON()
#undef DLOG
#undef DLOG_IF
#undef DLOG_ASSERT
#undef DVLOG
#define DLOG(severity) EXTERNAL_LOG_IF(severity, false)
#define DLOG_IF(verboselevel, condition) EXTERNAL_LOG_IF(INFO, false)
#define DLOG_ASSERT(condition) EXTERNAL_LOG_IF(FATAL, false && !(condition)) << #condition
#define DVLOG(verboselevel) EXTERNAL_LOG_IF(INFO, false)
#endif

#define EXTERNAL_PLOG(severity) GOOGLE_PLOG(severity, 0).stream()
#define EXTERNAL_PLOG_IF(severity, condition) \
  static_cast<void>(0), !(condition) ? static_cast<void>(0) : google::LogMessageVoidify() & EXTERNAL_PLOG(severity)

#undef PLOG
#define PLOG(severity) EXTERNAL_PLOG_IF(severity, FLAGS_minloglevel <= (google::severity))

// NOLINTEND

namespace logger {

void initialize();

void set_level(const char* log_level);

}  // namespace logger
