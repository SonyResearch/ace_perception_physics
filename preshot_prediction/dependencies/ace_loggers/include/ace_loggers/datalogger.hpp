// Confidential, Copyright 2026, Sony AI, All rights reserved.
#pragma once

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "ace_loggers/logging.hpp"
namespace datalogger {

constexpr std::array<std::string_view, 2> kRequiredMetadataKeys = {"module_name", "log_version"};
constexpr std::string_view kMagic = "ACE";
constexpr std::array<std::pair<const char*, const char*>, 14> kValidLoaders = {{
  {"arena_native", ".*state_estimator.*\\.ace"},
  {"ball_detector", ".*ball_detector.*\\.ace"},
  {"musashi_interface", ".*musashi.*\\.ace"},
  {"physics_layer", ".*physics.*\\.ace"},
  {"player_pose", ".*player_pose.*\\.ace"},
  {"pose_estimator", ".*pose_estimator.*\\.ace"},
  {"racket_pose_estimation", ".*racket_pose_estimator.*\\.ace"},
  {"racket_pose_vive", ".*racket_pose_vive.*\\.ace"},
  {"staubli_ethercat_interface", ".*staubli_60l.*\\.ace"},
  {"trajectory_estimation", ".*trajectory_estimation.*\\.ace"},
  {"triangulation", ".*triangulation.*\\.ace"},
  {"fpga_aps", ".*fpga_aps.*\\.ace"},
  {"optitrack", ".*optitrack.*\\.ace"},
  {"ace_loggers", ".*\\.ace"},  // Keep at the end, catch all
}};

enum class DataWriterFileMode {
  kOverwrite = 0,  // will override the log file if exist
  kExclusive,      // will throw an error if an existing log file with the same name exists
  kAutoIncrement   // will auto increment file name if the same log file exists (e.g. log.ace, log_1.ace, log_2.ace,...)
};

std::string ConstructFullLogName(const std::string& basename);

// injects a type to the file_path, e.g. file_path="./ace_logs/some_path/state_estimator.ace", type="physics",
// return will be "./ace_logs/some_path/state_stimator_physics.ace".
// Noe that if file_path is empty, then the return will be empty too
std::string ConstructSubLogName(const std::string& file_path, const std::string& type);

bool IsLoaderDefined(const std::string& module_name);
std::string GetModuleName(const std::string& path);

// See https://stackoverflow.com/a/50182372
template <typename F>
class DeferredCall {
 public:
  inline DeferredCall(const DeferredCall& that) = delete;
  inline DeferredCall& operator=(const DeferredCall& that) = delete;

  inline explicit DeferredCall(F&& f);
  inline DeferredCall(DeferredCall&& that) noexcept;

  inline virtual ~DeferredCall();

  inline bool Cancel();
  inline bool Execute();

 private:
  F func_;
  bool is_owner_{true};
};

template <typename F>
DeferredCall<F> defer(F&& f);

// TODO(cv3d): Follow the interface of std::ofstream
class FileWriter {
  FILE* file_ = nullptr;

 public:
  inline explicit FileWriter();
  inline virtual ~FileWriter();

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline void open(std::string& filename, std::ios_base::openmode mode);
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] inline bool is_open() const;
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline void write(const void* data, size_t size);
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline void close();
};

class DataBuffer {
 public:
  inline DataBuffer(const std::string& filename, std::string rt_profile, size_t const& buffer_size,
                    size_t const& flush_size);
  inline void Open(const std::string& filename);
  inline void SetFileName(const std::string& filename);
  [[nodiscard]] inline std::string GetFileName() const;
  inline virtual ~DataBuffer();
  inline virtual void Release();

  inline virtual bool Write(const char* data, size_t const& len);

 protected:
  std::string filename_;
  FileWriter file_;

 private:
#ifdef DISABLE_THREADING
  static constexpr bool kEnableThreading = false;
#else
  static constexpr bool kEnableThreading = true;
#endif
  const std::string rt_profile_;
  const size_t flush_size_;
  const size_t buffer_size_;

  std::vector<char> mem_;
  alignas(64) std::atomic_size_t read_idx_ = {0};
  alignas(64) std::atomic_size_t write_idx_ = {0};

  mutable std::mutex write_mutex_;
  mutable std::mutex read_mutex_;

  std::condition_variable write_event_;
  std::condition_variable read_event_;

  [[nodiscard]] inline size_t GetCapacity() const;
  [[nodiscard]] inline size_t GetUsedSize() const;
  [[nodiscard]] inline size_t GetRemainingSize() const;

  inline bool Dump(size_t const& len);
  inline void WorkerThread();
  std::list<std::pair<size_t, std::string>> split_queue_;
  std::atomic_bool is_running_;
  std::thread worker_;
};  // namespace datalogger

class DataWriter {
 public:
  using SharedPtr = std::shared_ptr<DataWriter>;

  inline explicit DataWriter(std::string const& filename, nlohmann::json metadata, const std::string& rt_profile = "",
                             size_t const& buffer_size = 1024 * 1024 * 200, size_t const& flush_size = 1024 * 100);
  inline virtual ~DataWriter();
  inline virtual void Release();

  template <typename T>
  inline friend DataWriter& operator<<(DataWriter& writer, T const& data);

  inline DataWriter& Write(const char* data, size_t const& len);

  template <typename T>
  inline DataWriter& Write(T const& data);

  inline void SetFileName(const std::string& filename);
  [[nodiscard]] inline std::string GetFileName() const;
  [[nodiscard]] inline const nlohmann::json& GetMetadata() const;

 private:
  inline DataWriter& WriterHeader();
  const std::string handler_id_ = "DataWriter";
  const nlohmann::json metadata_;
  DataBuffer buffer_;
};

class DataReader {
 public:
  using SharedPtr = std::shared_ptr<DataReader>;

  inline explicit DataReader(std::string const& filename);
  inline virtual ~DataReader();
  inline virtual void Release();
  inline virtual bool IsEndOfFile() const;
  inline virtual int64_t GetPosition() const;

  template <typename T>
  inline friend DataReader& operator>>(DataReader& reader, T& data);

  inline DataReader& Read(char* data, size_t const& len);

  template <typename T>
  inline DataReader& Read(T& data);
  template <typename T>
  inline T ReadAndReturn();

  [[nodiscard]] inline const nlohmann::json& GetMetadata() const;
  [[nodiscard]] inline std::string GetModuleName() const;
  [[nodiscard]] inline std::string GetLogVersion() const;

 private:
  const size_t file_size_;
  std::ifstream file_;
  nlohmann::json metadata_;
};

}  // namespace datalogger

#include "ace_loggers/datalogger__impl.hpp"
