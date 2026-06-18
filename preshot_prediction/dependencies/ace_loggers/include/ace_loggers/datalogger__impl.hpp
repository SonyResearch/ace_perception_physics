// Confidential, Copyright 2026, Sony AI, All rights reserved.
#pragma once

#include <Eigen/Dense>
#include <ace_rt_profiles/profile_manager.hpp>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <regex>
#include <sstream>

#include "ace_helpers/SignalHandler.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "ace_loggers/boost_time.hpp"
#include "ace_loggers/datalogger.hpp"
#include "ace_loggers/logging.hpp"

namespace datalogger {

inline void LoggerCreateDirectories(const std::string& dir) {
  std::filesystem::path path(dir);
  std::filesystem::path sub_dir;
  for (const auto& p : path) {
    sub_dir = sub_dir / p;
    if (std::filesystem::exists(sub_dir)) {
      continue;
    }
    try {
      std::filesystem::create_directory(sub_dir);
      std::filesystem::permissions(sub_dir, std::filesystem::perms::all, std::filesystem::perm_options::replace);
    } catch (std::filesystem::filesystem_error& e) {
      // ignore this error
    }
  }
}

inline std::string ConstructFullLogName(const std::string& filename) {
  if (filename.empty()) {
    return "";  // return empty for empty filenames
  }

  std::string logs_dir = "./ace_logs/";
  auto* const env_path = std::getenv("ACE_DATALOGGER_PATH");
  if (env_path != nullptr && strcmp(env_path, "") != 0) {
    logs_dir = env_path;
  }

  std::string logs_timezone = "UTC";
  auto* const env_timezone = std::getenv("ACE_DATALOGGER_TIMEZONE");
  if (env_timezone != nullptr && strcmp(env_timezone, "") != 0) {
    logs_timezone = env_timezone;
  }

  const auto local_tm = datalogger::GetLocalTime(logs_timezone);
  std::stringstream file_path;
  file_path << logs_dir << std::put_time(&local_tm, "%Y%m%d") << "/";
  LoggerCreateDirectories(file_path.str());

  file_path << "log_" << std::put_time(&local_tm, "%Y-%m-%dT%H-%M-%S") << "_" << filename << ".ace";
  return file_path.str();
}

inline std::string ConstructSubLogName(const std::string& file_path, const std::string& type) {
  if (file_path.empty()) {
    return "";
  }
  std::filesystem::path fs_path(file_path);
  const std::string dirname = fs_path.parent_path().string();
  LoggerCreateDirectories(dirname);

  const std::string filename = fs_path.filename().string();
  auto ext_index = filename.find_last_of('.');
  std::string ext;
  std::string log_fullname;
  if (ext_index != std::string::npos) {
    ext = filename.substr(ext_index);
    log_fullname = file_path.substr(0, file_path.size() + ext_index - filename.size());
  } else {
    ext = ".ace";
    log_fullname = file_path;
  }
  log_fullname += "_" + type + ext;
  return log_fullname;
}

inline bool IsLoaderDefined(const std::string& module_name) {
  return std::any_of(kValidLoaders.begin(), kValidLoaders.end(),
                     [module_name](const auto& item) { return item.first == module_name; });
}

inline std::string GetModuleName(const std::string& path) {
  {
    DataReader reader{std::string(path)};
    std::string module_name = reader.GetModuleName();
    if (!module_name.empty()) {
      return module_name;
    }
  }

  const std::string filename = std::filesystem::path(path).filename().string();

  for (const auto& [module_name, log_pattern] : kValidLoaders) {
    const std::regex regex_pattern(log_pattern);
    if (std::regex_match(filename, regex_pattern)) {
      return module_name;
    }
  }

  return {};
}

template <typename F>
DeferredCall<F>::DeferredCall(F&& f) : func_(std::forward<F>(f)) {}

template <typename F>
DeferredCall<F>::DeferredCall(DeferredCall&& that) noexcept : func_(std::move(that.func_)), is_owner_(that.is_owner_) {
  that.is_owner_ = false;
}

template <typename F>
DeferredCall<F>::~DeferredCall() {
  Execute();
}

template <typename F>
bool DeferredCall<F>::Cancel() {
  const auto was_owner = is_owner_;
  is_owner_ = false;
  return was_owner;
}

template <typename F>
bool DeferredCall<F>::Execute() {
  const auto was_owner = is_owner_;

  if (is_owner_) {
    is_owner_ = false;
    func_();
  }

  return was_owner;
}

template <typename F>
DeferredCall<F> defer(F&& f) {
  return DeferredCall<F>(std::forward<F>(f));
}

FileWriter::FileWriter() = default;

FileWriter::~FileWriter() { close(); }

// NOLINTNEXTLINE(readability-identifier-naming)
void FileWriter::open(std::string& filename, std::ios_base::openmode) {
  const DataWriterFileMode file_mode = DataWriterFileMode::kExclusive;
  int counter = 0;
  bool exclusive = (file_mode == DataWriterFileMode::kExclusive) || (file_mode == DataWriterFileMode::kAutoIncrement);

  auto ext_index = filename.find_last_of('.');
  std::string ext = ".ace";
  if (ext_index != std::string::npos) {
    ext = filename.substr(ext_index);
    filename = filename.substr(0, ext_index);
  }
  std::string file_to_open = filename + ext;

  do {
    file_ = fopen(file_to_open.c_str(), exclusive ? "wx" : "w");
    if (file_) {
      filename = file_to_open;
    }
    if (!file_ && file_mode == DataWriterFileMode::kAutoIncrement) {
      ++counter;
      file_to_open = filename + "_" + std::to_string(counter) + ext;
    } else {
      break;
    }
  } while (true);

  if (!file_ && errno != 0) {
    LOG(ERROR) << "Error Code: " << errno << ": " << strerror(errno) << " - Error opening file: " << file_to_open;
  }
}

// NOLINTNEXTLINE(readability-identifier-naming)
bool FileWriter::is_open() const { return file_ != nullptr; }

// NOLINTNEXTLINE(readability-identifier-naming)
void FileWriter::write(const void* data, size_t size) {
  if (!is_open()) {
    return;
  }
  fwrite(data, size, 1, file_);
}

// NOLINTNEXTLINE(readability-identifier-naming)
void FileWriter::close() {
  if (file_) {
    fclose(file_);
    file_ = nullptr;
  }
}

DataBuffer::DataBuffer(const std::string& filename, std::string rt_profile, size_t const& buffer_size,
                       size_t const& flush_size)
  : rt_profile_(std::move(rt_profile)), flush_size_(flush_size), buffer_size_(buffer_size), mem_(2 * flush_size) {
  auto* const env_val = std::getenv("ACE_DATALOGGER_DISABLE");
  is_running_ = env_val == nullptr || std::strcmp(env_val, "1") != 0;
  if constexpr (kEnableThreading) {
    worker_ = std::thread([this]() { WorkerThread(); });
  }
  if (!is_running_) {
    LOG(ERROR) << "Datalogger is disabled!" << std::endl;
    return;
  }
  Open(filename);
}

void DataBuffer::Open(const std::string& filename) {
  if (filename.empty()) {
    LOG(WARNING) << "Log filename is empty, no logs will be generated for this instance" << std::endl;
    return;
  }
  if (mem_.size() < buffer_size_) {
    mem_.resize(buffer_size_);
  }

  filename_ = filename;
  file_.open(filename_, std::ios::out | std::ios::binary);
  if (!file_.is_open()) {
    throw std::runtime_error(std::string("Cannot open file for writing!: ") + filename);
  }
  LOG(INFO) << "Started logging into file:\n - \"" << filename << "\"" << std::endl;
}

void DataBuffer::SetFileName(const std::string& filename) { split_queue_.emplace_back(GetUsedSize(), filename); }

std::string DataBuffer::GetFileName() const { return filename_; }
DataBuffer::~DataBuffer() {
  // LOG(INFO) << "Datalogger is destrying its buffer!" << std::endl;
  Release();
}

void DataBuffer::Release() {
  is_running_ = false;
  if (worker_.joinable()) {
    worker_.join();
  }
  file_.close();
}

size_t DataBuffer::GetCapacity() const { return mem_.size(); }

size_t DataBuffer::GetUsedSize() const {
  // FIXME(cv3d): These two loads shall be cached as well
  const auto write_idx_local = write_idx_.load(std::memory_order_relaxed);
  const auto read_idx_local = read_idx_.load(std::memory_order_relaxed);
  return ((GetCapacity() + write_idx_local) - read_idx_local) % GetCapacity();
}

size_t DataBuffer::GetRemainingSize() const { return GetCapacity() - (GetUsedSize() + 1); }

bool DataBuffer::Write(const char* data, size_t const& len) {
  if (!is_running_) {
    return false;
  }
  if constexpr (kEnableThreading) {
    std::unique_lock<std::mutex> lock(write_mutex_);
    write_event_.wait(lock, [&] { return GetRemainingSize() >= len; });

    auto write_idx_local = write_idx_.load(std::memory_order_relaxed);
    const size_t linear_len = GetCapacity() - write_idx_local;
    if (linear_len < len) {
      std::memcpy(mem_.data() + write_idx_local, data, linear_len);
      std::memcpy(mem_.data(), data + linear_len, len - linear_len);
    } else {
      std::memcpy(mem_.data() + write_idx_local, data, len);
    }
    write_idx_local = (write_idx_local + len) % GetCapacity();
    write_idx_.store(write_idx_local, std::memory_order_release);
    if (GetUsedSize() >= flush_size_) {
      read_event_.notify_one();
    }
  } else {
    file_.write(data, static_cast<std::streamsize>(len));
  }
  return true;
}

bool DataBuffer::Dump(size_t const& len) {
  if (len == 0) {
    return false;
  }

  if (!split_queue_.empty()) {
    // SetFileName was called, so close previous file and open a new one
    // Since we use the stack, access split_queue_ in a LIFO order
    auto [curr_len, curr_filename] = split_queue_.back();
    split_queue_.pop_back();
    Dump(curr_len);
    file_.close();
    Open(curr_filename);
    Dump(len - curr_len);
    return true;
  }

  auto read_idx_local = read_idx_.load(std::memory_order_relaxed);
  if (read_idx_local > write_idx_) {
    const size_t linear_len = GetCapacity() - read_idx_local;
    file_.write(mem_.data() + read_idx_local, static_cast<std::streamsize>(linear_len));
    file_.write(mem_.data(), static_cast<std::streamsize>(len - linear_len));
  } else {
    file_.write(mem_.data() + read_idx_local, static_cast<std::streamsize>(len));
  }
  read_idx_local = (read_idx_local + len) % GetCapacity();
  read_idx_.store(read_idx_local, std::memory_order_release);
  write_event_.notify_one();
  return true;
}

void DataBuffer::WorkerThread() {
  // LOG(INFO) << "Datalogger starting..." << std::endl;
  if (!rt_profile_.empty()) {
    ace_rt_profiles::ProfileManager::GetInstance().Apply(rt_profile_);
  }
  // LOG(INFO) << "Datalogger started!" << std::endl;

  while (is_running_) {
    std::unique_lock<std::mutex> lock(read_mutex_);
    if (!read_event_.wait_for(lock, std::chrono::milliseconds(50), [&] { return GetUsedSize() >= flush_size_; })) {
      continue;
    }
    Dump(GetUsedSize());
  }

  // LOG(INFO) << "Datalogger stopping..." << std::endl;
  Dump(GetUsedSize());  // Flush remaining data
  // LOG(INFO) << "Datalogger stopped!" << std::endl;
}

// Forward declarations
inline DataWriter& operator<<(DataWriter& writer, std::string const& str);
inline DataReader& operator>>(DataReader& reader, std::string& str);

DataWriter::DataWriter(std::string const& filename, nlohmann::json metadata, const std::string& rt_profile,
                       size_t const& buffer_size, size_t const& flush_size)
  : metadata_(std::move(metadata)), buffer_(filename, rt_profile, buffer_size, flush_size) {
  for (const auto& key : kRequiredMetadataKeys) {
    if (!metadata_.contains(key)) {
      throw std::runtime_error("Metadata must contain a \"" + static_cast<std::string>(key) + "\" key");
    }
  }
  if (!IsLoaderDefined(static_cast<std::string>(metadata_["module_name"]))) {
    throw std::runtime_error("module_name=\"" + static_cast<std::string>(metadata_["module_name"]) +
                             "\" must be registered in kValidLoaders. "
                             "See \"ace_loggers/datalogger.hpp\" for details.");
  }

  auto handler = [&](int) { Release(); };
  utils::SignalHandler::AddSignalHandler(handler_id_, SIGKILL, handler);
  utils::SignalHandler::AddSignalHandler(handler_id_, SIGTERM, handler);

  WriterHeader();
}

DataWriter& DataWriter::WriterHeader() {
  Write(kMagic.data(), kMagic.size());

  *this << static_cast<std::string>(metadata_.dump());
  return *this;
}

void DataWriter::SetFileName(const std::string& filename) {
  buffer_.SetFileName(filename);
  WriterHeader();
}

std::string DataWriter::GetFileName() const { return buffer_.GetFileName(); }

const nlohmann::json& DataWriter::GetMetadata() const { return metadata_; }

DataReader::DataReader(std::string const& filename) : file_size_(std::filesystem::file_size(filename)) {
  file_.open(filename, std::ios::in | std::ios::binary);
  if (!file_.is_open()) {
    throw std::runtime_error("Cannot open file for reading!");
  }

  std::string magic(kMagic.size(), ' ');
  Read(magic.data(), magic.size());
  if (magic == kMagic) {
    std::string metadata_str;
    *this >> metadata_str;
    if (nlohmann::json::accept(metadata_str)) {
      metadata_ = nlohmann::json::parse(metadata_str);
    } else {
      // Before using JSON for meta-data, we only stored version as string
      metadata_["module_name"] = "";
      metadata_["log_version"] = metadata_str;
    }
  } else {
    // Before introducing kMagic, there was no meta-data at all
    file_.seekg(0);
    metadata_["module_name"] = "";
    metadata_["log_version"] = "1.0.0";
  }
}

DataWriter::~DataWriter() {
  // LOG(INFO) << "Datalogger is destrying its writer!" << std::endl;
  utils::SignalHandler::RemoveSignalHandler(handler_id_, SIGKILL);
  utils::SignalHandler::RemoveSignalHandler(handler_id_, SIGTERM);

  Release();
}

DataReader::~DataReader() { Release(); }

void DataWriter::Release() { buffer_.Release(); }

void DataReader::Release() {
  if (file_.is_open()) {
    file_.close();
  }
}

bool DataReader::IsEndOfFile() const {
  // Will not use file_.eof() as it does not set unless an error occures
  auto pos = GetPosition();
  return pos < 0 || pos >= static_cast<std::streamoff>(file_size_);
}

int64_t DataReader::GetPosition() const {
  // const_cast<> as tellg() might updates the object
  return const_cast<std::ifstream*>(&file_)->tellg();
}

DataWriter& DataWriter::Write(const char* data, size_t const& len) {
  buffer_.Write(data, len);
  return *this;
}

DataReader& DataReader::Read(char* data, size_t const& len) {
  file_.read(data, static_cast<std::streamsize>(len));
  return *this;
}

template <typename T>
DataWriter& DataWriter::Write(T const& data) {
  return (*this << data);
}

template <typename T>
DataReader& DataReader::Read(T& data) {
  return (*this >> data);
}

template <typename T>
T DataReader::ReadAndReturn() {
  T data;
  *this >> data;
  return data;
}

const nlohmann::json& DataReader::GetMetadata() const { return metadata_; }

std::string DataReader::GetModuleName() const { return metadata_["module_name"]; }

std::string DataReader::GetLogVersion() const { return metadata_["log_version"]; }

// #define LOG_STRUCT_LEN

template <typename T>
inline DataWriter& operator<<(DataWriter& writer, T const& data) {
  const size_t len = sizeof(data);
#ifdef LOG_STRUCT_LEN
  writer.Write(reinterpret_cast<const char*>(&len), sizeof(len));
#endif
  writer.Write(reinterpret_cast<const char*>(&data), len);
  return writer;
}

template <typename T>
inline DataReader& operator>>(DataReader& reader, T& data) {
#ifdef LOG_STRUCT_LEN
  size_t len;
  reader.Read(reinterpret_cast<char*>(&len), sizeof(len));
  if (sizeof(data) != len) {
    throw std::runtime_error("Cannot read structure due to size mismatch!");
  }
#endif
  reader.Read(reinterpret_cast<char*>(&data), sizeof(data));
  return reader;
}

template <typename T>
inline std::shared_ptr<DataWriter> operator<<(std::shared_ptr<DataWriter> const& writer, T const& data) {
  if (writer != nullptr) {
    *writer << data;
  }
  return writer;
}

template <typename T>
inline DataReader& operator>>(std::shared_ptr<DataReader> const& reader, T& data) {
  return (*reader >> data);
}

template <typename T>
inline DataWriter& operator<<(std::unique_ptr<DataWriter> const& writer, T const& data) {
  return (*writer << data);
}

template <typename T>
inline DataReader& operator>>(std::unique_ptr<DataReader> const& reader, T& data) {
  return (*reader >> data);
}

template <typename T>
inline DataWriter& operator<<(DataWriter& writer, std::shared_ptr<const T> const& data) {
  return (writer << *data);
}

template <typename T>
inline DataWriter& operator<<(DataWriter& writer, std::shared_ptr<T> const& data) {
  return (writer << static_cast<std::shared_ptr<const T>>(data));
}

template <typename T>
inline DataReader& operator>>(DataReader& reader, std::shared_ptr<T> const& data) {
  return (reader >> *data);
}

template <typename T>
inline DataReader& operator>>(DataReader& reader, std::shared_ptr<T>& data) {
  return (reader >> *data);
}

template <typename T>
inline DataWriter& operator<<(DataWriter& writer, std::unique_ptr<const T> const& data) {
  return (writer << *data);
}

template <typename T>
inline DataWriter& operator<<(DataWriter& writer, std::unique_ptr<T> const& data) {
  return (writer << static_cast<std::unique_ptr<const T>>(data));
}

template <typename T>
inline DataReader& operator>>(DataReader& reader, std::unique_ptr<T> const& data) {
  return (reader >> *data);
}

template <typename T>
inline DataReader& operator>>(DataReader& reader, std::unique_ptr<T>& data) {
  return (reader >> *data);
}

inline DataWriter& operator<<(DataWriter& writer, std::string const& str) {
  writer << str.size();
  writer.Write(str.data(), str.size());
  return writer;
}

inline DataReader& operator>>(DataReader& reader, std::string& str) {
  size_t len;
  reader >> len;
  if (len > 1024 * 1024 * 100) {
    LOG(ERROR) << "Memory allocation error with len=" << len << "!" << std::endl;
    len = 0;
  }
  str.resize(len);
  reader.Read(str.data(), len);
  return reader;
}

template <typename T, int Rows, int Cols>
inline DataWriter& operator<<(DataWriter& writer, Eigen::Matrix<T, Rows, Cols> const& data) {
  if (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic) {
    writer << data.rows() << data.cols();
  }
  writer.Write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
  return writer;
}

template <typename T, int Rows, int Cols>
inline DataReader& operator>>(DataReader& reader, Eigen::Matrix<T, Rows, Cols>& data) {
  if (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic) {
    Eigen::Index rows;
    Eigen::Index cols;
    reader >> rows;
    reader >> cols;
    if (rows * cols * sizeof(T) > 1024 * 1024 * 100) {
      LOG(ERROR) << "Memory allocation error with rows=" << rows << ", cols=" << cols << "!" << std::endl;
      if (Rows == Eigen::Dynamic) {
        rows = 0;
      }
      if (Cols == Eigen::Dynamic) {
        cols = 0;
      }
    }
    data.resize(rows, cols);
  }
  reader.Read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
  return reader;
}

template <typename T>
inline DataWriter& operator<<(DataWriter& writer, std::vector<T> const& vec) {
  writer << vec.size();
  for (const auto& data : vec) {
    writer << data;
  }
  return writer;
}

template <typename T>
inline DataReader& operator>>(DataReader& reader, std::vector<T>& vec) {
  size_t len;
  reader >> len;
  if (len * sizeof(T) > 1024 * 1024 * 100) {
    LOG(ERROR) << "Memory allocation error with len=" << len << "!" << std::endl;
    len = 0;
  }
  vec.resize(len);
  for (size_t i = 0; i < len; ++i) {
    reader >> vec[i];
  }
  return reader;
}

template <typename T1, typename T2>
inline DataWriter& operator<<(DataWriter& writer, std::pair<T1, T2> const& pair) {
  writer << pair.first;
  writer << pair.second;
  return writer;
}

template <typename T1, typename T2>
inline DataReader& operator>>(DataReader& reader, std::pair<T1, T2>& pair) {
  reader >> pair.first;
  reader >> pair.second;
  return reader;
}

template <typename... Tp>
inline DataWriter& operator<<(DataWriter& writer, std::tuple<Tp...> const& tuple) {
  std::apply([&writer](auto&&... elem) { ((writer << elem), ...); }, tuple);
  return writer;
}

template <typename... Tp>
inline DataReader& operator>>(DataReader& reader, std::tuple<Tp...>& tuple) {
  std::apply([&reader](auto&&... elem) { ((reader >> elem), ...); }, tuple);
  return reader;
}

}  // namespace datalogger
