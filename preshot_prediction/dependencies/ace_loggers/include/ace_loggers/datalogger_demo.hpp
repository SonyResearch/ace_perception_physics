// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include "ace_loggers/ace_loggers.hpp"

namespace datalogger_demo {

struct HasPrimitivesStruct {
  using SharedPtr = std::shared_ptr<HasPrimitivesStruct>;

  HasPrimitivesStruct(int64_t const& var_int, double const& var_float) : var_int(var_int), var_float(var_float) {}
  HasPrimitivesStruct() = default;

  int64_t var_int = {0};
  double var_float = {0.0};
};

struct HasPointerStruct : HasPrimitivesStruct {
  using SharedPtr = std::shared_ptr<HasPointerStruct>;

  HasPointerStruct(int64_t const& var_int, double const& var_float, const std::string var_str)
    : HasPrimitivesStruct(var_int, var_float), var_str(std::move(var_str)) {}
  HasPointerStruct() = default;

  std::string var_str;
};

class DataWriter : public ::datalogger::DataWriter {
 public:
  using SharedPtr = std::shared_ptr<DataWriter>;

  explicit DataWriter(std::string const& file_name)
    : ::datalogger::DataWriter(file_name, {{"module_name", "ace_loggers"}, {"log_version", "1.0.0"}}) {}
};

class DataReader : public ::datalogger::DataReader {
 public:
  using SharedPtr = std::shared_ptr<DataReader>;

  explicit DataReader(std::string const& file_name) : ::datalogger::DataReader(file_name) {}
};

}  // namespace datalogger_demo

// Define how to read and write HasPointerStruct
namespace datalogger {
DataWriter& operator<<(DataWriter& writer, datalogger_demo::HasPointerStruct const& data) {
  writer << data.var_int;
  writer << data.var_float;
  writer << data.var_str;
  return writer;
}

DataReader& operator>>(DataReader& reader, datalogger_demo::HasPointerStruct& data) {
  reader >> data.var_int;
  reader >> data.var_float;
  reader >> data.var_str;
  return reader;
}
}  // namespace datalogger
