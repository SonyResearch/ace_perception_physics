// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "ace_loggers/ace_loggers.hpp"
#include "triangulation/triangulation.hpp"

namespace triangulation::datalogger {

class DataWriter : public ::datalogger::DataWriter {
 public:
  using SharedPtr = std::shared_ptr<DataWriter>;

  explicit DataWriter(std::string const& file_name, std::string const& triangulation_params,
                      std::string const& camera_calib_params)
    : ::datalogger::DataWriter(file_name, {{"module_name", "triangulation"},
                                           {"log_version", "1.0.1"},
                                           {"triangulation_params", triangulation_params},
                                           {"camera_calib_params", camera_calib_params}}) {}
};

class DataReader : public ::datalogger::DataReader {
 public:
  using SharedPtr = std::shared_ptr<DataReader>;

  explicit DataReader(std::string const& file_name) : ::datalogger::DataReader(file_name) {}
};

}  // namespace triangulation::datalogger

// Define how to read and write TriangulatedPoint
namespace datalogger {
DataWriter& operator<<(DataWriter& writer, triangulation::TriangulatedPoint const& data) {
  writer << data.position;
  writer << data.covariance;
  return writer;
}

DataReader& operator>>(DataReader& reader, triangulation::TriangulatedPoint& data) {
  reader >> data.position;
  reader >> data.covariance;
  return reader;
}
}  // namespace datalogger
