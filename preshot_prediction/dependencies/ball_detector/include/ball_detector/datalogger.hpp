// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "ace_loggers/ace_loggers.hpp"
#include "ball_detector/ball_detector_opencv.hpp"

namespace ball_detector::datalogger {

class DataWriter : public ::datalogger::DataWriter {
 public:
  using SharedPtr = std::shared_ptr<DataWriter>;

  explicit DataWriter(std::string const& file_name, std::string const& camera_name,
                      std::string const& ball_detector_params)
    : ::datalogger::DataWriter(file_name, {{"module_name", "ball_detector"},
                                           {"log_version", "1.0.1"},
                                           {"camera_name", camera_name},
                                           {"ball_detector_params", ball_detector_params}}) {}
};

class DataReader : public ::datalogger::DataReader {
 public:
  using SharedPtr = std::shared_ptr<DataReader>;

  explicit DataReader(std::string const& file_name) : ::datalogger::DataReader(file_name) {}
};

}  // namespace ball_detector::datalogger

// Define how to read and write cv::Mat
namespace datalogger {
DataWriter& operator<<(DataWriter& writer, cv::Mat const& data) {
  std::vector<int> sizes(&data.size[0], &data.size[0] + data.dims);
  writer << sizes << data.type();

  const cv::Mat tmp = data.isContinuous() ? data : data.clone();
  writer.Write(reinterpret_cast<const char*>(tmp.data), tmp.total() * tmp.elemSize());

  return writer;
}

DataReader& operator>>(DataReader& reader, cv::Mat& data) {
  std::vector<int> sizes;
  int type;
  reader >> sizes >> type;

  cv::Mat tmp(sizes, type);
  reader.Read(reinterpret_cast<char*>(tmp.data), tmp.total() * tmp.elemSize());
  tmp.copyTo(data);

  return reader;
}
}  // namespace datalogger
