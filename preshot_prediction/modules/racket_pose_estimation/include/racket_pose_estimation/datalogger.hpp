// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "ace_loggers/ace_loggers.hpp"
#include "racket_pose_estimation/RacketFeatures.hpp"

namespace racket_pose_estimation::datalogger {

class DataWriter : public ::datalogger::DataWriter {
 public:
  using SharedPtr = std::shared_ptr<DataWriter>;

  explicit DataWriter(std::string const& file_name, const std::vector<int>& racket_ids)
    : ::datalogger::DataWriter(
        file_name, {{"module_name", "racket_pose_estimation"}, {"log_version", "1.2.0"}, {"racket_ids", racket_ids}}) {}
};

class DataReader : public ::datalogger::DataReader {
 public:
  using SharedPtr = std::shared_ptr<DataReader>;

  explicit DataReader(std::string const& file_name) : ::datalogger::DataReader(file_name) {}
};

}  // namespace racket_pose_estimation::datalogger

namespace datalogger {
enum class RacketPoseEstimationDataType : size_t { kRacketPoseFeatures, kRacketEstimatedPose, kRacketFrameFeatures };

DataWriter& operator<<(DataWriter& writer, const perception::RacketPoseFeatures& data) {
  writer << data.sequence_number;
  writer << data.camera_index;
  writer << data.confidence;
  writer << data.bbox;
  writer << data.center;
  writer << data.keypoints;
  return writer;
}

DataReader& operator>>(DataReader& reader, perception::RacketPoseFeatures& data) {
  reader >> data.sequence_number;
  reader >> data.camera_index;
  reader >> data.confidence;
  reader >> data.bbox;
  reader >> data.center;
  reader >> data.keypoints;
  return reader;
}

DataWriter& operator<<(DataWriter& writer, const perception::RacketFrameFeatures& data) {
  writer << data.sequence_number;

  size_t count = std::count_if(data.features.begin(), data.features.end(),
                               [](const perception::RacketPoseFeatures::SharedPtr& f) { return f != nullptr; });

  writer << count;
  for (const auto& v : data.features) {
    if (v) {
      writer << *v;
    }
  }
  writer << data.estimated_rackets.size();
  for (const auto& v : data.estimated_rackets) {
    writer << *v;
  }

  return writer;
}

DataReader& operator>>(DataReader& reader, perception::RacketFrameFeatures& data) {
  reader >> data.sequence_number;

  size_t size = 0;
  reader >> size;
  data.features.resize(size);
  for (size_t i = 0; i < size; ++i) {
    data.features[i] = std::make_shared<perception::RacketPoseFeatures>();
    reader >> data.features[i];
  }
  reader >> size;
  data.estimated_rackets.resize(size);
  for (size_t i = 0; i < size; ++i) {
    data.estimated_rackets[i] = std::make_shared<perception::RacketEstimatedPose>();
    reader >> data.estimated_rackets[i];
  }
  return reader;
}
}  // namespace datalogger
