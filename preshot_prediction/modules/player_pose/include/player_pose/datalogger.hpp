// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include "ace_loggers/ace_loggers.hpp"
#include "player_pose/PlayerFeatures.hpp"

namespace player_pose::datalogger {

class DataWriter : public ::datalogger::DataWriter {
 public:
  using SharedPtr = std::shared_ptr<DataWriter>;

  explicit DataWriter(std::string const& file_name, const std::vector<int>& player_ids)
    : ::datalogger::DataWriter(
        file_name, {{"module_name", "player_pose"}, {"log_version", "1.1.0"}, {"player_ids", player_ids}}) {}
};

class DataReader : public ::datalogger::DataReader {
 public:
  using SharedPtr = std::shared_ptr<DataReader>;

  explicit DataReader(std::string const& file_name) : ::datalogger::DataReader(file_name) {}
};

}  // namespace player_pose::datalogger

namespace datalogger {
enum class PlayerPoseDataType : size_t {
  kPlayerBBoxDetection,
  kPlayerPoseDetection,
  kPlayerPoseEstimate,
  kPlayerFrameFeatures
};
DataWriter& operator<<(DataWriter& writer, const perception::PlayerPoseDetection& data) {
  writer << data.sequence_number;
  writer << data.camera_index;
  writer << data.bbox;
  writer << data.keypoints;
  return writer;
}

DataReader& operator>>(DataReader& reader, perception::PlayerPoseDetection& data) {
  reader >> data.sequence_number;
  reader >> data.camera_index;
  reader >> data.bbox;
  reader >> data.keypoints;
  return reader;
}
DataWriter& operator<<(DataWriter& writer, const perception::PlayerPoseEstimate& data) {
  writer << data.player_id;
  writer << data.keypoints;
  writer << data.projection_error;
  writer << data.confidences;
  return writer;
}

DataReader& operator>>(DataReader& reader, perception::PlayerPoseEstimate& data) {
  reader >> data.player_id;
  reader >> data.keypoints;
  reader >> data.projection_error;
  reader >> data.confidences;
  return reader;
}
DataWriter& operator<<(DataWriter& writer, const perception::PlayerFrameFeatures& data) {
  writer << data.sequence_number;

  size_t count = std::count_if(data.features.begin(), data.features.end(),
                               [](const perception::PlayerPoseDetection::SharedPtr& f) { return f != nullptr; });

  writer << count;
  for (const auto& v : data.features) {
    if (v) {
      writer << *v;
    }
  }
  writer << data.estimated_players.size();
  for (const auto& v : data.estimated_players) {
    writer << v;
  }

  return writer;
}

DataReader& operator>>(DataReader& reader, perception::PlayerFrameFeatures& data) {
  reader >> data.sequence_number;

  size_t size = 0;
  reader >> size;
  data.features.resize(size);
  for (size_t i = 0; i < size; ++i) {
    data.features[i] = std::make_shared<perception::PlayerPoseDetection>();
    reader >> data.features[i];
  }
  reader >> size;
  data.estimated_players.resize(size);
  for (size_t i = 0; i < size; ++i) {
    data.estimated_players[i] = std::make_shared<perception::PlayerPoseEstimate>();
    reader >> data.estimated_players[i];
  }
  return reader;
}
}  // namespace datalogger
