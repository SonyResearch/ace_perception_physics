// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <chrono>

#include "ace_loggers/datalogger_tests.hpp"
#include "ace_yaml/ace_yaml.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "racket_pose_estimation/datalogger.hpp"

namespace racket_pose_estimation::datalogger {
TEST_F(DataLoggerTests, TestStateEstimator) {
  const std::string file_name = ::datalogger::ConstructFullLogName("test_racket_pose_estimation_datalogger");
  const std::vector<int> racket_ids = {0};

  perception::RacketEstimatedPose est_pose;
  perception::RacketPoseFeatures pose_features;
  perception::RacketFrameFeatures frame_features;

  est_pose.racket_id = 1;
  est_pose.position = Eigen::Vector3f(-3, 0, 1);
  est_pose.orientation = Eigen::Quaternionf(0, 0.7, 0.7, 0.7);
  est_pose.iterations_count = 100;
  est_pose.orientation_confidence = 0.1;
  est_pose.orientation_error = 0.5;
  est_pose.reprojection_err = 7;

  pose_features.sequence_number = 3030;
  pose_features.camera_index = 0;
  pose_features.confidence = 0.7;
  pose_features.center = Eigen::Vector3i(50, 70, 80);
  pose_features.bbox = Eigen::Vector4i(-1, 0, 1, 1);
  pose_features.keypoints.resize(4);
  pose_features.keypoints[0] = Eigen::Vector3f(-1, 0, 1);
  pose_features.keypoints[1] = Eigen::Vector3f(-1, 1, 1);
  pose_features.keypoints[2] = Eigen::Vector3f(1, 0, 1);
  pose_features.keypoints[3] = Eigen::Vector3f(1, 1, 1);

  frame_features.sequence_number = 3030;
  frame_features.estimated_rackets.resize(1);
  frame_features.estimated_rackets[0] = std::make_shared<perception::RacketEstimatedPose>();
  *frame_features.estimated_rackets[0].get() = est_pose;
  frame_features.features.resize(1);
  frame_features.features[0] = std::make_shared<perception::RacketPoseFeatures>();
  *frame_features.features[0].get() = pose_features;

  {
    std::remove(file_name.c_str());
    DataWriter datawriter(file_name, racket_ids);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1; i++) {
      datawriter << est_pose;
      datawriter << pose_features;
      datawriter << frame_features;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << ms_float.count() << "ms\n";
    std::cout << "==================================================================================== writer done\n";
  }

  {
    DataReader reader(file_name);
    EXPECT_EQ(racket_ids, reader.GetMetadata()["racket_ids"]);

    perception::RacketEstimatedPose est_pose_read;
    perception::RacketPoseFeatures pose_features_read;
    perception::RacketFrameFeatures frame_features_read;

    for (int i = 0; i < 1; i++) {
      reader >> est_pose_read;
      EXPECT_EQ(est_pose_read.racket_id, est_pose.racket_id);
      EXPECT_EQ(est_pose_read.orientation, est_pose.orientation);
      EXPECT_EQ(est_pose_read.position, est_pose.position);
      EXPECT_EQ(est_pose_read.iterations_count, est_pose.iterations_count);
      EXPECT_EQ(est_pose_read.reprojection_err, est_pose.reprojection_err);
      EXPECT_EQ(est_pose_read.orientation_confidence, est_pose.orientation_confidence);
      EXPECT_EQ(est_pose_read.orientation_error, est_pose.orientation_error);

      reader >> pose_features_read;
      EXPECT_EQ(pose_features_read.sequence_number, pose_features.sequence_number);
      EXPECT_EQ(pose_features_read.camera_index, pose_features.camera_index);
      EXPECT_EQ(pose_features_read.center, pose_features.center);
      EXPECT_EQ(pose_features_read.confidence, pose_features.confidence);
      EXPECT_EQ(pose_features_read.bbox, pose_features.bbox);
      EXPECT_EQ(pose_features_read.keypoints.size(), pose_features.keypoints.size());
      for (size_t j = 0; j < pose_features.keypoints.size(); ++j) {
        EXPECT_EQ(pose_features_read.keypoints[j], pose_features.keypoints[j]);
      }

      reader >> frame_features_read;
      EXPECT_EQ(frame_features_read.sequence_number, frame_features_read.sequence_number);
      EXPECT_EQ(frame_features_read.features.size(), frame_features_read.features.size());
      EXPECT_EQ(frame_features_read.estimated_rackets.size(), frame_features_read.estimated_rackets.size());
      for (size_t j = 0; j < frame_features_read.features.size(); ++j) {
        EXPECT_EQ(frame_features_read.features[j]->bbox, frame_features.features[j]->bbox);
      }
      for (size_t j = 0; j < frame_features_read.estimated_rackets.size(); ++j) {
        EXPECT_EQ(frame_features_read.estimated_rackets[j]->position, frame_features.estimated_rackets[j]->position);
      }
    }
  }

  SUCCEED();
}
}  // namespace racket_pose_estimation::datalogger

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
