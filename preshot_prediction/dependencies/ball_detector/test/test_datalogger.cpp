// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ace_loggers/datalogger_tests.hpp"
#include "ball_detector/datalogger.hpp"

using namespace ball_detector;
using namespace ball_detector::datalogger;

TEST_F(DataLoggerTests, TestStructuresLogging) {
  const std::string file_name = "test.ace";
  const std::string camera_name = "aps0000000";

  std::vector<std::pair<cv::Mat, cv::Rect>> write_vec;
  for (size_t i = 0; i < 200; ++i) {
    cv::Mat write_mat(128, 128, CV_8U);
    cv::randu(write_mat, 0, 255);

    cv::Rect write_rect;
    write_rect.x = UniformIntSample(0, 100);
    write_rect.y = UniformIntSample(0, 100);
    write_rect.width = UniformIntSample(10, 100);
    write_rect.height = UniformIntSample(10, 100);

    write_vec.emplace_back(write_mat, write_rect);
  }

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, camera_name, "");
    writer << write_vec;
  }

  {
    DataReader reader(file_name);
    EXPECT_EQ(camera_name, reader.GetMetadata()["camera_name"]);

    std::vector<std::pair<cv::Mat, cv::Rect>> read_vec;
    reader >> read_vec;

    EXPECT_EQ(write_vec.size(), read_vec.size());
    for (size_t i = 0; i < read_vec.size(); ++i) {
      auto& write_mat = write_vec[i].first;
      auto& write_rect = write_vec[i].second;
      auto& read_mat = read_vec[i].first;
      auto& read_rect = read_vec[i].second;

      cv::Mat diff;
      cv::compare(write_mat, read_mat, diff, cv::CMP_NE);
      EXPECT_EQ(cv::countNonZero(diff), 0);

      EXPECT_EQ(write_rect.x, read_rect.x);
      EXPECT_EQ(write_rect.y, read_rect.y);
      EXPECT_EQ(write_rect.width, read_rect.width);
      EXPECT_EQ(write_rect.height, read_rect.height);
    }
  }

  SUCCEED();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
