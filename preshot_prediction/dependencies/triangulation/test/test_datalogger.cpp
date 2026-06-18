// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "ace_loggers/datalogger_tests.hpp"
#include "triangulation/datalogger.hpp"

using namespace triangulation;              // NOLINT
using namespace triangulation::datalogger;  // NOLINT

TEST_F(DataLoggerTests, TestTriangulatedPoint) {
  const std::string file_name = "test.ace";
  TriangulatedPoint write_data;
  write_data.position = Eigen::Vector3f::Random();
  write_data.covariance = Eigen::Matrix3f::Random();

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, "", "");
    writer << write_data;
  }

  {
    DataReader reader(file_name);

    TriangulatedPoint read_data;
    reader >> read_data;
    EXPECT_EQ(write_data.position, read_data.position);
    EXPECT_EQ(write_data.covariance, read_data.covariance);
  }

  SUCCEED();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
