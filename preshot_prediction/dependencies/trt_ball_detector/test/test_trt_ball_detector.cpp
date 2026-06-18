// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include "trt_ball_detector/trt_ball_detector.hpp"
////
#include "gtest/gtest.h"
#include "test/root_dir.hpp"

namespace test {

class trt_ball_detectorTests : public ::testing::Test {};

// TODO: Rename TestDummy here to a meaningful name, e.g. TestMyAmazingFeature, TestFlightDynamics, ..etc
// Use EXPECT_FALSE, EXPECT_TRUE, EXPECT_LE, ..etc function to check for success conditions
TEST_F(trt_ball_detectorTests, TestDummy) {
  SUCCEED();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}  // namespace test
