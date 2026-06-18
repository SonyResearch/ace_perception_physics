// Confidential, Copyright 2025, Sony AI, All rights reserved.

#include "ace_yaml/ace_yaml.hpp"
#include "gtest/gtest.h"
#include "test/root_dir.hpp"

namespace ace_yaml::test {

class AceYamlTests : public ::testing::Test {};

TEST_F(AceYamlTests, TestLoadFile) {
  YAML::Node params = ace_yaml::LoadFile(kRootDir / "test/test_params/level03_01.yaml");

  EXPECT_FALSE(params[ace_yaml::kInheritanceNodeName]);
  EXPECT_EQ(params["param1"].as<std::string>(), "level03_01");
  EXPECT_EQ(params["param2"].as<std::string>(), "level03_01");
  EXPECT_EQ(params["param3"].as<std::string>(), "level02_01");
  EXPECT_EQ(params["param4"].as<std::string>(), "level01_02");
  EXPECT_EQ(params["param5"]["x"].as<std::string>(), "level01_02");
  EXPECT_EQ(params["param5"]["y"].as<std::string>(), "level01_01");
  EXPECT_EQ(params["param5"]["z"].as<std::string>(), "level01_02");
  EXPECT_TRUE(params["param6"].IsNull());
  EXPECT_TRUE(params["param7"].IsNull());
  EXPECT_TRUE(params["param8"]["foo"].IsNull());

  SUCCEED();
}
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}  // namespace ace_yaml::test
