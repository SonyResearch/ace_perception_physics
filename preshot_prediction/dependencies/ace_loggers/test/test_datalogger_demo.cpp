// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ace_loggers/datalogger_demo.hpp"
#include "ace_loggers/datalogger_tests.hpp"

using namespace datalogger_demo;

TEST_F(DataLoggerTests, TestPrimitive) {
  const std::string file_name = "test.ace";
  const HasPrimitivesStruct write_data(UniformIntSample(0, 1000), NormalSample(0, 100));

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name);
    writer << write_data;
  }

  {
    DataReader reader(file_name);

    HasPrimitivesStruct read_data;
    reader >> read_data;
    EXPECT_EQ(write_data.var_int, read_data.var_int);
    EXPECT_EQ(write_data.var_float, read_data.var_float);
  }

  SUCCEED();
}

TEST_F(DataLoggerTests, TestOverriding) {
  const std::string file_name = "test.ace";
  const HasPointerStruct write_data(UniformIntSample(0, 1000), NormalSample(0, 100),
                                    RandomString(UniformIntSample(12, 32)));

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name);
    writer << write_data;
  }

  {
    HasPointerStruct read_data;
    {
      DataReader reader(file_name);
      reader >> read_data;
    }

    EXPECT_EQ(write_data.var_int, read_data.var_int);
    EXPECT_EQ(write_data.var_float, read_data.var_float);
    EXPECT_EQ(write_data.var_str, read_data.var_str);
  }

  SUCCEED();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
