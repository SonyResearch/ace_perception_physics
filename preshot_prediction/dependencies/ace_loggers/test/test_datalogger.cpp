// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include <chrono>

#include "ace_loggers/datalogger_tests.hpp"

using namespace datalogger;

TEST_F(DataLoggerTests, TestInt) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const std::string file_name = "test.ace";

  const int write_int = UniformIntSample(0, 1000);

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, metadata);
    writer << write_int;
  }

  {
    int read_int;
    {
      DataReader reader(file_name);
      EXPECT_EQ(reader.GetModuleName(), metadata["module_name"]);
      EXPECT_EQ(reader.GetLogVersion(), metadata["log_version"]);
      reader >> read_int;
    }

    EXPECT_EQ(write_int, read_int);
  }

  SUCCEED();
}

TEST_F(DataLoggerTests, TestString) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const std::string file_name = "test.ace";

  const std::string write_str = RandomString(UniformIntSample(12, 32));

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, metadata);
    writer << write_str;
  }

  {
    std::string read_str;
    {
      DataReader reader(file_name);
      EXPECT_EQ(reader.GetModuleName(), metadata["module_name"]);
      EXPECT_EQ(reader.GetLogVersion(), metadata["log_version"]);
      reader >> read_str;
    }

    EXPECT_EQ(write_str, read_str);
  }

  SUCCEED();
}

TEST_F(DataLoggerTests, TestTimePoint) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const std::string file_name = "test.ace";

  const auto write_timepoint = std::chrono::high_resolution_clock::now();

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, metadata);
    writer << write_timepoint;
  }

  {
    DataReader reader(file_name);
    EXPECT_EQ(reader.GetModuleName(), metadata["module_name"]);
    EXPECT_EQ(reader.GetLogVersion(), metadata["log_version"]);

    std::chrono::time_point<std::chrono::high_resolution_clock> read_timepoint;
    reader >> read_timepoint;

    EXPECT_EQ(write_timepoint, read_timepoint);
  }

  SUCCEED();
}

TEST_F(DataLoggerTests, TestVector) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const std::string file_name = "test.ace";

  const size_t vec_len = 256;

  std::vector<int> write_vec;
  write_vec.reserve(vec_len);
  for (size_t i = 0; i < vec_len; ++i) {
    write_vec.emplace_back(UniformIntSample(0, 1000));
  }

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, metadata);
    writer << write_vec;
  }

  {
    std::vector<int> read_vec;
    {
      DataReader reader(file_name);
      EXPECT_EQ(reader.GetModuleName(), metadata["module_name"]);
      EXPECT_EQ(reader.GetLogVersion(), metadata["log_version"]);
      reader >> read_vec;
    }

    for (size_t i = 0; i < read_vec.size(); ++i) {
      EXPECT_EQ(write_vec[i], read_vec[i]);
    }
  }

  SUCCEED();
}

TEST_F(DataLoggerTests, TestPair) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const std::string file_name = "test.ace";

  std::pair<int, float> write_pair;
  write_pair.first = UniformIntSample(0, 1000);
  write_pair.second = NormalSample(0, 1000);

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, metadata);
    writer << write_pair;
  }

  {
    DataReader reader(file_name);
    EXPECT_EQ(reader.GetModuleName(), metadata["module_name"]);
    EXPECT_EQ(reader.GetLogVersion(), metadata["log_version"]);

    std::pair<int, float> read_pair;
    reader >> read_pair;

    EXPECT_EQ(write_pair.first, read_pair.first);
    EXPECT_EQ(write_pair.second, read_pair.second);
  }

  SUCCEED();
}

TEST_F(DataLoggerTests, TestTuple) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const std::string file_name = "test.ace";

  std::tuple<int, float, std::string> write_tuple;
  std::get<0>(write_tuple) = UniformIntSample(0, 1000);
  std::get<1>(write_tuple) = NormalSample(0, 1000);
  std::get<2>(write_tuple) = RandomString(UniformIntSample(12, 32));

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, metadata);
    writer << write_tuple;
  }

  {
    DataReader reader(file_name);
    EXPECT_EQ(reader.GetModuleName(), metadata["module_name"]);
    EXPECT_EQ(reader.GetLogVersion(), metadata["log_version"]);

    std::tuple<int, float, std::string> read_tuple;
    reader >> read_tuple;

    EXPECT_EQ(std::get<0>(write_tuple), std::get<0>(read_tuple));
    EXPECT_EQ(std::get<1>(write_tuple), std::get<1>(read_tuple));
    EXPECT_EQ(std::get<2>(write_tuple), std::get<2>(read_tuple));
  }

  SUCCEED();
}

TEST_F(DataLoggerTests, TestDefer) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const std::string file_name = "test.ace";

  const int write_data1 = UniformIntSample(0, 1000);
  const int write_data2 = UniformIntSample(0, 1000);
  const int write_data3 = UniformIntSample(0, 1000);
  const int write_data4 = UniformIntSample(0, 1000);

  {
    std::remove(file_name.c_str());
    DataWriter writer(file_name, metadata);

    auto logger_hook1 = ::datalogger::defer([&writer, write_data1] { writer << write_data1; });
    auto logger_hook2 = ::datalogger::defer([&writer, write_data2] { writer << write_data2; });
    auto logger_hook3 = ::datalogger::defer([&writer, write_data3] { writer << write_data3; });
    auto logger_hook4 = ::datalogger::defer([&writer, write_data4] { writer << write_data4; });

    logger_hook1.Execute();
    logger_hook3.Cancel();
    // logger_hook4's deferred block is called first
    // logger_hook2's deferred block is called next
  }

  {
    int read_data1;
    int read_data2;
    int read_data4;
    {
      DataReader reader(file_name);
      EXPECT_EQ(reader.GetModuleName(), metadata["module_name"]);
      EXPECT_EQ(reader.GetLogVersion(), metadata["log_version"]);
      reader >> read_data1;
      reader >> read_data4;
      reader >> read_data2;
    }

    EXPECT_EQ(write_data1, read_data1);
    EXPECT_EQ(write_data2, read_data2);
    EXPECT_EQ(write_data4, read_data4);
  }

  SUCCEED();
}

TEST_P(DataLoggerTestsWithParam, TestSetFileName) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const auto params = GetParam();

  std::vector<std::string> write_strs;
  for (auto param : params) {
    write_strs.push_back(RandomString(param.first));
    if (!param.second.empty()) {
      std::remove(param.second.c_str());
    }
  }

  {
    DataWriter writer(params[0].second, metadata);
    writer << write_strs[0];
    for (size_t i = 1; i < params.size(); ++i) {
      auto start_time = std::chrono::high_resolution_clock::now();
      std::atomic_signal_fence(std::memory_order_seq_cst);
      writer.SetFileName(params[i].second);
      std::atomic_signal_fence(std::memory_order_seq_cst);
      auto stop_time = std::chrono::high_resolution_clock::now();
      writer << write_strs[i];
      auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();
      std::cout << "[" << i << "] Took " << elapsed_time << " µs" << std::endl;
    }
  }

  for (size_t i = 0; i < params.size(); ++i) {
    if (params[i].second.empty()) {
      continue;
    }
    std::string read_str;
    {
      DataReader reader(params[i].second);
      EXPECT_EQ(reader.GetModuleName(), metadata["module_name"]);
      EXPECT_EQ(reader.GetLogVersion(), metadata["log_version"]);
      reader >> read_str;
    }

    EXPECT_EQ(write_strs[i], read_str);
  }

  SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(FileNames, DataLoggerTestsWithParam,
                         ::testing::Values(TestParamType{{1024, "test1.ace"}, {1024, "test2.ace"}, {1024, "test3.ace"}},
                                           TestParamType{{1024, "test1.ace"}, {1024, ""}, {1024, "test3.ace"}},
                                           TestParamType{{1024, "test1.ace"}, {0, ""}, {0, ""}},
                                           TestParamType{{1024, ""}, {1024, "test2.ace"}, {1024, ""}},
                                           TestParamType{{0, ""}, {1024, "test2.ace"}, {1024, "test3.ace"}}));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
