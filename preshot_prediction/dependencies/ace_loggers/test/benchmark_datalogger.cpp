// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <benchmark/benchmark.h>

#include "ace_loggers/ace_loggers.hpp"

using namespace datalogger;

static void BM_WriteString(benchmark::State& state) {
  const nlohmann::json metadata = {{"module_name", "ace_loggers"}, {"log_version", "0.0.0"}};
  const std::string file_name = "test.ace";

  std::remove(file_name.c_str());
  DataWriter writer(file_name, metadata);

  std::string write_str;
  write_str.resize(state.range(0), 'x');

  for (auto _ : state) {
    for (auto written = 0; written < (1 << 22); written += write_str.size()) {
      writer << write_str;
    }
  }
}
BENCHMARK(BM_WriteString)->Arg(1 << 4)->Arg(1 << 6)->Arg(1 << 8)->Arg(1 << 10)->Arg(1 << 12);

BENCHMARK_MAIN();
