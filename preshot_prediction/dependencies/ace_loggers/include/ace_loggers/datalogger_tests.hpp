// Confidential, Copyright 2025, Sony AI, All rights reserved.
#pragma once

#include <array>
#include <random>
#include <string>
#include <string_view>

#include "ace_loggers/ace_loggers.hpp"
#include "gtest/gtest.h"

// Testing support class
class DataLoggerTestsBase {
 public:
  explicit DataLoggerTestsBase() : random_number_generator_(random_device_()) {
    // Set up random number generator.
    random_number_generator_.seed(std::time(nullptr));
  }

 protected:
  float UniformSample(float low, float high) {
    std::uniform_real_distribution<float> uniform_(low, high);
    return uniform_(random_number_generator_);
  }

  size_t UniformIntSample(size_t lower, size_t upper) {
    std::uniform_int_distribution<size_t> uniform_int_(lower, upper);
    return uniform_int_(random_number_generator_);
  }

  float NormalSample(float mean, float stddev) {
    std::normal_distribution<float> sampler{mean, stddev};
    return sampler(random_number_generator_);
  }

  std::string RandomString(size_t len) {
    using namespace std::string_view_literals;
    static constexpr auto kAlphaNum =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"sv;

    std::string output;
    output.reserve(len);
    std::ranges::sample(kAlphaNum, std::back_inserter(output), len, random_number_generator_);
    return output;
  }

 private:
  std::random_device random_device_;
  std::mt19937 random_number_generator_;
};

// Testing support classes
class DataLoggerTests : public ::testing::Test, public DataLoggerTestsBase {};

using TestParamType = std::vector<std::pair<size_t, std::string>>;
class DataLoggerTestsWithParam : public ::testing::TestWithParam<TestParamType>, public DataLoggerTestsBase {};
