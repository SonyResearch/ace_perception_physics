// Confidential, Copyright 2026, Sony AI, All rights reserved.
#pragma once

#ifndef __clang_analysis__
#include <boost/date_time/local_time/local_time.hpp>
#include <boost/make_shared.hpp>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <string_view>
#include <utility>

namespace datalogger {
inline constexpr struct ParseTimeZoneT {
  static constexpr auto kTimeZones = []() {
    using std::string_view_literals::operator""sv;
    using Pair = std::pair<std::string_view, std::string_view>;
    return std::array{
      Pair{"JST"sv, "UTC+9"sv},
      Pair{"JST-9"sv, "UTC+9"sv},
      Pair{"Asia/Tokyo"sv, "UTC+9"sv},
      Pair{"Europe/Zurich"sv, "CET+1CEST,M3.5.0,M10.5.0/3"sv},
    };
  }();

  constexpr std::string_view operator()(std::string_view search) const noexcept {
    for (const auto& p : kTimeZones) {
      if (p.first == search) return p.second;
    }
    return search;
  }
  // No lint, treat this as a function instead of a constant
} ParseTimeZone;  // NOLINT

static_assert(ParseTimeZone("jst") == "jst", "case sensitive");
static_assert(ParseTimeZone("asia/tokyo") == "asia/tokyo", "Case sensitive");
static_assert(ParseTimeZone("jst ") == "jst ", "Space sensitive");
static_assert(ParseTimeZone(" jst ") == " jst ", "Space sensitive");
static_assert(ParseTimeZone(" jst") == " jst", "Space sensitive");
static_assert(ParseTimeZone("").empty(), "Empty string returns empty string");
static_assert(ParseTimeZone("abcd") == "abcd", "Returns argument when no map entry");
static_assert(ParseTimeZone("CET+1CEST,M3.5.0,M10.5.0/3") == "CET+1CEST,M3.5.0,M10.5.0/3");
namespace detail {
constexpr bool AllTimeZonesResolveCorrectly() {
  for (const auto& p : ParseTimeZoneT::kTimeZones) {
    if (ParseTimeZone(p.first) != p.second) return false;
  }
  return true;
}
}  // namespace detail
static_assert(detail::AllTimeZonesResolveCorrectly(), "All timezones found");

std::tm GetLocalTime(std::string_view time_zone);

#ifndef __clang_analysis__
// Boost is broken in later compilers due weird lexing/parsing issue

inline std::tm GetLocalTime(std::string_view time_zone) {
  const auto now = std::chrono::system_clock::now();
  const auto now_t = std::chrono::system_clock::to_time_t(now);
  const auto current_time = boost::posix_time::from_time_t(now_t);

  using boost::local_time::local_date_time;
  using boost::local_time::posix_time_zone;

  const auto stime_zone = std::string(datalogger::ParseTimeZone(time_zone));
  // exception will be thrown if stime_zone is invalid
  const auto pz_ptr = boost::make_shared<posix_time_zone>(stime_zone);
  const local_date_time local_time(current_time, std::move(pz_ptr));
  return boost::local_time::to_tm(local_time);
}
#endif
}  // namespace datalogger
