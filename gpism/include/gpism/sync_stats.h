#pragma once

#include <cstddef>

namespace gpism {

class SyncStats {
public:
  static void enable(bool on);
  static bool enabled();
  static void reset();
  static void record_h2d(std::size_t bytes);
  static void record_d2h(std::size_t bytes);
  static void record_h2d_misc(std::size_t bytes);
  static void record_d2h_misc(std::size_t bytes);
  static std::size_t h2d_calls();
  static std::size_t d2h_calls();
  static std::size_t h2d_bytes();
  static std::size_t d2h_bytes();
  static std::size_t h2d_misc_calls();
  static std::size_t d2h_misc_calls();
  static std::size_t h2d_misc_bytes();
  static std::size_t d2h_misc_bytes();
};

}  // namespace gpism
