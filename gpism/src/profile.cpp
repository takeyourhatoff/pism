#include "gpism/profile.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gpism/context.h"

namespace gpism {
namespace {

struct ProfileEntry {
  double total_ms = 0.0;
  double max_ms = 0.0;
  std::size_t count = 0;
};

std::vector<std::pair<std::string, ProfileEntry>>& entries() {
  static std::vector<std::pair<std::string, ProfileEntry>> data;
  return data;
}

ProfileEntry* find_entry(const char* name) {
  auto& data = entries();
  for (auto& entry : data) {
    if (entry.first == name) {
      return &entry.second;
    }
  }
  data.emplace_back(name, ProfileEntry{});
  return &data.back().second;
}

bool env_enabled(const char* var) {
  const char* value = std::getenv(var);
  if (!value) {
    return false;
  }
  return std::atoi(value) != 0;
}

bool profile_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached != 0;
  }
  cached = env_enabled("GPISM_PROFILE") ? 1 : 0;
  return cached != 0;
}

bool profile_all_ranks() {
  static int cached = -1;
  if (cached >= 0) {
    return cached != 0;
  }
  cached = env_enabled("GPISM_PROFILE_ALL_RANKS") ? 1 : 0;
  return cached != 0;
}

}  // namespace

bool Profiler::enabled() { return profile_enabled(); }

void Profiler::add(const char* name, double milliseconds) {
  if (!profile_enabled() || !name) {
    return;
  }
  ProfileEntry* entry = find_entry(name);
  entry->total_ms += milliseconds;
  entry->count += 1;
  entry->max_ms = std::max(entry->max_ms, milliseconds);
}

void Profiler::report(const Context* context) {
  if (!profile_enabled()) {
    return;
  }
  if (entries().empty()) {
    return;
  }
  if (context && context->mpi_enabled() && context->rank() != 0 &&
      !profile_all_ranks()) {
    return;
  }

  auto data = entries();
  std::sort(data.begin(), data.end(),
            [](const auto& a, const auto& b) {
              return a.second.total_ms > b.second.total_ms;
            });

  const int rank = context ? context->rank() : 0;
  std::cout << "gpism profile (rank " << rank << "):\n";
  std::cout << std::left << std::setw(36) << "  name"
            << std::right << std::setw(8) << "count"
            << std::setw(12) << "total_ms"
            << std::setw(12) << "avg_ms"
            << std::setw(12) << "max_ms" << "\n";
  for (const auto& item : data) {
    const auto& entry = item.second;
    const double avg_ms =
        (entry.count > 0) ? entry.total_ms / entry.count : 0.0;
    std::cout << std::left << std::setw(36) << ("  " + item.first)
              << std::right << std::setw(8) << entry.count
              << std::setw(12) << std::fixed << std::setprecision(3)
              << entry.total_ms << std::setw(12) << avg_ms << std::setw(12)
              << entry.max_ms << "\n";
  }
}

ScopedTimer::ScopedTimer(const char* name)
    : name_(name),
      active_(Profiler::enabled()),
      start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
  if (!active_) {
    return;
  }
  const auto end = std::chrono::steady_clock::now();
  const auto delta =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start_);
  Profiler::add(name_, delta.count());
}

#if GPISM_HAVE_CUDA
CudaEventTimer::CudaEventTimer(const char* name)
    : name_(name), active_(Profiler::enabled()), start_{}, stop_{} {
  if (!active_) {
    return;
  }
  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
  cudaEventRecord(start_, 0);
}

CudaEventTimer::~CudaEventTimer() {
  if (!active_) {
    return;
  }
  cudaEventRecord(stop_, 0);
  cudaEventSynchronize(stop_);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start_, stop_);
  Profiler::add(name_, static_cast<double>(ms));
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}
#endif

}  // namespace gpism
