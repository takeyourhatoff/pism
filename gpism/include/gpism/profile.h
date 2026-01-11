#pragma once

#include <chrono>

#include "gpism/config.h"

#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gpism {

class Context;

class Profiler {
public:
  static bool enabled();
  static void add(const char* name, double milliseconds);
  static void report(const Context* context = nullptr);
};

class ScopedTimer {
public:
  explicit ScopedTimer(const char* name);
  ~ScopedTimer();

private:
  const char* name_;
  bool active_;
  std::chrono::steady_clock::time_point start_;
};

#if GPISM_HAVE_CUDA
class CudaEventTimer {
public:
  explicit CudaEventTimer(const char* name);
  ~CudaEventTimer();

private:
  const char* name_;
  bool active_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
#else
class CudaEventTimer {
public:
  explicit CudaEventTimer(const char*) {}
};
#endif

}  // namespace gpism
