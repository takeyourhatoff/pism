#pragma once

#include <vector>

#include <cuda_runtime.h>

namespace gpism {

class ExecutionContext {
public:
  static ExecutionContext& instance();

  void configure(bool enable_cuda_graphs, int compute_stream_count);

  bool cuda_graphs_enabled() const { return cuda_graphs_enabled_; }
  int compute_stream_count() const {
    return static_cast<int>(compute_streams_.size());
  }

  cudaStream_t compute_stream(int index = 0) const;
  cudaStream_t default_stream() const { return compute_stream(0); }

private:
  ExecutionContext();
  ~ExecutionContext();

  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  bool cuda_graphs_enabled_ = true;
  std::vector<cudaStream_t> compute_streams_;
};

}  // namespace gpism
