#include "gpism/execution_context.h"

#include <algorithm>

namespace gpism {

ExecutionContext& ExecutionContext::instance() {
  static ExecutionContext context;
  return context;
}

ExecutionContext::ExecutionContext() {
  compute_streams_.resize(1, nullptr);
  cudaStreamCreateWithFlags(&compute_streams_[0], cudaStreamNonBlocking);
}

ExecutionContext::~ExecutionContext() {
  for (auto stream : compute_streams_) {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
}

void ExecutionContext::configure(bool enable_cuda_graphs,
                                 int compute_stream_count) {
  cuda_graphs_enabled_ = enable_cuda_graphs;
  const int desired = std::max(1, compute_stream_count);
  const int current = static_cast<int>(compute_streams_.size());
  if (desired == current) {
    return;
  }
  if (desired > current) {
    compute_streams_.reserve(static_cast<std::size_t>(desired));
    for (int i = current; i < desired; ++i) {
      cudaStream_t stream = nullptr;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      compute_streams_.push_back(stream);
    }
    return;
  }
  for (int i = desired; i < current; ++i) {
    if (compute_streams_[static_cast<std::size_t>(i)] != nullptr) {
      cudaStreamDestroy(compute_streams_[static_cast<std::size_t>(i)]);
    }
  }
  compute_streams_.resize(static_cast<std::size_t>(desired), nullptr);
}

cudaStream_t ExecutionContext::compute_stream(int index) const {
  if (compute_streams_.empty()) {
    return nullptr;
  }
  const int n = static_cast<int>(compute_streams_.size());
  const int clamped = std::max(0, std::min(index, n - 1));
  return compute_streams_[static_cast<std::size_t>(clamped)];
}

}  // namespace gpism
