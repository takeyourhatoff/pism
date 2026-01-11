#include "gpism/context.h"

#include <cstdlib>

#include "gpism/config.h"
#include "gpism/profile.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace gpism {

Context::Context(int* argc, char*** argv)
    : owns_mpi_(false), rank_(0), size_(1), device_id_(0), device_count_(1) {
#if GPISM_HAVE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(argc, argv);
    owns_mpi_ = true;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
#else
  (void)argc;
  (void)argv;
#endif

  const char* device_count_env = std::getenv("GPISM_NUM_DEVICES");
  if (device_count_env) {
    int parsed = std::atoi(device_count_env);
    if (parsed > 0) {
      device_count_ = parsed;
    }
  }

  device_id_ = (device_count_ > 0) ? (rank_ % device_count_) : 0;

  const char* device_env = std::getenv("GPISM_DEVICE");
  if (device_env) {
    int parsed = std::atoi(device_env);
    if (parsed >= 0) {
      device_id_ = parsed;
    }
  }
}

Context::~Context() {
  Profiler::report(this);
#if GPISM_HAVE_MPI
  if (owns_mpi_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
#endif
}

int Context::rank() const { return rank_; }

int Context::size() const { return size_; }

bool Context::mpi_enabled() const {
#if GPISM_HAVE_MPI
  return true;
#else
  return false;
#endif
}

int Context::device_id() const { return device_id_; }

int Context::device_count() const { return device_count_; }

bool Context::cuda_aware_mpi() const {
#if GPISM_HAVE_MPI && GPISM_HAVE_CUDA
  const char* env = std::getenv("GPISM_CUDA_AWARE_MPI");
  if (!env) {
    return false;
  }
  if (std::atoi(env) == 0) {
    return false;
  }
#ifdef MPIX_CUDA_AWARE_SUPPORT
  return MPIX_Query_cuda_support() != 0;
#else
  return true;
#endif
#else
  return false;
#endif
}

Context::NeighborRanks Context::neighbors_2d(int dims_x, int dims_y) const {
  NeighborRanks neighbors{};
  neighbors.dims_x = dims_x;
  neighbors.dims_y = dims_y;
  neighbors.coord_x = (dims_x > 0) ? (rank_ % dims_x) : 0;
  neighbors.coord_y = (dims_x > 0) ? (rank_ / dims_x) : 0;

  neighbors.west = (neighbors.coord_x > 0) ? rank_ - 1 : -1;
  neighbors.east =
      (neighbors.coord_x + 1 < dims_x) ? rank_ + 1 : -1;
  neighbors.south =
      (neighbors.coord_y > 0) ? rank_ - dims_x : -1;
  neighbors.north =
      (neighbors.coord_y + 1 < dims_y) ? rank_ + dims_x : -1;

  return neighbors;
}

}  // namespace gpism
