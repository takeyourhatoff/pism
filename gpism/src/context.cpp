#include "gpism/context.h"

#include <cstdlib>

#include "gpism/config.h"

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

}  // namespace gpism
