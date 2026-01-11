#include "gpism/context.h"

#include "gpism/config.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace gpism {

Context::Context(int* argc, char*** argv)
    : owns_mpi_(false), rank_(0), size_(1) {
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

}  // namespace gpism
