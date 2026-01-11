#include <cmath>
#include <iostream>

#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/field2d.h"
#include "gpism/grid2d.h"
#include "gpism/halo_exchange.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace {

bool check_ghosts(const gpism::Field2D<double>& field, const gpism::Grid2D& grid) {
  const int gw = field.ghost_width();
  const int local_mx = field.local_mx();
  const int local_my = field.local_my();

  if (gw <= 0) {
    return true;
  }

  const double sentinel = -1.0;

  if (grid.neighbor_west() >= 0) {
    const double expected = static_cast<double>(grid.neighbor_west());
    for (int j = 0; j < local_my; ++j) {
      for (int i = -gw; i < 0; ++i) {
        if (std::fabs(field(i, j) - expected) > 0.0) {
          return false;
        }
      }
    }
  } else {
    for (int j = 0; j < local_my; ++j) {
      for (int i = -gw; i < 0; ++i) {
        if (std::fabs(field(i, j) - sentinel) > 0.0) {
          return false;
        }
      }
    }
  }

  if (grid.neighbor_east() >= 0) {
    const double expected = static_cast<double>(grid.neighbor_east());
    for (int j = 0; j < local_my; ++j) {
      for (int i = local_mx; i < local_mx + gw; ++i) {
        if (std::fabs(field(i, j) - expected) > 0.0) {
          return false;
        }
      }
    }
  } else {
    for (int j = 0; j < local_my; ++j) {
      for (int i = local_mx; i < local_mx + gw; ++i) {
        if (std::fabs(field(i, j) - sentinel) > 0.0) {
          return false;
        }
      }
    }
  }

  if (grid.neighbor_south() >= 0) {
    const double expected = static_cast<double>(grid.neighbor_south());
    for (int j = -gw; j < 0; ++j) {
      for (int i = 0; i < local_mx; ++i) {
        if (std::fabs(field(i, j) - expected) > 0.0) {
          return false;
        }
      }
    }
  } else {
    for (int j = -gw; j < 0; ++j) {
      for (int i = 0; i < local_mx; ++i) {
        if (std::fabs(field(i, j) - sentinel) > 0.0) {
          return false;
        }
      }
    }
  }

  if (grid.neighbor_north() >= 0) {
    const double expected = static_cast<double>(grid.neighbor_north());
    for (int j = local_my; j < local_my + gw; ++j) {
      for (int i = 0; i < local_mx; ++i) {
        if (std::fabs(field(i, j) - expected) > 0.0) {
          return false;
        }
      }
    }
  } else {
    for (int j = local_my; j < local_my + gw; ++j) {
      for (int i = 0; i < local_mx; ++i) {
        if (std::fabs(field(i, j) - sentinel) > 0.0) {
          return false;
        }
      }
    }
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  gpism::Context context(&argc, &argv);

#if !GPISM_HAVE_MPI
  std::cout << "MPI not enabled; skipping halo test.\n";
  return 0;
#else
  gpism::Grid2D grid(8, 8, 1.0, 1.0, 1, context.rank(), context.size());
  gpism::Field2D<double> field(grid.local_mx(), grid.local_my(), grid.ghost_width());

  const double sentinel = -1.0;
  field.fill(sentinel);
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      field(i, j) = static_cast<double>(context.rank());
    }
  }

  gpism::HaloExchange2D exchange;
  exchange.exchange(field, grid, context);

  bool local_ok = check_ghosts(field, grid);
  int local_val = local_ok ? 1 : 0;
  int global_val = 0;
  MPI_Allreduce(&local_val, &global_val, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

  if (context.rank() == 0) {
    if (global_val) {
      std::cout << "Halo exchange smoke test: OK\n";
    } else {
      std::cout << "Halo exchange smoke test: FAILED\n";
    }
  }

  return global_val ? 0 : 1;
#endif
}
