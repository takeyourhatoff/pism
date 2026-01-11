#include <cmath>
#include <iostream>

#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"
#include "gpism/halo_exchange.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif

namespace {

bool check_ghosts(const gpism::Field2D<double>& field, const gpism::Grid2D& grid,
                  double offset, double sentinel) {
  const int gw = field.ghost_width();
  const int local_mx = field.local_mx();
  const int local_my = field.local_my();

  if (gw <= 0) {
    return true;
  }

  if (grid.neighbor_west() >= 0) {
    const double expected = static_cast<double>(grid.neighbor_west()) + offset;
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
    const double expected = static_cast<double>(grid.neighbor_east()) + offset;
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
    const double expected = static_cast<double>(grid.neighbor_south()) + offset;
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
    const double expected = static_cast<double>(grid.neighbor_north()) + offset;
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

  gpism::Grid2D grid(8, 8, 1.0, 1.0, 1, context.rank(), context.size());
  gpism::Field2D<double> field(grid.local_mx(), grid.local_my(), grid.ghost_width());
  gpism::FieldStag2D<double> stag(grid.local_mx(), grid.local_my(), grid.ghost_width());

  const double sentinel = -1.0;
  field.fill(sentinel);
  stag.fill(sentinel);
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      field(i, j) = static_cast<double>(context.rank());
      stag(i, j, 0) = static_cast<double>(context.rank()) + 10.0;
      stag(i, j, 1) = static_cast<double>(context.rank()) + 20.0;
    }
  }

  gpism::HaloExchange2D exchange;
  exchange.exchange(field, grid, context);
  exchange.exchange(stag, grid, context);

  bool local_ok =
      check_ghosts(field, grid, 0.0, sentinel) &&
      check_ghosts(stag.component(0), grid, 10.0, sentinel) &&
      check_ghosts(stag.component(1), grid, 20.0, sentinel);

#if !GPISM_HAVE_MPI
  if (local_ok) {
    std::cout << "Halo exchange smoke test (single rank): OK\n";
  } else {
    std::cout << "Halo exchange smoke test (single rank): FAILED\n";
  }
  return local_ok ? 0 : 1;
#else
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
