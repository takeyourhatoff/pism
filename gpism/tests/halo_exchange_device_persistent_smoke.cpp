#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/field_sync.h"
#include "gpism/grid2d.h"
#include "gpism/halo_exchange.h"

#include <iostream>

#include <cuda_runtime.h>

int main(int argc, char** argv) {
#if !GPISM_HAVE_MPI
  (void)argc;
  (void)argv;
  std::cout << "halo_exchange_device_persistent_smoke skipped (MPI disabled)\n";
  return 0;
#else
  gpism::Context context(&argc, &argv);
  cudaSetDevice(context.device_id());

  if (!context.mpi_enabled() || context.size() < 2) {
    if (context.rank() == 0) {
      std::cout << "halo_exchange_device_persistent_smoke skipped "
                   "(requires >=2 MPI ranks)\n";
    }
    return 0;
  }
  if (!context.cuda_aware_mpi()) {
    if (context.rank() == 0) {
      std::cout << "halo_exchange_device_persistent_smoke skipped "
                   "(CUDA-aware MPI disabled)\n";
    }
    return 0;
  }

  const int mx_global = 8;
  const int my_global = 8;
  const int gw = 1;
  gpism::Grid2D grid(mx_global, my_global, 1.0, 1.0, gw, context.rank(),
                     context.size());
  gpism::Field2D<double> field(grid.local_mx(), grid.local_my(), gw);
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      field(i, j) = static_cast<double>(context.rank() + 1);
    }
  }
  gpism::sync_host_to_device(field);

  gpism::HaloExchange2D exchange;
  auto first =
      exchange.start_exchange(field, grid, context, gpism::HaloExchange2D::Mode::Device);
  if (!first.active || !first.use_device) {
    std::cerr << "expected active device halo exchange on rank "
              << context.rank() << "\n";
    return 1;
  }

  const double* send_west_1 = first.send_west_dev;
  const double* send_east_1 = first.send_east_dev;
  const double* send_south_1 = first.send_south_dev;
  const double* send_north_1 = first.send_north_dev;
  const double* recv_west_1 = first.recv_west_dev;
  const double* recv_east_1 = first.recv_east_dev;
  const double* recv_south_1 = first.recv_south_dev;
  const double* recv_north_1 = first.recv_north_dev;
  exchange.finish_exchange(first);

  auto second =
      exchange.start_exchange(field, grid, context, gpism::HaloExchange2D::Mode::Device);
  if (!second.active || !second.use_device) {
    std::cerr << "expected active device halo exchange on rank "
              << context.rank() << " (second)\n";
    return 1;
  }

  bool reused_any = false;
  auto check_reuse = [&](const double* a, const double* b, int neighbor) {
    if (neighbor < 0 || a == nullptr || b == nullptr) {
      return;
    }
    if (a == b) {
      reused_any = true;
    }
  };
  check_reuse(send_west_1, second.send_west_dev, second.west);
  check_reuse(send_east_1, second.send_east_dev, second.east);
  check_reuse(send_south_1, second.send_south_dev, second.south);
  check_reuse(send_north_1, second.send_north_dev, second.north);
  check_reuse(recv_west_1, second.recv_west_dev, second.west);
  check_reuse(recv_east_1, second.recv_east_dev, second.east);
  check_reuse(recv_south_1, second.recv_south_dev, second.south);
  check_reuse(recv_north_1, second.recv_north_dev, second.north);
  exchange.finish_exchange(second);

  if (!reused_any) {
    std::cerr << "expected persistent device halo buffers to be reused on rank "
              << context.rank() << "\n";
    return 1;
  }

  if (context.rank() == 0) {
    std::cout << "halo_exchange_device_persistent_smoke passed\n";
  }
  return 0;
#endif
}
