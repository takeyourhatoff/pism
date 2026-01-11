#pragma once

#include <vector>

#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif
#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gpism {

#if GPISM_HAVE_MPI
template <typename T>
struct MpiType;

template <>
struct MpiType<float> {
  static MPI_Datatype value() { return MPI_FLOAT; }
};

template <>
struct MpiType<double> {
  static MPI_Datatype value() { return MPI_DOUBLE; }
};
#endif

class HaloExchange2D {
public:
  template <typename T>
  void exchange(Field2D<T>& field, const Grid2D& grid, const Context& context) {
    if (!context.mpi_enabled()) {
      return;
    }
#if GPISM_HAVE_MPI
    const int gw = field.ghost_width();
    if (gw <= 0) {
      return;
    }

    const int local_mx = field.local_mx();
    const int local_my = field.local_my();

    const int west = grid.neighbor_west();
    const int east = grid.neighbor_east();
    const int south = grid.neighbor_south();
    const int north = grid.neighbor_north();

    const int tag_west = 100;
    const int tag_east = 101;
    const int tag_south = 102;
    const int tag_north = 103;

#if GPISM_HAVE_CUDA
    const bool use_device =
        context.cuda_aware_mpi() && field.device_data() != nullptr;
#else
    const bool use_device = false;
#endif

#if GPISM_HAVE_CUDA
    if (use_device) {
      const int stride = field.stride();

      auto idx = [&](int i, int j) {
        return (j + gw) * stride + (i + gw);
      };

      const std::size_t west_east_count = static_cast<std::size_t>(gw) *
                                          static_cast<std::size_t>(local_my);
      const std::size_t north_south_count = static_cast<std::size_t>(gw) *
                                            static_cast<std::size_t>(local_mx);

      T* send_west = nullptr;
      T* send_east = nullptr;
      T* send_south = nullptr;
      T* send_north = nullptr;
      T* recv_west = nullptr;
      T* recv_east = nullptr;
      T* recv_south = nullptr;
      T* recv_north = nullptr;

      auto alloc = [](std::size_t count, T** ptr) {
        if (count == 0) {
          *ptr = nullptr;
          return;
        }
        cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T));
      };

      auto release = [](T** ptr) {
        if (*ptr) {
          cudaFree(*ptr);
          *ptr = nullptr;
        }
      };

      if (west >= 0) {
        alloc(west_east_count, &send_west);
        alloc(west_east_count, &recv_west);
      }
      if (east >= 0) {
        alloc(west_east_count, &send_east);
        alloc(west_east_count, &recv_east);
      }
      if (south >= 0) {
        alloc(north_south_count, &send_south);
        alloc(north_south_count, &recv_south);
      }
      if (north >= 0) {
        alloc(north_south_count, &send_north);
        alloc(north_south_count, &recv_north);
      }

      const T* device_ptr = field.device_data();

      if (west >= 0) {
        const T* src = device_ptr + idx(0, 0);
        cudaMemcpy2D(send_west, gw * sizeof(T), src, stride * sizeof(T),
                     gw * sizeof(T), local_my, cudaMemcpyDeviceToDevice);
      }
      if (east >= 0) {
        const T* src = device_ptr + idx(local_mx - gw, 0);
        cudaMemcpy2D(send_east, gw * sizeof(T), src, stride * sizeof(T),
                     gw * sizeof(T), local_my, cudaMemcpyDeviceToDevice);
      }
      if (south >= 0) {
        const T* src = device_ptr + idx(0, 0);
        cudaMemcpy2D(send_south, local_mx * sizeof(T), src, stride * sizeof(T),
                     local_mx * sizeof(T), gw, cudaMemcpyDeviceToDevice);
      }
      if (north >= 0) {
        const T* src = device_ptr + idx(0, local_my - gw);
        cudaMemcpy2D(send_north, local_mx * sizeof(T), src, stride * sizeof(T),
                     local_mx * sizeof(T), gw, cudaMemcpyDeviceToDevice);
      }

      std::vector<MPI_Request> requests;
      requests.reserve(8);

      auto post_recv = [&](int neighbor, int tag, T* buffer, std::size_t count) {
        if (neighbor < 0 || count == 0) {
          return;
        }
        MPI_Request req{};
        MPI_Irecv(buffer, static_cast<int>(count), MpiType<T>::value(), neighbor,
                  tag, MPI_COMM_WORLD, &req);
        requests.push_back(req);
      };

      auto post_send = [&](int neighbor, int tag, T* buffer, std::size_t count) {
        if (neighbor < 0 || count == 0) {
          return;
        }
        MPI_Request req{};
        MPI_Isend(buffer, static_cast<int>(count), MpiType<T>::value(), neighbor,
                  tag, MPI_COMM_WORLD, &req);
        requests.push_back(req);
      };

      post_recv(west, tag_east, recv_west, west_east_count);
      post_recv(east, tag_west, recv_east, west_east_count);
      post_recv(south, tag_north, recv_south, north_south_count);
      post_recv(north, tag_south, recv_north, north_south_count);

      post_send(west, tag_west, send_west, west_east_count);
      post_send(east, tag_east, send_east, west_east_count);
      post_send(south, tag_south, send_south, north_south_count);
      post_send(north, tag_north, send_north, north_south_count);

      if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                    MPI_STATUSES_IGNORE);
      }

      T* device_dest = field.device_data();

      if (west >= 0) {
        T* dst = device_dest + idx(-gw, 0);
        cudaMemcpy2D(dst, stride * sizeof(T), recv_west, gw * sizeof(T),
                     gw * sizeof(T), local_my, cudaMemcpyDeviceToDevice);
      }
      if (east >= 0) {
        T* dst = device_dest + idx(local_mx, 0);
        cudaMemcpy2D(dst, stride * sizeof(T), recv_east, gw * sizeof(T),
                     gw * sizeof(T), local_my, cudaMemcpyDeviceToDevice);
      }
      if (south >= 0) {
        T* dst = device_dest + idx(0, -gw);
        cudaMemcpy2D(dst, stride * sizeof(T), recv_south, local_mx * sizeof(T),
                     local_mx * sizeof(T), gw, cudaMemcpyDeviceToDevice);
      }
      if (north >= 0) {
        T* dst = device_dest + idx(0, local_my);
        cudaMemcpy2D(dst, stride * sizeof(T), recv_north, local_mx * sizeof(T),
                     local_mx * sizeof(T), gw, cudaMemcpyDeviceToDevice);
      }

      release(&send_west);
      release(&send_east);
      release(&send_south);
      release(&send_north);
      release(&recv_west);
      release(&recv_east);
      release(&recv_south);
      release(&recv_north);

      return;
    }
#endif

    std::vector<T> send_west;
    std::vector<T> send_east;
    std::vector<T> send_south;
    std::vector<T> send_north;
    std::vector<T> recv_west;
    std::vector<T> recv_east;
    std::vector<T> recv_south;
    std::vector<T> recv_north;

    if (west >= 0) {
      send_west.resize(static_cast<size_t>(gw) * local_my);
      recv_west.resize(static_cast<size_t>(gw) * local_my);
    }
    if (east >= 0) {
      send_east.resize(static_cast<size_t>(gw) * local_my);
      recv_east.resize(static_cast<size_t>(gw) * local_my);
    }
    if (south >= 0) {
      send_south.resize(static_cast<size_t>(gw) * local_mx);
      recv_south.resize(static_cast<size_t>(gw) * local_mx);
    }
    if (north >= 0) {
      send_north.resize(static_cast<size_t>(gw) * local_mx);
      recv_north.resize(static_cast<size_t>(gw) * local_mx);
    }

    if (west >= 0) {
      int idx = 0;
      for (int j = 0; j < local_my; ++j) {
        for (int i = 0; i < gw; ++i) {
          send_west[idx++] = field(i, j);
        }
      }
    }
    if (east >= 0) {
      int idx = 0;
      for (int j = 0; j < local_my; ++j) {
        for (int i = local_mx - gw; i < local_mx; ++i) {
          send_east[idx++] = field(i, j);
        }
      }
    }
    if (south >= 0) {
      int idx = 0;
      for (int j = 0; j < gw; ++j) {
        for (int i = 0; i < local_mx; ++i) {
          send_south[idx++] = field(i, j);
        }
      }
    }
    if (north >= 0) {
      int idx = 0;
      for (int j = local_my - gw; j < local_my; ++j) {
        for (int i = 0; i < local_mx; ++i) {
          send_north[idx++] = field(i, j);
        }
      }
    }

    std::vector<MPI_Request> requests;
    requests.reserve(8);

    auto post_recv = [&](int neighbor, int tag, std::vector<T>& buffer) {
      if (neighbor < 0) {
        return;
      }
      MPI_Request req{};
      MPI_Irecv(buffer.data(), static_cast<int>(buffer.size()), MpiType<T>::value(),
                neighbor, tag, MPI_COMM_WORLD, &req);
      requests.push_back(req);
    };

    auto post_send = [&](int neighbor, int tag, std::vector<T>& buffer) {
      if (neighbor < 0) {
        return;
      }
      MPI_Request req{};
      MPI_Isend(buffer.data(), static_cast<int>(buffer.size()), MpiType<T>::value(),
                neighbor, tag, MPI_COMM_WORLD, &req);
      requests.push_back(req);
    };

    post_recv(west, tag_east, recv_west);
    post_recv(east, tag_west, recv_east);
    post_recv(south, tag_north, recv_south);
    post_recv(north, tag_south, recv_north);

    post_send(west, tag_west, send_west);
    post_send(east, tag_east, send_east);
    post_send(south, tag_south, send_south);
    post_send(north, tag_north, send_north);

    if (!requests.empty()) {
      MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }

    if (west >= 0) {
      int idx = 0;
      for (int j = 0; j < local_my; ++j) {
        for (int i = -gw; i < 0; ++i) {
          field(i, j) = recv_west[idx++];
        }
      }
    }
    if (east >= 0) {
      int idx = 0;
      for (int j = 0; j < local_my; ++j) {
        for (int i = local_mx; i < local_mx + gw; ++i) {
          field(i, j) = recv_east[idx++];
        }
      }
    }
    if (south >= 0) {
      int idx = 0;
      for (int j = -gw; j < 0; ++j) {
        for (int i = 0; i < local_mx; ++i) {
          field(i, j) = recv_south[idx++];
        }
      }
    }
    if (north >= 0) {
      int idx = 0;
      for (int j = local_my; j < local_my + gw; ++j) {
        for (int i = 0; i < local_mx; ++i) {
          field(i, j) = recv_north[idx++];
        }
      }
    }
#else
    (void)field;
    (void)grid;
    (void)context;
#endif
  }

  template <typename T>
  void exchange(FieldStag2D<T>& field, const Grid2D& grid, const Context& context) {
    exchange(field.component(0), grid, context);
    exchange(field.component(1), grid, context);
  }

private:
};

}  // namespace gpism
