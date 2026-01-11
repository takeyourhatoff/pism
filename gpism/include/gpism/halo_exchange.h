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
