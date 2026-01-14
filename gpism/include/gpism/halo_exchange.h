#pragma once

#include <vector>

#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/device_policy.h"
#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"
#include "gpism/profile.h"

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

template <>
struct MpiType<int> {
  static MPI_Datatype value() { return MPI_INT; }
};
#endif

class HaloExchange2D {
public:
  enum class Mode { Auto, Host, Device };

  template <typename T>
  struct ExchangeHandle {
    bool active = false;
    bool use_device = false;
    Field2D<T>* field = nullptr;
    int gw = 0;
    int local_mx = 0;
    int local_my = 0;
    int stride = 0;
    int west = -1;
    int east = -1;
    int south = -1;
    int north = -1;
#if GPISM_HAVE_MPI
    std::vector<MPI_Request> requests;
#endif
#if GPISM_HAVE_CUDA
    T* send_west_dev = nullptr;
    T* send_east_dev = nullptr;
    T* send_south_dev = nullptr;
    T* send_north_dev = nullptr;
    T* recv_west_dev = nullptr;
    T* recv_east_dev = nullptr;
    T* recv_south_dev = nullptr;
    T* recv_north_dev = nullptr;
#endif
    struct HostBuffer {
      T* ptr = nullptr;
      std::size_t count = 0;
      std::vector<T> storage;
#if GPISM_HAVE_CUDA
      bool pinned = false;
#endif
      void allocate(std::size_t n) {
        count = n;
        if (n == 0) {
          return;
        }
#if GPISM_HAVE_CUDA
        if (device_enabled()) {
          cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                                          n * sizeof(T), cudaHostAllocDefault);
          if (err == cudaSuccess) {
            pinned = true;
            return;
          }
          ptr = nullptr;
          pinned = false;
        }
#endif
        storage.resize(n);
        ptr = storage.data();
      }
      void release() {
#if GPISM_HAVE_CUDA
        if (pinned && ptr) {
          cudaFreeHost(ptr);
        }
        pinned = false;
#endif
        storage.clear();
        ptr = nullptr;
        count = 0;
      }
    };

    HostBuffer send_west;
    HostBuffer send_east;
    HostBuffer send_south;
    HostBuffer send_north;
    HostBuffer recv_west;
    HostBuffer recv_east;
    HostBuffer recv_south;
    HostBuffer recv_north;
  };

  template <typename T>
  struct StagExchangeHandle {
    ExchangeHandle<T> u;
    ExchangeHandle<T> v;
  };

  template <typename T>
  ExchangeHandle<T> start_exchange(Field2D<T>& field, const Grid2D& grid,
                                   const Context& context,
                                   Mode mode = Mode::Auto) {
    ExchangeHandle<T> handle{};
    if (!context.mpi_enabled()) {
      return handle;
    }
#if GPISM_HAVE_MPI
    const int gw = field.ghost_width();
    if (gw <= 0) {
      return handle;
    }

    handle.active = true;
    handle.field = &field;
    handle.gw = gw;
    handle.local_mx = field.local_mx();
    handle.local_my = field.local_my();
    handle.stride = field.stride();
    handle.west = grid.neighbor_west();
    handle.east = grid.neighbor_east();
    handle.south = grid.neighbor_south();
    handle.north = grid.neighbor_north();

    const int tag_west = 100;
    const int tag_east = 101;
    const int tag_south = 102;
    const int tag_north = 103;

#if GPISM_HAVE_CUDA
    const bool can_device =
        context.cuda_aware_mpi() && field.device_data() != nullptr;
    handle.use_device =
        (mode == Mode::Device) ? can_device :
        (mode == Mode::Host) ? false : can_device;
#else
    handle.use_device = false;
    (void)mode;
#endif

#if GPISM_HAVE_CUDA
    if (handle.use_device) {
      const int stride = handle.stride;
      auto idx = [&](int i, int j) {
        return (j + gw) * stride + (i + gw);
      };

      const std::size_t west_east_count = static_cast<std::size_t>(gw) *
                                          static_cast<std::size_t>(handle.local_my);
      const std::size_t north_south_count = static_cast<std::size_t>(gw) *
                                            static_cast<std::size_t>(handle.local_mx);

      auto alloc = [](std::size_t count, T** ptr) {
        if (count == 0) {
          *ptr = nullptr;
          return;
        }
        cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T));
      };

      if (handle.west >= 0) {
        alloc(west_east_count, &handle.send_west_dev);
        alloc(west_east_count, &handle.recv_west_dev);
      }
      if (handle.east >= 0) {
        alloc(west_east_count, &handle.send_east_dev);
        alloc(west_east_count, &handle.recv_east_dev);
      }
      if (handle.south >= 0) {
        alloc(north_south_count, &handle.send_south_dev);
        alloc(north_south_count, &handle.recv_south_dev);
      }
      if (handle.north >= 0) {
        alloc(north_south_count, &handle.send_north_dev);
        alloc(north_south_count, &handle.recv_north_dev);
      }

      const T* device_ptr = field.device_data();

      if (handle.west >= 0) {
        const T* src = device_ptr + idx(0, 0);
        cudaMemcpy2D(handle.send_west_dev, gw * sizeof(T), src,
                     stride * sizeof(T), gw * sizeof(T), handle.local_my,
                     cudaMemcpyDeviceToDevice);
      }
      if (handle.east >= 0) {
        const T* src = device_ptr + idx(handle.local_mx - gw, 0);
        cudaMemcpy2D(handle.send_east_dev, gw * sizeof(T), src,
                     stride * sizeof(T), gw * sizeof(T), handle.local_my,
                     cudaMemcpyDeviceToDevice);
      }
      if (handle.south >= 0) {
        const T* src = device_ptr + idx(0, 0);
        cudaMemcpy2D(handle.send_south_dev, handle.local_mx * sizeof(T), src,
                     stride * sizeof(T), handle.local_mx * sizeof(T), gw,
                     cudaMemcpyDeviceToDevice);
      }
      if (handle.north >= 0) {
        const T* src = device_ptr + idx(0, handle.local_my - gw);
        cudaMemcpy2D(handle.send_north_dev, handle.local_mx * sizeof(T), src,
                     stride * sizeof(T), handle.local_mx * sizeof(T), gw,
                     cudaMemcpyDeviceToDevice);
      }

      handle.requests.reserve(8);
      auto post_recv = [&](int neighbor, int tag, T* buffer,
                           std::size_t count) {
        if (neighbor < 0 || count == 0) {
          return;
        }
        MPI_Request req{};
        MPI_Irecv(buffer, static_cast<int>(count), MpiType<T>::value(), neighbor,
                  tag, MPI_COMM_WORLD, &req);
        handle.requests.push_back(req);
      };
      auto post_send = [&](int neighbor, int tag, T* buffer,
                           std::size_t count) {
        if (neighbor < 0 || count == 0) {
          return;
        }
        MPI_Request req{};
        MPI_Isend(buffer, static_cast<int>(count), MpiType<T>::value(), neighbor,
                  tag, MPI_COMM_WORLD, &req);
        handle.requests.push_back(req);
      };

      post_recv(handle.west, tag_east, handle.recv_west_dev, west_east_count);
      post_recv(handle.east, tag_west, handle.recv_east_dev, west_east_count);
      post_recv(handle.south, tag_north, handle.recv_south_dev, north_south_count);
      post_recv(handle.north, tag_south, handle.recv_north_dev, north_south_count);

      post_send(handle.west, tag_west, handle.send_west_dev, west_east_count);
      post_send(handle.east, tag_east, handle.send_east_dev, west_east_count);
      post_send(handle.south, tag_south, handle.send_south_dev, north_south_count);
      post_send(handle.north, tag_north, handle.send_north_dev, north_south_count);

      return handle;
    }
#endif

    const std::size_t west_east_count = static_cast<std::size_t>(gw) *
                                        static_cast<std::size_t>(handle.local_my);
    const std::size_t north_south_count = static_cast<std::size_t>(gw) *
                                          static_cast<std::size_t>(handle.local_mx);
    if (handle.west >= 0) {
      handle.send_west.allocate(west_east_count);
      handle.recv_west.allocate(west_east_count);
    }
    if (handle.east >= 0) {
      handle.send_east.allocate(west_east_count);
      handle.recv_east.allocate(west_east_count);
    }
    if (handle.south >= 0) {
      handle.send_south.allocate(north_south_count);
      handle.recv_south.allocate(north_south_count);
    }
    if (handle.north >= 0) {
      handle.send_north.allocate(north_south_count);
      handle.recv_north.allocate(north_south_count);
    }

    if (handle.west >= 0) {
      int idx = 0;
      for (int j = 0; j < handle.local_my; ++j) {
        for (int i = 0; i < gw; ++i) {
          handle.send_west.ptr[idx++] = field(i, j);
        }
      }
    }
    if (handle.east >= 0) {
      int idx = 0;
      for (int j = 0; j < handle.local_my; ++j) {
        for (int i = handle.local_mx - gw; i < handle.local_mx; ++i) {
          handle.send_east.ptr[idx++] = field(i, j);
        }
      }
    }
    if (handle.south >= 0) {
      int idx = 0;
      for (int j = 0; j < gw; ++j) {
        for (int i = 0; i < handle.local_mx; ++i) {
          handle.send_south.ptr[idx++] = field(i, j);
        }
      }
    }
    if (handle.north >= 0) {
      int idx = 0;
      for (int j = handle.local_my - gw; j < handle.local_my; ++j) {
        for (int i = 0; i < handle.local_mx; ++i) {
          handle.send_north.ptr[idx++] = field(i, j);
        }
      }
    }

    handle.requests.reserve(8);
    auto post_recv = [&](int neighbor, int tag, typename ExchangeHandle<T>::HostBuffer& buffer) {
      if (neighbor < 0) {
        return;
      }
      MPI_Request req{};
      MPI_Irecv(buffer.ptr, static_cast<int>(buffer.count), MpiType<T>::value(),
                neighbor, tag, MPI_COMM_WORLD, &req);
      handle.requests.push_back(req);
    };
    auto post_send = [&](int neighbor, int tag, typename ExchangeHandle<T>::HostBuffer& buffer) {
      if (neighbor < 0) {
        return;
      }
      MPI_Request req{};
      MPI_Isend(buffer.ptr, static_cast<int>(buffer.count), MpiType<T>::value(),
                neighbor, tag, MPI_COMM_WORLD, &req);
      handle.requests.push_back(req);
    };

    post_recv(handle.west, tag_east, handle.recv_west);
    post_recv(handle.east, tag_west, handle.recv_east);
    post_recv(handle.south, tag_north, handle.recv_south);
    post_recv(handle.north, tag_south, handle.recv_north);

    post_send(handle.west, tag_west, handle.send_west);
    post_send(handle.east, tag_east, handle.send_east);
    post_send(handle.south, tag_south, handle.send_south);
    post_send(handle.north, tag_north, handle.send_north);
#else
    (void)field;
    (void)grid;
    (void)context;
    (void)mode;
#endif
    return handle;
  }

  template <typename T>
  void finish_exchange(ExchangeHandle<T>& handle) {
    if (!handle.active) {
      return;
    }
#if GPISM_HAVE_MPI
    if (!handle.requests.empty()) {
      MPI_Waitall(static_cast<int>(handle.requests.size()),
                  handle.requests.data(), MPI_STATUSES_IGNORE);
    }

#if GPISM_HAVE_CUDA
    if (handle.use_device) {
      if (!handle.field || handle.field->device_data() == nullptr) {
        handle.active = false;
        return;
      }
      const int gw = handle.gw;
      const int stride = handle.stride;
      auto idx = [&](int i, int j) {
        return (j + gw) * stride + (i + gw);
      };

      T* device_dest = handle.field->device_data();
      if (handle.west >= 0 && handle.recv_west_dev) {
        T* dst = device_dest + idx(-gw, 0);
        cudaMemcpy2D(dst, stride * sizeof(T), handle.recv_west_dev,
                     gw * sizeof(T), gw * sizeof(T), handle.local_my,
                     cudaMemcpyDeviceToDevice);
      }
      if (handle.east >= 0 && handle.recv_east_dev) {
        T* dst = device_dest + idx(handle.local_mx, 0);
        cudaMemcpy2D(dst, stride * sizeof(T), handle.recv_east_dev,
                     gw * sizeof(T), gw * sizeof(T), handle.local_my,
                     cudaMemcpyDeviceToDevice);
      }
      if (handle.south >= 0 && handle.recv_south_dev) {
        T* dst = device_dest + idx(0, -gw);
        cudaMemcpy2D(dst, stride * sizeof(T), handle.recv_south_dev,
                     handle.local_mx * sizeof(T),
                     handle.local_mx * sizeof(T), gw,
                     cudaMemcpyDeviceToDevice);
      }
      if (handle.north >= 0 && handle.recv_north_dev) {
        T* dst = device_dest + idx(0, handle.local_my);
        cudaMemcpy2D(dst, stride * sizeof(T), handle.recv_north_dev,
                     handle.local_mx * sizeof(T),
                     handle.local_mx * sizeof(T), gw,
                     cudaMemcpyDeviceToDevice);
      }

      auto release = [](T** ptr) {
        if (*ptr) {
          cudaFree(*ptr);
          *ptr = nullptr;
        }
      };
      release(&handle.send_west_dev);
      release(&handle.send_east_dev);
      release(&handle.send_south_dev);
      release(&handle.send_north_dev);
      release(&handle.recv_west_dev);
      release(&handle.recv_east_dev);
      release(&handle.recv_south_dev);
      release(&handle.recv_north_dev);
      handle.active = false;
      return;
    }
#endif
    if (!handle.field) {
      handle.active = false;
      return;
    }
    const int gw = handle.gw;
    if (handle.west >= 0) {
      int idx = 0;
      for (int j = 0; j < handle.local_my; ++j) {
        for (int i = -gw; i < 0; ++i) {
          (*handle.field)(i, j) = handle.recv_west.ptr[idx++];
        }
      }
    }
    if (handle.east >= 0) {
      int idx = 0;
      for (int j = 0; j < handle.local_my; ++j) {
        for (int i = handle.local_mx; i < handle.local_mx + gw; ++i) {
          (*handle.field)(i, j) = handle.recv_east.ptr[idx++];
        }
      }
    }
    if (handle.south >= 0) {
      int idx = 0;
      for (int j = -gw; j < 0; ++j) {
        for (int i = 0; i < handle.local_mx; ++i) {
          (*handle.field)(i, j) = handle.recv_south.ptr[idx++];
        }
      }
    }
    if (handle.north >= 0) {
      int idx = 0;
      for (int j = handle.local_my; j < handle.local_my + gw; ++j) {
        for (int i = 0; i < handle.local_mx; ++i) {
          (*handle.field)(i, j) = handle.recv_north.ptr[idx++];
        }
      }
    }
    handle.send_west.release();
    handle.send_east.release();
    handle.send_south.release();
    handle.send_north.release();
    handle.recv_west.release();
    handle.recv_east.release();
    handle.recv_south.release();
    handle.recv_north.release();
    handle.active = false;
#endif
  }

  template <typename T>
  StagExchangeHandle<T> start_exchange(FieldStag2D<T>& field, const Grid2D& grid,
                                       const Context& context,
                                       Mode mode = Mode::Auto) {
    StagExchangeHandle<T> handle{};
    handle.u = start_exchange(field.component(0), grid, context, mode);
    handle.v = start_exchange(field.component(1), grid, context, mode);
    return handle;
  }

  template <typename T>
  void finish_exchange(StagExchangeHandle<T>& handle) {
    finish_exchange(handle.u);
    finish_exchange(handle.v);
  }

  template <typename T>
  void exchange(Field2D<T>& field, const Grid2D& grid, const Context& context,
                Mode mode = Mode::Auto) {
    auto handle = start_exchange(field, grid, context, mode);
    if (!handle.active) {
      return;
    }
    ScopedTimer timer(handle.use_device ? "halo_exchange_device"
                                        : "halo_exchange_host");
    finish_exchange(handle);
  }

  template <typename T>
  void exchange(FieldStag2D<T>& field, const Grid2D& grid, const Context& context,
                Mode mode = Mode::Auto) {
    exchange(field.component(0), grid, context, mode);
    exchange(field.component(1), grid, context, mode);
  }

private:
};

}  // namespace gpism
