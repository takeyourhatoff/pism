#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"
#include "gpism/profile.h"

#include <cuda_runtime.h>

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

template <>
struct MpiType<int> {
  static MPI_Datatype value() { return MPI_INT; }
};
#endif

class HaloExchange2D {
public:
  enum class Mode { Device };

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
    T* send_west_dev = nullptr;
    T* send_east_dev = nullptr;
    T* send_south_dev = nullptr;
    T* send_north_dev = nullptr;
    T* recv_west_dev = nullptr;
    T* recv_east_dev = nullptr;
    T* recv_south_dev = nullptr;
    T* recv_north_dev = nullptr;
  };

  template <typename T>
  struct StagExchangeHandle {
    ExchangeHandle<T> u;
    ExchangeHandle<T> v;
  };

  template <typename T>
  ExchangeHandle<T> start_exchange(Field2D<T>& field, const Grid2D& grid,
                                   const Context& context,
                                   Mode mode = Mode::Device) {
    ExchangeHandle<T> handle{};
    if (!context.mpi_enabled() || context.size() <= 1) {
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

    (void)mode;

    const bool can_device =
        context.cuda_aware_mpi() && field.device_data() != nullptr;
    if (!can_device) {
      throw std::runtime_error(
          "HaloExchange2D requires CUDA-aware MPI with device buffers");
    }
    handle.use_device = true;

    const int stride = handle.stride;
    auto idx = [&](int i, int j) { return (j + gw) * stride + (i + gw); };

    const std::size_t west_east_count =
        static_cast<std::size_t>(gw) * static_cast<std::size_t>(handle.local_my);
    const std::size_t north_south_count =
        static_cast<std::size_t>(gw) * static_cast<std::size_t>(handle.local_mx);

    const void* owner = static_cast<const void*>(field.device_data());

    if (handle.west >= 0) {
      acquire_device_buffer(make_buffer_key(owner, 0), west_east_count,
                            &handle.send_west_dev);
      acquire_device_buffer(make_buffer_key(owner, 1), west_east_count,
                            &handle.recv_west_dev);
    }
    if (handle.east >= 0) {
      acquire_device_buffer(make_buffer_key(owner, 2), west_east_count,
                            &handle.send_east_dev);
      acquire_device_buffer(make_buffer_key(owner, 3), west_east_count,
                            &handle.recv_east_dev);
    }
    if (handle.south >= 0) {
      acquire_device_buffer(make_buffer_key(owner, 4), north_south_count,
                            &handle.send_south_dev);
      acquire_device_buffer(make_buffer_key(owner, 5), north_south_count,
                            &handle.recv_south_dev);
    }
    if (handle.north >= 0) {
      acquire_device_buffer(make_buffer_key(owner, 6), north_south_count,
                            &handle.send_north_dev);
      acquire_device_buffer(make_buffer_key(owner, 7), north_south_count,
                            &handle.recv_north_dev);
    }

    const T* device_ptr = field.device_data();

    if (handle.west >= 0) {
      const T* src = device_ptr + idx(0, 0);
      cudaMemcpy2D(handle.send_west_dev, gw * sizeof(T), src, stride * sizeof(T),
                   gw * sizeof(T), handle.local_my, cudaMemcpyDeviceToDevice);
    }
    if (handle.east >= 0) {
      const T* src = device_ptr + idx(handle.local_mx - gw, 0);
      cudaMemcpy2D(handle.send_east_dev, gw * sizeof(T), src, stride * sizeof(T),
                   gw * sizeof(T), handle.local_my, cudaMemcpyDeviceToDevice);
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
    auto post_recv = [&](int neighbor, int tag, T* buffer, std::size_t count) {
      if (neighbor < 0 || count == 0) {
        return;
      }
      MPI_Request req{};
      MPI_Irecv(buffer, static_cast<int>(count), MpiType<T>::value(), neighbor,
                tag, MPI_COMM_WORLD, &req);
      handle.requests.push_back(req);
    };
    auto post_send = [&](int neighbor, int tag, T* buffer, std::size_t count) {
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
      MPI_Waitall(static_cast<int>(handle.requests.size()), handle.requests.data(),
                  MPI_STATUSES_IGNORE);
    }

    if (!handle.field || handle.field->device_data() == nullptr) {
      handle.active = false;
      return;
    }
    const int gw = handle.gw;
    const int stride = handle.stride;
    auto idx = [&](int i, int j) { return (j + gw) * stride + (i + gw); };

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
                   handle.local_mx * sizeof(T), handle.local_mx * sizeof(T), gw,
                   cudaMemcpyDeviceToDevice);
    }
    if (handle.north >= 0 && handle.recv_north_dev) {
      T* dst = device_dest + idx(0, handle.local_my);
      cudaMemcpy2D(dst, stride * sizeof(T), handle.recv_north_dev,
                   handle.local_mx * sizeof(T), handle.local_mx * sizeof(T), gw,
                   cudaMemcpyDeviceToDevice);
    }

    handle.send_west_dev = nullptr;
    handle.send_east_dev = nullptr;
    handle.send_south_dev = nullptr;
    handle.send_north_dev = nullptr;
    handle.recv_west_dev = nullptr;
    handle.recv_east_dev = nullptr;
    handle.recv_south_dev = nullptr;
    handle.recv_north_dev = nullptr;
#endif
    handle.active = false;
  }

  template <typename T>
  StagExchangeHandle<T> start_exchange(FieldStag2D<T>& field, const Grid2D& grid,
                                       const Context& context,
                                       Mode mode = Mode::Device) {
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
                Mode mode = Mode::Device) {
    auto handle = start_exchange(field, grid, context, mode);
    if (!handle.active) {
      return;
    }
    ScopedTimer timer("halo_exchange_device");
    finish_exchange(handle);
  }

  template <typename T>
  void exchange(FieldStag2D<T>& field, const Grid2D& grid, const Context& context,
                Mode mode = Mode::Device) {
    exchange(field.component(0), grid, context, mode);
    exchange(field.component(1), grid, context, mode);
  }

private:
  template <typename T>
  struct DeviceBufferCacheEntry {
    T* ptr = nullptr;
    std::size_t count = 0;
  };

  static std::uint64_t make_buffer_key(const void* owner, int slot) {
    const auto owner_value =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(owner));
    const auto slot_value = static_cast<std::uint64_t>(slot & 0xff);
    return (owner_value << 8) ^ slot_value;
  }

  template <typename T>
  static std::unordered_map<std::uint64_t, DeviceBufferCacheEntry<T>>&
  device_buffer_cache() {
    static std::unordered_map<std::uint64_t, DeviceBufferCacheEntry<T>> cache;
    return cache;
  }

  template <typename T>
  static void acquire_device_buffer(std::uint64_t key, std::size_t count,
                                    T** ptr) {
    if (count == 0) {
      *ptr = nullptr;
      return;
    }
    auto& cache = device_buffer_cache<T>();
    auto& entry = cache[key];
    if (entry.ptr == nullptr || entry.count < count) {
      if (entry.ptr != nullptr) {
        cudaFree(entry.ptr);
      }
      cudaMalloc(reinterpret_cast<void**>(&entry.ptr), count * sizeof(T));
      entry.count = count;
    }
    *ptr = entry.ptr;
  }
};

}  // namespace gpism
