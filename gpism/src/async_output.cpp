#include "gpism/async_output.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "gpism/field_sync.h"

namespace gpism {
namespace {

template <typename T>
void copy_interior(const Grid2D& grid, const Field2D<T>& src,
                   Field2D<T>& dst) {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      dst(i, j) = src(i, j);
    }
  }
}

template <typename T>
bool stage_field_from_device_async(const Grid2D& grid, const Field2D<T>& src,
                                   Field2D<T>& dst, cudaStream_t stream) {
  if (!src.has_device_data()) {
    return false;
  }
  dst.ensure_host_staging();
  T* dst_staging = dst.host_staging_data();
  if (!dst_staging) {
    return false;
  }
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const std::size_t row_bytes = static_cast<std::size_t>(mx) * sizeof(T);
  const std::size_t row_count = static_cast<std::size_t>(my);
  const std::size_t total_bytes = row_bytes * row_count;
  if (row_bytes == 0 || row_count == 0) {
    return true;
  }
  const std::size_t src_offset =
      static_cast<std::size_t>(src.ghost_width()) *
          static_cast<std::size_t>(src.stride()) +
      static_cast<std::size_t>(src.ghost_width());
  const cudaError_t err = cudaMemcpy2DAsync(
      dst_staging, static_cast<std::size_t>(dst.stride()) * sizeof(T),
      src.device_data() + src_offset,
      static_cast<std::size_t>(src.stride()) * sizeof(T), row_bytes, row_count,
      cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) {
    SyncAudit::record(SyncDirection::DeviceToHost, total_bytes,
                      "AsyncOutputWriter");
    SyncStats::record_d2h(total_bytes);
  }
  return err == cudaSuccess;
}

}  // namespace

AsyncOutputWriter::AsyncOutputWriter(bool enabled) : enabled_(enabled) {}

AsyncOutputWriter::~AsyncOutputWriter() {
  (void)flush();
  for (auto& frame : buffers_) {
    if (frame.ready_event) {
      cudaEventDestroy(frame.ready_event);
      frame.ready_event = nullptr;
    }
  }
  if (staging_stream_) {
    cudaStreamDestroy(staging_stream_);
    staging_stream_ = nullptr;
  }
}

void AsyncOutputWriter::configure(const Grid2D& grid,
                                  const IOFields2D& /*prototype*/,
                                  bool enabled, int device_ring_depth,
                                  bool async_stage_from_device) {
  enabled_ = enabled;
  ring_depth_ = std::max(2, device_ring_depth);
  async_stage_from_device_ = async_stage_from_device;
  if (staging_stream_ == nullptr) {
    cudaStreamCreateWithFlags(&staging_stream_, cudaStreamNonBlocking);
  }
  for (auto& frame : buffers_) {
    if (frame.ready_event) {
      cudaEventDestroy(frame.ready_event);
      frame.ready_event = nullptr;
    }
  }
  buffers_.clear();
  buffers_.resize(static_cast<std::size_t>(ring_depth_));
  for (auto& frame : buffers_) {
    init_frame(frame, grid);
  }
  next_index_ = 0;
  in_flight_index_ = -1;
  configured_ = true;
}

void AsyncOutputWriter::init_frame(Frame& frame, const Grid2D& grid) {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  frame.fields.thk.resize(mx, my, 0);
  frame.fields.topg.resize(mx, my, 0);
  frame.fields.tauc.resize(mx, my, 0);
  frame.fields.u_bc.resize(mx, my, 0);
  frame.fields.v_bc.resize(mx, my, 0);
  frame.fields.vel_bc_mask.resize(mx, my, 0);
  frame.fields.uvel.resize(mx, my, 0);
  frame.fields.vvel.resize(mx, my, 0);
  frame.fields.u_ssa.resize(mx, my, 0);
  frame.fields.v_ssa.resize(mx, my, 0);
  frame.fields.usurf.resize(mx, my, 0);
  if (!frame.ready_event) {
    cudaEventCreateWithFlags(&frame.ready_event, cudaEventDisableTiming);
  }
}

void AsyncOutputWriter::stage_fields(Frame& frame, const Grid2D& grid,
                                     IOFields2D& fields) {
  frame.fields.has_tauc = fields.has_tauc;
  frame.fields.has_vel_bc = fields.has_vel_bc;
  frame.fields.has_velocity = fields.has_velocity;
  frame.fields.has_ssa_velocity = fields.has_ssa_velocity;
  frame.fields.has_usurf = fields.has_usurf;

  frame.pending_device_stage = false;
  frame.staged_double.clear();
  frame.staged_int.clear();

  auto stage_field_double = [&](Field2D<double>& src, Field2D<double>& dst) {
    if (async_stage_from_device_ && src.has_device_data() && staging_stream_) {
      if (stage_field_from_device_async(grid, src, dst, staging_stream_)) {
        frame.pending_device_stage = true;
        frame.staged_double.push_back(&dst);
        return;
      }
    }
    sync_device_to_host(src);
    copy_interior(grid, src, dst);
  };

  auto stage_field_int = [&](Field2D<int>& src, Field2D<int>& dst) {
    if (async_stage_from_device_ && src.has_device_data() && staging_stream_) {
      if (stage_field_from_device_async(grid, src, dst, staging_stream_)) {
        frame.pending_device_stage = true;
        frame.staged_int.push_back(&dst);
        return;
      }
    }
    sync_device_to_host(src);
    copy_interior(grid, src, dst);
  };

  stage_field_double(fields.thk, frame.fields.thk);
  stage_field_double(fields.topg, frame.fields.topg);
  stage_field_double(fields.tauc, frame.fields.tauc);

  if (fields.has_velocity) {
    stage_field_double(fields.uvel, frame.fields.uvel);
    stage_field_double(fields.vvel, frame.fields.vvel);
  }

  if (fields.has_ssa_velocity) {
    stage_field_double(fields.u_ssa, frame.fields.u_ssa);
    stage_field_double(fields.v_ssa, frame.fields.v_ssa);
  }

  if (fields.has_usurf) {
    stage_field_double(fields.usurf, frame.fields.usurf);
  }

  if (fields.has_vel_bc) {
    stage_field_double(fields.u_bc, frame.fields.u_bc);
    stage_field_double(fields.v_bc, frame.fields.v_bc);
    stage_field_int(fields.vel_bc_mask, frame.fields.vel_bc_mask);
  }

  if (frame.pending_device_stage && frame.ready_event) {
    cudaEventRecord(frame.ready_event, staging_stream_);
  }
}

void AsyncOutputWriter::finalize_device_stage(Frame& frame) {
  if (frame.pending_device_stage && frame.ready_event) {
    cudaEventSynchronize(frame.ready_event);
    for (auto* field : frame.staged_double) {
      const std::size_t bytes = field->elements() * sizeof(double);
      std::memcpy(field->data(), field->host_staging_data(), bytes);
    }
    for (auto* field : frame.staged_int) {
      const std::size_t bytes = field->elements() * sizeof(int);
      std::memcpy(field->data(), field->host_staging_data(), bytes);
    }
  }
  frame.pending_device_stage = false;
  frame.staged_double.clear();
  frame.staged_int.clear();
}

void AsyncOutputWriter::wait_for_pending_frame() {
  if (pending_.valid()) {
    last_ok_ = pending_.get() && last_ok_;
  }
}

bool AsyncOutputWriter::enqueue(const std::string& path, const Context& context,
                                const Grid2D& grid, IOFields2D& fields,
                                double time_value) {
  if (!enabled_ || (context.mpi_enabled() && context.size() > 1)) {
    sync_device_to_host(fields.thk);
    sync_device_to_host(fields.topg);
    sync_device_to_host(fields.tauc);
    if (fields.has_velocity) {
      sync_device_to_host(fields.uvel);
      sync_device_to_host(fields.vvel);
    }
    if (fields.has_ssa_velocity) {
      sync_device_to_host(fields.u_ssa);
      sync_device_to_host(fields.v_ssa);
    }
    if (fields.has_usurf) {
      sync_device_to_host(fields.usurf);
    }
    if (fields.has_vel_bc) {
      sync_device_to_host(fields.u_bc);
      sync_device_to_host(fields.v_bc);
      sync_device_to_host(fields.vel_bc_mask);
    }
    NetcdfIO io;
    return io.write_output_append(path, context, grid, fields, time_value);
  }

  if (!configured_) {
    configure(grid, fields, enabled_, ring_depth_, async_stage_from_device_);
  }

  const int buffer_index = (in_flight_index_ >= 0)
                               ? (in_flight_index_ + 1) % ring_depth_
                               : next_index_;
  Frame& frame = buffers_[static_cast<std::size_t>(buffer_index)];
  stage_fields(frame, grid, fields);
  frame.time_value = time_value;
  frame.has_data = true;

  wait_for_pending_frame();

  in_flight_index_ = buffer_index;
  next_index_ = (buffer_index + 1) % ring_depth_;
  Frame* frame_ptr = &frame;
  const Context* context_ptr = &context;
  pending_ = std::async(std::launch::async,
                        [this, frame_ptr, path, context_ptr, grid]() {
                          cudaSetDevice(context_ptr->device_id());
                          finalize_device_stage(*frame_ptr);
                          NetcdfIO io;
                          return io.write_output_append(path, *context_ptr,
                                                        grid, frame_ptr->fields,
                                                        frame_ptr->time_value);
                        });

  return last_ok_;
}

bool AsyncOutputWriter::flush() {
  wait_for_pending_frame();
  in_flight_index_ = -1;
  return last_ok_;
}

}  // namespace gpism
