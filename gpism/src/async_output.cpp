#include "gpism/async_output.h"

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

}  // namespace

AsyncOutputWriter::AsyncOutputWriter(bool enabled) : enabled_(enabled) {}

void AsyncOutputWriter::configure(const Grid2D& grid,
                                  const IOFields2D& /*prototype*/,
                                  bool enabled) {
  enabled_ = enabled;
  for (auto& frame : {&buffers_[0], &buffers_[1]}) {
    init_frame(*frame, grid);
  }
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
  frame.fields.usurf.resize(mx, my, 0);
}

void AsyncOutputWriter::stage_fields(Frame& frame, const Grid2D& grid,
                                     IOFields2D& fields) {
  frame.fields.has_tauc = fields.has_tauc;
  frame.fields.has_vel_bc = fields.has_vel_bc;
  frame.fields.has_velocity = fields.has_velocity;
  frame.fields.has_usurf = fields.has_usurf;

  sync_device_to_host(fields.thk);
  sync_device_to_host(fields.topg);
  sync_device_to_host(fields.tauc);
  copy_interior(grid, fields.thk, frame.fields.thk);
  copy_interior(grid, fields.topg, frame.fields.topg);
  copy_interior(grid, fields.tauc, frame.fields.tauc);

  if (fields.has_velocity) {
    sync_device_to_host(fields.uvel);
    sync_device_to_host(fields.vvel);
    copy_interior(grid, fields.uvel, frame.fields.uvel);
    copy_interior(grid, fields.vvel, frame.fields.vvel);
  }

  if (fields.has_usurf) {
    sync_device_to_host(fields.usurf);
    copy_interior(grid, fields.usurf, frame.fields.usurf);
  }

  if (fields.has_vel_bc) {
    sync_device_to_host(fields.u_bc);
    sync_device_to_host(fields.v_bc);
    sync_device_to_host(fields.vel_bc_mask);
    copy_interior(grid, fields.u_bc, frame.fields.u_bc);
    copy_interior(grid, fields.v_bc, frame.fields.v_bc);
    copy_interior(grid, fields.vel_bc_mask, frame.fields.vel_bc_mask);
  }
}

bool AsyncOutputWriter::enqueue(const std::string& path, const Context& context,
                                const Grid2D& grid, IOFields2D& fields,
                                double time_value) {
  if (!enabled_ || (context.mpi_enabled() && context.size() > 1)) {
    NetcdfIO io;
    return io.write_output_append(path, context, grid, fields, time_value);
  }

  if (!configured_) {
    configure(grid, fields, enabled_);
  }

  const int buffer_index =
      (in_flight_index_ >= 0) ? (in_flight_index_ + 1) % 2 : next_index_;
  Frame& frame = buffers_[buffer_index];
  stage_fields(frame, grid, fields);
  frame.time_value = time_value;
  frame.has_data = true;

  if (pending_.valid()) {
    last_ok_ = pending_.get() && last_ok_;
  }

  in_flight_index_ = buffer_index;
  Frame* frame_ptr = &frame;
  const Context* context_ptr = &context;
  pending_ = std::async(std::launch::async,
                        [frame_ptr, path, context_ptr, grid]() {
                          NetcdfIO io;
                          return io.write_output_append(path, *context_ptr,
                                                        grid, frame_ptr->fields,
                                                        frame_ptr->time_value);
                        });

  return last_ok_;
}

bool AsyncOutputWriter::flush() {
  if (pending_.valid()) {
    last_ok_ = pending_.get() && last_ok_;
  }
  in_flight_index_ = -1;
  return last_ok_;
}

}  // namespace gpism
