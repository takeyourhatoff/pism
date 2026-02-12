#pragma once

#include <future>
#include <string>
#include <vector>

#include "gpism/context.h"
#include "gpism/grid2d.h"
#include "gpism/netcdf_io.h"

#include <cuda_runtime.h>

namespace gpism {

class AsyncOutputWriter {
public:
  explicit AsyncOutputWriter(bool enabled = true);
  ~AsyncOutputWriter();

  void configure(const Grid2D& grid, const IOFields2D& prototype,
                 bool enabled, int device_ring_depth = 2,
                 bool async_stage_from_device = true);

  bool enqueue(const std::string& path, const Context& context,
               const Grid2D& grid, IOFields2D& fields, double time_value);

  bool flush();

private:
  struct Frame {
    IOFields2D fields;
    double time_value = 0.0;
    bool has_data = false;
    bool pending_device_stage = false;
    std::vector<Field2D<double>*> staged_double;
    std::vector<Field2D<int>*> staged_int;
    cudaEvent_t ready_event = nullptr;
  };

  std::vector<Frame> buffers_;
  void init_frame(Frame& frame, const Grid2D& grid);
  void stage_fields(Frame& frame, const Grid2D& grid, IOFields2D& fields);
  void finalize_device_stage(Frame& frame);
  void wait_for_pending_frame();

  int next_index_ = 0;
  int in_flight_index_ = -1;
  bool enabled_ = false;
  bool configured_ = false;
  bool last_ok_ = true;
  int ring_depth_ = 2;
  bool async_stage_from_device_ = true;
  std::future<bool> pending_;
  cudaStream_t staging_stream_ = nullptr;
};

}  // namespace gpism
