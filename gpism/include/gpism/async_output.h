#pragma once

#include <future>
#include <string>

#include "gpism/context.h"
#include "gpism/grid2d.h"
#include "gpism/netcdf_io.h"

namespace gpism {

class AsyncOutputWriter {
public:
  explicit AsyncOutputWriter(bool enabled = true);

  void configure(const Grid2D& grid, const IOFields2D& prototype,
                 bool enabled);

  bool enqueue(const std::string& path, const Context& context,
               const Grid2D& grid, IOFields2D& fields, double time_value);

  bool flush();

private:
  struct Frame {
    IOFields2D fields;
    double time_value = 0.0;
    bool has_data = false;
  };

  Frame buffers_[2];
  void init_frame(Frame& frame, const Grid2D& grid);
  void stage_fields(Frame& frame, const Grid2D& grid, IOFields2D& fields);

  int next_index_ = 0;
  int in_flight_index_ = -1;
  bool enabled_ = false;
  bool configured_ = false;
  bool last_ok_ = true;
  std::future<bool> pending_;
};

}  // namespace gpism
