#pragma once

#include <string>

#include "gpism/context.h"
#include "gpism/field2d.h"
#include "gpism/grid2d.h"

namespace gpism {

struct IOFields2D {
  Field2D<double> thk;
  Field2D<double> topg;
  Field2D<double> tauc;
};

class NetcdfIO {
public:
  bool read_restart(const std::string& path, Grid2D& grid, IOFields2D& fields,
                    int time_index = -1);
  bool read_restart(const std::string& path, const Context& context, Grid2D& grid,
                    IOFields2D& fields, int time_index = -1);
  bool write_output(const std::string& path, const Grid2D& grid, const IOFields2D& fields,
                    double time_value = 0.0);
  bool write_output(const std::string& path, const Context& context, const Grid2D& grid,
                    const IOFields2D& fields, double time_value = 0.0);
};

}  // namespace gpism
