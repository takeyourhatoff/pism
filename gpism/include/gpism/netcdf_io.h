#pragma once

#include <string>

#include "gpism/context.h"
#include "gpism/field2d.h"
#include "gpism/field_stag2d.h"
#include "gpism/grid2d.h"

namespace gpism {

struct IOFields2D {
  Field2D<double> thk;
  Field2D<double> topg;
  Field2D<double> tauc;
  bool has_tauc = false;
  Field2D<double> u_bc;
  Field2D<double> v_bc;
  Field2D<int> vel_bc_mask;
  bool has_vel_bc = false;
  Field2D<double> uvel;
  Field2D<double> vvel;
  Field2D<double> u_ssa;
  Field2D<double> v_ssa;
  Field2D<double> usurf;
  bool has_velocity = false;
  bool has_ssa_velocity = false;
  bool has_usurf = false;
};

// SSA solver debug dump. Unlike IOFields2D, this includes internal solver
// intermediates and staggered fields.
struct SSADebugBundle2D {
  const Field2D<double>* thk = nullptr;
  const Field2D<double>* topg = nullptr;
  const Field2D<double>* usurf = nullptr;
  const Field2D<double>* dhdx = nullptr;
  const Field2D<double>* dhdy = nullptr;
  const Field2D<int>* cell_type = nullptr;
  const FieldStag2D<double>* beta = nullptr;
  const FieldStag2D<double>* rhs = nullptr;
  const FieldStag2D<double>* nuH = nullptr;
  const FieldStag2D<double>* vel_prev = nullptr;
  const FieldStag2D<double>* vel = nullptr;
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
  bool write_output_append(const std::string& path, const Grid2D& grid,
                           const IOFields2D& fields, double time_value);
  bool write_output_append(const std::string& path, const Context& context,
                           const Grid2D& grid, const IOFields2D& fields,
                           double time_value);

  bool write_ssa_debug_bundle(const std::string& path, const Grid2D& grid,
                              const SSADebugBundle2D& bundle,
                              double time_value = 0.0);
  bool write_ssa_debug_bundle(const std::string& path, const Context& context,
                              const Grid2D& grid, const SSADebugBundle2D& bundle,
                              double time_value = 0.0);
};

}  // namespace gpism
