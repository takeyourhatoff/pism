// Copyright (C) 2024 PISM Authors
//
// This file is part of PISM.
//
// PISM is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// PISM is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License
// along with PISM; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

#ifndef PISM_STRESSBALANCE_TIMESTEPPING_CUDA_HH
#define PISM_STRESSBALANCE_TIMESTEPPING_CUDA_HH

#include "pism/util/array/Array.hh"
#include "pism/util/array/Array3D.hh"
#include "pism/util/array/CellType.hh"
#include "pism/util/array/Scalar.hh"
#include "pism/util/array/Vector.hh"

namespace pism {
namespace stressbalance {
namespace cuda {

bool vec_is_cuda(const array::Array &array);

void max_timestep_cfl_3d_local(const array::Scalar &ice_thickness,
                               const array::CellType &cell_type,
                               const array::Array3D &u3,
                               const array::Array3D &v3,
                               const array::Array3D &w3,
                               int xs, int ys, int xm, int ym, int Mz,
                               const double *z,
                               double one_over_dx,
                               double one_over_dy,
                               double dt_max_default,
                               double *u_max,
                               double *v_max,
                               double *w_max,
                               double *dt_min);

void max_timestep_cfl_2d_local(const array::CellType &cell_type,
                               const array::Vector &velocity,
                               int xs, int ys, int xm, int ym,
                               double dx, double dy,
                               double dt_max_default,
                               double *u_max,
                               double *v_max,
                               double *dt_min);

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#endif
