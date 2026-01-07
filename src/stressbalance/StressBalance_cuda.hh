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

#ifndef PISM_STRESSBALANCE_CUDA_HH
#define PISM_STRESSBALANCE_CUDA_HH

#include "pism/util/array/Array.hh"
#include "pism/util/array/Array3D.hh"
#include "pism/util/array/CellType.hh"
#include "pism/util/array/Scalar.hh"

namespace pism {
namespace stressbalance {
namespace cuda {

bool vec_is_cuda(const array::Array &array);

void compute_vertical_velocity(const array::CellType1 &mask,
                               const array::Array3D &u,
                               const array::Array3D &v,
                               const array::Scalar *basal_melt_rate,
                               array::Array3D &result,
                               int xs, int ys, int xm, int ym, int Mz,
                               const double *z,
                               double dx, double dy,
                               int use_upstream_fd);

void compute_volumetric_strain_heating(const array::CellType1 &mask,
                                       const array::Array3D &u,
                                       const array::Array3D &v,
                                       const array::Scalar &thickness,
                                       const array::Array3D &enthalpy,
                                       array::Array3D &strain_heating,
                                       int xs, int ys, int xm, int ym, int Mz,
                                       const double *z,
                                       double dx, double dy,
                                       double exponent,
                                       double e_to_a_power,
                                       int flow_law_mode,
                                       double A_cold, double A_warm,
                                       double Q_cold, double Q_warm, double T_crit,
                                       double gas_const, double n,
                                       double T_melting, double beta, double c_i,
                                       double T_0, double rho_i, double g, double p_air);

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#endif
