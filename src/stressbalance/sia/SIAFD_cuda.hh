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

#ifndef PISM_SIAFD_CUDA_HH
#define PISM_SIAFD_CUDA_HH

#include "pism/util/array/Array.hh"
#include "pism/util/array/Array3D.hh"
#include "pism/util/array/CellType.hh"
#include "pism/util/array/Scalar.hh"
#include "pism/util/array/Staggered.hh"
#include "pism/util/array/Vector.hh"

namespace pism {
namespace stressbalance {
namespace cuda {

bool vec_is_cuda(const array::Array &array);

void surface_gradient_mahaffy(const array::Scalar &ice_surface_elevation,
                              array::Staggered1 &h_x,
                              array::Staggered1 &h_y,
                              int xs, int ys, int xm, int ym,
                              double dx, double dy);

void surface_gradient_eta(const array::Scalar2 &ice_thickness,
                          const array::Scalar2 &bed_elevation,
                          array::Scalar2 &eta,
                          array::Staggered1 &h_x,
                          array::Staggered1 &h_y,
                          int xs, int ys, int xm, int ym,
                          int ghosts, double dx, double dy,
                          double invpow, double dinvpow, double etapow);

void surface_gradient_haseloff(const array::Scalar2 &ice_surface_elevation,
                               const array::CellType2 &cell_type,
                               array::Scalar1 &w_i,
                               array::Scalar1 &w_j,
                               array::Staggered1 &h_x,
                               array::Staggered1 &h_y,
                               int xs, int ys, int xm, int ym,
                               double dx, double dy);

void diffusive_flux(const array::Staggered &h_x,
                    const array::Staggered &h_y,
                    const array::Staggered &diffusivity,
                    array::Staggered &result,
                    int xs, int ys, int xm, int ym,
                    int o);

void bed_smoother_smoothed_thk(const array::Scalar &usurf,
                               const array::Scalar &thk,
                               const array::Scalar &maxtl,
                               const array::Scalar &topgsmooth,
                               const array::CellType2 &mask,
                               array::Scalar &result,
                               int xs, int ys, int xm, int ym,
                               int ghosts);

void bed_smoother_theta(const array::Scalar &usurf,
                        const array::Scalar &maxtl,
                        const array::Scalar &topgsmooth,
                        const array::Scalar &C2,
                        const array::Scalar &C3,
                        const array::Scalar &C4,
                        array::Scalar &result,
                        int xs, int ys, int xm, int ym,
                        int ghosts,
                        double theta_min, double theta_max,
                        double Glen_exponent);

void update_vertical_grid(const double *z, int Mz);

void compute_I(const array::Scalar &thk_smooth,
               const array::Array3D &delta,
               array::Array3D &I,
               int xs, int ys, int xm, int ym,
               int ghosts, int o);

void compute_3d_horizontal_velocity(const array::Array3D &I0,
                                    const array::Array3D &I1,
                                    const array::Staggered &h_x,
                                    const array::Staggered &h_y,
                                    const array::Vector &sliding_velocity,
                                    array::Array3D &u_out,
                                    array::Array3D &v_out,
                                    int xs, int ys, int xm, int ym,
                                    int Mz);

void compute_3d_horizontal_velocity_from_delta(const array::Array3D &delta0,
                                               const array::Array3D &delta1,
                                               const array::Staggered &h_x,
                                               const array::Staggered &h_y,
                                               const array::Vector &sliding_velocity,
                                               array::Array3D &u_out,
                                               array::Array3D &v_out,
                                               int xs, int ys, int xm, int ym,
                                               int Mz);

void compute_diffusivity_pb(const array::Scalar &thk_smooth,
                            const array::Scalar &theta,
                            const array::Staggered &h_x,
                            const array::Staggered &h_y,
                            const array::Array3D &enthalpy,
                            array::Staggered &result,
                            array::Array3D &delta,
                            array::Array3D &I,
                            int xs, int ys, int xm, int ym,
                            int Mx, int My,
                            int ghosts, int o, int full_update, int compute_I,
                            int flow_law_mode,
                            int periodic_x, int periodic_y,
                            int limit_diffusivity,
                            double D_limit, double e_factor,
                            double A_cold, double A_warm,
                            double Q_cold, double Q_warm, double T_crit,
                            double gas_const, double n,
                            double T_melting, double beta, double c_i,
                            double T_0, double rho_i, double g, double p_air,
                            double *D_max, int *high_diffusivity_counter);

} // namespace cuda
} // namespace stressbalance
} // namespace pism

#endif // PISM_SIAFD_CUDA_HH
