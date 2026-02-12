#include "gpism/viscosity.h"

#include <cmath>
#include <stdexcept>

namespace gpism {
void viscosity_compute_nuH_cuda(int mx, int my, int gw, int stride_thk,
                                int stride_u, int stride_v, int stride_nuH_u,
                                int stride_nuH_v, const double* thk,
                                const double* u, const double* v, double* nuH_u,
                                double* nuH_v, double nuH_regularization,
                                double strength_extension_nu,
                                double strength_extension_min_thickness,
                                const double* enthalpy, int nz, int enthalpy_gw,
                                int enthalpy_stride, double enthalpy_gamma,
                                double enthalpy_ref, double B, double n_eff,
                                double eps0, double inv_dx, double inv_dy,
                                int periodic);
}  // namespace gpism

namespace gpism {
namespace {

double clamp_positive(double value, double floor) {
  return value < floor ? floor : value;
}

}  // namespace

ViscosityModel::ViscosityModel(double A, double n, double eps0,
                               double enhancement)
    : A_(A), n_(n), eps0_(eps0), enhancement_(enhancement) {}

void ViscosityModel::compute_nuH(const Grid2D& grid, const Field2D<double>& thk,
                                 const FieldStag2D<double>& vel,
                                 FieldStag2D<double>& nuH,
                                 double nuH_regularization,
                                 double strength_extension_nu,
                                 double strength_extension_min_thickness,
                                 const Field3D<double>* enthalpy,
                                 double enthalpy_gamma,
                                 double enthalpy_ref) const {
  const int mx = grid.local_mx();
  const int my = grid.local_my();
  const double dx = grid.dx();
  const double dy = grid.dy();
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;

  if (!(thk.has_device_data() && vel.component(0).has_device_data() &&
        vel.component(1).has_device_data() &&
        nuH.component(0).has_device_data() &&
        nuH.component(1).has_device_data())) {
    throw std::runtime_error(
        "ViscosityModel::compute_nuH requires device-resident thk/vel/nuH");
  }
  if (enthalpy && enthalpy_gamma != 0.0 && !enthalpy->has_device_data()) {
    throw std::runtime_error(
        "ViscosityModel::compute_nuH requires device-resident enthalpy");
  }

  // Note: PISM uses B = A^{-1/n}. Do not clamp A to an overly-large floor
  // (Glen softness A is typically ~1e-24 in SI units), otherwise viscosity
  // becomes far too small and velocities blow up.
  const double enhancement = clamp_positive(enhancement_, 1e-12);
  const double A_eff = clamp_positive(A_ * enhancement, 1e-60);
  const double n_eff = clamp_positive(n_, 1.0);
  const double B = std::pow(A_eff, -1.0 / n_eff);

  const bool use_temp =
      enthalpy && enthalpy->has_device_data() && enthalpy_gamma != 0.0;
  const double* enthalpy_ptr = use_temp ? enthalpy->device_data() : nullptr;
  const int nz = use_temp ? enthalpy->local_mz() : 0;
  const int enthalpy_gw = use_temp ? enthalpy->ghost_width() : 0;
  const int enthalpy_stride = use_temp ? enthalpy->stride() : 0;
  const int periodic = (grid.dims_x() == 1 && grid.dims_y() == 1) ? 1 : 0;
  viscosity_compute_nuH_cuda(
      mx, my, thk.ghost_width(), thk.stride(), vel.component(0).stride(),
      vel.component(1).stride(), nuH.component(0).stride(),
      nuH.component(1).stride(), thk.device_data(), vel.component(0).device_data(),
      vel.component(1).device_data(), nuH.component(0).device_data(),
      nuH.component(1).device_data(), nuH_regularization, strength_extension_nu,
      strength_extension_min_thickness, enthalpy_ptr, nz, enthalpy_gw,
      enthalpy_stride, enthalpy_gamma, enthalpy_ref, B, n_eff, eps0_, inv_dx,
      inv_dy, periodic);
}

}  // namespace gpism
