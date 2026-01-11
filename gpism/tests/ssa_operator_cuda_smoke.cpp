#include "gpism/ssa_operator.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace {

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) <= tol;
}

template <typename T>
void sync_host_to_device(gpism::Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  std::memcpy(field.host_staging_data(), field.data(),
              field.elements() * sizeof(T));
  field.copy_host_to_device();
#else
  (void)field;
#endif
}

template <typename T>
void sync_device_to_host(gpism::Field2D<T>& field) {
#if GPISM_HAVE_CUDA
  if (!field.host_staging_data() || field.elements() == 0) {
    return;
  }
  field.copy_device_to_host();
  std::memcpy(field.data(), field.host_staging_data(),
              field.elements() * sizeof(T));
#else
  (void)field;
#endif
}

template <typename T>
void sync_host_to_device(gpism::FieldStag2D<T>& field) {
  sync_host_to_device(field.component(0));
  sync_host_to_device(field.component(1));
}

template <typename T>
void sync_device_to_host(gpism::FieldStag2D<T>& field) {
  sync_device_to_host(field.component(0));
  sync_device_to_host(field.component(1));
}

double avg2(double a, double b) { return 0.5 * (a + b); }

bool is_dirichlet(const gpism::SSABoundaryCondition* bc, int i, int j, int comp) {
  if (!bc || !bc->mask) {
    return false;
  }
  return (*bc->mask)(i, j, comp) != 0;
}

void compute_basal_drag_ref(const gpism::Grid2D& grid,
                            const gpism::Field2D<double>& tauc,
                            gpism::FieldStag2D<double>& beta,
                            double u_threshold) {
  const double denom = std::max(u_threshold, 1e-6);
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const double tauc_u = avg2(tauc(i, j), tauc(i + 1, j));
      const double tauc_v = avg2(tauc(i, j), tauc(i, j + 1));
      beta(i, j, 0) = tauc_u / denom;
      beta(i, j, 1) = tauc_v / denom;
    }
  }
}

void assemble_rhs_ref(const gpism::Grid2D& grid,
                      const gpism::Field2D<double>& thk,
                      const gpism::Field2D<double>& dhdx,
                      const gpism::Field2D<double>& dhdy,
                      gpism::FieldStag2D<double>& rhs, double rho,
                      double g, const gpism::SSABoundaryCondition* bc) {
  const double scale = rho * g;
  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      const double H_u = avg2(thk(i, j), thk(i + 1, j));
      const double H_v = avg2(thk(i, j), thk(i, j + 1));
      const double slope_x = avg2(dhdx(i, j), dhdx(i + 1, j));
      const double slope_y = avg2(dhdy(i, j), dhdy(i, j + 1));

      rhs(i, j, 0) = scale * H_u * slope_x;
      rhs(i, j, 1) = scale * H_v * slope_y;

      if (is_dirichlet(bc, i, j, 0) && bc->values) {
        rhs(i, j, 0) = (*bc->values)(i, j, 0);
      }
      if (is_dirichlet(bc, i, j, 1) && bc->values) {
        rhs(i, j, 1) = (*bc->values)(i, j, 1);
      }
    }
  }
}

void apply_ref(const gpism::Grid2D& grid, const gpism::FieldStag2D<double>& nuH,
               const gpism::FieldStag2D<double>& beta,
               const gpism::FieldStag2D<double>& vel,
               gpism::FieldStag2D<double>& out,
               const gpism::SSABoundaryCondition* bc) {
  const double dx = grid.dx();
  const double dy = grid.dy();
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  const double inv_2dx = 1.0 / (2.0 * dx);
  const double inv_2dy = 1.0 / (2.0 * dy);

  const gpism::Field2D<double>& u = vel.component(0);
  const gpism::Field2D<double>& v = vel.component(1);
  const gpism::Field2D<double>& nu_u = nuH.component(0);
  const gpism::Field2D<double>& nu_v = nuH.component(1);
  const gpism::Field2D<double>& beta_u = beta.component(0);
  const gpism::Field2D<double>& beta_v = beta.component(1);

  auto shear = [&](int i, int j) {
    const double du_dy = (u(i, j + 1) - u(i, j - 1)) * inv_2dy;
    const double dv_dx = (v(i + 1, j) - v(i - 1, j)) * inv_2dx;
    return du_dy + dv_dx;
  };

  for (int j = 0; j < grid.local_my(); ++j) {
    for (int i = 0; i < grid.local_mx(); ++i) {
      if (is_dirichlet(bc, i, j, 0)) {
        out(i, j, 0) = u(i, j);
      } else {
        const double u_c = u(i, j);
        const double flux_x =
            nu_u(i + 1, j) * (u(i + 1, j) - u_c) -
            nu_u(i - 1, j) * (u_c - u(i - 1, j));
        const double flux_y =
            nu_u(i, j + 1) * (u(i, j + 1) - u_c) -
            nu_u(i, j - 1) * (u_c - u(i, j - 1));
        double coupling = 0.0;
        if (j >= 1 && j <= grid.local_my() - 2) {
          const double shear_p = shear(i, j + 1);
          const double shear_m = shear(i, j - 1);
          coupling = nu_u(i, j) * (shear_p - shear_m) * inv_2dy;
        }
        out(i, j, 0) = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                       beta_u(i, j) * u_c;
      }

      if (is_dirichlet(bc, i, j, 1)) {
        out(i, j, 1) = v(i, j);
      } else {
        const double v_c = v(i, j);
        const double flux_x =
            nu_v(i + 1, j) * (v(i + 1, j) - v_c) -
            nu_v(i - 1, j) * (v_c - v(i - 1, j));
        const double flux_y =
            nu_v(i, j + 1) * (v(i, j + 1) - v_c) -
            nu_v(i, j - 1) * (v_c - v(i, j - 1));
        double coupling = 0.0;
        if (i >= 1 && i <= grid.local_mx() - 2) {
          const double shear_p = shear(i + 1, j);
          const double shear_m = shear(i - 1, j);
          coupling = nu_v(i, j) * (shear_p - shear_m) * inv_2dx;
        }
        out(i, j, 1) = flux_x * inv_dx2 + flux_y * inv_dy2 + coupling +
                       beta_v(i, j) * v_c;
      }
    }
  }
}

bool compare_stag(const gpism::FieldStag2D<double>& a,
                  const gpism::FieldStag2D<double>& b, const char* name,
                  double tol = 1e-10) {
  for (int j = 0; j < a.local_my(); ++j) {
    for (int i = 0; i < a.local_mx(); ++i) {
      for (int comp = 0; comp < 2; ++comp) {
        if (!nearly_equal(a(i, j, comp), b(i, j, comp), tol)) {
          std::cerr << name << " mismatch at " << i << "," << j << "," << comp
                    << " (" << a(i, j, comp) << " vs " << b(i, j, comp)
                    << ")\n";
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

int main() {
#if !GPISM_HAVE_CUDA
  std::cout << "ssa_operator_cuda_smoke skipped (CUDA disabled)\n";
  return 0;
#else
  const int mx = 5;
  const int my = 4;
  const int gw = 1;
  gpism::Grid2D grid(mx, my, 2.5, 3.5, gw, 0, 1);

  const double rho = 910.0;
  const double g = 9.81;
  const double u_threshold = 100.0;
  gpism::SSAOperator op(rho, g, u_threshold);

  gpism::Field2D<double> tauc(mx, my, gw);
  tauc.fill(100.0);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      tauc(i, j) = 50.0 + i + 2.0 * j;
    }
  }

  gpism::Field2D<double> thk(mx, my, gw);
  gpism::Field2D<double> dhdx(mx, my, gw);
  gpism::Field2D<double> dhdy(mx, my, gw);
  thk.fill(2.0);
  dhdx.fill(0.5);
  dhdy.fill(-0.25);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      thk(i, j) = 2.0 + 0.1 * i - 0.05 * j;
      dhdx(i, j) = 0.5 + 0.01 * i;
      dhdy(i, j) = -0.25 + 0.02 * j;
    }
  }

  gpism::FieldStag2D<int> bc_mask(mx, my, gw);
  gpism::FieldStag2D<double> bc_values(mx, my, gw);
  bc_mask.fill(0);
  bc_values.fill(0.0);
  bc_mask(2, 1, 0) = 1;
  bc_values(2, 1, 0) = 3.25;
  bc_mask(1, 2, 1) = 1;
  bc_values(1, 2, 1) = -1.75;
  gpism::SSABoundaryCondition bc{&bc_mask, &bc_values};

  gpism::FieldStag2D<double> beta_ref(mx, my, gw);
  gpism::FieldStag2D<double> rhs_ref(mx, my, gw);
  gpism::FieldStag2D<double> nuH(mx, my, gw);
  gpism::FieldStag2D<double> vel(mx, my, gw);
  gpism::FieldStag2D<double> out_ref(mx, my, gw);

  nuH.fill(1.0);
  vel.fill(0.0);
  for (int j = 0; j < my; ++j) {
    for (int i = 0; i < mx; ++i) {
      nuH(i, j, 0) = 1.0 + 0.01 * i;
      nuH(i, j, 1) = 1.2 + 0.02 * j;
      vel(i, j, 0) = static_cast<double>(i - j);
      vel(i, j, 1) = static_cast<double>(2 * i + j);
    }
  }

  compute_basal_drag_ref(grid, tauc, beta_ref, u_threshold);
  assemble_rhs_ref(grid, thk, dhdx, dhdy, rhs_ref, rho, g, &bc);
  apply_ref(grid, nuH, beta_ref, vel, out_ref, &bc);

  gpism::FieldStag2D<double> beta_gpu(mx, my, gw);
  gpism::FieldStag2D<double> rhs_gpu(mx, my, gw);
  gpism::FieldStag2D<double> out_gpu(mx, my, gw);

  sync_host_to_device(tauc);
  sync_host_to_device(thk);
  sync_host_to_device(dhdx);
  sync_host_to_device(dhdy);
  sync_host_to_device(bc_mask);
  sync_host_to_device(bc_values);

  op.compute_basal_drag(grid, tauc, beta_gpu);
  sync_device_to_host(beta_gpu);
  if (!compare_stag(beta_gpu, beta_ref, "basal_drag")) {
    return 1;
  }

  sync_host_to_device(beta_ref);
  op.assemble_rhs(grid, thk, dhdx, dhdy, rhs_gpu, &bc);
  sync_device_to_host(rhs_gpu);
  if (!compare_stag(rhs_gpu, rhs_ref, "rhs")) {
    return 1;
  }

  sync_host_to_device(nuH);
  sync_host_to_device(beta_ref);
  sync_host_to_device(vel);
  op.apply(grid, nuH, beta_ref, vel, out_gpu, &bc);
  sync_device_to_host(out_gpu);
  if (!compare_stag(out_gpu, out_ref, "apply")) {
    return 1;
  }

  std::cout << "ssa_operator_cuda_smoke passed\n";
  return 0;
#endif
}
