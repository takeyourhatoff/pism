#include "gpism/gmres.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/linear_algebra.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif
#include <cuda_runtime.h>
namespace gpism {
void orthogonalize_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                             double* w_u, double* w_v, const double** V_u,
                             const double** V_v, int count, double* hij);
}  // namespace gpism

namespace gpism {
namespace {

struct GMRESWorkspace {
  int mx = 0;
  int my = 0;
  int gw = 0;
  int restart_cap = 0;
  bool initialized = false;
  FieldStag2D<double> Ax;
  FieldStag2D<double> r;
  FieldStag2D<double> z;
  FieldStag2D<double> w;
  std::vector<std::unique_ptr<FieldStag2D<double>>> V;
  std::vector<std::unique_ptr<FieldStag2D<double>>> Z;
  std::vector<double> H;
  std::vector<double> cs;
  std::vector<double> sn;
  std::vector<double> g;
  std::vector<double> y;
  DeviceScalarBuffer hessenberg_dev;
  DeviceScalarBuffer cs_dev;
  DeviceScalarBuffer sn_dev;
  DeviceScalarBuffer g_dev;
  DeviceScalarBuffer y_dev;
  std::vector<const double*> V_u_host;
  std::vector<const double*> V_v_host;
  const double** V_u_dev = nullptr;
  const double** V_v_dev = nullptr;
  int ptr_cap = 0;
  bool ptr_tables_dirty = true;

  void ensure(const FieldStag2D<double>& ref, int restart) {
    const int mx_new = ref.local_mx();
    const int my_new = ref.local_my();
    const int gw_new = ref.ghost_width();
    const bool dims_changed =
        (!initialized || mx != mx_new || my != my_new || gw != gw_new);
    mx = mx_new;
    my = my_new;
    gw = gw_new;
    if (!initialized) {
      Ax.resize(mx, my, gw);
      r.resize(mx, my, gw);
      z.resize(mx, my, gw);
      w.resize(mx, my, gw);
    } else if (dims_changed) {
      Ax.resize(mx, my, gw);
      r.resize(mx, my, gw);
      z.resize(mx, my, gw);
      w.resize(mx, my, gw);
    }

    const int needed_v = restart + 1;
    if (static_cast<int>(V.size()) < needed_v) {
      V.reserve(static_cast<std::size_t>(needed_v));
      while (static_cast<int>(V.size()) < needed_v) {
        V.emplace_back(std::make_unique<FieldStag2D<double>>(mx, my, gw));
      }
      ptr_tables_dirty = true;
    }
    const int needed_z = restart;
    if (static_cast<int>(Z.size()) < needed_z) {
      Z.reserve(static_cast<std::size_t>(needed_z));
      while (static_cast<int>(Z.size()) < needed_z) {
        Z.emplace_back(std::make_unique<FieldStag2D<double>>(mx, my, gw));
      }
    }
    if (dims_changed) {
      for (auto& vec : V) {
        if (vec) {
          vec->resize(mx, my, gw);
        }
      }
      for (auto& vec : Z) {
        if (vec) {
          vec->resize(mx, my, gw);
        }
      }
      ptr_tables_dirty = true;
    }

    restart_cap = std::max(restart_cap, restart);
    const std::size_t hsize =
        static_cast<std::size_t>(restart + 1) * static_cast<std::size_t>(restart);
    if (H.size() < hsize) {
      H.resize(hsize);
    }
    if (cs.size() < static_cast<std::size_t>(restart)) {
      cs.resize(static_cast<std::size_t>(restart));
    }
    if (sn.size() < static_cast<std::size_t>(restart)) {
      sn.resize(static_cast<std::size_t>(restart));
    }
    if (g.size() < static_cast<std::size_t>(restart + 1)) {
      g.resize(static_cast<std::size_t>(restart + 1));
    }
    if (y.size() < static_cast<std::size_t>(restart)) {
      y.resize(static_cast<std::size_t>(restart));
    }
    hessenberg_dev.ensure(static_cast<int>(hsize));
    cs_dev.ensure(restart);
    sn_dev.ensure(restart);
    g_dev.ensure(restart + 1);
    y_dev.ensure(restart);
    initialized = true;

    const int needed = restart + 1;
    if (static_cast<int>(V_u_host.size()) < needed) {
      V_u_host.resize(static_cast<std::size_t>(needed));
      V_v_host.resize(static_cast<std::size_t>(needed));
    }
    if (ptr_cap < needed) {
      if (V_u_dev) {
        cudaFree(const_cast<double**>(V_u_dev));
        cudaFree(const_cast<double**>(V_v_dev));
      }
      cudaMalloc(reinterpret_cast<void**>(&V_u_dev),
                 static_cast<std::size_t>(needed) * sizeof(double*));
      cudaMalloc(reinterpret_cast<void**>(&V_v_dev),
                 static_cast<std::size_t>(needed) * sizeof(double*));
      ptr_cap = needed;
      ptr_tables_dirty = true;
    }
  }
};

GMRESWorkspace& gmres_workspace() {
  static GMRESWorkspace workspace;
  return workspace;
}

void apply_givens(double c, double s, double& v0, double& v1) {
  const double temp = c * v0 + s * v1;
  v1 = -s * v0 + c * v1;
  v0 = temp;
}

void compute_givens(double a, double b, double& c, double& s) {
  if (b == 0.0) {
    c = 1.0;
    s = 0.0;
    return;
  }
  if (std::abs(b) > std::abs(a)) {
    const double tau = a / b;
    s = 1.0 / std::sqrt(1.0 + tau * tau);
    c = s * tau;
  } else {
    const double tau = b / a;
    c = 1.0 / std::sqrt(1.0 + tau * tau);
    s = c * tau;
  }
}

double global_sum(const Context* context, double local_value) {
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled() && context->size() > 1) {
    double global_value = 0.0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_value;
  }
#endif
  return local_value;
}

bool is_rank0(const Context* context) {
  if (!context || !context->mpi_enabled()) {
    return true;
  }
  return context->rank() == 0;
}

}  // namespace

void IdentityPreconditioner::apply(const FieldStag2D<double>& x,
                                   FieldStag2D<double>& y) const {
  copy(x, y);
}

GMRESResult gmres_solve(const LinearOperator& op, const FieldStag2D<double>& b,
                        FieldStag2D<double>& x, const GMRESOptions& options,
                        const Preconditioner* precond) {
  GMRESResult result{};
  const int restart = std::max(1, options.restart);
  const int max_iter = std::max(1, options.max_iter);
  const int residual_check_interval = std::max(1, options.residual_check_interval);

  IdentityPreconditioner identity;
  const Preconditioner* M = precond ? precond : &identity;

  GMRESWorkspace& workspace = gmres_workspace();
  workspace.ensure(x, restart);
  FieldStag2D<double>& Ax = workspace.Ax;
  FieldStag2D<double>& r = workspace.r;
  FieldStag2D<double>& z = workspace.z;
  FieldStag2D<double>& w = workspace.w;
  auto& V = workspace.V;
  const int needed = restart + 1;
  if (workspace.ptr_tables_dirty) {
    for (int i = 0; i < needed; ++i) {
      const auto& Vi = *V[static_cast<std::size_t>(i)];
      workspace.V_u_host[static_cast<std::size_t>(i)] =
          Vi.component(0).device_data();
      workspace.V_v_host[static_cast<std::size_t>(i)] =
          Vi.component(1).device_data();
    }
    cudaMemcpy(workspace.V_u_dev, workspace.V_u_host.data(),
               static_cast<std::size_t>(needed) * sizeof(double*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(workspace.V_v_dev, workspace.V_v_host.data(),
               static_cast<std::size_t>(needed) * sizeof(double*),
               cudaMemcpyHostToDevice);
    workspace.ptr_tables_dirty = false;
  }

  // Left-preconditioned GMRES (PETSc KSPGMRES default): solve (M^{-1} A) x =
  // (M^{-1} b), monitoring ||M^{-1}(b - A x)||. This matches PISM SSAFD's
  // default convergence behavior and keeps the GMRES residual in velocity-like
  // units.

  double b_norm = 0.0;
  if (options.tol_relative && options.tol_relative_to_rhs) {
    M->apply(b, z);
    b_norm = std::sqrt(global_sum(options.context, dot(z, z)));
  }

  op.apply(x, Ax);
  copy(b, r);
  axpy(-1.0, Ax, r);

  // Preconditioned residual z = M^{-1} r.
  M->apply(r, z);
  double beta = std::sqrt(global_sum(options.context, dot(z, z)));
  result.residual = beta;
  result.true_residual = std::sqrt(global_sum(options.context, dot(r, r)));
  result.residuals.push_back(result.residual);
  if (!std::isfinite(beta)) {
    if (options.verbose && is_rank0(options.context)) {
      std::cout << "GMRES init: beta0=" << beta << " (non-finite)\n";
    }
    return result;
  }
  const double beta0 = beta;

  if (options.precond_diagnostic) {
    const double z_norm = beta0;
    // Check how close M^{-1}A is to identity for the initial residual:
    // p = M^{-1} A z, then compare ||z - p|| / ||z||.
    op.apply(z, w);
    M->apply(w, Ax);  // reuse Ax as p
    copy(z, w);
    axpy(-1.0, Ax, w);
    const double z_mp_norm =
        std::sqrt(global_sum(options.context, dot(w, w)));
    if (is_rank0(options.context)) {
      std::cout << "GMRES preconditioning: left (Krylov on M^{-1} A), "
                   "monitoring ||M^{-1}(b-Ax)||\n";
      std::cout << "GMRES preconditioner diagnostic: ||M^{-1} r0||=" << z_norm
                << " ||z - M^{-1}A z||=" << z_mp_norm;
      if (z_norm > 0.0) {
        std::cout << " ratio=" << (z_mp_norm / z_norm);
      }
      std::cout << '\n';
    }
  }

  double tol_abs = options.tol;
  if (options.tol_relative && options.tol_relative_to_rhs) {
    tol_abs = options.tol * std::max(beta0, b_norm);
  } else if (options.tol_relative) {
    tol_abs = options.tol * beta0;
  }
  if (options.verbose && is_rank0(options.context)) {
    std::cout << "GMRES init: beta0=" << beta0
              << " r0=" << result.true_residual;
    if (options.tol_relative_to_rhs) {
      std::cout << " b=" << b_norm;
    }
    std::cout << " tol=" << options.tol
              << " tol_relative=" << (options.tol_relative ? "yes" : "no")
              << " tol_relative_to_rhs="
              << (options.tol_relative_to_rhs ? "yes" : "no")
              << " tol_abs=" << tol_abs
              << " restart=" << restart
              << " max_iter=" << max_iter
              << '\n';
  }
  if (beta <= tol_abs) {
    result.converged = true;
    return result;
  }

  int total_iter = 0;
  while (total_iter < max_iter) {
    auto& H = workspace.H;
    auto& cs = workspace.cs;
    auto& sn = workspace.sn;
    auto& g = workspace.g;
    const std::size_t hsize =
        static_cast<std::size_t>(restart + 1) * static_cast<std::size_t>(restart);
    const bool multi_rank =
#if GPISM_HAVE_MPI
        (options.context && options.context->mpi_enabled() &&
         options.context->size() > 1);
#else
        false;
#endif
    const bool gpu_ortho = z.component(0).has_device_data() &&
                           z.component(1).has_device_data() && !multi_rank;
    if (gpu_ortho) {
      cudaMemset(workspace.hessenberg_dev.device_data(), 0, hsize * sizeof(double));
      cudaMemset(workspace.cs_dev.device_data(), 0,
                 static_cast<std::size_t>(restart) * sizeof(double));
      cudaMemset(workspace.sn_dev.device_data(), 0,
                 static_cast<std::size_t>(restart) * sizeof(double));
      cudaMemset(workspace.g_dev.device_data(), 0,
                 static_cast<std::size_t>(restart + 1) * sizeof(double));
      cudaMemcpy(workspace.g_dev.device_data(), &beta, sizeof(double),
                 cudaMemcpyHostToDevice);
    } else {
      std::fill(H.begin(), H.begin() + hsize, 0.0);
      std::fill(cs.begin(), cs.begin() + static_cast<std::size_t>(restart), 0.0);
      std::fill(sn.begin(), sn.begin() + static_cast<std::size_t>(restart), 0.0);
      std::fill(g.begin(), g.begin() + static_cast<std::size_t>(restart + 1), 0.0);
      g[0] = beta;
    }

    copy(z, *V[0]);
    if (beta != 0.0) {
      scal(1.0 / beta, *V[0]);
    }

    int inner_iters = 0;
    for (int j = 0; j < restart && total_iter < max_iter; ++j) {
      // Left preconditioning:
      //   w = A v_j
      //   z = M^{-1} w
      op.apply(*V[static_cast<std::size_t>(j)], w);
      M->apply(w, z);

      if (gpu_ortho) {
        const int count = j + 1;
        const std::size_t col_offset =
            static_cast<std::size_t>(j) * static_cast<std::size_t>(restart + 1);
        double* const h_col_dev =
            workspace.hessenberg_dev.device_data() + col_offset;
        orthogonalize_stag_cuda(
            z.local_mx(), z.local_my(), z.ghost_width(),
            z.component(0).stride(), z.component(1).stride(),
            z.component(0).device_data(), z.component(1).device_data(),
            workspace.V_u_dev, workspace.V_v_dev, count, h_col_dev);
        normalize_stag_from_norm2_pointer_device(
            z, *V[static_cast<std::size_t>(j + 1)], h_col_dev + count);
        gmres_apply_givens_device(workspace.hessenberg_dev, workspace.cs_dev,
                                  workspace.sn_dev, workspace.g_dev, restart, j);
      } else {
        double h_next = 0.0;
        for (int i = 0; i <= j; ++i) {
          const double hij = global_sum(options.context, dot(
              z, *V[static_cast<std::size_t>(i)]));
          H[static_cast<std::size_t>(i) +
            static_cast<std::size_t>(restart + 1) * j] = hij;
          axpy(-hij, *V[static_cast<std::size_t>(i)], z);
        }
        h_next = std::sqrt(global_sum(options.context, dot(z, z)));
        H[static_cast<std::size_t>(j + 1) +
          static_cast<std::size_t>(restart + 1) * j] = h_next;
        if (h_next != 0.0) {
          copy(z, *V[static_cast<std::size_t>(j + 1)]);
          scal(1.0 / h_next, *V[static_cast<std::size_t>(j + 1)]);
        }

        for (int i = 0; i < j; ++i) {
          double& h0 =
              H[static_cast<std::size_t>(i) +
                static_cast<std::size_t>(restart + 1) * j];
          double& h1 =
              H[static_cast<std::size_t>(i + 1) +
                static_cast<std::size_t>(restart + 1) * j];
          apply_givens(cs[static_cast<std::size_t>(i)],
                       sn[static_cast<std::size_t>(i)], h0, h1);
        }

        double& h00 =
            H[static_cast<std::size_t>(j) +
              static_cast<std::size_t>(restart + 1) * j];
        double& h10 =
            H[static_cast<std::size_t>(j + 1) +
              static_cast<std::size_t>(restart + 1) * j];
        compute_givens(h00, h10, cs[static_cast<std::size_t>(j)],
                       sn[static_cast<std::size_t>(j)]);
        apply_givens(cs[static_cast<std::size_t>(j)],
                     sn[static_cast<std::size_t>(j)], h00, h10);

        apply_givens(cs[static_cast<std::size_t>(j)],
                     sn[static_cast<std::size_t>(j)],
                     g[static_cast<std::size_t>(j)],
                     g[static_cast<std::size_t>(j + 1)]);

        result.residual = std::abs(g[static_cast<std::size_t>(j + 1)]);
      }
      ++total_iter;
      ++inner_iters;
      if (gpu_ortho) {
        const bool do_check = (total_iter % residual_check_interval == 0) ||
                              (j + 1 == restart) || (total_iter == max_iter);
        if (do_check) {
          cudaMemcpy(workspace.g_dev.host_data(),
                     workspace.g_dev.device_data() + (j + 1), sizeof(double),
                     cudaMemcpyDeviceToHost);
          result.residual = std::abs(workspace.g_dev.host_data()[0]);
          result.residuals.push_back(result.residual);
          if (result.residual <= tol_abs) {
            break;
          }
        }
      } else {
        if (total_iter % residual_check_interval == 0 ||
            result.residual <= tol_abs || j + 1 == restart ||
            total_iter == max_iter) {
          result.residuals.push_back(result.residual);
        }

        if (result.residual <= tol_abs) {
          break;
        }
      }
    }

    const int k = inner_iters;
    if (gpu_ortho) {
      gmres_backsolve_device(workspace.hessenberg_dev, workspace.g_dev,
                             workspace.y_dev, restart, k);
      gmres_update_solution_device(workspace.y_dev, k, *V[0], workspace.V_u_dev,
                                   workspace.V_v_dev, x);
    } else {
      auto& y = workspace.y;
      if (y.size() < static_cast<std::size_t>(k)) {
        y.resize(static_cast<std::size_t>(k));
      }
      std::fill(y.begin(), y.begin() + static_cast<std::size_t>(k), 0.0);
      for (int i = k - 1; i >= 0; --i) {
        double sum = g[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < k; ++j) {
          sum -= H[static_cast<std::size_t>(i) +
                   static_cast<std::size_t>(restart + 1) * j] *
                 y[static_cast<std::size_t>(j)];
        }
        const double h_ii =
            H[static_cast<std::size_t>(i) +
              static_cast<std::size_t>(restart + 1) * i];
        y[static_cast<std::size_t>(i)] =
            (h_ii == 0.0) ? 0.0 : (sum / h_ii);
      }

      for (int i = 0; i < k; ++i) {
        axpy(y[static_cast<std::size_t>(i)],
             *V[static_cast<std::size_t>(i)], x);
      }
    }

    op.apply(x, Ax);
    copy(b, r);
    axpy(-1.0, Ax, r);
    M->apply(r, z);
    beta = std::sqrt(global_sum(options.context, dot(z, z)));
    result.residual = beta;
    result.true_residual = std::sqrt(global_sum(options.context, dot(r, r)));
    if (total_iter % residual_check_interval == 0 || beta <= tol_abs ||
        total_iter == max_iter) {
      result.residuals.push_back(result.residual);
    }
    if (!std::isfinite(beta)) {
      break;
    }
    if (beta <= tol_abs) {
      break;
    }
  }

  result.iterations = total_iter;
  result.converged =
      std::isfinite(result.residual) && (result.residual <= tol_abs);
  if (options.verbose && is_rank0(options.context)) {
    std::cout << "GMRES done: iters=" << result.iterations
              << " residual=" << result.residual
              << " true_residual=" << result.true_residual
              << " converged=" << (result.converged ? "yes" : "no")
              << '\n';
  }
  return result;
}

}  // namespace gpism
