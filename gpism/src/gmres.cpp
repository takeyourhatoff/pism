#include "gpism/gmres.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/device_policy.h"
#include "gpism/linear_algebra.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
#endif
#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
namespace gpism {
void orthogonalize_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                             double* w_u, double* w_v, const double** V_u,
                             const double** V_v, int count,
                             const double* hij);
void dot_stag_batch_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                         const double* w_u, const double* w_v,
                         const double** V_u, const double** V_v, int count,
                         double* hij);
void dot_stag_self_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                        const double* w_u, const double* w_v, double* out);
void normalize_stag_cuda(int mx, int my, int gw, int stride_u, int stride_v,
                          const double* w_u, const double* w_v, double* out_u,
                          double* out_v, const double* norm_in);
void gmres_update_cuda(int restart, int j, double* H, double* cs, double* sn,
                       double* g, const double* hij);
}  // namespace gpism
#endif

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
  std::vector<double> H;
  std::vector<double> cs;
  std::vector<double> sn;
  std::vector<double> g;
  std::vector<double> y;
  std::vector<double> hij_host;
#if GPISM_HAVE_CUDA
  std::vector<const double*> V_u_host;
  std::vector<const double*> V_v_host;
  const double** V_u_dev = nullptr;
  const double** V_v_dev = nullptr;
  double* hij_dev = nullptr;
  double* H_dev = nullptr;
  double* g_dev = nullptr;
  double* cs_dev = nullptr;
  double* sn_dev = nullptr;
  int hij_cap = 0;
  int ptr_cap = 0;
  int h_cap = 0;
  int g_cap = 0;
#endif

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

    const int needed = restart + 1;
    if (static_cast<int>(V.size()) < needed) {
      V.reserve(static_cast<std::size_t>(needed));
      while (static_cast<int>(V.size()) < needed) {
        V.emplace_back(std::make_unique<FieldStag2D<double>>(mx, my, gw));
      }
    }
    if (dims_changed) {
      for (auto& vec : V) {
        if (vec) {
          vec->resize(mx, my, gw);
        }
      }
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
    if (hij_host.size() < static_cast<std::size_t>(restart + 1)) {
      hij_host.resize(static_cast<std::size_t>(restart + 1));
    }
    initialized = true;

#if GPISM_HAVE_CUDA
    if (device_enabled()) {
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
      }
      if (hij_cap < needed) {
        if (hij_dev) {
          cudaFree(hij_dev);
        }
        cudaMalloc(reinterpret_cast<void**>(&hij_dev),
                   static_cast<std::size_t>(needed) * sizeof(double));
        hij_cap = needed;
      }
      const int h_needed = (restart + 1) * restart;
      if (h_cap < h_needed) {
        if (H_dev) {
          cudaFree(H_dev);
        }
        cudaMalloc(reinterpret_cast<void**>(&H_dev),
                   static_cast<std::size_t>(h_needed) * sizeof(double));
        h_cap = h_needed;
      }
      if (g_cap < needed) {
        if (g_dev) {
          cudaFree(g_dev);
        }
        if (cs_dev) {
          cudaFree(cs_dev);
        }
        if (sn_dev) {
          cudaFree(sn_dev);
        }
        cudaMalloc(reinterpret_cast<void**>(&g_dev),
                   static_cast<std::size_t>(needed) * sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&cs_dev),
                   static_cast<std::size_t>(restart) * sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&sn_dev),
                   static_cast<std::size_t>(restart) * sizeof(double));
        g_cap = needed;
      }
    }
#endif
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
    const double tau = -a / b;
    s = 1.0 / std::sqrt(1.0 + tau * tau);
    c = s * tau;
  } else {
    const double tau = -b / a;
    c = 1.0 / std::sqrt(1.0 + tau * tau);
    s = c * tau;
  }
}

double global_sum(const Context* context, double local_value) {
#if GPISM_HAVE_MPI
  if (context && context->mpi_enabled()) {
    double global_value = 0.0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_value;
  }
#endif
  return local_value;
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

  IdentityPreconditioner identity;
  const Preconditioner* M = precond ? precond : &identity;

  GMRESWorkspace& workspace = gmres_workspace();
  workspace.ensure(x, restart);
  FieldStag2D<double>& Ax = workspace.Ax;
  FieldStag2D<double>& r = workspace.r;
  FieldStag2D<double>& z = workspace.z;
  FieldStag2D<double>& w = workspace.w;

  op.apply(x, Ax);
  copy(b, r);
  axpy(-1.0, Ax, r);
  M->apply(r, z);

  double beta = std::sqrt(global_sum(options.context, dot(z, z)));
  result.residual = beta;
  result.residuals.push_back(result.residual);
  if (beta <= options.tol) {
    result.converged = true;
    return result;
  }

  auto& V = workspace.V;

  int total_iter = 0;
  while (total_iter < max_iter) {
    auto& H = workspace.H;
    auto& cs = workspace.cs;
    auto& sn = workspace.sn;
    auto& g = workspace.g;
    const std::size_t hsize =
        static_cast<std::size_t>(restart + 1) * static_cast<std::size_t>(restart);
    std::fill(H.begin(), H.begin() + hsize, 0.0);
    std::fill(cs.begin(), cs.begin() + static_cast<std::size_t>(restart), 0.0);
    std::fill(sn.begin(), sn.begin() + static_cast<std::size_t>(restart), 0.0);
    std::fill(g.begin(), g.begin() + static_cast<std::size_t>(restart + 1), 0.0);

    g[0] = beta;
    copy(z, *V[0]);
    scal(1.0 / beta, *V[0]);

    int inner_iters = 0;
#if GPISM_HAVE_CUDA
    bool gpu_ortho = w.component(0).has_device_data() &&
                     w.component(1).has_device_data();
    if (gpu_ortho) {
      const int count_all = restart + 1;
      for (int i = 0; i < count_all; ++i) {
        const auto& Vi = *V[static_cast<std::size_t>(i)];
        workspace.V_u_host[static_cast<std::size_t>(i)] =
            Vi.component(0).device_data();
        workspace.V_v_host[static_cast<std::size_t>(i)] =
            Vi.component(1).device_data();
      }
      cudaMemcpy(workspace.V_u_dev, workspace.V_u_host.data(),
                 static_cast<std::size_t>(count_all) * sizeof(double*),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(workspace.V_v_dev, workspace.V_v_host.data(),
                 static_cast<std::size_t>(count_all) * sizeof(double*),
                 cudaMemcpyHostToDevice);
      cudaMemset(workspace.g_dev, 0,
                 static_cast<std::size_t>(count_all) * sizeof(double));
      cudaMemcpy(workspace.g_dev, &beta, sizeof(double),
                 cudaMemcpyHostToDevice);
      cudaMemset(workspace.cs_dev, 0,
                 static_cast<std::size_t>(restart) * sizeof(double));
      cudaMemset(workspace.sn_dev, 0,
                 static_cast<std::size_t>(restart) * sizeof(double));
    }
#endif
    for (int j = 0; j < restart && total_iter < max_iter; ++j) {
      op.apply(*V[static_cast<std::size_t>(j)], Ax);
      M->apply(Ax, w);

#if GPISM_HAVE_CUDA
      if (gpu_ortho) {
        const int count = j + 1;
        dot_stag_batch_cuda(
            w.local_mx(), w.local_my(), w.ghost_width(),
            w.component(0).stride(), w.component(1).stride(),
            w.component(0).device_data(), w.component(1).device_data(),
            workspace.V_u_dev, workspace.V_v_dev, count, workspace.hij_dev);
        if (options.context && options.context->mpi_enabled()) {
          if (options.context->cuda_aware_mpi()) {
            MPI_Allreduce(MPI_IN_PLACE, workspace.hij_dev, count,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          } else {
            cudaMemcpy(workspace.hij_host.data(), workspace.hij_dev,
                       static_cast<std::size_t>(count) * sizeof(double),
                       cudaMemcpyDeviceToHost);
            MPI_Allreduce(MPI_IN_PLACE, workspace.hij_host.data(), count,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cudaMemcpy(workspace.hij_dev, workspace.hij_host.data(),
                       static_cast<std::size_t>(count) * sizeof(double),
                       cudaMemcpyHostToDevice);
          }
        }
        orthogonalize_stag_cuda(
            w.local_mx(), w.local_my(), w.ghost_width(),
            w.component(0).stride(), w.component(1).stride(),
            w.component(0).device_data(), w.component(1).device_data(),
            workspace.V_u_dev, workspace.V_v_dev, count, workspace.hij_dev);
        dot_stag_self_cuda(
            w.local_mx(), w.local_my(), w.ghost_width(),
            w.component(0).stride(), w.component(1).stride(),
            w.component(0).device_data(), w.component(1).device_data(),
            workspace.hij_dev + count);
        if (options.context && options.context->mpi_enabled()) {
          if (options.context->cuda_aware_mpi()) {
            MPI_Allreduce(MPI_IN_PLACE, workspace.hij_dev + count, 1,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          } else {
            cudaMemcpy(workspace.hij_host.data() + count,
                       workspace.hij_dev + count, sizeof(double),
                       cudaMemcpyDeviceToHost);
            MPI_Allreduce(MPI_IN_PLACE, workspace.hij_host.data() + count, 1,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cudaMemcpy(workspace.hij_dev + count,
                       workspace.hij_host.data() + count, sizeof(double),
                       cudaMemcpyHostToDevice);
          }
        }
        if (count < restart + 1) {
          auto& v_next = *V[static_cast<std::size_t>(count)];
          normalize_stag_cuda(
              w.local_mx(), w.local_my(), w.ghost_width(),
              w.component(0).stride(), w.component(1).stride(),
              w.component(0).device_data(), w.component(1).device_data(),
              v_next.component(0).device_data(),
              v_next.component(1).device_data(),
              workspace.hij_dev + count);
        }
        gmres_update_cuda(restart, j, workspace.H_dev, workspace.cs_dev,
                          workspace.sn_dev, workspace.g_dev,
                          workspace.hij_dev);
        double g_next = 0.0;
        cudaMemcpy(&g_next, workspace.g_dev + j + 1, sizeof(double),
                   cudaMemcpyDeviceToHost);
        result.residual = std::abs(g_next);
        result.residuals.push_back(result.residual);
        ++total_iter;
        ++inner_iters;
        if (result.residual <= options.tol) {
          break;
        }
        const double h_next = 1.0;
        (void)h_next;
      } else
#endif
      {
        for (int i = 0; i <= j; ++i) {
          const double hij = global_sum(options.context, dot(
              w, *V[static_cast<std::size_t>(i)]));
          H[static_cast<std::size_t>(i) +
            static_cast<std::size_t>(restart + 1) * j] = hij;
          axpy(-hij, *V[static_cast<std::size_t>(i)], w);
        }

        const double h_next = std::sqrt(global_sum(options.context, dot(w, w)));
        H[static_cast<std::size_t>(j + 1) +
          static_cast<std::size_t>(restart + 1) * j] = h_next;
        if (h_next != 0.0) {
          copy(w, *V[static_cast<std::size_t>(j + 1)]);
          scal(1.0 / h_next, *V[static_cast<std::size_t>(j + 1)]);
        }
      }

#if GPISM_HAVE_CUDA
      if (gpu_ortho) {
        continue;
      }
#endif

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
                   sn[static_cast<std::size_t>(j)], g[static_cast<std::size_t>(j)],
                   g[static_cast<std::size_t>(j + 1)]);

      result.residual = std::abs(g[static_cast<std::size_t>(j + 1)]);
      result.residuals.push_back(result.residual);
      ++total_iter;
      ++inner_iters;

      if (result.residual <= options.tol) {
        break;
      }
    }

    const int k = inner_iters;
    auto& y = workspace.y;
    if (y.size() < static_cast<std::size_t>(k)) {
      y.resize(static_cast<std::size_t>(k));
    }
    std::fill(y.begin(), y.begin() + static_cast<std::size_t>(k), 0.0);
    if (k > 0) {
#if GPISM_HAVE_CUDA
      if (gpu_ortho) {
        const std::size_t hsize_full =
            static_cast<std::size_t>(restart + 1) * static_cast<std::size_t>(restart);
        cudaMemcpy(H.data(), workspace.H_dev, hsize_full * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(g.data(), workspace.g_dev,
                   static_cast<std::size_t>(restart + 1) * sizeof(double),
                   cudaMemcpyDeviceToHost);
      }
#endif
    }
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

    op.apply(x, Ax);
    copy(b, r);
    axpy(-1.0, Ax, r);
    M->apply(r, z);
    beta = std::sqrt(global_sum(options.context, dot(z, z)));
    result.residual = beta;
    result.residuals.push_back(result.residual);
    if (result.residual <= options.tol) {
      break;
    }
  }

  result.iterations = total_iter;
  result.converged = (result.residual <= options.tol);
  return result;
}

}  // namespace gpism
