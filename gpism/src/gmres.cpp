#include "gpism/gmres.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gpism/config.h"
#include "gpism/context.h"
#include "gpism/linear_algebra.h"

#if GPISM_HAVE_MPI
#include <mpi.h>
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
    initialized = true;
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
    for (int j = 0; j < restart && total_iter < max_iter; ++j) {
      op.apply(*V[static_cast<std::size_t>(j)], Ax);
      M->apply(Ax, w);

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
