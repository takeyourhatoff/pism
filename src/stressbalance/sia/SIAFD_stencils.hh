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

#ifndef PISM_SIAFD_STENCILS_HH
#define PISM_SIAFD_STENCILS_HH

#include <cmath>

#include "pism/util/error_handling.hh"

#ifdef __CUDACC__
#define PISM_HOST_DEVICE __host__ __device__
#else
#define PISM_HOST_DEVICE
#endif

namespace pism {
namespace stressbalance {
namespace sia_kernels {

PISM_HOST_DEVICE inline double pism_max(double a, double b) {
  return (a > b) ? a : b;
}

PISM_HOST_DEVICE inline double pism_min(double a, double b) {
  return (a < b) ? a : b;
}

PISM_HOST_DEVICE inline double pism_clip(double value, double lo, double hi) {
  return pism_max(lo, pism_min(value, hi));
}

struct MaskOps {
  PISM_HOST_DEVICE static inline bool ocean(int M) {
    return M >= 3;
  }
  PISM_HOST_DEVICE static inline bool grounded(int M) {
    return !ocean(M);
  }
  PISM_HOST_DEVICE static inline bool icy(int M) {
    return (M == 2) || (M == 3);
  }
  PISM_HOST_DEVICE static inline bool grounded_ice(int M) {
    return icy(M) && grounded(M);
  }
  PISM_HOST_DEVICE static inline bool floating_ice(int M) {
    return icy(M) && ocean(M);
  }
  PISM_HOST_DEVICE static inline bool ice_free(int M) {
    return !icy(M);
  }
  PISM_HOST_DEVICE static inline bool ice_free_ocean(int M) {
    return ocean(M) && ice_free(M);
  }
};

struct HostArray2DView {
  double **data;
  PISM_HOST_DEVICE inline double &operator()(int i, int j, int d = 0) const {
    (void)d;
    return data[j][i];
  }
};

struct HostArray2DConstView {
  const double *const *data;
  PISM_HOST_DEVICE inline double operator()(int i, int j, int d = 0) const {
    (void)d;
    return data[j][i];
  }
};

struct HostStaggeredView {
  double ***data;
  PISM_HOST_DEVICE inline double &operator()(int i, int j, int d) const {
    return data[j][i][d];
  }
};

struct HostStaggeredConstView {
  const double *const *const *data;
  PISM_HOST_DEVICE inline double operator()(int i, int j, int d) const {
    return data[j][i][d];
  }
};

struct HostMaskView {
  const double *const *data;
  PISM_HOST_DEVICE inline int value(int i, int j) const {
    return static_cast<int>(data[j][i]);
  }
};

struct DeviceArray2DView {
  double *data;
  int gxs;
  int gys;
  int gxm;
  int dof;
  PISM_HOST_DEVICE inline double &operator()(int i, int j, int d = 0) const {
    return data[((j - gys) * gxm + (i - gxs)) * dof + d];
  }
};

struct DeviceArray2DConstView {
  const double *data;
  int gxs;
  int gys;
  int gxm;
  int dof;
  PISM_HOST_DEVICE inline double operator()(int i, int j, int d = 0) const {
    return data[((j - gys) * gxm + (i - gxs)) * dof + d];
  }
};

struct DeviceMaskView {
  const double *data;
  int gxs;
  int gys;
  int gxm;
  int dof;
  PISM_HOST_DEVICE inline int value(int i, int j) const {
    return static_cast<int>(data[((j - gys) * gxm + (i - gxs)) * dof]);
  }
};

template <typename ScalarView, typename StaggeredView>
struct SurfaceGradientMahaffy {
  ScalarView h;
  StaggeredView h_x;
  StaggeredView h_y;
  double dx;
  double dy;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    h_x(i, j, 0) = (h(i + 1, j) - h(i, j)) / dx;
    h_y(i, j, 0) = (h(i + 1, j + 1) + h(i, j + 1) -
                    h(i + 1, j - 1) - h(i, j - 1)) / (4.0 * dy);
    h_y(i, j, 1) = (h(i, j + 1) - h(i, j)) / dy;
    h_x(i, j, 1) = (h(i + 1, j + 1) + h(i + 1, j) -
                    h(i - 1, j + 1) - h(i - 1, j)) / (4.0 * dx);
  }
};

template <typename InputView, typename OutputView>
struct EtaFromThickness {
  InputView H;
  OutputView eta;
  double etapow;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    eta(i, j) = pow(H(i, j), etapow);
  }
};

template <typename ScalarView, typename StaggeredView>
struct SurfaceGradientEta {
  ScalarView bed;
  ScalarView eta;
  StaggeredView h_x;
  StaggeredView h_y;
  double dx;
  double dy;
  double invpow;
  double dinvpow;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    const double b_c = bed(i, j);
    const double b_e = bed(i + 1, j);
    const double b_w = bed(i - 1, j);
    const double b_n = bed(i, j + 1);
    const double b_s = bed(i, j - 1);
    const double b_ne = bed(i + 1, j + 1);
    const double b_nw = bed(i - 1, j + 1);
    const double b_se = bed(i + 1, j - 1);

    const double e_c = eta(i, j);
    const double e_e = eta(i + 1, j);
    const double e_w = eta(i - 1, j);
    const double e_n = eta(i, j + 1);
    const double e_s = eta(i, j - 1);
    const double e_ne = eta(i + 1, j + 1);
    const double e_nw = eta(i - 1, j + 1);
    const double e_se = eta(i + 1, j - 1);

    {
      const double mean_eta = 0.5 * (e_e + e_c);
      if (mean_eta > 0.0) {
        const double factor = invpow * pow(mean_eta, dinvpow);
        h_x(i, j, 0) = factor * (e_e - e_c) / dx;
        h_y(i, j, 0) = factor * (e_ne + e_n - e_se - e_s) / (4.0 * dy);
      } else {
        h_x(i, j, 0) = 0.0;
        h_y(i, j, 0) = 0.0;
      }
      h_x(i, j, 0) += (b_e - b_c) / dx;
      h_y(i, j, 0) += (b_ne + b_n - b_se - b_s) / (4.0 * dy);
    }

    {
      const double mean_eta = 0.5 * (e_n + e_c);
      if (mean_eta > 0.0) {
        const double factor = invpow * pow(mean_eta, dinvpow);
        h_x(i, j, 1) = factor * (e_ne + e_e - e_nw - e_w) / (4.0 * dx);
        h_y(i, j, 1) = factor * (e_n - e_c) / dy;
      } else {
        h_x(i, j, 1) = 0.0;
        h_y(i, j, 1) = 0.0;
      }
      h_x(i, j, 1) += (b_ne + b_e - b_nw - b_w) / (4.0 * dx);
      h_y(i, j, 1) += (b_n - b_c) / dy;
    }
  }
};

template <typename ScalarInView, typename ScalarOutView, typename StaggeredView, typename MaskView>
struct SurfaceGradientHaseloffPrimary {
  ScalarInView h;
  StaggeredView h_x;
  StaggeredView h_y;
  ScalarOutView w_i;
  ScalarOutView w_j;
  MaskView mask;
  double dx;
  double dy;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    const int m_c = mask.value(i, j);
    const int m_e = mask.value(i + 1, j);
    const int m_n = mask.value(i, j + 1);

    if ((MaskOps::floating_ice(m_c) && MaskOps::ice_free_ocean(m_e)) ||
        (MaskOps::ice_free_ocean(m_c) && MaskOps::floating_ice(m_e))) {
      h_x(i, j, 0) = 0.0;
      w_i(i, j) = 0.0;
    } else if ((MaskOps::icy(m_c) && MaskOps::ice_free(m_e) && h(i + 1, j) > h(i, j)) ||
               (MaskOps::ice_free(m_c) && MaskOps::icy(m_e) && h(i, j) > h(i + 1, j))) {
      h_x(i, j, 0) = 0.0;
      w_i(i, j) = 0.0;
    } else {
      h_x(i, j, 0) = (h(i + 1, j) - h(i, j)) / dx;
      w_i(i, j) = 1.0;
    }

    if ((MaskOps::floating_ice(m_c) && MaskOps::ice_free_ocean(m_n)) ||
        (MaskOps::ice_free_ocean(m_c) && MaskOps::floating_ice(m_n))) {
      h_y(i, j, 1) = 0.0;
      w_j(i, j) = 0.0;
    } else if ((MaskOps::icy(m_c) && MaskOps::ice_free(m_n) && h(i, j + 1) > h(i, j)) ||
               (MaskOps::ice_free(m_c) && MaskOps::icy(m_n) && h(i, j) > h(i, j + 1))) {
      h_y(i, j, 1) = 0.0;
      w_j(i, j) = 0.0;
    } else {
      h_y(i, j, 1) = (h(i, j + 1) - h(i, j)) / dy;
      w_j(i, j) = 1.0;
    }
  }
};

template <typename ScalarInView, typename StaggeredView, typename MaskView>
struct SurfaceGradientHaseloffSecondary {
  StaggeredView h_x;
  StaggeredView h_y;
  ScalarInView w_i;
  ScalarInView w_j;
  MaskView mask;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    if (w_j(i, j) > 0.0) {
      const double W = w_i(i, j) + w_i(i - 1, j) + w_i(i - 1, j + 1) + w_i(i, j + 1);
      if (W > 0.0) {
        h_x(i, j, 1) = (h_x(i, j, 0) + h_x(i - 1, j, 0) +
                        h_x(i - 1, j + 1, 0) + h_x(i, j + 1, 0)) / W;
      } else {
        h_x(i, j, 1) = 0.0;
      }
    } else {
      if (MaskOps::icy(mask.value(i, j))) {
        const double W = w_i(i, j) + w_i(i - 1, j);
        if (W > 0.0) {
          h_x(i, j, 1) = (h_x(i, j, 0) + h_x(i - 1, j, 0)) / W;
        } else {
          h_x(i, j, 1) = 0.0;
        }
      } else {
        const double W = w_i(i, j + 1) + w_i(i - 1, j + 1);
        if (W > 0.0) {
          h_x(i, j, 1) = (h_x(i - 1, j + 1, 0) + h_x(i, j + 1, 0)) / W;
        } else {
          h_x(i, j, 1) = 0.0;
        }
      }
    }

    if (w_i(i, j) > 0.0) {
      const double W = w_j(i, j) + w_j(i, j - 1) + w_j(i + 1, j - 1) + w_j(i + 1, j);
      if (W > 0.0) {
        h_y(i, j, 0) = (h_y(i, j, 1) + h_y(i, j - 1, 1) +
                        h_y(i + 1, j - 1, 1) + h_y(i + 1, j, 1)) / W;
      } else {
        h_y(i, j, 0) = 0.0;
      }
    } else {
      if (MaskOps::icy(mask.value(i, j))) {
        const double W = w_j(i, j) + w_j(i, j - 1);
        if (W > 0.0) {
          h_y(i, j, 0) = (h_y(i, j, 1) + h_y(i, j - 1, 1)) / W;
        } else {
          h_y(i, j, 0) = 0.0;
        }
      } else {
        const double W = w_j(i + 1, j - 1) + w_j(i + 1, j);
        if (W > 0.0) {
          h_y(i, j, 0) = (h_y(i + 1, j - 1, 1) + h_y(i + 1, j, 1)) / W;
        } else {
          h_y(i, j, 0) = 0.0;
        }
      }
    }
  }
};

template <typename StaggeredInView, typename StaggeredOutView>
struct DiffusiveFlux {
  StaggeredInView h_x;
  StaggeredInView h_y;
  StaggeredInView D;
  StaggeredOutView result;
  int o;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    const double slope = (o == 0) ? h_x(i, j, o) : h_y(i, j, o);
    result(i, j, o) = -D(i, j, o) * slope;
  }
};

template <typename ScalarInView, typename ScalarOutView, typename MaskView>
struct BedSmoothThk {
  ScalarInView usurf;
  ScalarInView thk;
  ScalarInView maxtl;
  ScalarInView topgsmooth;
  MaskView mask;
  ScalarOutView result;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    const double H = thk(i, j);
    if (H == 0.0) {
      result(i, j) = 0.0;
    } else if (maxtl(i, j) >= H) {
      result(i, j) = H;
    } else {
      if (MaskOps::grounded(mask.value(i, j))) {
        const double val = usurf(i, j) - topgsmooth(i, j);
        result(i, j) = pism_max(val, 0.0);
      } else {
        result(i, j) = H;
      }
    }
  }
};

template <typename ScalarInView, typename ScalarOutView>
struct BedSmoothTheta {
  ScalarInView usurf;
  ScalarInView maxtl;
  ScalarInView topgsmooth;
  ScalarInView C2;
  ScalarInView C3;
  ScalarInView C4;
  ScalarOutView result;
  double theta_min;
  double theta_max;
  double Glen_exponent;
  PISM_HOST_DEVICE inline void operator()(int i, int j) const {
    const double H = usurf(i, j) - topgsmooth(i, j);
    if (H > maxtl(i, j)) {
      const double Hinv = 1.0 / pism_max(H, 1.0);
      double omega = 1.0 + Hinv * Hinv *
        (C2(i, j) + Hinv * (C3(i, j) + Hinv * C4(i, j)));
      if (omega <= 0.0) {
#ifndef __CUDA_ARCH__
        throw RuntimeError::formatted(PISM_ERROR_LOCATION,
                                      "omega is negative for i=%d, j=%d\n"
                                      "in BedSmoother.theta()", i, j);
#else
        omega = 0.001;
#endif
      }
      if (omega < 0.001) {
        omega = 0.001;
      }
      result(i, j) = pow(omega, -Glen_exponent);
    } else {
      result(i, j) = 0.0;
    }
    result(i, j) = pism_clip(result(i, j), theta_min, theta_max);
  }
};

} // namespace sia_kernels
} // namespace stressbalance
} // namespace pism

#endif // PISM_SIAFD_STENCILS_HH
