// Copyright (C) 2004--2023, 2025 Jed Brown, Craig Lingle, Ed Bueler and Constantine Khroulev
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

#include <cstdlib>
#include <cassert>

#include "pism/stressbalance/sia/BedSmoother.hh"
#include "pism/stressbalance/sia/SIAFD.hh"
#include "pism/stressbalance/sia/SIAFD_stencils.hh"
#include "pism/pism_config.hh"
#if Pism_USE_CUDA_SIA
#include "pism/stressbalance/sia/SIAFD_cuda.hh"
#endif
#include "pism/geometry/Geometry.hh"
#include "pism/rheology/FlowLawFactory.hh"
#include "pism/rheology/grain_size_vostok.hh"
#include "pism/stressbalance/StressBalance.hh"
#include "pism/util/EnthalpyConverter.hh"
#include "pism/util/Grid.hh"
#include "pism/util/Profiling.hh"
#include "pism/util/Time.hh"
#include "pism/util/array/CellType.hh"
#include "pism/util/array/Scalar.hh"
#include "pism/util/error_handling.hh"
#include "pism/util/petscwrappers/Vec.hh"
#include "pism/util/pism_utilities.hh"
#include "pism/util/array/Vector.hh"

namespace {
constexpr unsigned int GEOM_SURFACE = 1u << 0;
constexpr unsigned int GEOM_THICKNESS = 1u << 1;
constexpr unsigned int GEOM_BED = 1u << 2;
constexpr unsigned int GEOM_CELL_TYPE = 1u << 3;
} // namespace

namespace pism {
namespace stressbalance {

SIAFD::SIAFD(std::shared_ptr<const Grid> g)
    : SSB_Modifier(std::move(g)),
      m_stencil_width(m_config->get_number("grid.max_stencil_width")),
      m_work_2d_0(m_grid, "work_vector_2d_0"),
      m_work_2d_1(m_grid, "work_vector_2d_1"),
      m_h_x(m_grid, "h_x"),
      m_h_y(m_grid, "h_y"),
      m_D(m_grid, "diffusivity"),
      m_delta_0(m_grid, "delta_0", array::WITH_GHOSTS, m_grid->z()),
      m_delta_1(m_grid, "delta_1", array::WITH_GHOSTS, m_grid->z()),
      m_work_3d_0(m_grid, "work_3d_0", array::WITH_GHOSTS, m_grid->z()),
      m_work_3d_1(m_grid, "work_3d_1", array::WITH_GHOSTS, m_grid->z()),
      m_I_valid(false) {
  // bed smoother
  m_bed_smoother = new BedSmoother(m_grid);

  m_seconds_per_year = units::convert(m_sys, 1, "second", "years");

  m_e_factor = m_config->get_number("stress_balance.sia.enhancement_factor");
  m_e_factor_interglacial =
      m_config->get_number("stress_balance.sia.enhancement_factor_interglacial");

  {
    rheology::FlowLawFactory ice_factory("stress_balance.sia.", m_config, m_EC);
    m_flow_law = ice_factory.create();
  }

  const bool compute_grain_size_using_age =
      m_config->get_flag("stress_balance.sia.grain_size_age_coupling");
  const bool age_model_enabled = m_config->get_flag("age.enabled");
  const bool e_age_coupling    = m_config->get_flag("stress_balance.sia.e_age_coupling");

  if (compute_grain_size_using_age) {
    if (not FlowLawUsesGrainSize(*m_flow_law)) {
      throw RuntimeError::formatted(PISM_ERROR_LOCATION,
                                    "flow law %s does not use grain size "
                                    "but sia.grain_size_age_coupling was set",
                                    m_flow_law->name().c_str());
    }

    if (not age_model_enabled) {
      throw RuntimeError::formatted(PISM_ERROR_LOCATION,
                                    "SIAFD: age model is not active but\n"
                                    "age is needed for grain-size-based flow law %s",
                                    m_flow_law->name().c_str());
    }
  }

  if (e_age_coupling and not age_model_enabled) {
    throw RuntimeError(PISM_ERROR_LOCATION, "SIAFD: age model is not active but\n"
                                            "age is needed for age-dependent flow enhancement");
  }

  m_eemian_start   = m_config->get_number("time.eemian_start", "seconds");
  m_eemian_end     = m_config->get_number("time.eemian_end", "seconds");
  m_holocene_start = m_config->get_number("time.holocene_start", "seconds");
}

SIAFD::~SIAFD() {
  delete m_bed_smoother;
}

//! \brief Initialize the SIA module.
void SIAFD::init() {

  SSB_Modifier::init();

  m_log->message(2, "* Initializing the SIA stress balance modifier...\n");
  m_log->message(2, "  [using the %s flow law]\n", m_flow_law->name().c_str());


  // implements an option e.g. described in @ref Greve97Greenland that is the
  // enhancement factor is coupled to the age of the ice
  if (m_config->get_flag("stress_balance.sia.e_age_coupling")) {
    m_log->message(2,
                   "  using age-dependent enhancement factor:\n"
                   "  e=%f for ice accumulated during interglacial periods\n"
                   "  e=%f for ice accumulated during glacial periods\n",
                   m_e_factor_interglacial, m_e_factor);
  }
}

//! \brief Do the update; if full_update == false skip the update of 3D velocities and strain
//! heating.
void SIAFD::update(const array::Vector &sliding_velocity, const Inputs &inputs, bool full_update) {
  m_I_valid = false;
  m_geometry_ghosts_valid_mask = 0;
  m_diffusive_flux_valid = false;

  // Check if the smoothed bed computed by BedSmoother is out of date and
  // recompute if necessary.
  if (inputs.new_bed_elevation) {
    profiling().begin("sia.bed_smoother");
    m_bed_smoother->preprocess_bed(inputs.geometry->bed_elevation);
    profiling().end("sia.bed_smoother");
  }

  profiling().begin("sia.gradient");
  const std::string gradient_method =
      m_config->get_string("stress_balance.sia.surface_gradient_method");
  const bool fuse_mahaffy =
      (gradient_method == "mahaffy") &&
      can_use_cuda_diffusivity(*inputs.geometry, inputs.enthalpy, inputs.age, m_h_x, m_h_y, m_D);
  if (!fuse_mahaffy) {
    compute_surface_gradient(inputs, m_h_x, m_h_y);
  }
  profiling().end("sia.gradient");

  profiling().begin("sia.flux");
  compute_diffusivity(full_update, *inputs.geometry, inputs.enthalpy, inputs.age, m_h_x, m_h_y,
                      m_D, fuse_mahaffy);
  if (!m_diffusive_flux_valid) {
    compute_diffusive_flux(m_h_x, m_h_y, m_D, m_diffusive_flux);
  }
  profiling().end("sia.flux");

  if (full_update) {
    profiling().begin("sia.3d_velocity");
    compute_3d_horizontal_velocity(*inputs.geometry, m_h_x, m_h_y, sliding_velocity, m_u, m_v);
    profiling().end("sia.3d_velocity");
  }

  m_geometry_ghosts_valid_mask = 0;
}


//! \brief Compute the ice surface gradient for the SIA.
/*!
  There are three methods for computing the surface gradient. Which method is
  controlled by configuration parameter `sia.surface_gradient_method` which can
  have values `haseloff`, `mahaffy`, or `eta`.

  The most traditional method is to directly differentiate the surface
  elevation \f$h\f$ by the Mahaffy method [\ref Mahaffy]. The `haseloff` method,
  suggested by Marianne Haseloff, modifies the Mahaffy method only where
  ice-free adjacent bedrock points are above the ice surface, and in those
  cases the returned gradient component is zero.

  The alternative method, when `sia.surface_gradient_method` = `eta`, transforms
  the thickness to something more regular and differentiates that. We get back
  to the gradient of the surface by applying the chain rule. In particular, as
  shown in [\ref Vazquezetal2003] for the flat bed and \f$n=3\f$ case, if we define

  \f[\eta = H^{(2n+2)/n}\f]

  then \f$\eta\f$ is more regular near the margin than \f$H\f$. So we compute
  the surface gradient by

  \f[\nabla h = \frac{n}{(2n+2)} \eta^{(-n-2)/(2n+2)} \nabla \eta + \nabla b,\f]

  recalling that \f$h = H + b\f$. This method is only applied when \f$\eta >
  0\f$ at a given point; otherwise \f$\nabla h = \nabla b\f$.

  In all cases we are computing the gradient by finite differences onto a
  staggered grid. In the method with \f$\eta\f$ we apply centered differences
  using (roughly) the same method for \f$\eta\f$ and \f$b\f$ that applies
  directly to the surface elevation \f$h\f$ in the `mahaffy` and `haseloff`
  methods.

  \param[out] h_x the X-component of the surface gradient, on the staggered grid
  \param[out] h_y the Y-component of the surface gradient, on the staggered grid
*/
void SIAFD::compute_surface_gradient(const Inputs &inputs, array::Staggered &h_x,
                                     array::Staggered &h_y) {
  const std::string method = m_config->get_string("stress_balance.sia.surface_gradient_method");
  unsigned int geom_mask = 0;
  if (method == "eta") {
    geom_mask = GEOM_THICKNESS | GEOM_BED;
  } else if (method == "haseloff") {
    geom_mask = GEOM_SURFACE | GEOM_CELL_TYPE;
  } else if (method == "mahaffy") {
    geom_mask = GEOM_SURFACE;
  }
  ensure_geometry_ghosts(*inputs.geometry, geom_mask);

  if (method == "eta") {

    surface_gradient_eta(inputs.geometry->ice_thickness, inputs.geometry->bed_elevation, h_x, h_y);

  } else if (method == "haseloff") {

    surface_gradient_haseloff(inputs.geometry->ice_surface_elevation, inputs.geometry->cell_type,
                              h_x, h_y);

  } else if (method == "mahaffy") {

    surface_gradient_mahaffy(inputs.geometry->ice_surface_elevation, h_x, h_y);

  } else {
    throw RuntimeError::formatted(
        PISM_ERROR_LOCATION,
        "value of sia.surface_gradient_method, option '-gradient %s', is not valid",
        method.c_str());
  }
}

void SIAFD::ensure_geometry_ghosts(const Geometry &geometry, unsigned int mask) {
#if Pism_USE_CUDA_SIA
  if (mask == 0) {
    return;
  }

  if ((m_geometry_ghosts_valid_mask & mask) == mask) {
    return;
  }

  const bool surface_cuda = (mask & GEOM_SURFACE) &&
      cuda::vec_is_cuda(geometry.ice_surface_elevation);
  const bool thickness_cuda = (mask & GEOM_THICKNESS) &&
      cuda::vec_is_cuda(geometry.ice_thickness);
  const bool bed_cuda = (mask & GEOM_BED) &&
      cuda::vec_is_cuda(geometry.bed_elevation);
  const bool cell_cuda = (mask & GEOM_CELL_TYPE) &&
      cuda::vec_is_cuda(geometry.cell_type);

  if (!(surface_cuda || thickness_cuda || bed_cuda || cell_cuda)) {
    return;
  }

  if ((mask & GEOM_SURFACE) && !(m_geometry_ghosts_valid_mask & GEOM_SURFACE)) {
    const_cast<array::Scalar2 &>(geometry.ice_surface_elevation).update_ghosts();
    m_geometry_ghosts_valid_mask |= GEOM_SURFACE;
  }
  if ((mask & GEOM_THICKNESS) && !(m_geometry_ghosts_valid_mask & GEOM_THICKNESS)) {
    const_cast<array::Scalar2 &>(geometry.ice_thickness).update_ghosts();
    m_geometry_ghosts_valid_mask |= GEOM_THICKNESS;
  }
  if ((mask & GEOM_BED) && !(m_geometry_ghosts_valid_mask & GEOM_BED)) {
    const_cast<array::Scalar2 &>(geometry.bed_elevation).update_ghosts();
    m_geometry_ghosts_valid_mask |= GEOM_BED;
  }
  if ((mask & GEOM_CELL_TYPE) && !(m_geometry_ghosts_valid_mask & GEOM_CELL_TYPE)) {
    const_cast<array::CellType2 &>(geometry.cell_type).update_ghosts();
    m_geometry_ghosts_valid_mask |= GEOM_CELL_TYPE;
  }
#else
  (void)geometry;
  (void)mask;
#endif
}

bool SIAFD::can_use_cuda_diffusivity(const Geometry &geometry,
                                     const array::Array3D *enthalpy,
                                     const array::Array3D *age,
                                     const array::Staggered &h_x,
                                     const array::Staggered &h_y,
                                     const array::Staggered1 &result) const {
#if Pism_USE_CUDA_SIA
  (void)age;
  if (enthalpy == nullptr) {
    return false;
  }
  const bool compute_grain_size_using_age =
      m_config->get_flag("stress_balance.sia.grain_size_age_coupling");
  const bool e_age_coupling = m_config->get_flag("stress_balance.sia.e_age_coupling");
  if (compute_grain_size_using_age || e_age_coupling) {
    return false;
  }

  if (!cuda::vec_is_cuda(geometry.ice_surface_elevation) ||
      !cuda::vec_is_cuda(geometry.ice_thickness) ||
      !cuda::vec_is_cuda(geometry.bed_elevation) ||
      !cuda::vec_is_cuda(geometry.cell_type)) {
    return false;
  }

  if (!cuda::vec_is_cuda(m_work_2d_0) || !cuda::vec_is_cuda(m_work_2d_1) ||
      !cuda::vec_is_cuda(h_x) || !cuda::vec_is_cuda(h_y) ||
      !cuda::vec_is_cuda(result) || !cuda::vec_is_cuda(m_diffusive_flux) ||
      !cuda::vec_is_cuda(*enthalpy) ||
      !cuda::vec_is_cuda(m_delta_0) || !cuda::vec_is_cuda(m_delta_1) ||
      !cuda::vec_is_cuda(m_work_3d_0) || !cuda::vec_is_cuda(m_work_3d_1)) {
    return false;
  }

  const std::string flow_law = m_config->get_string("stress_balance.sia.flow_law");
  if (flow_law != "pb" && flow_law != "arr" && flow_law != "arrwarm") {
    return false;
  }

  return true;
#else
  (void)geometry;
  (void)enthalpy;
  (void)age;
  (void)h_x;
  (void)h_y;
  (void)result;
  return false;
#endif
}

//! \brief Compute the ice surface gradient using the eta-transformation.
void SIAFD::surface_gradient_eta(const array::Scalar2 &ice_thickness,
                                 const array::Scalar2 &bed_elevation, array::Staggered &h_x,
                                 array::Staggered &h_y) {
  const double n = m_flow_law->exponent(), // presumably 3.0
      etapow     = (2.0 * n + 2.0) / n,    // = 8/3 if n = 3
      invpow = 1.0 / etapow, dinvpow = (-n - 2.0) / (2.0 * n + 2.0);
  const double dx = m_grid->dx(), dy = m_grid->dy(); // convenience

  array::Scalar2 &eta = m_work_2d_0;

  const int xs = m_grid->xs();
  const int ys = m_grid->ys();
  const int xm = m_grid->xm();
  const int ym = m_grid->ym();

#if Pism_USE_CUDA_SIA
  if (cuda::vec_is_cuda(eta) && cuda::vec_is_cuda(h_x) && cuda::vec_is_cuda(h_y) &&
      cuda::vec_is_cuda(ice_thickness) && cuda::vec_is_cuda(bed_elevation)) {
    const unsigned int needed = GEOM_THICKNESS | GEOM_BED;
    if ((m_geometry_ghosts_valid_mask & needed) != needed) {
      if (!(m_geometry_ghosts_valid_mask & GEOM_THICKNESS)) {
        const_cast<array::Scalar2 &>(ice_thickness).update_ghosts();
        m_geometry_ghosts_valid_mask |= GEOM_THICKNESS;
      }
      if (!(m_geometry_ghosts_valid_mask & GEOM_BED)) {
        const_cast<array::Scalar2 &>(bed_elevation).update_ghosts();
        m_geometry_ghosts_valid_mask |= GEOM_BED;
      }
    }
    cuda::surface_gradient_eta(ice_thickness, bed_elevation, eta, h_x, h_y,
                               xs, ys, xm, ym, eta.stencil_width(),
                               dx, dy, invpow, dinvpow, etapow);
    return;
  }
#endif

  auto GHOSTS = eta.stencil_width();

  petsc::DMDAVecArray thk_arr(ice_thickness.dm(), ice_thickness.vec());
  petsc::DMDAVecArray bed_arr(bed_elevation.dm(), bed_elevation.vec());
  petsc::DMDAVecArray eta_arr(eta.dm(), eta.vec());
  petsc::DMDAVecArrayDOF hx_arr(h_x.dm(), h_x.vec());
  petsc::DMDAVecArrayDOF hy_arr(h_y.dm(), h_y.vec());

  {
    sia_kernels::EtaFromThickness<sia_kernels::HostArray2DConstView,
                                  sia_kernels::HostArray2DView>
      functor{{static_cast<const double *const *>(thk_arr.get())},
              {static_cast<double **>(eta_arr.get())},
              etapow};
    for (auto p = m_grid->points(GHOSTS); p; p.next()) {
      functor(p.i(), p.j());
    }
  }

  assert(eta.stencil_width() >= 2);
  assert(h_x.stencil_width() >= 1);
  assert(h_y.stencil_width() >= 1);

  {
    sia_kernels::SurfaceGradientEta<sia_kernels::HostArray2DConstView,
                                    sia_kernels::HostStaggeredView>
      functor{{static_cast<const double *const *>(bed_arr.get())},
              {static_cast<const double *const *>(eta_arr.get())},
              {static_cast<double ***>(hx_arr.get())},
              {static_cast<double ***>(hy_arr.get())},
              dx, dy, invpow, dinvpow};
    for (auto p = m_grid->points(1); p; p.next()) {
      functor(p.i(), p.j());
    }
  }
}


//! \brief Compute the ice surface gradient using the Mary Anne Mahaffy method;
//! see [\ref Mahaffy].
void SIAFD::surface_gradient_mahaffy(const array::Scalar &ice_surface_elevation,
                                     array::Staggered &h_x, array::Staggered &h_y) {
  const double dx = m_grid->dx(), dy = m_grid->dy(); // convenience

  const array::Scalar &h = ice_surface_elevation;

  const int xs = m_grid->xs();
  const int ys = m_grid->ys();
  const int xm = m_grid->xm();
  const int ym = m_grid->ym();

#if Pism_USE_CUDA_SIA
  if (cuda::vec_is_cuda(h) && cuda::vec_is_cuda(h_x) && cuda::vec_is_cuda(h_y)) {
    if (!(m_geometry_ghosts_valid_mask & GEOM_SURFACE)) {
      const_cast<array::Scalar &>(h).update_ghosts();
      m_geometry_ghosts_valid_mask |= GEOM_SURFACE;
    }
    cuda::surface_gradient_mahaffy(h, h_x, h_y, xs, ys, xm, ym, dx, dy);
    return;
  }
#endif

  assert(h_x.stencil_width() >= 1);
  assert(h_y.stencil_width() >= 1);
  assert(h.stencil_width() >= 2);

  petsc::DMDAVecArray h_arr(h.dm(), h.vec());
  petsc::DMDAVecArrayDOF hx_arr(h_x.dm(), h_x.vec());
  petsc::DMDAVecArrayDOF hy_arr(h_y.dm(), h_y.vec());

  sia_kernels::SurfaceGradientMahaffy<sia_kernels::HostArray2DConstView,
                                      sia_kernels::HostStaggeredView>
    functor{{static_cast<const double *const *>(h_arr.get())},
            {static_cast<double ***>(hx_arr.get())},
            {static_cast<double ***>(hy_arr.get())},
            dx, dy};

  for (auto p = m_grid->points(1); p; p.next()) {
    functor(p.i(), p.j());
  }
}

//! \brief Compute the ice surface gradient using a modification of Marianne Haseloff's approach.
/*!
 * The original code deals correctly with adjacent ice-free points with bed
 * elevations which are above the surface of the ice nearby. This is done by
 * setting surface gradient at the margin to zero at such locations.
 *
 * This code also deals with shelf fronts: sharp surface elevation change at
 * the ice shelf front would otherwise cause abnormally high diffusivity
 * values, which forces PISM to take shorter time-steps than necessary. (Note
 * that the mass continuity code does not use SIA fluxes in floating areas.)
 * This is done by assuming that the ice surface near shelf fronts is
 * horizontal (i.e. here the surface gradient is set to zero also).
 *
 * The code below uses an interpretation of the standard Mahaffy scheme. We
 * compute components of the surface gradient at staggered grid locations. The
 * field h_x stores the x-component on the i-offset and j-offset grids, h_y ---
 * the y-component.
 *
 * The Mahaffy scheme for the x-component at grid points on the i-offset grid
 * (offset in the x-direction) is just the centered finite difference using
 * adjacent regular-grid points. (Similarly for the y-component at j-offset
 * locations.)
 *
 * Mahaffy's prescription for computing the y-component on the i-offset can be
 * interpreted as:
 *
 * - compute the y-component at four surrounding j-offset staggered grid locations,
 * - compute the average of these four.
 *
 * The code below does just that.
 *
 * - The first double for-loop computes x-components at i-offset
 *   locations and y-components at j-offset locations. Each computed
 *   number is assigned a weight (w_i and w_j) that is used below
 *
 * - The second double for-loop computes x-components at j-offset
 *   locations and y-components at i-offset locations as averages of
 *   quantities computed earlier. The weight are used to keep track of
 *   the number of values used in the averaging process.
 *
 * This method communicates ghost values of h_x and h_y. They cannot be
 * computed locally because the first loop uses width=2 stencil of surface,
 * mask, and bed to compute values at all grid points including width=1 ghosts,
 * then the second loop uses width=1 stencil to compute local values. (In other
 * words, a purely local computation would require width=3 stencil of surface,
 * mask, and bed fields.) The CUDA path computes the secondary step on the
 * width=1 ghost region using width=2 storage to avoid a ghost exchange.
 */
void SIAFD::surface_gradient_haseloff(const array::Scalar2 &ice_surface_elevation,
                                      const array::CellType2 &cell_type, array::Staggered &h_x,
                                      array::Staggered &h_y) {
  const double dx         = m_grid->dx(),
               dy         = m_grid->dy(); // convenience
  const array::Scalar2 &h = ice_surface_elevation;
  array::Scalar1 &w_i     = m_work_2d_0,
                 &w_j     = m_work_2d_1; // averaging weights

  const auto &mask = cell_type;

  const int xs = m_grid->xs();
  const int ys = m_grid->ys();
  const int xm = m_grid->xm();
  const int ym = m_grid->ym();

#if Pism_USE_CUDA_SIA
  if (cuda::vec_is_cuda(h) && cuda::vec_is_cuda(mask) && cuda::vec_is_cuda(w_i) &&
      cuda::vec_is_cuda(w_j) && cuda::vec_is_cuda(h_x) && cuda::vec_is_cuda(h_y)) {
    if (h_x.stencil_width() < 2 || h_y.stencil_width() < 2) {
      // GPU Haseloff gradients compute ghost values locally and require width=2 storage.
      goto cpu_path;
    }
    const unsigned int needed = GEOM_SURFACE | GEOM_CELL_TYPE;
    if ((m_geometry_ghosts_valid_mask & needed) != needed) {
      if (!(m_geometry_ghosts_valid_mask & GEOM_SURFACE)) {
        const_cast<array::Scalar2 &>(h).update_ghosts();
        m_geometry_ghosts_valid_mask |= GEOM_SURFACE;
      }
      if (!(m_geometry_ghosts_valid_mask & GEOM_CELL_TYPE)) {
        const_cast<array::CellType2 &>(mask).update_ghosts();
        m_geometry_ghosts_valid_mask |= GEOM_CELL_TYPE;
      }
    }
    cuda::surface_gradient_haseloff(h, mask, w_i, w_j, h_x, h_y, xs, ys, xm, ym, dx, dy);
    return;
  }
#endif
cpu_path:

  assert(mask.stencil_width() >= 2);
  assert(h.stencil_width() >= 2);
  assert(h_x.stencil_width() >= 1);
  assert(h_y.stencil_width() >= 1);
  assert(w_i.stencil_width() >= 1);
  assert(w_j.stencil_width() >= 1);

  petsc::DMDAVecArray h_arr(h.dm(), h.vec());
  petsc::DMDAVecArray mask_arr(mask.dm(), mask.vec());
  petsc::DMDAVecArray wi_arr(w_i.dm(), w_i.vec());
  petsc::DMDAVecArray wj_arr(w_j.dm(), w_j.vec());
  petsc::DMDAVecArrayDOF hx_arr(h_x.dm(), h_x.vec());
  petsc::DMDAVecArrayDOF hy_arr(h_y.dm(), h_y.vec());

  {
    sia_kernels::SurfaceGradientHaseloffPrimary<sia_kernels::HostArray2DConstView,
                                                sia_kernels::HostArray2DView,
                                                sia_kernels::HostStaggeredView,
                                                sia_kernels::HostMaskView>
      functor{{static_cast<const double *const *>(h_arr.get())},
              {static_cast<double ***>(hx_arr.get())},
              {static_cast<double ***>(hy_arr.get())},
              {static_cast<double **>(wi_arr.get())},
              {static_cast<double **>(wj_arr.get())},
              {static_cast<const double *const *>(mask_arr.get())},
              dx, dy};
    for (auto p = m_grid->points(1); p; p.next()) {
      functor(p.i(), p.j());
    }
  }

  {
    sia_kernels::SurfaceGradientHaseloffSecondary<sia_kernels::HostArray2DConstView,
                                                  sia_kernels::HostStaggeredView,
                                                  sia_kernels::HostMaskView>
      functor{{static_cast<double ***>(hx_arr.get())},
              {static_cast<double ***>(hy_arr.get())},
              {static_cast<const double *const *>(wi_arr.get())},
              {static_cast<const double *const *>(wj_arr.get())},
              {static_cast<const double *const *>(mask_arr.get())}};
    for (auto p = m_grid->points(); p; p.next()) {
      functor(p.i(), p.j());
    }
  }

  h_x.update_ghosts();
  h_y.update_ghosts();
}


//! \brief Compute the SIA diffusivity. If full_update, also store delta on the staggered grid.
/*!
 * Recall that \f$ Q = -D \nabla h \f$ is the diffusive flux in the mass-continuity equation
 *
 * \f[ \frac{\partial H}{\partial t} = M - S - \nabla \cdot (Q + \mathbf{U}_b H),\f]
 *
 * where \f$h\f$ is the ice surface elevation, \f$M\f$ is the top surface
 * accumulation/ablation rate, \f$S\f$ is the basal mass balance and
 * \f$\mathbf{U}_b\f$ is the thickness-advective (in PISM: usually SSA) ice
 * velocity.
 *
 * Recall also that at any particular point in the map-plane (i.e. if \f$x\f$
 * and \f$y\f$ are fixed)
 *
 * \f[ D = 2\int_b^h F(z)P(z)(h-z)dz, \f]
 *
 * where \f$F(z)\f$ is a constitutive function and \f$P(z)\f$ is the pressure
 * at a level \f$z\f$.
 *
 * By defining
 *
 * \f[ \delta(z) = 2F(z)P(z) \f]
 *
 * one can write
 *
 * \f[D = \int_b^h\delta(z)(h-z)dz. \f]
 *
 * The advantage is that it is then possible to avoid re-evaluating
 * \f$F(z)\f$ (which is computationally expensive) in the horizontal ice
 * velocity (see compute_3d_horizontal_velocity()) computation.
 *
 * This method computes \f$D\f$ and stores \f$\delta\f$ in delta[0,1] if full_update is true.
 *
 * The trapezoidal rule is used to approximate the integral.
 *
 * \param[in]  full_update the flag specitying if we're doing a "full" update.
 * \param[in]  h_x x-component of the surface gradient, on the staggered grid
 * \param[in]  h_y y-component of the surface gradient, on the staggered grid
 * \param[out] result diffusivity of the SIA flow
 */
void SIAFD::compute_diffusivity(bool full_update, const Geometry &geometry,
                                const array::Array3D *enthalpy, const array::Array3D *age,
                                array::Staggered &h_x, array::Staggered &h_y,
                                array::Staggered1 &result,
                                bool fuse_mahaffy) {
  array::Scalar2 &thk_smooth = m_work_2d_0, &theta = m_work_2d_1;

  array::Array3D *delta[] = { &m_delta_0, &m_delta_1 };
  array::Array3D *I[]     = { &m_work_3d_0, &m_work_3d_1 };

  const double current_time = time().current(),
               D_limit      = m_config->get_number("stress_balance.sia.max_diffusivity");

  const bool compute_grain_size_using_age =
                 m_config->get_flag("stress_balance.sia.grain_size_age_coupling"),
             e_age_coupling    = m_config->get_flag("stress_balance.sia.e_age_coupling"),
             limit_diffusivity = m_config->get_flag("stress_balance.sia.limit_diffusivity"),
             use_age           = compute_grain_size_using_age or e_age_coupling;

#if !Pism_USE_CUDA_SIA
  (void)fuse_mahaffy;
#endif

  rheology::grain_size_vostok gs_vostok;

  // get "theta" from Schoof (2003) bed smoothness calculation and the
  // thickness relative to the smoothed bed; each array::Scalar involved must
  // have stencil width WIDE_GHOSTS for this too work
  ensure_geometry_ghosts(geometry, GEOM_SURFACE | GEOM_THICKNESS | GEOM_CELL_TYPE);

  m_bed_smoother->theta(geometry.ice_surface_elevation, theta, false);

  m_bed_smoother->smoothed_thk(geometry.ice_surface_elevation, geometry.ice_thickness,
                               geometry.cell_type, thk_smooth, false);

#if Pism_USE_CUDA_SIA
  if (!use_age &&
      cuda::vec_is_cuda(thk_smooth) && cuda::vec_is_cuda(theta) &&
      cuda::vec_is_cuda(h_x) && cuda::vec_is_cuda(h_y) &&
      cuda::vec_is_cuda(*enthalpy) && cuda::vec_is_cuda(result) &&
      cuda::vec_is_cuda(*delta[0]) && cuda::vec_is_cuda(*delta[1])) {
    const std::string flow_law = m_config->get_string("stress_balance.sia.flow_law");
    int flow_law_mode = -1;
    if (flow_law == "pb") {
      flow_law_mode = 0;
    } else if (flow_law == "arr") {
      flow_law_mode = 1;
    } else if (flow_law == "arrwarm") {
      flow_law_mode = 2;
    }
    if (flow_law_mode < 0) {
      // Unsupported flow law for GPU diffusivity path.
      goto cpu_path;
    }
    const auto &z = m_grid->z();
    cuda::update_vertical_grid(z.data(), static_cast<int>(z.size()));

    const int xs = m_grid->xs();
    const int ys = m_grid->ys();
    const int xm = m_grid->xm();
    const int ym = m_grid->ym();
    const int Mx = m_grid->Mx();
    const int My = m_grid->My();
    const int ghosts = 1;
    const int periodic_x = (m_grid->periodicity() & grid::X_PERIODIC) ? 1 : 0;
    const int periodic_y = (m_grid->periodicity() & grid::Y_PERIODIC) ? 1 : 0;
    const double dx = m_grid->dx();
    const double dy = m_grid->dy();

    const double A_cold = m_config->get_number("flow_law.Paterson_Budd.A_cold");
    const double A_warm = m_config->get_number("flow_law.Paterson_Budd.A_warm");
    const double Q_cold = m_config->get_number("flow_law.Paterson_Budd.Q_cold");
    const double Q_warm = m_config->get_number("flow_law.Paterson_Budd.Q_warm");
    const double T_crit = m_config->get_number("flow_law.Paterson_Budd.T_critical");
    const double gas_const = m_config->get_number("constants.ideal_gas_constant");
    const double T_melting = m_config->get_number("constants.fresh_water.melting_point_temperature");
    const double beta = m_config->get_number("constants.ice.beta_Clausius_Clapeyron");
    const double c_i = m_config->get_number("constants.ice.specific_heat_capacity");
    const double T_0 = m_config->get_number("enthalpy_converter.T_reference");
    const double rho_i = m_config->get_number("constants.ice.density");
    const double g = m_config->get_number("constants.standard_gravity");
    const double p_air = m_config->get_number("surface.pressure");
    const double n = m_flow_law->exponent();

    double D_max_local = 0.0;
    int high_diffusivity_counter_local = 0;
    const int gpu_full_update = full_update ? 1 : 0;
    const int gpu_compute_I = full_update ? 1 : 0;
    const int gpu_compute_mahaffy = fuse_mahaffy ? 1 : 0;
    cuda::compute_diffusivity_pb_both(thk_smooth, theta, geometry.ice_surface_elevation,
                                      h_x, h_y, *enthalpy,
                                      result, m_diffusive_flux, *delta[0], *delta[1], *I[0], *I[1],
                                      xs, ys, xm, ym, Mx, My, ghosts, gpu_full_update, gpu_compute_I,
                                      gpu_compute_mahaffy, dx, dy, flow_law_mode,
                                      periodic_x, periodic_y, limit_diffusivity ? 1 : 0,
                                      D_limit, m_e_factor, A_cold, A_warm, Q_cold, Q_warm, T_crit,
                                      gas_const, n, T_melting, beta, c_i, T_0, rho_i, g, p_air,
                                      &D_max_local, &high_diffusivity_counter_local);

    m_D_max = GlobalMax(m_grid->com, D_max_local);
    high_diffusivity_counter_local = GlobalSum(m_grid->com, high_diffusivity_counter_local);

    if (m_D_max > D_limit) {
      throw RuntimeError::formatted(
          PISM_ERROR_LOCATION,
          "Maximum diffusivity of SIA flow (%f m2/s) is too high.\n"
          "This probably means that the bed elevation or the ice thickness is "
          "too rough.\n"
          "Increase stress_balance.sia.max_diffusivity to suppress this message.",
          m_D_max);
    }

    if (high_diffusivity_counter_local > 0) {
      m_log->message(2, "  SIA diffusivity was capped at %.2f m2/s at %d locations.\n",
                     D_limit, high_diffusivity_counter_local);
    }
    m_I_valid = full_update;
    m_diffusive_flux_valid = true;
    return;
  }
#endif

cpu_path:

  result.set(0.0);

  array::AccessScope list{ &result, &theta, &thk_smooth, &h_x, &h_y, enthalpy };

  if (use_age) {
    assert(age->stencil_width() >= 2);
    list.add(*age);
  }

  if (full_update) {
    list.add({ delta[0], delta[1] });
    assert(m_delta_0.stencil_width() >= 1);
    assert(m_delta_1.stencil_width() >= 1);
  }

  assert(theta.stencil_width() >= 2);
  assert(thk_smooth.stencil_width() >= 2);
  assert(result.stencil_width() >= 1);
  assert(h_x.stencil_width() >= 1);
  assert(h_y.stencil_width() >= 1);
  assert(enthalpy->stencil_width() >= 2);

  const std::vector<double> &z = m_grid->z();
  const unsigned int Mx = m_grid->Mx(), My = m_grid->My(), Mz = m_grid->Mz();

  std::vector<double> depth(Mz), stress(Mz), pressure(Mz), E(Mz), flow(Mz);
  std::vector<double> delta_ij(Mz);
  std::vector<double> A(Mz),
      ice_grain_size(Mz, m_config->get_number("constants.ice.grain_size", "m"));
  std::vector<double> e_factor(Mz, m_e_factor);

  double D_max                 = 0.0;
  int high_diffusivity_counter = 0;
  for (int o = 0; o < 2; o++) {
    ParallelSection loop(m_grid->com);
    try {
      for (auto p = m_grid->points(1); p; p.next()) {
        const int i = p.i(), j = p.j();

        // staggered point: o=0 is i+1/2, o=1 is j+1/2, (i, j) and (i+oi, j+oj)
        //   are regular grid neighbors of a staggered point:
        const int oi = 1 - o, oj = o;

        const double thk = 0.5 * (thk_smooth(i, j) + thk_smooth(i + oi, j + oj));

        // zero thickness case:
        if (thk == 0.0) {
          result(i, j, o) = 0.0;
          if (full_update) {
            delta[o]->set_column(i, j, 0.0);
          }
          continue;
        }

        const int ks = m_grid->kBelowHeight(thk);

        for (int k = 0; k <= ks; ++k) {
          depth[k] = thk - z[k];
        }

        // pressure added by the ice (i.e. pressure difference between the
        // current level and the top of the column)
        m_EC->pressure(depth, ks, pressure); // FIXME issue #15

        if (use_age) {
          const double *age_ij     = age->get_column(i, j),
                       *age_offset = age->get_column(i + oi, j + oj);

          for (int k = 0; k <= ks; ++k) {
            A[k] = 0.5 * (age_ij[k] + age_offset[k]);
          }

          if (compute_grain_size_using_age) {
            for (int k = 0; k <= ks; ++k) {
              // convert age from seconds to years:
              ice_grain_size[k] = gs_vostok(A[k] * m_seconds_per_year);
            }
          }

          if (e_age_coupling) {
            for (int k = 0; k <= ks; ++k) {
              const double accumulation_time = current_time - A[k];
              if (interglacial(accumulation_time)) {
                e_factor[k] = m_e_factor_interglacial;
              } else {
                e_factor[k] = m_e_factor;
              }
            }
          }
        }

        {
          const double *E_ij     = enthalpy->get_column(i, j),
                       *E_offset = enthalpy->get_column(i + oi, j + oj);
          for (int k = 0; k <= ks; ++k) {
            E[k] = 0.5 * (E_ij[k] + E_offset[k]);
          }
        }

        const double alpha = sqrt(PetscSqr(h_x(i, j, o)) + PetscSqr(h_y(i, j, o)));
        for (int k = 0; k <= ks; ++k) {
          stress[k] = alpha * pressure[k];
        }

        m_flow_law->flow_n(stress.data(), E.data(), pressure.data(), ice_grain_size.data(), ks + 1,
                           flow.data());

        const double theta_local = 0.5 * (theta(i, j) + theta(i + oi, j + oj));
        for (int k = 0; k <= ks; ++k) {
          delta_ij[k] = e_factor[k] * theta_local * 2.0 * pressure[k] * flow[k];
        }

        double D = 0.0; // diffusivity for deformational SIA flow
        {
          for (int k = 1; k <= ks; ++k) {
            // trapezoidal rule
            const double dz = z[k] - z[k - 1];
            D += 0.5 * dz * ((depth[k] + dz) * delta_ij[k - 1] + depth[k] * delta_ij[k]);
          }
          // finish off D with (1/2) dz (0 + (H-z[ks])*delta_ij[ks]), but dz=H-z[ks]:
          const double dz = thk - z[ks];
          D += 0.5 * dz * dz * delta_ij[ks];
        }

        // Override diffusivity at the edges of the domain. (At these
        // locations PISM uses ghost cells *beyond* the boundary of
        // the computational domain. This does not matter if the ice
        // does not extend all the way to the domain boundary, as in
        // whole-ice-sheet simulations. In a regional setup, though,
        // this adjustment lets us avoid taking very small time-steps
        // because of the possible thickness and bed elevation
        // "discontinuities" at the boundary.)
        {
          if ((i < 0 or i >= (int)Mx - 1) and not(m_grid->periodicity() & grid::X_PERIODIC)) {
            D = 0.0;
          }
          if ((j < 0 or j >= (int)My - 1) and not(m_grid->periodicity() & grid::Y_PERIODIC)) {
            D = 0.0;
          }
        }

        if (limit_diffusivity and D >= D_limit) {
          D = D_limit;
          high_diffusivity_counter += 1;
        }

        D_max = std::max(D_max, D);

        result(i, j, o) = D;

        // if doing the full update, fill the delta column above the ice and
        // store it:
        if (full_update) {
          for (unsigned int k = ks + 1; k < Mz; ++k) {
            delta_ij[k] = 0.0;
          }
          delta[o]->set_column(i, j, delta_ij.data());
        }
      } // i, j-loop
    } catch (...) {
      loop.failed();
    }
    loop.check();
  } // o-loop

  m_D_max = GlobalMax(m_grid->com, D_max);

  high_diffusivity_counter = GlobalSum(m_grid->com, high_diffusivity_counter);

  if (m_D_max > D_limit) {
    // This can happen only if stress_balance.sia.limit_diffusivity is false (m_D_max <=
    // D_limit when limiting is enabled).

    throw RuntimeError::formatted(
        PISM_ERROR_LOCATION,
        "Maximum diffusivity of SIA flow (%f m2/s) is too high.\n"
        "This probably means that the bed elevation or the ice thickness is "
        "too rough.\n"
        "Increase stress_balance.sia.max_diffusivity to suppress this message.",
        m_D_max);
  }

  if (high_diffusivity_counter > 0) {
    // This can happen only if stress_balance.sia.limit_diffusivity is true and this
    // limiting mechanism was active (high_diffusivity_counter is incremented only if
    // limit_diffusivity is true).

    m_log->message(2, "  SIA diffusivity was capped at %.2f m2/s at %d locations.\n", D_limit,
                   high_diffusivity_counter);
  }
}

void SIAFD::compute_diffusive_flux(const array::Staggered &h_x, const array::Staggered &h_y,
                                   const array::Staggered &diffusivity, array::Staggered &result) {

  const int xs = m_grid->xs();
  const int ys = m_grid->ys();
  const int xm = m_grid->xm();
  const int ym = m_grid->ym();

#if Pism_USE_CUDA_SIA
  if (cuda::vec_is_cuda(h_x) && cuda::vec_is_cuda(h_y) &&
      cuda::vec_is_cuda(diffusivity) && cuda::vec_is_cuda(result)) {
    for (int o = 0; o < 2; o++) {
      cuda::diffusive_flux(h_x, h_y, diffusivity, result, xs, ys, xm, ym, o);
    }
    return;
  }
#endif

  petsc::DMDAVecArrayDOF hx_arr(h_x.dm(), h_x.vec());
  petsc::DMDAVecArrayDOF hy_arr(h_y.dm(), h_y.vec());
  petsc::DMDAVecArrayDOF D_arr(diffusivity.dm(), diffusivity.vec());
  petsc::DMDAVecArrayDOF out_arr(result.dm(), result.vec());

  for (int o = 0; o < 2; o++) {
    ParallelSection loop(m_grid->com);
    try {
      sia_kernels::DiffusiveFlux<sia_kernels::HostStaggeredConstView,
                                 sia_kernels::HostStaggeredView>
        functor{{static_cast<const double *const *const *>(hx_arr.get())},
                {static_cast<const double *const *const *>(hy_arr.get())},
                {static_cast<const double *const *const *>(D_arr.get())},
                {static_cast<double ***>(out_arr.get())},
                o};
      for (auto p = m_grid->points(1); p; p.next()) {
        functor(p.i(), p.j());
      }
    } catch (...) {
      loop.failed();
    }
    loop.check();
  } // o-loop
}

//! \brief Compute I.
/*!
 * This computes
 * \f[ I(z) = \int_b^z\delta(s)ds.\f]
 *
 * Uses the trapezoidal rule to approximate the integral.
 *
 * See compute_diffusive_flux() for the definition of \f$\delta\f$.
 *
 * The result is stored in work_3d[0,1] and is used to compute the SIA component
 * of the 3D-distributed horizontal ice velocity.
 */
void SIAFD::compute_I(const Geometry &geometry) {
  if (m_I_valid) {
    return;
  }

  ensure_geometry_ghosts(geometry, GEOM_SURFACE | GEOM_THICKNESS | GEOM_CELL_TYPE);

  array::Scalar &thk_smooth = m_work_2d_0;
  array::Array3D *I[]       = { &m_work_3d_0, &m_work_3d_1 };
  array::Array3D *delta[]   = { &m_delta_0, &m_delta_1 };

  const array::Scalar &h = geometry.ice_surface_elevation, &H = geometry.ice_thickness;

  const auto &mask = geometry.cell_type;

  m_bed_smoother->smoothed_thk(h, H, mask, thk_smooth, false);

#if Pism_USE_CUDA_SIA
  if (cuda::vec_is_cuda(thk_smooth) && cuda::vec_is_cuda(*delta[0]) &&
      cuda::vec_is_cuda(*delta[1]) && cuda::vec_is_cuda(*I[0]) &&
      cuda::vec_is_cuda(*I[1])) {
    const auto &z = m_grid->z();
    cuda::update_vertical_grid(z.data(), static_cast<int>(z.size()));
    const int xs = m_grid->xs();
    const int ys = m_grid->ys();
    const int xm = m_grid->xm();
    const int ym = m_grid->ym();
    const int ghosts = 1;
    for (int o = 0; o < 2; ++o) {
      cuda::compute_I(thk_smooth, *delta[o], *I[o], xs, ys, xm, ym, ghosts, o);
    }
    return;
  }
#endif

  array::AccessScope list{ delta[0], delta[1], I[0], I[1], &thk_smooth };

  assert(I[0]->stencil_width() >= 1);
  assert(I[1]->stencil_width() >= 1);
  assert(delta[0]->stencil_width() >= 1);
  assert(delta[1]->stencil_width() >= 1);
  assert(thk_smooth.stencil_width() >= 2);

  const unsigned int Mz = m_grid->Mz();

  std::vector<double> dz(Mz);
  for (unsigned int k = 1; k < Mz; ++k) {
    dz[k] = m_grid->z(k) - m_grid->z(k - 1);
  }

  for (int o = 0; o < 2; ++o) {
    ParallelSection loop(m_grid->com);
    try {
      for (auto p = m_grid->points(1); p; p.next()) {
        const int i = p.i(), j = p.j();

        const int oi = 1 - o, oj = o;
        const double thk = 0.5 * (thk_smooth(i, j) + thk_smooth(i + oi, j + oj));

        const double *delta_ij = delta[o]->get_column(i, j);
        double *I_ij           = I[o]->get_column(i, j);

        const unsigned int ks = m_grid->kBelowHeight(thk);

        // within the ice:
        I_ij[0]          = 0.0;
        double I_current = 0.0;
        for (unsigned int k = 1; k <= ks; ++k) {
          // trapezoidal rule
          I_current += 0.5 * dz[k] * (delta_ij[k - 1] + delta_ij[k]);
          I_ij[k] = I_current;
        }

        // above the ice:
        for (unsigned int k = ks + 1; k < Mz; ++k) {
          I_ij[k] = I_current;
        }
      }
    } catch (...) {
      loop.failed();
    }
    loop.check();
  } // o-loop
  m_I_valid = true;
}

//! \brief Compute horizontal components of the SIA velocity (in 3D).
/*!
 * Recall that
 *
 * \f[ \mathbf{U}(z) = -2 \nabla h \int_b^z F(s)P(s)ds + \mathbf{U}_b,\f]
 *
 * which can be written in terms of \f$I(z)\f$ defined in compute_I():
 *
 * \f[ \mathbf{U}(z) = -I(z) \nabla h + \mathbf{U}_b. \f]
 *
 * \note This is one of the places where "hybridization" is done.
 *
 * \param[in] h_x the X-component of the surface gradient, on the staggered grid
 * \param[in] h_y the Y-component of the surface gradient, on the staggered grid
 * \param[in] sliding_velocity the thickness-advective velocity from the underlying stress balance module
 * \param[out] u_out the X-component of the resulting horizontal velocity field
 * \param[out] v_out the Y-component of the resulting horizontal velocity field
 */
void SIAFD::compute_3d_horizontal_velocity(const Geometry &geometry, const array::Staggered &h_x,
                                           const array::Staggered &h_y,
                                           const array::Vector &sliding_velocity,
                                           array::Array3D &u_out, array::Array3D &v_out) {

  array::Array3D *I[] = { &m_work_3d_0, &m_work_3d_1 };

#if Pism_USE_CUDA_SIA
  if (cuda::vec_is_cuda(m_delta_0) && cuda::vec_is_cuda(m_delta_1) &&
      cuda::vec_is_cuda(h_x) && cuda::vec_is_cuda(h_y) &&
      cuda::vec_is_cuda(sliding_velocity) && cuda::vec_is_cuda(u_out) &&
      cuda::vec_is_cuda(v_out)) {
    const auto &z = m_grid->z();
    cuda::update_vertical_grid(z.data(), static_cast<int>(z.size()));
    const int xs = m_grid->xs();
    const int ys = m_grid->ys();
    const int xm = m_grid->xm();
    const int ym = m_grid->ym();
    const int Mx = m_grid->Mx();
    const int My = m_grid->My();
    const int Mz = m_grid->Mz();
    const char *use_I_env = std::getenv("PISM_SIAFD_CUDA_USE_I");
    bool force_I = false;
    bool force_delta = false;
    if (use_I_env && use_I_env[0] != '\0') {
      if (use_I_env[0] == '0') {
        force_delta = true;
      } else {
        force_I = true;
      }
    }
    bool use_I = m_I_valid || force_I;
    if (force_delta) {
      use_I = false;
    }
    if (use_I) {
      if (!m_I_valid) {
        const int ghosts = 1;
        cuda::compute_I(m_work_2d_0, m_delta_0, m_work_3d_0, xs, ys, xm, ym, ghosts, 0);
        cuda::compute_I(m_work_2d_0, m_delta_1, m_work_3d_1, xs, ys, xm, ym, ghosts, 1);
      }
      cuda::compute_3d_horizontal_velocity(m_work_3d_0, m_work_3d_1, h_x, h_y,
                                           sliding_velocity, u_out, v_out,
                                           xs, ys, xm, ym, Mz);
    } else {
      cuda::compute_3d_horizontal_velocity_from_delta(m_delta_0, m_delta_1, h_x, h_y,
                                                      sliding_velocity, u_out, v_out,
                                                      xs, ys, xm, ym, Mz);
    }
    if (m_grid->size() == 1) {
      const int ghosts = static_cast<int>(u_out.stencil_width());
      cuda::update_periodic_ghosts(u_out, xs, ys, xm, ym, Mx, My, ghosts);
      cuda::update_periodic_ghosts(v_out, xs, ys, xm, ym, Mx, My, ghosts);
    } else {
      u_out.update_ghosts();
      v_out.update_ghosts();
    }
    return;
  }
#endif

  compute_I(geometry);
  // after the compute_I() call work_3d[0,1] contains I on the staggered grid

  array::AccessScope list{ &u_out, &v_out, &h_x, &h_y, &sliding_velocity, I[0], I[1] };

  const unsigned int Mz = m_grid->Mz();

  for (auto p = m_grid->points(); p; p.next()) {
    const int i = p.i(), j = p.j();

    const double
      *I_e = I[0]->get_column(i, j),
      *I_w = I[0]->get_column(i - 1, j),
      *I_n = I[1]->get_column(i, j),
      *I_s = I[1]->get_column(i, j - 1);

    // Fetch values from 2D fields *outside* of the k-loop:
    const double
      h_x_w = h_x(i - 1, j, 0),
      h_x_e = h_x(i, j, 0),
      h_x_n = h_x(i, j, 1),
      h_x_s = h_x(i, j - 1, 1);

    const double
      h_y_w = h_y(i - 1, j, 0),
      h_y_e = h_y(i, j, 0),
      h_y_n = h_y(i, j, 1),
      h_y_s = h_y(i, j - 1, 1);

    const double
      sliding_velocity_u = sliding_velocity(i, j).u,
      sliding_velocity_v = sliding_velocity(i, j).v;

    double
      *u_ij = u_out.get_column(i, j),
      *v_ij = v_out.get_column(i, j);

    // split into two loops to encourage auto-vectorization
    for (unsigned int k = 0; k < Mz; ++k) {
      u_ij[k] = sliding_velocity_u - 0.25 * (I_e[k] * h_x_e + I_w[k] * h_x_w +
                                             I_n[k] * h_x_n + I_s[k] * h_x_s);
    }
    for (unsigned int k = 0; k < Mz; ++k) {
      v_ij[k] = sliding_velocity_v - 0.25 * (I_e[k] * h_y_e + I_w[k] * h_y_w +
                                             I_n[k] * h_y_n + I_s[k] * h_y_s);
    }
  }

  // Communicate to get ghosts:
  u_out.update_ghosts();
  v_out.update_ghosts();
}

//! Determine if `accumulation_time` corresponds to an interglacial period.
bool SIAFD::interglacial(double accumulation_time) const {
  if (accumulation_time < m_eemian_start) {
    return false;
  }

  if (accumulation_time < m_eemian_end) {
    return true;
  }

  return (accumulation_time >= m_holocene_start);
}

const array::Staggered& SIAFD::surface_gradient_x() const {
  return m_h_x;
}

const array::Staggered& SIAFD::surface_gradient_y() const {
  return m_h_y;
}

const array::Staggered1& SIAFD::diffusivity() const {
  return m_D;
}

const BedSmoother& SIAFD::bed_smoother() const {
  return *m_bed_smoother;
}


} // end of namespace stressbalance
} // end of namespace pism
