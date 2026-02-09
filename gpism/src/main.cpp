#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gpism/context.h"
#include "gpism/device_policy.h"
#include "gpism/field_sync.h"
#include "gpism/runtime_config.h"
#include "gpism/time_manager.h"
#include "gpism/config.h"
#include "gpism/version.h"

#if GPISM_HAVE_NETCDF
#include "gpism/async_output.h"
#include "gpism/netcdf_io.h"
#endif
#include "gpism/geometry.h"
#include "gpism/halo_exchange.h"
#include "gpism/ssa_solver.h"
#include "gpism/thickness.h"
#include "gpism/thermodynamics.h"
#include "gpism/viscosity.h"

#if GPISM_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

void print_help() {
  std::cout
      << "gpism: GPU-first, PISM-compatible ice-sheet model (scaffold)\n"
      << "\n"
      << "Usage:\n"
      << "  gpism [--help] [--version] [options]\n"
      << "\n"
      << "Options:\n"
      << "  --help       Show this help text\n"
      << "  --version    Show build information\n"
      << "  -i FILE      Input (restart) NetCDF file\n"
      << "  -o FILE      Output NetCDF file\n"
      << "  -y YEARS     Run length in years\n"
      << "  -time YEARS  Run length in years (alias)\n"
      << "  -config FILE Full config file (replaces defaults)\n"
      << "  -config_override FILE  Config overrides (partial)\n"
      << "  -dry_run     Print resolved config and exit\n";
}

void print_version() {
  std::cout << "gpism " << GPISM_VERSION;
  if (std::strcmp(GPISM_GIT_SHA, "unknown") != 0) {
    std::cout << " (" << GPISM_GIT_SHA << ")";
  }
  std::cout << "\n"
            << "  build type: " << GPISM_BUILD_TYPE << "\n"
            << "  backend:    " << GPISM_BACKEND << "\n"
            << "  precision:  " << GPISM_PRECISION << "\n"
            << "  MPI:        " << GPISM_MPI << "\n"
            << "  NetCDF:     " << GPISM_NETCDF << "\n";
}

struct Options {
  std::string input;
  std::string output;
  std::string config_path;
  std::string config_override_path;
  std::string run_years;
  bool dry_run = false;
  std::vector<std::string> gpism_opts;
};

bool parse_args(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_help();
      return false;
    }
    if (arg == "--version" || arg == "-V") {
      print_version();
      return false;
    }
    if (arg == "-i") {
      options->input = require_value(arg);
      continue;
    }
    if (arg == "-o") {
      options->output = require_value(arg);
      continue;
    }
    if (arg == "-y" || arg == "-time") {
      options->run_years = require_value(arg);
      continue;
    }
    if (arg == "-config") {
      options->config_path = require_value(arg);
      continue;
    }
    if (arg == "-config_override") {
      options->config_override_path = require_value(arg);
      continue;
    }
    if (arg == "-dry_run") {
      options->dry_run = true;
      continue;
    }
    if (arg.rfind("-gpism_", 0) == 0) {
      options->gpism_opts.push_back(arg);
      continue;
    }
    if (arg.rfind("-", 0) == 0) {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  return true;
}

void log_rank0(const gpism::Context& context, const std::string& message) {
  if (context.rank() == 0) {
    std::cout << message << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  gpism::Context context(&argc, &argv);
#if GPISM_HAVE_CUDA
  cudaSetDevice(context.device_id());
#endif
  Options options;
  try {
    if (!parse_args(argc, argv, &options)) {
      return 0;
    }
  } catch (const std::exception& exc) {
    std::cerr << "Error: " << exc.what() << '\n';
    return 2;
  }

  gpism::RuntimeConfig config;
  if (!options.config_path.empty()) {
    if (!config.load_file(options.config_path, true)) {
      std::cerr << "Error: failed to read config file " << options.config_path
                << '\n';
      return 2;
    }
  }
  if (!options.config_override_path.empty()) {
    if (!config.apply_override(options.config_override_path)) {
      std::cerr << "Error: failed to read override config file "
                << options.config_override_path << '\n';
      return 2;
    }
  }
  if (!options.run_years.empty()) {
    config.set("time.years", options.run_years);
  }

  // Allow forcing the host code path even in CUDA builds. This is useful for
  // debugging (e.g. parity checks vs CPU kernels).
  gpism::set_device_enabled(config.get_bool("device.enabled"));

  if (!options.gpism_opts.empty() && context.rank() == 0) {
    std::cout << "Ignoring gpism-only options:";
    for (const auto& opt : options.gpism_opts) {
      std::cout << ' ' << opt;
    }
    std::cout << '\n';
  }

  if (options.dry_run) {
    log_rank0(context, "gpism dry run configuration:");
    log_rank0(context, config.summary());
    return 0;
  }

  if (!options.input.empty() && !options.output.empty()) {
#if GPISM_HAVE_NETCDF
    gpism::Grid2D grid(0, 0, 1.0, 1.0, 1, context.rank(), context.size());
    gpism::IOFields2D fields;
    gpism::NetcdfIO io;
    gpism::AsyncOutputWriter output_writer;
    int time_index = -1;
    if (config.has("io.time_index")) {
      time_index = config.get_int("io.time_index");
    }
    if (!io.read_restart(options.input, context, grid, fields, time_index)) {
      std::cerr << "Error: failed to read input file " << options.input << '\n';
      return 2;
    }
    const bool async_output = config.get_bool("io.async_output");
    output_writer.configure(grid, fields, async_output);

    const double tauc_default = config.get_double("ssa.tauc_default");
    const double tauc_floor = config.get_double("ssa.tauc_floor");
    if (!fields.has_tauc) {
      if (tauc_default != 0.0) {
        fields.tauc.fill(tauc_default);
        log_rank0(context,
                  "Input missing tauc; using ssa.tauc_default=" +
                      std::to_string(tauc_default));
      } else {
        log_rank0(context, "Input missing tauc; defaulting to 0.0");
      }
    }
    if (tauc_floor > 0.0) {
      for (int j = 0; j < grid.local_my(); ++j) {
        for (int i = 0; i < grid.local_mx(); ++i) {
          fields.tauc(i, j) = std::max(fields.tauc(i, j), tauc_floor);
        }
      }
      log_rank0(context,
                "Applied ssa.tauc_floor=" + std::to_string(tauc_floor));
    }

    const double start_year = config.get_double("time.start_year");
    const double run_years = config.get_double("time.years");
    const double dt = config.get_double("time.dt");
    const double output_interval = config.get_double("time.output_interval");
    gpism::TimeManager clock(start_year, dt, start_year + run_years,
                             output_interval);

    const double smb_constant = config.get_double("forcing.smb_constant");
    gpism::Field2D<double> smb(grid.local_mx(), grid.local_my(), grid.ghost_width());
    smb.fill(smb_constant);

    gpism::FieldStag2D<double> vel_cc(grid.local_mx(), grid.local_my(),
                                      grid.ghost_width());
    gpism::FieldStag2D<double> vel_face(grid.local_mx(), grid.local_my(),
                                        grid.ghost_width());
    auto init_vel_cc_from_center_fields =
        [&](const gpism::Field2D<double>& u_center,
            const gpism::Field2D<double>& v_center) {
          const int mx = grid.local_mx();
          const int my = grid.local_my();
          for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
              vel_cc(i, j, 0) = u_center(i, j);
              vel_cc(i, j, 1) = v_center(i, j);
            }
          }
        };
    if (fields.has_ssa_velocity) {
      init_vel_cc_from_center_fields(fields.u_ssa, fields.v_ssa);
      log_rank0(context,
                "Initialized SSA velocity guess from input u_ssa/v_ssa.");
    } else if (fields.has_velocity) {
      init_vel_cc_from_center_fields(fields.uvel, fields.vvel);
      log_rank0(context, "Initialized velocity guess from input uvel/vvel.");
    } else {
      vel_cc.fill(0.0);
    }

    gpism::FieldStag2D<double> flux(grid.local_mx(), grid.local_my(),
                                    grid.ghost_width());
    gpism::Field2D<int> cell_type(grid.local_mx(), grid.local_my(),
                                  grid.ghost_width());

    const double rho_ice = config.get_double("constants.ice.density");
    const double rho_water = config.get_double("constants.sea_water.density");
    const double gravity = config.get_double("constants.standard_gravity");
    const double sea_level = config.get_double("constants.sea_level");
    const double ice_free_thickness_standard =
        config.get_double("stress_balance.ice_free_thickness_standard");
    const double seconds_per_year = config.get_double("constants.seconds_per_year");
    const double glen_n = config.get_double("stress_balance.ssa.Glen_exponent");
    const std::string flow_law = config.get_string("stress_balance.ssa.flow_law");
    const double softness =
        config.get_double("flow_law.isothermal_Glen.ice_softness");
    const double enhancement =
        config.get_double("stress_balance.ssa.enhancement_factor");

    // PISM config uses "meter / year" for several velocity-scale parameters. gpism
    // uses SI (seconds) for the SSA solve, so convert to m/s.
    const double seconds_per_year_safe = std::max(1.0, seconds_per_year);
    const double u_threshold_year =
        config.get_double("basal_resistance.pseudo_plastic.u_threshold");
    const double u_threshold = u_threshold_year / seconds_per_year_safe;  // m/s
    const double plastic_reg_year =
        config.get_double("basal_resistance.plastic.regularization");
    const double plastic_reg = plastic_reg_year / seconds_per_year_safe;  // m/s
    const double schoof_vel_year =
        config.get_double("flow_law.Schoof_regularizing_velocity");
    const double schoof_vel = schoof_vel_year / seconds_per_year_safe;  // m/s
    const double schoof_length_km =
        config.get_double("flow_law.Schoof_regularizing_length");
    const double schoof_length = schoof_length_km * 1000.0;
    const double eps0 =
        (schoof_length > 0.0) ? (schoof_vel / schoof_length) : 0.0;

    const double A = softness;

    if (flow_law != "isothermal_glen" && context.rank() == 0) {
      std::cout << "Warning: SSA flow law '" << flow_law
                << "' not implemented; using isothermal_glen.\n";
    }

    gpism::ViscosityModel viscosity(A, glen_n, eps0, enhancement);
    gpism::SSASolver solver(grid, rho_ice, gravity, u_threshold, viscosity);

    gpism::SSASolverOptions ssa_options;
    ssa_options.max_picard = config.get_int("ssa.max_picard");
    ssa_options.gmres_max_iter = config.get_int("ssa.gmres_max_iter");
    ssa_options.tol_nuH = config.get_double("ssa.tol_nuH");
    ssa_options.tol_vel = config.get_double("ssa.tol_vel");
    ssa_options.gmres_tol = config.get_double("ssa.gmres_tol");
    if (config.has("ssa.gmres_tol_relative_to_rhs")) {
      ssa_options.gmres_tol_relative_to_rhs =
          config.get_bool("ssa.gmres_tol_relative_to_rhs");
    }
    ssa_options.vel_relax = config.get_double("ssa.vel_relax");
    ssa_options.nuH_relax = config.get_double("ssa.nuH_relax");
    ssa_options.max_speed = config.get_double("ssa.max_speed");
    ssa_options.gmres_verbose = config.get_bool("ssa.gmres_verbose");
    ssa_options.diagnostic = config.get_bool("ssa.diagnostic");
    ssa_options.force_host_convergence = config.get_bool("ssa.force_host_convergence");
    // PISM's stress_balance.ssa.epsilon has units Pa*s*m (regularization added to nu*H).
    ssa_options.nuH_regularization = config.get_double("stress_balance.ssa.epsilon");
    ssa_options.use_mg_precond = config.get_bool("ssa.mg.enabled");
    ssa_options.mg_pre_iters = config.get_int("ssa.mg.pre_iters");
    ssa_options.mg_post_iters = config.get_int("ssa.mg.post_iters");
    ssa_options.mg_coarse_iters = config.get_int("ssa.mg.coarse_iters");
    ssa_options.mg_omega = config.get_double("ssa.mg.omega");
    ssa_options.mg_min_size = config.get_int("ssa.mg.min_size");
    const std::string mg_smoother = config.get_string("ssa.mg.smoother");
    if (mg_smoother == "chebyshev") {
      ssa_options.mg_smoother = gpism::MGSmoother::Chebyshev;
    } else if (mg_smoother == "jacobi") {
      ssa_options.mg_smoother = gpism::MGSmoother::Jacobi;
    } else {
      throw std::runtime_error("Invalid ssa.mg.smoother value: " + mg_smoother);
    }
    ssa_options.mg_cheby_lambda_min =
        config.get_double("ssa.mg.chebyshev.lambda_min");
    ssa_options.mg_cheby_lambda_max =
        config.get_double("ssa.mg.chebyshev.lambda_max");
    ssa_options.mg_cheby_estimate =
        config.get_bool("ssa.mg.chebyshev.estimate");
    ssa_options.mg_cheby_estimate_iters =
        config.get_int("ssa.mg.chebyshev.estimate_iters");
    ssa_options.mg_cheby_estimate_min_factor =
        config.get_double("ssa.mg.chebyshev.estimate_min_factor");
    ssa_options.mg_cheby_estimate_max_factor =
        config.get_double("ssa.mg.chebyshev.estimate_max_factor");
    ssa_options.mg_diagnostic = config.get_bool("ssa.mg.diagnostic");
    ssa_options.gmres_precond_diagnostic =
        config.get_bool("ssa.gmres.precond_diagnostic");
    ssa_options.use_bc = fields.has_vel_bc;
    ssa_options.enforce_ice_free_bc = config.get_bool("ssa.enforce_ice_free_bc");
    ssa_options.fail_fast = config.get_bool("ssa.fail_fast");
    ssa_options.fail_fast_require_converged =
        config.get_bool("ssa.fail_fast_require_converged");
    ssa_options.fail_fast_residual_max = config.get_double("ssa.fail_fast_residual_max");
    ssa_options.fail_fast_dump_prefix = config.get_string("ssa.fail_fast_dump_prefix");
    ssa_options.config_override_path = options.config_override_path;
    ssa_options.sea_level = sea_level;
    ssa_options.rho_ice = rho_ice;
    ssa_options.rho_water = rho_water;
    ssa_options.ice_free_thickness_standard = ice_free_thickness_standard;
    ssa_options.surface_gradient_inward =
        config.get_bool("stress_balance.ssa.compute_surface_gradient_inward");
    ssa_options.surface_slope_uphill =
        config.get_bool("stress_balance.ssa.fd.upstream_surface_slope_approximation");
    ssa_options.extrapolate_at_margins =
        config.get_bool("stress_balance.ssa.fd.extrapolate_at_margins");
    ssa_options.use_cfbc =
        config.get_bool("stress_balance.calving_front_stress_bc");
    // PISM's SSA strength extension (keeps SSA elliptic in thin-ice regions).
    if (config.has("stress_balance.ssa.strength_extension.constant_nu")) {
      ssa_options.strength_extension_nu =
          config.get_double("stress_balance.ssa.strength_extension.constant_nu");
    }
    if (config.has("stress_balance.ssa.strength_extension.min_thickness")) {
      ssa_options.strength_extension_min_thickness = config.get_double(
          "stress_balance.ssa.strength_extension.min_thickness");
    }
    ssa_options.context = &context;

    gpism::BasalResistanceParams basal_params;
    basal_params.q =
        config.get_double("basal_resistance.pseudo_plastic.q");
    basal_params.u_threshold = u_threshold;
    basal_params.plastic_regularization = plastic_reg;
    basal_params.sliding_scale_factor =
        config.get_double("basal_resistance.pseudo_plastic.sliding_scale_factor");
    basal_params.beta_ice_free_bedrock =
        config.get_double("basal_resistance.beta_ice_free_bedrock");
    basal_params.beta_lateral_margin =
        config.get_double("basal_resistance.beta_lateral_margin");
    basal_params.law =
        config.get_bool("basal_resistance.pseudo_plastic.enabled")
            ? gpism::BasalResistanceLaw::PseudoPlastic
            : gpism::BasalResistanceLaw::Plastic;
    ssa_options.basal_params = basal_params;

    const bool thermo_enabled = config.get_bool("thermo.enabled");
    gpism::Field3D<double> enthalpy;
    gpism::Field3D<double> enthalpy_next;
    gpism::VerticalDiffusionOptions thermo_opts;
    double dz = 0.0;
    if (thermo_enabled) {
      const int mz = config.get_int("grid.Mz");
      const double Lz = config.get_double("grid.Lz");
      dz = (mz > 0) ? (Lz / static_cast<double>(mz)) : 1.0;
      enthalpy.resize(grid.local_mx(), grid.local_my(), mz, grid.ghost_width());
      enthalpy_next.resize(grid.local_mx(), grid.local_my(), mz,
                           grid.ghost_width());
      thermo_opts.kappa = config.get_double("thermo.kappa");
      thermo_opts.surface_value = config.get_double("thermo.surface_value");
      thermo_opts.basal_value = config.get_double("thermo.basal_value");
      thermo_opts.dirichlet = true;
      ssa_options.enthalpy = &enthalpy;
      ssa_options.enthalpy_gamma =
          config.get_double("thermo.enthalpy_gamma");
      ssa_options.enthalpy_ref = config.get_double("thermo.enthalpy_ref");

      for (int j = 0; j < grid.local_my(); ++j) {
        for (int i = 0; i < grid.local_mx(); ++i) {
          for (int k = 0; k < mz; ++k) {
            const double t = (mz > 1) ? static_cast<double>(k) / (mz - 1) : 0.0;
            enthalpy(i, j, k) =
                (1.0 - t) * thermo_opts.surface_value +
                t * thermo_opts.basal_value;
          }
        }
      }
    }

    gpism::ThicknessUpdateOptions thickness_opts;
    const bool evolve_thickness = config.get_bool("thickness.evolve");
    const bool run_ssa = config.get_bool("ssa.enabled");

    gpism::GeometryDiagnostics geometry;

    // If the input file does not provide a velocity field, use a simple heuristic
    // initial guess aligned with the driving stress direction. This helps Picard
    // iterations avoid locking into a near-zero-velocity solution when basal drag
    // is (regularized) plastic.
    if (!fields.has_ssa_velocity && !fields.has_velocity) {
      const double guess_speed = std::max(0.0, config.get_double("ssa.initial_guess_speed"));
      if (guess_speed > 0.0) {
        // This block intentionally uses host data structures. At this point we
        // have not synced input fields to the device yet, so CUDA kernels would
        // read uninitialized device buffers and leave host arrays (used below)
        // unchanged, resulting in a near-zero initial guess.
        const bool was_device_enabled = gpism::device_enabled();
        if (was_device_enabled) {
          gpism::set_device_enabled(false);
        }

        gpism::Field2D<int> cell_type_guess(grid.local_mx(), grid.local_my(),
                                            grid.ghost_width());
        gpism::Field2D<double> usurf_guess(grid.local_mx(), grid.local_my(),
                                           grid.ghost_width());
        gpism::Field2D<double> dhdx_guess(grid.local_mx(), grid.local_my(),
                                          grid.ghost_width());
        gpism::Field2D<double> dhdy_guess(grid.local_mx(), grid.local_my(),
                                          grid.ghost_width());
        gpism::Field2D<double> u_guess(grid.local_mx(), grid.local_my(),
                                       grid.ghost_width());
        gpism::Field2D<double> v_guess(grid.local_mx(), grid.local_my(),
                                       grid.ghost_width());

        gpism::compute_cell_type(grid, fields.thk, fields.topg, sea_level, rho_ice,
                                 rho_water, ice_free_thickness_standard,
                                 cell_type_guess);
        gpism::compute_usurf_flotation(grid, fields.thk, fields.topg,
                                       cell_type_guess, sea_level, rho_ice,
                                       rho_water, usurf_guess);
        gpism::compute_surface_slopes_pism(
            grid, usurf_guess, cell_type_guess, dhdx_guess, dhdy_guess,
            ssa_options.surface_gradient_inward, ssa_options.surface_slope_uphill,
            ssa_options.use_cfbc);

        const double scale = -rho_ice * gravity;
        const int mx = grid.local_mx();
        const int my = grid.local_my();
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            const double tauc = std::max(1.0, fields.tauc(i, j));
            const double tau_x = scale * fields.thk(i, j) * dhdx_guess(i, j);
            const double tau_y = scale * fields.thk(i, j) * dhdy_guess(i, j);
            u_guess(i, j) = guess_speed * (tau_x / tauc);
            v_guess(i, j) = guess_speed * (tau_y / tauc);
          }
        }

        init_vel_cc_from_center_fields(u_guess, v_guess);
        log_rank0(context, "Initialized velocity guess from driving stress.");

        if (was_device_enabled) {
          gpism::set_device_enabled(true);
        }
      }
    }

    gpism::sync_host_to_device(fields.thk);
    gpism::sync_host_to_device(fields.topg);
    gpism::sync_host_to_device(fields.tauc);
    gpism::sync_host_to_device(smb);
    gpism::sync_host_to_device(vel_cc);
    gpism::compute_face_velocity_from_center(grid, vel_cc, vel_face);
    if (thermo_enabled) {
      gpism::sync_host_to_device(enthalpy);
      gpism::sync_host_to_device(enthalpy_next);
    }

    gpism::HaloExchange2D exchange;
    auto exchange_field2d = [&](auto& field) {
      if (!context.mpi_enabled() || context.size() <= 1) {
        return;
      }
#if GPISM_HAVE_CUDA
      if (context.cuda_aware_mpi() && field.device_data() != nullptr) {
        exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Device);
        return;
      }
      if (field.device_data() != nullptr) {
        gpism::sync_device_to_host(field);
        exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Host);
        gpism::sync_host_to_device(field);
        return;
      }
#endif
      exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Host);
    };

    auto exchange_field_stag = [&](auto& field) {
      if (!context.mpi_enabled() || context.size() <= 1) {
        return;
      }
#if GPISM_HAVE_CUDA
      const bool have_device =
          field.component(0).device_data() != nullptr &&
          field.component(1).device_data() != nullptr;
      if (context.cuda_aware_mpi() && have_device) {
        exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Device);
        return;
      }
      if (have_device) {
        gpism::sync_device_to_host(field);
        exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Host);
        gpism::sync_host_to_device(field);
        return;
      }
#endif
      exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Host);
    };

    double last_output_time = -1.0;
    while (!clock.done()) {
      if (clock.should_output()) {
        gpism::compute_cell_type(grid, fields.thk, fields.topg, sea_level,
                                 rho_ice, rho_water,
                                 ice_free_thickness_standard, cell_type);
        gpism::compute_usurf_flotation(grid, fields.thk, fields.topg,
                                       cell_type, sea_level, rho_ice,
                                       rho_water, fields.usurf);
        gpism::sync_device_to_host(vel_cc);
        for (int j = 0; j < grid.local_my(); ++j) {
          for (int i = 0; i < grid.local_mx(); ++i) {
            const double u = vel_cc(i, j, 0);
            const double v = vel_cc(i, j, 1);
            fields.uvel(i, j) = u;
            fields.vvel(i, j) = v;
            fields.u_ssa(i, j) = u;
            fields.v_ssa(i, j) = v;
          }
        }
        // Ensure device buffers are consistent with host-produced diagnostics
        // for async output staging.
        gpism::sync_host_to_device(fields.uvel);
        gpism::sync_host_to_device(fields.vvel);
        gpism::sync_host_to_device(fields.u_ssa);
        gpism::sync_host_to_device(fields.v_ssa);
        // NetCDF writers use host buffers; ensure derived fields computed on the
        // device are synced before enqueueing asynchronous output.
        gpism::sync_device_to_host(fields.usurf);
        gpism::sync_device_to_host(fields.uvel);
        gpism::sync_device_to_host(fields.vvel);
        gpism::sync_device_to_host(fields.u_ssa);
        gpism::sync_device_to_host(fields.v_ssa);
        fields.has_usurf = true;
        fields.has_velocity = true;
        fields.has_ssa_velocity = true;
        if (!output_writer.enqueue(options.output, context, grid, fields,
                                   clock.time())) {
          std::cerr << "Error: failed to write output file " << options.output
                    << '\n';
          return 2;
        }
        log_rank0(context, "Wrote output to " + options.output);
        clock.mark_output();
        last_output_time = clock.time();
      }

      if (run_ssa) {
        try {
          solver.solve(fields.thk, fields.topg, fields.tauc,
                       fields.has_vel_bc ? &fields.u_bc : nullptr,
                       fields.has_vel_bc ? &fields.v_bc : nullptr,
                       fields.has_vel_bc ? &fields.vel_bc_mask : nullptr,
                       vel_cc, ssa_options);
        } catch (const std::exception& exc) {
          std::cerr << "SSA failure: " << exc.what() << '\n';
          return 2;
        }
      }

      if (thermo_enabled) {
        const int mz = enthalpy.local_mz();
        gpism::vertical_diffusion_step(enthalpy, mz, dz, clock.dt(), thermo_opts,
                                       enthalpy_next);
        std::swap(enthalpy, enthalpy_next);
      }

      if (evolve_thickness) {
        exchange_field2d(fields.thk);
        exchange_field_stag(vel_cc);
        gpism::compute_face_velocity_from_center(grid, vel_cc, vel_face);
        exchange_field_stag(vel_face);
        gpism::compute_face_fluxes(grid, fields.thk, vel_face, flux);
        gpism::update_thickness(grid, flux, smb, clock.dt(), thickness_opts,
                                fields.thk);
        gpism::compute_cell_type(grid, fields.thk, fields.topg, sea_level,
                                 rho_ice, rho_water,
                                 ice_free_thickness_standard, cell_type);
      }

      clock.advance();
    }

    if (!options.output.empty() &&
        (last_output_time < clock.time() - 1e-12)) {
      gpism::compute_cell_type(grid, fields.thk, fields.topg, sea_level,
                               rho_ice, rho_water,
                               ice_free_thickness_standard, cell_type);
      gpism::compute_usurf_flotation(grid, fields.thk, fields.topg, cell_type,
                                     sea_level, rho_ice, rho_water,
                                     fields.usurf);
      gpism::sync_device_to_host(vel_cc);
      for (int j = 0; j < grid.local_my(); ++j) {
        for (int i = 0; i < grid.local_mx(); ++i) {
          const double u = vel_cc(i, j, 0);
          const double v = vel_cc(i, j, 1);
          fields.uvel(i, j) = u;
          fields.vvel(i, j) = v;
          fields.u_ssa(i, j) = u;
          fields.v_ssa(i, j) = v;
        }
      }
      gpism::sync_host_to_device(fields.uvel);
      gpism::sync_host_to_device(fields.vvel);
      gpism::sync_host_to_device(fields.u_ssa);
      gpism::sync_host_to_device(fields.v_ssa);
      gpism::sync_device_to_host(fields.usurf);
      gpism::sync_device_to_host(fields.uvel);
      gpism::sync_device_to_host(fields.vvel);
      gpism::sync_device_to_host(fields.u_ssa);
      gpism::sync_device_to_host(fields.v_ssa);
      fields.has_usurf = true;
      fields.has_velocity = true;
      fields.has_ssa_velocity = true;
      if (!output_writer.enqueue(options.output, context, grid, fields,
                                 clock.time())) {
        std::cerr << "Error: failed to write output file " << options.output
                  << '\n';
        return 2;
      }
      log_rank0(context, "Wrote final output to " + options.output);
    }

    if (!output_writer.flush()) {
      std::cerr << "Error: failed to flush output file " << options.output
                << '\n';
      return 2;
    }

    return 0;
#else
    std::cerr << "Error: NetCDF support not enabled in this build.\n";
    return 2;
#endif
  }

  log_rank0(context, "gpism scaffold: no simulation configured yet.");
  log_rank0(context, "Run `gpism --help` for available options.");
  return 0;
}
