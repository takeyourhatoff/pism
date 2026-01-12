#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gpism/context.h"
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

    gpism::FieldStag2D<double> vel(grid.local_mx(), grid.local_my(),
                                   grid.ghost_width());
    vel.fill(0.0);

    gpism::FieldStag2D<double> flux(grid.local_mx(), grid.local_my(),
                                    grid.ghost_width());
    gpism::Field2D<int> mask(grid.local_mx(), grid.local_my(), grid.ghost_width());
    gpism::ViscosityModel viscosity(1e-16, 3.0, 1.0);
    gpism::SSASolver solver(grid, 910.0, 9.81, 100.0, viscosity);

    gpism::SSASolverOptions ssa_options;
    ssa_options.max_picard = config.get_int("ssa.max_picard");
    ssa_options.gmres_max_iter = config.get_int("ssa.gmres_max_iter");
    ssa_options.tol_nuH = config.get_double("ssa.tol_nuH");
    ssa_options.tol_vel = config.get_double("ssa.tol_vel");
    ssa_options.gmres_tol = config.get_double("ssa.gmres_tol");
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
    ssa_options.context = &context;

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

    gpism::sync_host_to_device(fields.thk);
    gpism::sync_host_to_device(fields.topg);
    gpism::sync_host_to_device(fields.tauc);
    gpism::sync_host_to_device(smb);
    gpism::sync_host_to_device(vel);
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

    while (!clock.done()) {
      if (clock.should_output()) {
        geometry.compute_usurf(grid, fields.thk, fields.topg, fields.usurf);
        gpism::compute_cell_center_velocity(grid, vel, fields.uvel, fields.vvel);
        fields.has_usurf = true;
        fields.has_velocity = true;
        if (!output_writer.enqueue(options.output, context, grid, fields,
                                   clock.time())) {
          std::cerr << "Error: failed to write output file " << options.output
                    << '\n';
          return 2;
        }
        log_rank0(context, "Wrote output to " + options.output);
        clock.mark_output();
      }

      if (run_ssa) {
        solver.solve(fields.thk, fields.topg, fields.tauc,
                     fields.has_vel_bc ? &fields.u_bc : nullptr,
                     fields.has_vel_bc ? &fields.v_bc : nullptr,
                     fields.has_vel_bc ? &fields.vel_bc_mask : nullptr,
                     vel, ssa_options);
      }

      if (thermo_enabled) {
        const int mz = enthalpy.local_mz();
        gpism::vertical_diffusion_step(enthalpy, mz, dz, clock.dt(), thermo_opts,
                                       enthalpy_next);
        std::swap(enthalpy, enthalpy_next);
      }

      if (evolve_thickness) {
        exchange_field2d(fields.thk);
        exchange_field_stag(vel);
        gpism::compute_face_fluxes(grid, fields.thk, vel, flux);
        gpism::update_thickness(grid, flux, smb, clock.dt(), thickness_opts,
                                fields.thk);
        gpism::update_mask(grid, fields.thk, mask);
      }

      clock.advance();
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
