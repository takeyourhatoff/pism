#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gpism/context.h"
#include "gpism/execution_context.h"
#include "gpism/field_sync.h"
#include "gpism/linear_algebra.h"
#include "gpism/runtime_config.h"
#include "gpism/sync_audit.h"
#include "gpism/sync_stats.h"
#include "gpism/time_manager.h"
#include "gpism/config.h"
#include "gpism/version.h"

#include "gpism/async_output.h"
#include "gpism/netcdf_io.h"
#include "gpism/geometry.h"
#include "gpism/halo_exchange.h"
#include "gpism/ssa_solver.h"
#include "gpism/thickness.h"
#include "gpism/thermodynamics.h"
#include "gpism/viscosity.h"

#include <cuda_runtime.h>

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
  cudaSetDevice(context.device_id());
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
      std::cerr << "Error: failed to read config file " << options.config_path;
      if (!config.last_error().empty()) {
        std::cerr << ": " << config.last_error();
      }
      std::cerr << '\n';
      return 2;
    }
  }
  if (!options.config_override_path.empty()) {
    if (!config.apply_override(options.config_override_path)) {
      std::cerr << "Error: failed to read override config file "
                << options.config_override_path;
      if (!config.last_error().empty()) {
        std::cerr << ": " << config.last_error();
      }
      std::cerr << '\n';
      return 2;
    }
  }
  if (!options.run_years.empty()) {
    config.set("time.years", options.run_years);
  }

  const bool enforce_hotloop_residency =
      config.get_bool("device.enforce_hotloop_residency");
  const bool enable_cuda_graphs = config.get_bool("device.cuda_graphs");
  const int compute_streams = config.get_int("device.compute_streams");
  const std::string reduction_backend =
      config.get_string("linear_algebra.reduction_backend");
  if (reduction_backend == "cub") {
    gpism::set_reduction_backend(gpism::ReductionBackend::CUB);
  } else if (reduction_backend == "cublas") {
    gpism::set_reduction_backend(gpism::ReductionBackend::CUBLAS);
  } else {
    throw std::runtime_error("Invalid linear_algebra.reduction_backend value: " +
                             reduction_backend);
  }
  gpism::ExecutionContext::instance().configure(enable_cuda_graphs,
                                                compute_streams);
  gpism::SyncAudit::enable(enforce_hotloop_residency);
  gpism::SyncAudit::set_fail_fast(enforce_hotloop_residency);
  gpism::SyncAudit::reset();
  gpism::SyncStats::enable(enforce_hotloop_residency);
  gpism::SyncStats::reset();

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
    const int output_ring_depth = config.get_int("io.output.device_ring_depth");
    const bool async_stage_from_device =
        config.get_bool("io.output.async_stage_from_device");
    output_writer.configure(grid, fields, async_output, output_ring_depth,
                            async_stage_from_device);

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
    if (config.has("ssa.picard.convergence_check_interval")) {
      ssa_options.picard_convergence_check_interval = std::max(
          1, config.get_int("ssa.picard.convergence_check_interval"));
    }
    if (config.has("ssa.device_metrics_batch")) {
      ssa_options.device_metrics_batch =
          config.get_bool("ssa.device_metrics_batch");
    }
    ssa_options.gmres_max_iter = config.get_int("ssa.gmres_max_iter");
    ssa_options.tol_nuH = config.get_double("ssa.tol_nuH");
    ssa_options.tol_vel = config.get_double("ssa.tol_vel");
    ssa_options.gmres_tol = config.get_double("ssa.gmres_tol");
    if (config.has("ssa.gmres_tol_relative_to_rhs")) {
      ssa_options.gmres_tol_relative_to_rhs =
          config.get_bool("ssa.gmres_tol_relative_to_rhs");
    }
    if (config.has("ssa.gmres.residual_check_interval")) {
    ssa_options.gmres_residual_check_interval =
          std::max(1, config.get_int("ssa.gmres.residual_check_interval"));
    }
    ssa_options.vel_relax = config.get_double("ssa.vel_relax");
    ssa_options.nuH_relax = config.get_double("ssa.nuH_relax");
    ssa_options.gmres_verbose = config.get_bool("ssa.gmres_verbose");
    ssa_options.diagnostic = config.get_bool("ssa.diagnostic");
    // PISM's stress_balance.ssa.epsilon has units Pa*s*m (regularization added to nu*H).
    ssa_options.nuH_regularization = config.get_double("stress_balance.ssa.epsilon");
    ssa_options.use_mg_precond = config.get_bool("ssa.mg.enabled");
    ssa_options.mg_pre_iters = config.get_int("ssa.mg.pre_iters");
    ssa_options.mg_post_iters = config.get_int("ssa.mg.post_iters");
    ssa_options.mg_coarse_iters = config.get_int("ssa.mg.coarse_iters");
    ssa_options.mg_omega = config.get_double("ssa.mg.omega");
    ssa_options.mg_min_size = config.get_int("ssa.mg.min_size");
    ssa_options.mg_jacobi_sweeps_per_launch =
        std::max(1, config.get_int("ssa.mg.jacobi_sweeps_per_launch"));
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
    const std::string precond_precision =
        config.get_string("ssa.precond_precision");
    if (precond_precision == "fp32") {
      ssa_options.precond_precision = gpism::SSAPrecondPrecision::FP32;
    } else if (precond_precision == "fp64") {
      ssa_options.precond_precision = gpism::SSAPrecondPrecision::FP64;
    } else {
      throw std::runtime_error("Invalid ssa.precond_precision value: " +
                               precond_precision);
    }
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
    ssa_options.halo_mode = gpism::SSAHaloMode::Device;
    ssa_options.require_cuda_aware_mpi =
        config.get_bool("device.require_cuda_aware_mpi");

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
    const bool require_cuda_aware_mpi = ssa_options.require_cuda_aware_mpi;
    auto exchange_field2d = [&](auto& field) {
      if (!context.mpi_enabled() || context.size() <= 1) {
        return;
      }
      const bool have_device = field.device_data() != nullptr;
      const bool can_device = context.cuda_aware_mpi() && have_device;
      if (!can_device && require_cuda_aware_mpi) {
        throw std::runtime_error(
            "device.require_cuda_aware_mpi=1 but Field2D exchange cannot use "
            "device buffers");
      }
      if (!can_device) {
        throw std::runtime_error(
            "Multi-rank run requires CUDA-aware MPI device-buffer exchange");
      }
      exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Device);
    };

    auto exchange_field_stag = [&](auto& field) {
      if (!context.mpi_enabled() || context.size() <= 1) {
        return;
      }
      const bool have_device =
          field.component(0).device_data() != nullptr &&
          field.component(1).device_data() != nullptr;
      const bool can_device = context.cuda_aware_mpi() && have_device;
      if (!can_device && require_cuda_aware_mpi) {
        throw std::runtime_error(
            "device.require_cuda_aware_mpi=1 but FieldStag2D exchange cannot "
            "use device buffers");
      }
      if (!can_device) {
        throw std::runtime_error(
            "Multi-rank run requires CUDA-aware MPI device-buffer exchange");
      }
      exchange.exchange(field, grid, context, gpism::HaloExchange2D::Mode::Device);
    };

    auto populate_velocity_diagnostics = [&]() {
      auto copy_field2d_device =
          [&](const gpism::Field2D<double>& src, gpism::Field2D<double>& dst) {
            if (!src.has_device_data() || !dst.has_device_data()) {
              return false;
            }
            const std::size_t src_offset =
                static_cast<std::size_t>(src.ghost_width()) *
                    static_cast<std::size_t>(src.stride()) +
                static_cast<std::size_t>(src.ghost_width());
            const std::size_t dst_offset =
                static_cast<std::size_t>(dst.ghost_width()) *
                    static_cast<std::size_t>(dst.stride()) +
                static_cast<std::size_t>(dst.ghost_width());
            const std::size_t row_bytes =
                static_cast<std::size_t>(grid.local_mx()) * sizeof(double);
            const std::size_t row_count =
                static_cast<std::size_t>(grid.local_my());
            if (row_bytes == 0 || row_count == 0) {
              return true;
            }
            const cudaError_t err =
                cudaMemcpy2D(dst.device_data() + dst_offset,
                             static_cast<std::size_t>(dst.stride()) *
                                 sizeof(double),
                             src.device_data() + src_offset,
                             static_cast<std::size_t>(src.stride()) *
                                 sizeof(double),
                             row_bytes, row_count, cudaMemcpyDeviceToDevice);
            return err == cudaSuccess;
          };
      const bool copied_on_device =
          copy_field2d_device(vel_cc.component(0), fields.uvel) &&
          copy_field2d_device(vel_cc.component(1), fields.vvel) &&
          copy_field2d_device(vel_cc.component(0), fields.u_ssa) &&
          copy_field2d_device(vel_cc.component(1), fields.v_ssa);
      if (copied_on_device) {
        return;
      }
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
    };

    const bool allow_solver_sync_for_diagnostics =
        ssa_options.diagnostic || ssa_options.gmres_verbose ||
        ssa_options.mg_diagnostic || ssa_options.gmres_precond_diagnostic;
    double last_output_time = -1.0;
    while (!clock.done()) {
      if (clock.should_output()) {
        gpism::ScopedSyncAudit output_scope(true, "timestep.output");
        gpism::compute_cell_type(grid, fields.thk, fields.topg, sea_level,
                                 rho_ice, rho_water,
                                 ice_free_thickness_standard, cell_type);
        gpism::compute_usurf_flotation(grid, fields.thk, fields.topg,
                                       cell_type, sea_level, rho_ice,
                                       rho_water, fields.usurf);
        populate_velocity_diagnostics();
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

      {
        gpism::ScopedSyncAudit compute_scope(
            !enforce_hotloop_residency, "timestep.compute");
        if (run_ssa) {
          try {
            gpism::ScopedSyncAudit solver_scope(
                !enforce_hotloop_residency || allow_solver_sync_for_diagnostics,
                "timestep.ssa");
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
          gpism::vertical_diffusion_step(enthalpy, mz, dz, clock.dt(),
                                         thermo_opts, enthalpy_next);
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
      }

      clock.advance();
    }

    if (!options.output.empty() &&
        (last_output_time < clock.time() - 1e-12)) {
      gpism::ScopedSyncAudit output_scope(true, "timestep.output.final");
      gpism::compute_cell_type(grid, fields.thk, fields.topg, sea_level,
                               rho_ice, rho_water,
                               ice_free_thickness_standard, cell_type);
      gpism::compute_usurf_flotation(grid, fields.thk, fields.topg, cell_type,
                                     sea_level, rho_ice, rho_water,
                                     fields.usurf);
      populate_velocity_diagnostics();
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

    if (enforce_hotloop_residency) {
      if (gpism::SyncAudit::violations() != 0) {
        std::cerr << "Error: hot-loop residency audit failed (violations="
                  << gpism::SyncAudit::violations() << ")\n";
        return 2;
      }
      log_rank0(
          context,
          "Hot-loop residency audit passed: no disallowed syncs detected.");
    }

    return 0;
  }

  log_rank0(context, "gpism scaffold: no simulation configured yet.");
  log_rank0(context, "Run `gpism --help` for available options.");
  return 0;
}
