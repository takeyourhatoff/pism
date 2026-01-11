#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "gpism/context.h"
#include "gpism/runtime_config.h"
#include "gpism/config.h"
#include "gpism/version.h"

#if GPISM_HAVE_NETCDF
#include "gpism/netcdf_io.h"
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
    int time_index = -1;
    if (config.has("io.time_index")) {
      time_index = config.get_int("io.time_index");
    }
    if (!io.read_restart(options.input, context, grid, fields, time_index)) {
      std::cerr << "Error: failed to read input file " << options.input << '\n';
      return 2;
    }
    double time_value = 0.0;
    if (config.has("time.start_year")) {
      time_value = config.get_double("time.start_year");
    }
    if (!io.write_output(options.output, context, grid, fields, time_value)) {
      std::cerr << "Error: failed to write output file " << options.output << '\n';
      return 2;
    }
    log_rank0(context, "Wrote output to " + options.output);
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
