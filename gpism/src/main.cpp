#include <cstring>
#include <iostream>

#include "gpism/context.h"
#include "gpism/version.h"

namespace {

void print_help() {
  std::cout
      << "gpism: GPU-first, PISM-compatible ice-sheet model (scaffold)\n"
      << "\n"
      << "Usage:\n"
      << "  gpism [--help] [--version]\n"
      << "\n"
      << "Options:\n"
      << "  --help       Show this help text\n"
      << "  --version    Show build information\n";
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

}  // namespace

int main(int argc, char** argv) {
  gpism::Context context(&argc, &argv);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_help();
      return 0;
    }
    if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-V") == 0) {
      print_version();
      return 0;
    }
  }

  std::cout << "gpism scaffold: no simulation configured yet.\n";
  std::cout << "Run `gpism --help` for available options.\n";
  return 0;
}
