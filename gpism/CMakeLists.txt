cmake_minimum_required(VERSION 3.16)

project(gpism LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GPISM_VERSION "0.0.0-dev")

set(GPISM_BACKEND "cuda")
set(GPISM_PRECISION "double" CACHE STRING "Precision: double or mixed")

option(GPISM_ENABLE_MPI "Enable MPI support" ON)
option(GPISM_ENABLE_CLANG_TIDY "Enable clang-tidy analysis" OFF)
option(GPISM_BUILD_TESTS "Build gpism tests" ON)

find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GPISM_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT GPISM_GIT_SHA)
  set(GPISM_GIT_SHA "unknown")
endif()

set(GPISM_MPI "disabled")
set(GPISM_NETCDF "enabled")
set(GPISM_HAVE_MPI 0)

find_package(CUDAToolkit REQUIRED)
if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
endif()

if(GPISM_ENABLE_MPI)
  find_package(MPI QUIET COMPONENTS C CXX)
  if(MPI_CXX_FOUND)
    set(GPISM_MPI "enabled")
    set(GPISM_HAVE_MPI 1)
  endif()
endif()

find_package(NetCDF QUIET)
if(NOT NetCDF_FOUND)
  find_package(netCDF CONFIG QUIET)
  if(netCDF_FOUND)
    set(NetCDF_FOUND TRUE)
  endif()
endif()
if(NOT NetCDF_FOUND)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(NetCDF IMPORTED_TARGET QUIET netcdf)
  endif()
endif()
if(NOT NetCDF_FOUND)
  message(FATAL_ERROR "NetCDF is required for gpism GPU-only builds.")
endif()

if(GPISM_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  else()
    message(WARNING "clang-tidy requested but not found")
  endif()
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/gpism)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gpism/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/gpism/version.h
  @ONLY
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gpism/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/gpism/config.h
  @ONLY
)

set(GPISM_CORE_SOURCES
  src/context.cpp
  src/grid2d.cpp
  src/runtime_config.cpp
  src/execution_context.cpp
  src/time_manager.cpp
  src/profile.cpp
  src/sync_stats.cpp
  src/sync_audit.cpp
  src/thermodynamics.cpp
  src/geometry.cpp
  src/thickness.cpp
  src/viscosity.cpp
  src/ssa_operator.cpp
  src/linear_algebra.cpp
  src/gmres.cpp
  src/multigrid.cpp
  src/mg_preconditioner.cpp
  src/ssa_solver.cpp
)

list(APPEND GPISM_CORE_SOURCES src/netcdf_io.cpp src/async_output.cpp)
list(APPEND GPISM_CORE_SOURCES src/gpu_stub.cu)
list(APPEND GPISM_CORE_SOURCES src/geometry_cuda.cu)
list(APPEND GPISM_CORE_SOURCES src/thickness_cuda.cu)
list(APPEND GPISM_CORE_SOURCES src/viscosity_cuda.cu)
list(APPEND GPISM_CORE_SOURCES src/ssa_operator_cuda.cu)
list(APPEND GPISM_CORE_SOURCES src/ssa_solver_cuda.cu)
list(APPEND GPISM_CORE_SOURCES src/linear_algebra_cuda.cu)
list(APPEND GPISM_CORE_SOURCES src/multigrid_cuda.cu)
list(APPEND GPISM_CORE_SOURCES src/thermodynamics_cuda.cu)

add_library(gpism-core ${GPISM_CORE_SOURCES})

target_include_directories(gpism-core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND)
  target_link_libraries(gpism-core PUBLIC MPI::MPI_CXX)
endif()

target_link_libraries(gpism-core PUBLIC CUDA::cudart)
target_include_directories(gpism-core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

if(TARGET NetCDF::NetCDF)
  target_link_libraries(gpism-core PUBLIC NetCDF::NetCDF)
elseif(TARGET netCDF::netcdf)
  target_link_libraries(gpism-core PUBLIC netCDF::netcdf)
elseif(TARGET PkgConfig::NetCDF)
  target_link_libraries(gpism-core PUBLIC PkgConfig::NetCDF)
elseif(NetCDF_LIBRARIES)
  target_link_libraries(gpism-core PUBLIC ${NetCDF_LIBRARIES})
  target_include_directories(gpism-core PUBLIC ${NetCDF_INCLUDE_DIRS})
endif()
if(NetCDF_INCLUDE_DIRS)
  target_include_directories(gpism-core PUBLIC ${NetCDF_INCLUDE_DIRS})
endif()
find_package(HDF5 QUIET COMPONENTS C HL)
if(HDF5_FOUND)
  target_link_libraries(gpism-core PUBLIC HDF5::HDF5)
  if(TARGET HDF5::HL)
    target_link_libraries(gpism-core PUBLIC HDF5::HL)
  endif()
endif()

add_executable(gpism
  src/main.cpp
)

target_link_libraries(gpism PRIVATE gpism-core)

install(TARGETS gpism RUNTIME DESTINATION bin)

find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(CLANG_FORMAT_EXE)
  file(GLOB_RECURSE GPISM_FORMAT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
  )
  add_custom_target(gpism-format
    COMMAND ${CLANG_FORMAT_EXE} -i ${GPISM_FORMAT_FILES}
    COMMENT "Formatting gpism sources with clang-format"
  )
endif()

if(GPISM_BUILD_TESTS)
  enable_testing()
  if(MPIEXEC_EXECUTABLE)
    set(GPISM_MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS})
    list(APPEND GPISM_MPIEXEC_PREFLAGS "--oversubscribe")
  endif()

  function(gpism_add_smoke name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE gpism-core)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  gpism_add_smoke(gpism-field-smoke tests/field_smoke.cpp)
  gpism_add_smoke(gpism-io-smoke tests/io_smoke.cpp)
  gpism_add_smoke(gpism-io-legacy-dims-smoke tests/io_legacy_dims_smoke.cpp)
  gpism_add_smoke(gpism-timestep-io-smoke tests/timestep_io_smoke.cpp)
  gpism_add_smoke(gpism-ssa-operator-smoke tests/ssa_operator_smoke.cpp)
  gpism_add_smoke(gpism-ssa-rhs-centered-smoke tests/ssa_rhs_centered_smoke.cpp)
  gpism_add_smoke(gpism-ssa-beta-centered-smoke tests/ssa_beta_centered_smoke.cpp)
  gpism_add_smoke(gpism-linear-algebra-smoke tests/linear_algebra_smoke.cpp)
  gpism_add_smoke(gpism-thickness-smoke tests/thickness_smoke.cpp)
  gpism_add_smoke(gpism-timestep-gpu-smoke tests/timestep_gpu_smoke.cpp)
  gpism_add_smoke(gpism-timestep-device-residency-smoke
                  tests/timestep_device_residency_smoke.cpp)
  gpism_add_smoke(gpism-gmres-device-orthogonalization-smoke
                  tests/gmres_device_orthogonalization_smoke.cpp)
  gpism_add_smoke(gpism-mg-mixed-precision-parity-smoke
                  tests/mg_mixed_precision_parity_smoke.cpp)
  gpism_add_smoke(gpism-gmres-smoke tests/gmres_smoke.cpp)
  gpism_add_smoke(gpism-gmres-givens-smoke tests/gmres_givens_smoke.cpp)
  gpism_add_smoke(gpism-multigrid-levels-smoke tests/multigrid_levels_smoke.cpp)
  gpism_add_smoke(gpism-multigrid-transfer-smoke tests/multigrid_transfer_smoke.cpp)
  gpism_add_smoke(gpism-multigrid-jacobi-smoke tests/multigrid_jacobi_smoke.cpp)
  gpism_add_smoke(gpism-multigrid-chebyshev-smoke tests/multigrid_chebyshev_smoke.cpp)
  gpism_add_smoke(gpism-multigrid-vcycle-smoke tests/multigrid_vcycle_smoke.cpp)
  gpism_add_smoke(gpism-mg-preconditioner-smoke tests/mg_preconditioner_smoke.cpp)
  gpism_add_smoke(gpism-multigrid-coeff-smoke tests/multigrid_coeff_smoke.cpp)
  gpism_add_smoke(gpism-multigrid-scaling-smoke tests/multigrid_scaling_smoke.cpp)
  gpism_add_smoke(gpism-ssa-picard-smoke tests/ssa_picard_smoke.cpp)
  gpism_add_smoke(gpism-ssa-picard-nontrivial-smoke tests/ssa_picard_nontrivial_smoke.cpp)

  if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
    add_test(
      NAME gpism-field-smoke-mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${GPISM_MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-field-smoke> ${MPIEXEC_POSTFLAGS}
    )
    add_test(
      NAME gpism-io-smoke-mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${GPISM_MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-io-smoke> ${MPIEXEC_POSTFLAGS}
    )
    gpism_add_smoke(gpism-ssa-operator-mpi-smoke tests/ssa_operator_mpi_smoke.cpp)
    add_test(
      NAME gpism-ssa-operator-mpi-smoke-2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${GPISM_MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-ssa-operator-mpi-smoke> ${MPIEXEC_POSTFLAGS}
    )
    gpism_add_smoke(gpism-halo-exchange-device-persistent-smoke
                    tests/halo_exchange_device_persistent_smoke.cpp)
    add_test(
      NAME gpism-halo-exchange-device-persistent-smoke-2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${GPISM_MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-halo-exchange-device-persistent-smoke> ${MPIEXEC_POSTFLAGS}
    )
  endif()

  add_test(
    NAME gpism-pism-ssa-compare
    COMMAND ${CMAKE_COMMAND} -E env
            GPISM_BIN=$<TARGET_FILE:gpism>
            COMPARE_STRICT=1
            COMPARE_TOL_RMS=0.02
            COMPARE_TOL_MAX=0.05
            COMPARE_REQUIRE_NONZERO=1
            COMPARE_ALLOW_MISSING=1
            MIN_WALL=0
            YEARS=0.05
            GPISM_DT=0.05
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_pism_gpism.sh
  )
endif()
