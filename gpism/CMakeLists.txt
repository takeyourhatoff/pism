cmake_minimum_required(VERSION 3.16)

project(gpism LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GPISM_VERSION "0.0.0-dev")

set(GPISM_BACKEND "cpu" CACHE STRING "Backend: cpu, cuda, hip, sycl")
set(GPISM_PRECISION "double" CACHE STRING "Precision: double or mixed")

option(GPISM_ENABLE_CUDA "Enable CUDA backend" OFF)
option(GPISM_ENABLE_MPI "Enable MPI support" ON)
option(GPISM_ENABLE_NETCDF "Enable NetCDF support" ON)
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
set(GPISM_NETCDF "disabled")
set(GPISM_HAVE_CUDA 0)
set(GPISM_HAVE_MPI 0)
set(GPISM_HAVE_NETCDF 0)

if(GPISM_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  set(GPISM_BACKEND "cuda" CACHE STRING "Backend: cpu, cuda, hip, sycl" FORCE)
  set(GPISM_HAVE_CUDA 1)
endif()

if(GPISM_ENABLE_MPI)
  find_package(MPI QUIET COMPONENTS C CXX)
  if(MPI_CXX_FOUND)
    set(GPISM_MPI "enabled")
    set(GPISM_HAVE_MPI 1)
  endif()
endif()

if(GPISM_ENABLE_NETCDF)
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
  if(NetCDF_FOUND)
    set(GPISM_NETCDF "enabled")
    set(GPISM_HAVE_NETCDF 1)
  endif()
endif()

if(GPISM_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  else()
    message(WARNING "clang-tidy requested but not found")
  endif()
endif()

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
  src/device_policy.cpp
  src/grid2d.cpp
  src/runtime_config.cpp
  src/time_manager.cpp
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

if(GPISM_ENABLE_NETCDF)
  list(APPEND GPISM_CORE_SOURCES src/netcdf_io.cpp)
endif()

if(GPISM_ENABLE_CUDA)
  list(APPEND GPISM_CORE_SOURCES src/gpu_stub.cu)
  list(APPEND GPISM_CORE_SOURCES src/geometry_cuda.cu)
  list(APPEND GPISM_CORE_SOURCES src/ssa_operator_cuda.cu)
  list(APPEND GPISM_CORE_SOURCES src/linear_algebra_cuda.cu)
endif()

add_library(gpism-core ${GPISM_CORE_SOURCES})

target_include_directories(gpism-core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND)
  target_link_libraries(gpism-core PUBLIC MPI::MPI_CXX)
endif()

if(GPISM_ENABLE_CUDA)
  target_link_libraries(gpism-core PUBLIC CUDA::cudart)
  target_include_directories(gpism-core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(GPISM_ENABLE_NETCDF AND NetCDF_FOUND)
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
  add_executable(gpism-field-smoke tests/field_smoke.cpp)
  target_link_libraries(gpism-field-smoke PRIVATE gpism-core)
  add_test(NAME gpism-field-smoke COMMAND gpism-field-smoke)
  if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
    add_test(
      NAME gpism-field-smoke-mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-field-smoke> ${MPIEXEC_POSTFLAGS}
    )
  endif()

  if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
    add_executable(gpism-gmres-mpi-smoke tests/gmres_mpi_smoke.cpp)
    target_link_libraries(gpism-gmres-mpi-smoke PRIVATE gpism-core)
    add_test(
      NAME gpism-gmres-mpi-smoke-1
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-gmres-mpi-smoke> ${MPIEXEC_POSTFLAGS}
    )
    add_test(
      NAME gpism-gmres-mpi-smoke-2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-gmres-mpi-smoke> ${MPIEXEC_POSTFLAGS}
    )
  endif()

  if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
    add_executable(gpism-ssa-picard-mpi-smoke tests/ssa_picard_mpi_smoke.cpp)
    target_link_libraries(gpism-ssa-picard-mpi-smoke PRIVATE gpism-core)
    add_test(
      NAME gpism-ssa-picard-mpi-smoke-1
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-ssa-picard-mpi-smoke> ${MPIEXEC_POSTFLAGS}
    )
    add_test(
      NAME gpism-ssa-picard-mpi-smoke-2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-ssa-picard-mpi-smoke> ${MPIEXEC_POSTFLAGS}
    )
  endif()

  if(NOT GPISM_ENABLE_CUDA)
    add_executable(gpism-gmres-mg-compare-smoke tests/gmres_mg_compare_smoke.cpp)
    target_link_libraries(gpism-gmres-mg-compare-smoke PRIVATE gpism-core)
    add_test(NAME gpism-gmres-mg-compare-smoke COMMAND gpism-gmres-mg-compare-smoke)
  endif()

  if(GPISM_ENABLE_NETCDF AND NetCDF_FOUND)
    add_executable(gpism-io-smoke tests/io_smoke.cpp)
    target_link_libraries(gpism-io-smoke PRIVATE gpism-core)
    add_test(NAME gpism-io-smoke COMMAND gpism-io-smoke)
    if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
      add_test(
        NAME gpism-io-smoke-mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:gpism-io-smoke> ${MPIEXEC_POSTFLAGS}
      )
    endif()
    add_executable(gpism-timestep-io-smoke tests/timestep_io_smoke.cpp)
    target_link_libraries(gpism-timestep-io-smoke PRIVATE gpism-core)
    add_test(NAME gpism-timestep-io-smoke COMMAND gpism-timestep-io-smoke)
  endif()

  if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE AND GPISM_ENABLE_CUDA)
    add_executable(gpism-ssa-operator-mpi-smoke tests/ssa_operator_mpi_smoke.cpp)
    target_link_libraries(gpism-ssa-operator-mpi-smoke PRIVATE gpism-core)
    add_test(
      NAME gpism-ssa-operator-mpi-smoke
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:gpism-ssa-operator-mpi-smoke> ${MPIEXEC_POSTFLAGS}
    )
  endif()

  add_executable(gpism-ssa-operator-smoke tests/ssa_operator_smoke.cpp)
  target_link_libraries(gpism-ssa-operator-smoke PRIVATE gpism-core)
  add_test(NAME gpism-ssa-operator-smoke COMMAND gpism-ssa-operator-smoke)

  add_executable(gpism-linear-algebra-smoke tests/linear_algebra_smoke.cpp)
  target_link_libraries(gpism-linear-algebra-smoke PRIVATE gpism-core)
  add_test(NAME gpism-linear-algebra-smoke COMMAND gpism-linear-algebra-smoke)

  add_executable(gpism-thickness-smoke tests/thickness_smoke.cpp)
  target_link_libraries(gpism-thickness-smoke PRIVATE gpism-core)
  add_test(NAME gpism-thickness-smoke COMMAND gpism-thickness-smoke)

  add_executable(gpism-ssa-timestep-smoke tests/ssa_timestep_smoke.cpp)
  target_link_libraries(gpism-ssa-timestep-smoke PRIVATE gpism-core)
  add_test(NAME gpism-ssa-timestep-smoke COMMAND gpism-ssa-timestep-smoke)

  add_executable(gpism-gmres-smoke tests/gmres_smoke.cpp)
  target_link_libraries(gpism-gmres-smoke PRIVATE gpism-core)
  add_test(NAME gpism-gmres-smoke COMMAND gpism-gmres-smoke)

  add_executable(gpism-multigrid-levels-smoke tests/multigrid_levels_smoke.cpp)
  target_link_libraries(gpism-multigrid-levels-smoke PRIVATE gpism-core)
  add_test(NAME gpism-multigrid-levels-smoke COMMAND gpism-multigrid-levels-smoke)

  add_executable(gpism-multigrid-transfer-smoke tests/multigrid_transfer_smoke.cpp)
  target_link_libraries(gpism-multigrid-transfer-smoke PRIVATE gpism-core)
  add_test(NAME gpism-multigrid-transfer-smoke COMMAND gpism-multigrid-transfer-smoke)

  add_executable(gpism-multigrid-jacobi-smoke tests/multigrid_jacobi_smoke.cpp)
  target_link_libraries(gpism-multigrid-jacobi-smoke PRIVATE gpism-core)
  add_test(NAME gpism-multigrid-jacobi-smoke COMMAND gpism-multigrid-jacobi-smoke)

  add_executable(gpism-multigrid-chebyshev-smoke tests/multigrid_chebyshev_smoke.cpp)
  target_link_libraries(gpism-multigrid-chebyshev-smoke PRIVATE gpism-core)
  add_test(NAME gpism-multigrid-chebyshev-smoke COMMAND gpism-multigrid-chebyshev-smoke)

  add_executable(gpism-multigrid-vcycle-smoke tests/multigrid_vcycle_smoke.cpp)
  target_link_libraries(gpism-multigrid-vcycle-smoke PRIVATE gpism-core)
  add_test(NAME gpism-multigrid-vcycle-smoke COMMAND gpism-multigrid-vcycle-smoke)

  add_executable(gpism-mg-preconditioner-smoke tests/mg_preconditioner_smoke.cpp)
  target_link_libraries(gpism-mg-preconditioner-smoke PRIVATE gpism-core)
  add_test(NAME gpism-mg-preconditioner-smoke COMMAND gpism-mg-preconditioner-smoke)

  add_executable(gpism-multigrid-coeff-smoke tests/multigrid_coeff_smoke.cpp)
  target_link_libraries(gpism-multigrid-coeff-smoke PRIVATE gpism-core)
  add_test(NAME gpism-multigrid-coeff-smoke COMMAND gpism-multigrid-coeff-smoke)

  add_executable(gpism-multigrid-scaling-smoke tests/multigrid_scaling_smoke.cpp)
  target_link_libraries(gpism-multigrid-scaling-smoke PRIVATE gpism-core)
  add_test(NAME gpism-multigrid-scaling-smoke COMMAND gpism-multigrid-scaling-smoke)

  add_executable(gpism-ssa-picard-smoke tests/ssa_picard_smoke.cpp)
  target_link_libraries(gpism-ssa-picard-smoke PRIVATE gpism-core)
  add_test(NAME gpism-ssa-picard-smoke COMMAND gpism-ssa-picard-smoke)

  add_executable(gpism-ssa-picard-nontrivial-smoke tests/ssa_picard_nontrivial_smoke.cpp)
  target_link_libraries(gpism-ssa-picard-nontrivial-smoke PRIVATE gpism-core)
  add_test(NAME gpism-ssa-picard-nontrivial-smoke COMMAND gpism-ssa-picard-nontrivial-smoke)

  if(GPISM_ENABLE_CUDA)
    add_executable(gpism-ssa-operator-cuda-smoke tests/ssa_operator_cuda_smoke.cpp)
    target_link_libraries(gpism-ssa-operator-cuda-smoke PRIVATE gpism-core)
    add_test(NAME gpism-ssa-operator-cuda-smoke COMMAND gpism-ssa-operator-cuda-smoke)

    add_executable(gpism-ssa-picard-parity-smoke tests/ssa_picard_parity_smoke.cpp)
    target_link_libraries(gpism-ssa-picard-parity-smoke PRIVATE gpism-core)
    add_test(NAME gpism-ssa-picard-parity-smoke COMMAND gpism-ssa-picard-parity-smoke)
  endif()
endif()
