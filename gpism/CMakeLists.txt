cmake_minimum_required(VERSION 3.16)

project(gpism LANGUAGES CXX)

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
  find_package(MPI QUIET)
  if(MPI_CXX_FOUND)
    set(GPISM_MPI "enabled")
    set(GPISM_HAVE_MPI 1)
  endif()
endif()

if(GPISM_ENABLE_NETCDF)
  find_package(NetCDF QUIET)
  if(NOT NetCDF_FOUND)
    find_package(netCDF CONFIG QUIET)
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
  src/grid2d.cpp
  src/runtime_config.cpp
  src/geometry.cpp
)

if(GPISM_ENABLE_NETCDF)
  list(APPEND GPISM_CORE_SOURCES src/netcdf_io.cpp)
endif()

if(GPISM_ENABLE_CUDA)
  list(APPEND GPISM_CORE_SOURCES src/gpu_stub.cu)
  list(APPEND GPISM_CORE_SOURCES src/geometry_cuda.cu)
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
  endif()
endif()
