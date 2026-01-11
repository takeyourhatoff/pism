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

option(GPISM_ENABLE_MPI "Enable MPI support" ON)
option(GPISM_ENABLE_NETCDF "Enable NetCDF support" ON)

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

if(GPISM_ENABLE_MPI)
  find_package(MPI QUIET)
  if(MPI_CXX_FOUND)
    set(GPISM_MPI "enabled")
  endif()
endif()

if(GPISM_ENABLE_NETCDF)
  find_package(NetCDF QUIET)
  if(NetCDF_FOUND)
    set(GPISM_NETCDF "enabled")
  endif()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gpism/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/gpism/version.h
  @ONLY
)

add_executable(gpism
  src/main.cpp
)

target_include_directories(gpism
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

if(GPISM_ENABLE_MPI AND MPI_CXX_FOUND)
  target_link_libraries(gpism PRIVATE MPI::MPI_CXX)
endif()

if(GPISM_ENABLE_NETCDF AND NetCDF_FOUND)
  if(TARGET NetCDF::NetCDF)
    target_link_libraries(gpism PRIVATE NetCDF::NetCDF)
  elseif(NetCDF_LIBRARIES)
    target_link_libraries(gpism PRIVATE ${NetCDF_LIBRARIES})
    target_include_directories(gpism PRIVATE ${NetCDF_INCLUDE_DIRS})
  endif()
endif()

install(TARGETS gpism RUNTIME DESTINATION bin)
