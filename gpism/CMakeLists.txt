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
option(GPISM_ENABLE_CLANG_TIDY "Enable clang-tidy analysis" OFF)

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
