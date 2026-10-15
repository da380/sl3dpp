cmake_minimum_required(VERSION 3.24)

project(
  sl3dpp
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Forward and adjoint GIA modelling in 1D and 3D viscosity earth models")

# GSHTrans needs C++23 (GCC 13+ or Clang 18+).
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SL3DPP_BUILD_TESTS "Build the test suite" ON)

# ---------------------------------------------------------------------------
# Dependencies.
#
# The sibling header-only libraries are pinned by commit.  Declaration order
# matters: the first declaration of a name wins, so declaring all five here
# keeps GSHTrans's own (unpinned, main-tracking) declarations from taking
# over, and the whole dependency set moves only when these tags are edited.
#
# For local development against working copies, point FetchContent at them:
#
#   cmake -B build \
#     -DFETCHCONTENT_SOURCE_DIR_NUMERICCONCEPTS=~/dev/NumericConcepts \
#     -DFETCHCONTENT_SOURCE_DIR_GAUSSQUAD=~/dev/GaussQuad \
#     -DFETCHCONTENT_SOURCE_DIR_INTERPOLATION=~/dev/Interpolation \
#     -DFETCHCONTENT_SOURCE_DIR_FFTWPP=~/dev/FFTWpp \
#     -DFETCHCONTENT_SOURCE_DIR_GSHTRANS=~/dev/GSHTrans
#
# An installed copy is preferred when present (FIND_PACKAGE_ARGS).
# ---------------------------------------------------------------------------
include(FetchContent)

set(SL3DPP_NUMERICCONCEPTS_TAG "e03a2889d43a03660d0879f46dda1406d90c9517"
    CACHE STRING "NumericConcepts commit")
set(SL3DPP_GAUSSQUAD_TAG "7d4786d192f68077cd99a34dcdf1379ed1eeac9a"
    CACHE STRING "GaussQuad commit")
set(SL3DPP_INTERPOLATION_TAG "36a9539f4a94811f04d0d0c6f78bf15c72771b39"
    CACHE STRING "Interpolation commit")
set(SL3DPP_FFTWPP_TAG "423558c0842b25f6031f2ae58dcab36d8f5f8874"
    CACHE STRING "FFTWpp commit")
set(SL3DPP_GSHTRANS_TAG "c47b365bfa7f7cb6fdd1cc7e231a34b74247bb31"
    CACHE STRING "GSHTrans commit")

FetchContent_Declare(
  NumericConcepts
  GIT_REPOSITORY https://github.com/da380/NumericConcepts.git
  GIT_TAG ${SL3DPP_NUMERICCONCEPTS_TAG}
  FIND_PACKAGE_ARGS)

FetchContent_Declare(
  GaussQuad
  GIT_REPOSITORY https://github.com/da380/GaussQuad.git
  GIT_TAG ${SL3DPP_GAUSSQUAD_TAG}
  FIND_PACKAGE_ARGS)

FetchContent_Declare(
  Interpolation
  GIT_REPOSITORY https://github.com/da380/Interpolation.git
  GIT_TAG ${SL3DPP_INTERPOLATION_TAG}
  FIND_PACKAGE_ARGS)

FetchContent_Declare(
  FFTWpp
  GIT_REPOSITORY https://github.com/da380/FFTWpp.git
  GIT_TAG ${SL3DPP_FFTWPP_TAG}
  FIND_PACKAGE_ARGS)

FetchContent_Declare(
  GSHTrans
  GIT_REPOSITORY https://github.com/da380/GSHTrans.git
  GIT_TAG ${SL3DPP_GSHTRANS_TAG}
  FIND_PACKAGE_ARGS)

FetchContent_MakeAvailable(NumericConcepts GaussQuad Interpolation FFTWpp
                           GSHTrans)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# netCDF-C, for the 3D model files and the ice histories.
find_package(PkgConfig REQUIRED)
pkg_check_modules(NETCDF REQUIRED IMPORTED_TARGET netcdf)

# ---------------------------------------------------------------------------
# The library.
# ---------------------------------------------------------------------------
add_library(sl3dpp STATIC
  src/Deck.cpp
  src/DegreeOperator.cpp
  src/EarthModel.cpp
  src/LoveNumbers.cpp
  src/Material.cpp
  src/RadialMesh.cpp
  src/Version.cpp)

add_library(sl3dpp::sl3dpp ALIAS sl3dpp)

target_include_directories(sl3dpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(sl3dpp PUBLIC
  GSHTrans::GSHTrans
  Interpolation::Interpolation
  GaussQuad::GaussQuad
  OpenMP::OpenMP_CXX
  LAPACK::LAPACK
  BLAS::BLAS
  PkgConfig::NETCDF)

target_compile_options(sl3dpp PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)

# Clang 18 reports __cpp_concepts = 201907, which makes libstdc++ disable
# <expected> although the needed support is present; Clang 19 fixed the
# macro. The usual workaround until the floor rises (also in .clangd, for
# the editor tooling).
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
  target_compile_definitions(sl3dpp PUBLIC __cpp_concepts=202002L)
  target_compile_options(sl3dpp PUBLIC -Wno-builtin-macro-redefined)
endif()

# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
if(SL3DPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
