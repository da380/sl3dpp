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
option(SL3DPP_BUILD_APPS "Build the sl3d program" ON)
option(SL3DPP_BUILD_BENCHMARKS "Build the cross-code benchmark drivers" OFF)
option(SL3DPP_BUILD_EXAMPLES "Build the example drivers" ON)

# ---------------------------------------------------------------------------
# Dependencies.
#
# The sibling header-only libraries are pinned by commit.  Declaration order
# matters: the first declaration of a name wins, so declaring them all here
# keeps GSHTrans's own (unpinned, main-tracking) declarations from taking
# over — and planetmodel's Interpolation requirement (>= v1.1.2) is met by
# this file's own pin — so the whole dependency set moves only when these
# tags are edited.
#
# For local development against working copies, point FetchContent at them:
#
#   cmake -B build \
#     -DFETCHCONTENT_SOURCE_DIR_NUMERICCONCEPTS=~/dev/NumericConcepts \
#     -DFETCHCONTENT_SOURCE_DIR_GAUSSQUAD=~/dev/GaussQuad \
#     -DFETCHCONTENT_SOURCE_DIR_INTERPOLATION=~/dev/Interpolation \
#     -DFETCHCONTENT_SOURCE_DIR_FFTWPP=~/dev/FFTWpp \
#     -DFETCHCONTENT_SOURCE_DIR_GSHTRANS=~/dev/GSHTrans \
#     -DFETCHCONTENT_SOURCE_DIR_PLANETMODEL=~/dev/planetmodel
#
# An installed copy is preferred when present (FIND_PACKAGE_ARGS).
# ---------------------------------------------------------------------------
include(FetchContent)

set(SL3DPP_NUMERICCONCEPTS_TAG "e03a2889d43a03660d0879f46dda1406d90c9517"
    CACHE STRING "NumericConcepts commit")
set(SL3DPP_GAUSSQUAD_TAG "7d4786d192f68077cd99a34dcdf1379ed1eeac9a"
    CACHE STRING "GaussQuad commit")
set(SL3DPP_INTERPOLATION_TAG "d5a1711ca6a89fc8ad8148fe13212398fe569bc0"
    CACHE STRING "Interpolation commit (v1.1.2)")
set(SL3DPP_FFTWPP_TAG "423558c0842b25f6031f2ae58dcab36d8f5f8874"
    CACHE STRING "FFTWpp commit")
set(SL3DPP_GSHTRANS_TAG "8dd116df8df4222fe9b10a40c7213d34f2a274fb"
    CACHE STRING "GSHTrans commit")
set(SL3DPP_PLANETMODEL_TAG "v1.3.1"
    CACHE STRING "planetmodel release tag")

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

# The whole repository is fetched (SOURCE_SUBDIR, no shallow clone of cpp/
# alone): cpp/CMakeLists.txt reads the version from ../pyproject.toml.
FetchContent_Declare(
  planetmodel
  GIT_REPOSITORY https://github.com/da380/planetmodel.git
  GIT_TAG ${SL3DPP_PLANETMODEL_TAG}
  SOURCE_SUBDIR cpp)

FetchContent_MakeAvailable(NumericConcepts GaussQuad Interpolation FFTWpp
                           GSHTrans planetmodel)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# netCDF-C comes in through planetmodel, which finds it itself (CONFIG
# first, then pkg-config) and makes the netCDF::netcdf target global for
# any future direct use (the ice histories, say).

# ---------------------------------------------------------------------------
# The library.
# ---------------------------------------------------------------------------
add_library(sl3dpp STATIC
  src/Adjoint.cpp
  src/Adjoint3D.cpp
  src/Deck.cpp
  src/DegreeOperator.cpp
  src/EarthModel.cpp
  src/Evolution.cpp
  src/Evolution3D.cpp
  src/Fingerprint.cpp
  src/GridCoupling.cpp
  src/io/KernelWriter.cpp
  src/io/FieldHistory.cpp
  src/io/ModelViscosity.cpp
  src/LoveNumbers.cpp
  src/Material.cpp
  src/Measurements.cpp
  src/MemoryPoints.cpp
  src/RadialMesh.cpp
  src/Rheology.cpp
  src/SeaLevel.cpp
  src/ToroidalOperator.cpp
  src/Version.cpp
  src/Viscoelastic.cpp
  src/Viscoelastic3D.cpp)

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
  BLAS::BLAS)

# Private: planetmodel's types stay behind the io/ boundary (stage 11);
# sl3dpp's public headers never include its headers.
target_link_libraries(sl3dpp PRIVATE planetmodel::planetmodel)

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
if(SL3DPP_BUILD_APPS)
  add_subdirectory(apps)
endif()

# Tests.
# ---------------------------------------------------------------------------
if(SL3DPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SL3DPP_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(SL3DPP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
