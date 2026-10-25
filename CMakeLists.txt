cmake_minimum_required(VERSION 3.24)

project(
  sl3dpp
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Forward and adjoint GIA modelling in 1D and 3D viscosity earth models")

# GSHTrans needs C++23 (GCC 13+ or Clang 18+). The requirement also travels
# with the library target (target_compile_features below), for consumers.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Only for a build of this project itself: as a subproject, the build type is
# the parent's to choose.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE
   AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)

# The developer targets default on only when this is the top-level project, so
# that a code fetching the library builds the library. The install rules stay
# on as a subproject too: a code installing a library that links this one
# needs it in an export set.
option(SL3DPP_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(SL3DPP_BUILD_APPS "Build the sl3d program" ${PROJECT_IS_TOP_LEVEL})
option(SL3DPP_BUILD_BENCHMARKS "Build the cross-code benchmark drivers" OFF)
option(SL3DPP_BUILD_EXAMPLES "Build the example drivers" ${PROJECT_IS_TOP_LEVEL})
option(SL3DPP_BUILD_DOCS "Add the sl3dpp_docs target (Doxygen)" OFF)
option(SL3DPP_INSTALL "Generate the install and export rules" ON)

# ---------------------------------------------------------------------------
# Dependencies.
#
# The sibling libraries are pinned by commit, so that every commit of sl3dpp
# names a state that can be rebuilt: the same sources, and so the same
# numbers, whenever it is built. The libraries track each other's main
# between their releases; sl3dpp, which produces results, does not. The pins
# move only when edited, and scripts/bump_dependencies.py moves them all to
# the libraries' current main. The Upstream workflow
# (.github/workflows/upstream.yml) builds and tests against every main each
# week, so a pin that falls behind something breaking is seen before the bump.
#
# The pins are plain variables, so an existing build tree follows them when
# they move (a cache variable would keep the first value it saw). One build
# tree can override any of them with the matching cache variable, and an
# empty override returns it to the pin:
#
#   cmake -B build -DSL3DPP_GSHTRANS_TAG=main ...
#
# The first declaration of a name wins, so declaring them all here, before
# any library declares its own, decides the revision of each for the whole
# build.
#
# For local development against working copies, point FetchContent at them:
#
#   cmake -B build \
#     -DFETCHCONTENT_SOURCE_DIR_NUMERICCONCEPTS=<checkouts>/NumericConcepts \
#     -DFETCHCONTENT_SOURCE_DIR_GAUSSQUAD=<checkouts>/GaussQuad \
#     -DFETCHCONTENT_SOURCE_DIR_INTERPOLATION=<checkouts>/Interpolation \
#     -DFETCHCONTENT_SOURCE_DIR_FFTWPP=<checkouts>/FFTWpp \
#     -DFETCHCONTENT_SOURCE_DIR_GSHTRANS=<checkouts>/GSHTrans \
#     -DFETCHCONTENT_SOURCE_DIR_PLANETMODEL=<checkouts>/planetmodel
#
# An installed copy is preferred when present (FIND_PACKAGE_ARGS).
# ---------------------------------------------------------------------------
include(FetchContent)

set(SL3DPP_NUMERICCONCEPTS_PIN "e5ec506ab510b0790f94dded6196ab4df8979259")
set(SL3DPP_GAUSSQUAD_PIN "c95bac5339e47ee6d8722dd15f1ab7cb031382bd")
set(SL3DPP_INTERPOLATION_PIN "43619e2627fe302e244d321c994d5b5cc717cd74")
set(SL3DPP_FFTWPP_PIN "b3af30eaf3f06760cd1d97df4386af1a0c75dd76")
set(SL3DPP_GSHTRANS_PIN "991dd1eaba2d0c47e4200356a0329c51dfe5fb5e")
set(SL3DPP_PLANETMODEL_PIN "d0c55e23dac7610802496768402de2bc001b3259")  # v1.3.2

# The revision each library is fetched at: the override when one is set,
# else the pin.
foreach(_sl3dpp_lib IN ITEMS NUMERICCONCEPTS GAUSSQUAD INTERPOLATION FFTWPP
                             GSHTRANS PLANETMODEL)
  set(SL3DPP_${_sl3dpp_lib}_TAG "" CACHE STRING
      "Override of the pinned ${_sl3dpp_lib} revision (empty: the pin)")
  mark_as_advanced(SL3DPP_${_sl3dpp_lib}_TAG)
  if(SL3DPP_${_sl3dpp_lib}_TAG)
    set(SL3DPP_${_sl3dpp_lib}_REVISION ${SL3DPP_${_sl3dpp_lib}_TAG})
    if(NOT SL3DPP_${_sl3dpp_lib}_TAG STREQUAL SL3DPP_${_sl3dpp_lib}_PIN)
      message(STATUS "sl3dpp: ${_sl3dpp_lib} at ${SL3DPP_${_sl3dpp_lib}_TAG} "
                     "(overriding the pin)")
    endif()
  else()
    set(SL3DPP_${_sl3dpp_lib}_REVISION ${SL3DPP_${_sl3dpp_lib}_PIN})
  endif()
endforeach()

# GSHTrans and FFTWpp generate their install rules only as the top-level
# project; this library's package refers to GSHTrans's, and GSHTrans's to
# FFTWpp's, so ask both for them. GSHTrans asks FFTWpp itself, but too late
# here: FFTWpp is made available first, below, and its own default would be
# cached by then. Without FORCE, so that an explicit OFF still wins. The other
# libraries install as subprojects by default.
if(SL3DPP_INSTALL)
  set(GSHTRANS_INSTALL ON CACHE BOOL "Install GSHTrans alongside sl3dpp")
  set(FFTWPP_INSTALL ON CACHE BOOL "Install FFTWpp alongside sl3dpp")
endif()

FetchContent_Declare(
  NumericConcepts
  GIT_REPOSITORY https://github.com/da380/NumericConcepts.git
  GIT_TAG ${SL3DPP_NUMERICCONCEPTS_REVISION}
  FIND_PACKAGE_ARGS)

FetchContent_Declare(
  GaussQuad
  GIT_REPOSITORY https://github.com/da380/GaussQuad.git
  GIT_TAG ${SL3DPP_GAUSSQUAD_REVISION}
  FIND_PACKAGE_ARGS)

FetchContent_Declare(
  Interpolation
  GIT_REPOSITORY https://github.com/da380/Interpolation.git
  GIT_TAG ${SL3DPP_INTERPOLATION_REVISION}
  FIND_PACKAGE_ARGS)

FetchContent_Declare(
  FFTWpp
  GIT_REPOSITORY https://github.com/da380/FFTWpp.git
  GIT_TAG ${SL3DPP_FFTWPP_REVISION}
  FIND_PACKAGE_ARGS)

FetchContent_Declare(
  GSHTrans
  GIT_REPOSITORY https://github.com/da380/GSHTrans.git
  GIT_TAG ${SL3DPP_GSHTRANS_REVISION}
  FIND_PACKAGE_ARGS)

# The whole repository is fetched (SOURCE_SUBDIR, no shallow clone of cpp/
# alone): cpp/CMakeLists.txt reads the version from ../pyproject.toml.
FetchContent_Declare(
  planetmodel
  GIT_REPOSITORY https://github.com/da380/planetmodel.git
  GIT_TAG ${SL3DPP_PLANETMODEL_REVISION}
  SOURCE_SUBDIR cpp
  FIND_PACKAGE_ARGS)

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
  src/BandSolvers.cpp
  src/Deck.cpp
  src/DegreeOperator.cpp
  src/EarthModel.cpp
  src/Evolution.cpp
  src/Evolution3D.cpp
  src/Fingerprint.cpp
  src/GridCoupling.cpp
  src/InitialSeaLevel.cpp
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
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(sl3dpp PUBLIC cxx_std_23)

# planetmodel is public: the io/ classes keep its types out of the library's
# own headers, but a code using sl3dpp reads and writes planetmodel files
# (decks, surface files, kernels) beside it, and with a static library the
# link is part of the interface either way.
target_link_libraries(sl3dpp PUBLIC
  GSHTrans::GSHTrans
  Interpolation::Interpolation
  GaussQuad::GaussQuad
  planetmodel::planetmodel
  OpenMP::OpenMP_CXX
  LAPACK::LAPACK
  BLAS::BLAS)

# Version() reports the project version, so that the two cannot drift apart.
target_compile_definitions(sl3dpp PRIVATE
  SL3DPP_VERSION="${PROJECT_VERSION}")

target_compile_options(sl3dpp PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)

# Clang 18 reports __cpp_concepts = 201907, which makes libstdc++ disable
# <expected> although the needed support is present; Clang 19 fixed the
# macro. The usual workaround until the floor rises (also in .clangd, for
# the editor tooling). A generator expression rather than a configure-time
# test, so that an installed library applies it for the consumer's compiler
# rather than for the one that built it.
set(_sl3dpp_old_clang
    "$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19>>")
target_compile_definitions(sl3dpp PUBLIC
  $<${_sl3dpp_old_clang}:__cpp_concepts=202002L>)
target_compile_options(sl3dpp PUBLIC
  $<${_sl3dpp_old_clang}:-Wno-builtin-macro-redefined>)

# ---------------------------------------------------------------------------
# Install and export: the library, its headers, and a CMake package, so that
# a code finds the same target with find_package(sl3dpp).
#
# Every library the target links has to be in an export set too. planetmodel
# releases before its install rules (1.3.1 and earlier) have none, so with one
# of those the library is left out of the install, saying so, and only the
# sl3d program is installed.
# ---------------------------------------------------------------------------
if(SL3DPP_INSTALL AND TARGET planetmodel AND NOT PLANETMODEL_INSTALL)
  message(WARNING
    "sl3dpp: this planetmodel (${SL3DPP_PLANETMODEL_REVISION}) has no install "
    "rules, so the library cannot be exported; only the sl3d program will be "
    "installed. Use a planetmodel with PLANETMODEL_INSTALL.")
elseif(SL3DPP_INSTALL)
  include(CMakePackageConfigHelpers)
  set(SL3DPP_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/sl3dpp)

  install(TARGETS sl3dpp EXPORT sl3dppTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/sl3dpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT sl3dppTargets
          FILE sl3dppTargets.cmake
          NAMESPACE sl3dpp::
          DESTINATION ${SL3DPP_CMAKE_INSTALL_DIR})

  # Before 1.0 a minor release may change the API, so only a matching minor
  # version satisfies a request.
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sl3dppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sl3dppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sl3dppConfig.cmake
    INSTALL_DESTINATION ${SL3DPP_CMAKE_INSTALL_DIR})
  install(FILES
          ${CMAKE_CURRENT_BINARY_DIR}/sl3dppConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/sl3dppConfigVersion.cmake
          DESTINATION ${SL3DPP_CMAKE_INSTALL_DIR})
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

# The API reference, built on request: `cmake --build build --target
# sl3dpp_docs` writes build/docs/html. Prefixed, like the sibling libraries'
# docs targets, so that it cannot collide with a consumer's own.
if(SL3DPP_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  set(SL3DPP_DOCS_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/docs)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(sl3dpp_docs
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SL3DPP_DOCS_OUTPUT}
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building the sl3dpp API reference in ${SL3DPP_DOCS_OUTPUT}/html"
    VERBATIM)
endif()
