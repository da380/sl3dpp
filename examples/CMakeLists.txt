# The examples. Everything here runs FROM THE BUILD TREE: the inputs, the
# sl3d configs, the Python scripts and their poetry file are copied beside
# the executables at configure time, the synthetic Earth is written there by
# the build, and every output (data files, figures, the poetry environment,
# a downloaded ICE-7G file) lands in build/examples — the source tree is
# never written to. See examples/README.md.

# Held to the warnings the library is, as code meant to be copied.
function(sl3dpp_add_example name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE sl3dpp::sl3dpp)
  target_compile_options(${name} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)
endfunction()

sl3dpp_add_example(ExampleLoveNumbers)
sl3dpp_add_example(ExampleFingerprint)
sl3dpp_add_example(ExampleEvolution)
sl3dpp_add_example(ExampleLateralViscosity)
sl3dpp_add_example(ExampleModelViscosity)
sl3dpp_add_example(ExampleIceKernels)
sl3dpp_add_example(ExampleViscosityKernels3D)

# The synthetic Earth (an ice history and topography over a toy glacial
# cycle), written by the build so that the examples and the sl3d configs
# run with no data to fetch.
sl3dpp_add_example(MakeSyntheticEarth)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/synthetic_earth.nc
  COMMAND MakeSyntheticEarth synthetic_earth.nc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS MakeSyntheticEarth
  COMMENT "Writing the examples' synthetic Earth")
add_custom_target(sl3dpp_synthetic_earth ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/synthetic_earth.nc)

# The inputs: the PREM-like deck, and a planetmodel model file of a 3D
# viscosity on the same body (one of the test fixtures).
configure_file(${PROJECT_SOURCE_DIR}/tests/data/prem8.elastic.deck
               prem8.elastic.deck COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/tests/data/viscosity_log.nc
               viscosity_log.nc COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/README.md README.md COPYONLY)

# The Python side, copied flat beside the executables: the plotting scripts,
# the ICE-7G download, and the poetry file of their environment.
file(GLOB SL3DPP_EXAMPLE_PYTHON LIST_DIRECTORIES false CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/python/*)
foreach(python_file IN LISTS SL3DPP_EXAMPLE_PYTHON)
  get_filename_component(python_name ${python_file} NAME)
  configure_file(${python_file} ${python_name} COPYONLY)
endforeach()

# The sl3d configs, copied into build/examples/configs when they are not there
# yet, and never overwritten: they are there to be edited and experimented
# with. Deleting one (or the directory) and reconfiguring restores the
# original. Their relative paths resolve against that directory, so they
# reach the inputs above as ../.
file(GLOB SL3DPP_EXAMPLE_CONFIGS LIST_DIRECTORIES false CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/configs/*)
foreach(config_file IN LISTS SL3DPP_EXAMPLE_CONFIGS)
  get_filename_component(config_name ${config_file} NAME)
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/configs/${config_name})
    file(COPY ${config_file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/configs)
  endif()
endforeach()
