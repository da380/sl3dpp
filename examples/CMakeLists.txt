# The examples. Everything here runs FROM THE BUILD TREE, from build/examples,
# which holds the executables, the Python scripts and their poetry file, and
#
#   data/                      every input: copied at configure time, written
#                              by the build (the synthetic Earths) or by
#                              get_ice7g.py and the make_*.py scripts
#   output/library/<example>/  each library example's outputs and its log
#   figures/library/           the library examples' figures
#   figures/sl3d/<config>/     the figures of the sl3d configs' outputs
#   configs/, configs/out/     the sl3d configs and their outputs
#
# The source tree is never written to. See examples/README.md.

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
sl3dpp_add_example(ExampleStatic3D)

set(SL3DPP_EXAMPLE_DATA ${CMAKE_CURRENT_BINARY_DIR}/data)
file(MAKE_DIRECTORY ${SL3DPP_EXAMPLE_DATA})

# The inputs, into data/: the PREM-like deck and a planetmodel model file of
# a 3D viscosity on the same body (test fixtures), and the examples' own
# (examples/data: the deck in planetmodel's GIA format).
configure_file(${PROJECT_SOURCE_DIR}/tests/data/prem8.elastic.deck
               ${SL3DPP_EXAMPLE_DATA}/prem8.elastic.deck COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/tests/data/viscosity_log.nc
               ${SL3DPP_EXAMPLE_DATA}/viscosity_log.nc COPYONLY)
file(GLOB SL3DPP_EXAMPLE_INPUTS LIST_DIRECTORIES false CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/data/*)
foreach(input_file IN LISTS SL3DPP_EXAMPLE_INPUTS)
  get_filename_component(input_name ${input_file} NAME)
  configure_file(${input_file} ${SL3DPP_EXAMPLE_DATA}/${input_name} COPYONLY)
endforeach()

# The synthetic Earth (an ice history and topography over a toy glacial
# cycle), written into data/ by the build so that the examples and the sl3d
# configs run with no data to fetch.
sl3dpp_add_example(MakeSyntheticEarth)
add_custom_command(
  OUTPUT ${SL3DPP_EXAMPLE_DATA}/synthetic_earth.nc
  COMMAND MakeSyntheticEarth data/synthetic_earth.nc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS MakeSyntheticEarth
  COMMENT "Writing the examples' synthetic Earth")
add_custom_target(sl3dpp_synthetic_earth ALL
  DEPENDS ${SL3DPP_EXAMPLE_DATA}/synthetic_earth.nc)

# The synthetic 3D elastic Earth (the PREM-like deck's moduli with a
# lateral perturbation of the upper mantle as a planetmodel model file),
# likewise written into data/ by the build, for the 3D static example and
# the response configs.
sl3dpp_add_example(MakeSyntheticModuli)
add_custom_command(
  OUTPUT ${SL3DPP_EXAMPLE_DATA}/synthetic_moduli.nc
  COMMAND MakeSyntheticModuli data/synthetic_moduli.nc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS MakeSyntheticModuli ${SL3DPP_EXAMPLE_DATA}/prem8.elastic.deck
  COMMENT "Writing the examples' synthetic 3D elastic Earth")
add_custom_target(sl3dpp_synthetic_moduli ALL
  DEPENDS ${SL3DPP_EXAMPLE_DATA}/synthetic_moduli.nc)

# The examples' outputs, generated in the build: every driver run at the
# small degree CI uses (seconds each) by the target sl3dpp_example_outputs —
# part of ALL with SL3DPP_RUN_EXAMPLES=ON — each writing into its own
# output/library/<example>/, with what it prints as <example>.log there;
# and their figures, into figures/library/, by sl3dpp_example_figures
# through the poetry environment of build/examples (poetry install there
# first), never part of ALL. The full-size runs are the drivers' own
# defaults (examples/README.md). Each entry: the driver, its <example>, its
# arguments.
option(SL3DPP_RUN_EXAMPLES "Run the examples at a small degree as part of the build" OFF)
set(SL3DPP_EXAMPLE_RUNS
  "ExampleLoveNumbers love_numbers --lmax 16"
  "ExampleFingerprint fingerprint --lmax 32"
  "ExampleEvolution evolution --lmax 16"
  "ExampleLateralViscosity lateral_viscosity --lmax 8"
  "ExampleModelViscosity model_viscosity --lmax 8"
  "ExampleIceKernels ice_kernels --lmax 16"
  "ExampleViscosityKernels3D viscosity_kernels_3d --lmax 8"
  "ExampleStatic3D static_3d --lmax 8")
set(SL3DPP_EXAMPLE_STAMPS)
set(stamps ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/sl3dpp_example_runs)
file(MAKE_DIRECTORY ${stamps})
foreach(entry IN LISTS SL3DPP_EXAMPLE_RUNS)
  separate_arguments(run UNIX_COMMAND "${entry}")
  list(POP_FRONT run driver example)
  set(output ${CMAKE_CURRENT_BINARY_DIR}/output/library/${example})
  set(stamp ${stamps}/${driver}.done)
  add_custom_command(
    OUTPUT ${stamp}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${output}
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=4
            $<TARGET_FILE:${driver}> ${run} > ${output}/${example}.log
    COMMAND ${CMAKE_COMMAND} -E touch ${stamp}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${driver} ${SL3DPP_EXAMPLE_DATA}/synthetic_earth.nc
            ${SL3DPP_EXAMPLE_DATA}/synthetic_moduli.nc
    COMMENT "Running ${driver} at a small degree (output/library/${example})"
    VERBATIM)
  list(APPEND SL3DPP_EXAMPLE_STAMPS ${stamp})
endforeach()
if(SL3DPP_RUN_EXAMPLES)
  add_custom_target(sl3dpp_example_outputs ALL DEPENDS ${SL3DPP_EXAMPLE_STAMPS})
else()
  add_custom_target(sl3dpp_example_outputs DEPENDS ${SL3DPP_EXAMPLE_STAMPS})
endif()
find_program(SL3DPP_POETRY poetry)
if(SL3DPP_POETRY)
  add_custom_target(sl3dpp_example_figures
    COMMAND ${SL3DPP_POETRY} run python plot_love_numbers.py
    COMMAND ${SL3DPP_POETRY} run python plot_fingerprint.py
    COMMAND ${SL3DPP_POETRY} run python plot_evolution.py
    COMMAND ${SL3DPP_POETRY} run python plot_lateral.py
    COMMAND ${SL3DPP_POETRY} run python plot_model_viscosity.py
    COMMAND ${SL3DPP_POETRY} run python plot_kernels.py
    COMMAND ${SL3DPP_POETRY} run python plot_kernels_3d.py
    COMMAND ${SL3DPP_POETRY} run python plot_static_3d.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS sl3dpp_example_outputs
    COMMENT "Drawing the examples' figures into figures/library (poetry)")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/README.md README.md COPYONLY)

# The Python side, copied flat beside the executables: the plotting scripts,
# the ICE-7G download, the make_*.py scripts that write inputs for the sl3d
# program, and the poetry file of their environment.
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
# reach the inputs as ../data/ and write into out/<config>/.
file(GLOB SL3DPP_EXAMPLE_CONFIGS LIST_DIRECTORIES false CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/configs/*)
foreach(config_file IN LISTS SL3DPP_EXAMPLE_CONFIGS)
  get_filename_component(config_name ${config_file} NAME)
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/configs/${config_name})
    file(COPY ${config_file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/configs)
  endif()
endforeach()
