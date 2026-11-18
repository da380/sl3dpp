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
sl3dpp_add_example(ExampleSeismic)

set(SL3DPP_EXAMPLE_DATA ${CMAKE_CURRENT_BINARY_DIR}/data)
file(MAKE_DIRECTORY ${SL3DPP_EXAMPLE_DATA})

# The inputs, into data/: the PREM-like deck and a planetmodel model file of
# a 3D viscosity on the same body (test fixtures), and the examples' own
# (examples/data: the decks in planetmodel's GIA format).
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
# The generators replace an input only where its content changed
# (KeepUnchanged.hpp), so that a rebuild leaves its bytes, and the hash a
# run takes of them, as they were; its time is set all the same, the build
# then seeing it made.
add_custom_command(
  OUTPUT ${SL3DPP_EXAMPLE_DATA}/synthetic_earth.nc
  COMMAND MakeSyntheticEarth data/synthetic_earth.nc
  COMMAND ${CMAKE_COMMAND} -E touch_nocreate data/synthetic_earth.nc
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
  COMMAND ${CMAKE_COMMAND} -E touch_nocreate data/synthetic_moduli.nc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS MakeSyntheticModuli ${SL3DPP_EXAMPLE_DATA}/prem8.elastic.deck
  COMMENT "Writing the examples' synthetic 3D elastic Earth")
add_custom_target(sl3dpp_synthetic_moduli ALL
  DEPENDS ${SL3DPP_EXAMPLE_DATA}/synthetic_moduli.nc)

# The synthetic perturbations of the viscosity (planetmodel model files of
# a change of ln viscosity on the PREM-like deck's body: a weak zone of the
# upper mantle, radial, and a weak patch of it, lateral), likewise written
# into data/ by the build, for the sl3d tangent configs.
sl3dpp_add_example(MakeSyntheticPerturbation)
add_custom_command(
  OUTPUT ${SL3DPP_EXAMPLE_DATA}/synthetic_weak_zone.nc
         ${SL3DPP_EXAMPLE_DATA}/synthetic_weak_patch.nc
  COMMAND MakeSyntheticPerturbation data
  COMMAND ${CMAKE_COMMAND} -E touch_nocreate data/synthetic_weak_zone.nc
          data/synthetic_weak_patch.nc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS MakeSyntheticPerturbation ${SL3DPP_EXAMPLE_DATA}/prem8.elastic.deck
  COMMENT "Writing the examples' synthetic perturbations of the viscosity")
add_custom_target(sl3dpp_synthetic_perturbation ALL
  DEPENDS ${SL3DPP_EXAMPLE_DATA}/synthetic_weak_zone.nc
          ${SL3DPP_EXAMPLE_DATA}/synthetic_weak_patch.nc)

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
  "ExampleStatic3D static_3d --lmax 8"
  "ExampleSeismic seismic --lmax 32")
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
    COMMAND ${SL3DPP_POETRY} run python plot_seismic.py
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

# The sl3d configs, copied into build/examples/configs to be edited and
# experimented with there. A config is copied when its copy is
# missing, and again when its source has changed and its copy has not been
# edited: the source as last copied is kept under CMakeFiles/sl3dpp_configs,
# and a copy that differs from it is an edited one, kept, the configure step
# saying so. A copy made before that record was kept is set aside there
# (replaced/) and copied again. Deleting one (or the directory) and
# reconfiguring restores the original; a changed source makes the build
# reconfigure. Their relative paths resolve against that directory, so they
# reach the inputs as ../data/ and write into out/<config>/.
file(GLOB SL3DPP_EXAMPLE_CONFIGS LIST_DIRECTORIES false CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/configs/*)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             ${SL3DPP_EXAMPLE_CONFIGS})
set(copies ${CMAKE_CURRENT_BINARY_DIR}/configs)
set(copied ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/sl3dpp_configs)
file(MAKE_DIRECTORY ${copies} ${copied})
foreach(config_file IN LISTS SL3DPP_EXAMPLE_CONFIGS)
  get_filename_component(config_name ${config_file} NAME)
  set(copy ${copies}/${config_name})
  set(record ${copied}/${config_name})
  if(NOT EXISTS ${copy})
    file(COPY ${config_file} DESTINATION ${copies})
    file(COPY ${config_file} DESTINATION ${copied})
    continue()
  endif()
  file(SHA256 ${config_file} source_hash)
  file(SHA256 ${copy} copy_hash)
  set(record_hash "")
  if(EXISTS ${record})
    file(SHA256 ${record} record_hash)
  endif()
  if(copy_hash STREQUAL source_hash)
    if(NOT record_hash STREQUAL source_hash)
      file(COPY ${config_file} DESTINATION ${copied})
    endif()
  elseif(record_hash STREQUAL "")
    # copied before the record was kept: set aside, and copied again
    file(MAKE_DIRECTORY ${copied}/replaced)
    file(RENAME ${copy} ${copied}/replaced/${config_name})
    file(COPY ${config_file} DESTINATION ${copies})
    file(COPY ${config_file} DESTINATION ${copied})
    message(STATUS "examples/configs/${config_name}: copied again from its "
                   "source; the copy it replaces is in ${copied}/replaced")
  elseif(copy_hash STREQUAL record_hash)
    # unedited, its source changed: copied again
    file(COPY ${config_file} DESTINATION ${copies})
    file(COPY ${config_file} DESTINATION ${copied})
  elseif(NOT source_hash STREQUAL record_hash)
    message(STATUS "examples/configs/${config_name}: its source has changed "
                   "and this copy was edited, so it is kept; delete it and "
                   "reconfigure for the source's")
  endif()
endforeach()
