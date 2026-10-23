# The example drivers. Everything here runs FROM THE BUILD TREE: the
# deck, the plotting scripts and the poetry file are copied beside the
# executables at configure time, and all outputs (data files, figures,
# the poetry environment) land in build/examples — the source tree is
# never written to. See examples/README.md.

function(sl3dpp_add_example name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE sl3dpp::sl3dpp)
endfunction()

sl3dpp_add_example(ExampleFingerprint)
sl3dpp_add_example(ExampleEvolution)
sl3dpp_add_example(ExampleLateralViscosity)
sl3dpp_add_example(ExampleIceKernels)

configure_file(${PROJECT_SOURCE_DIR}/tests/data/prem8.elastic.deck
               prem8.elastic.deck COPYONLY)

# The present-day Earth state (ICE-7G ice + topography coefficients) is
# EXTRACTED FROM PYSLFP AS PART OF THE BUILD when a checkout is found —
# straight into the build tree, nothing written to the source. Point
# SL3DPP_PYSLFP_DIR elsewhere to override the default sibling checkout.
set(SL3DPP_PYSLFP_DIR "" CACHE PATH
    "pyslfp checkout used to extract the examples' ICE-7G Earth state")
if(NOT SL3DPP_PYSLFP_DIR AND EXISTS "$ENV{HOME}/dev/pyslfp")
  set(SL3DPP_PYSLFP_DIR "$ENV{HOME}/dev/pyslfp")
endif()
find_program(SL3DPP_POETRY poetry)
set(SL3DPP_EARTH_STATE ${CMAKE_CURRENT_BINARY_DIR}/earth_state_l256.txt)
if(SL3DPP_PYSLFP_DIR AND SL3DPP_POETRY)
  add_custom_command(
    OUTPUT ${SL3DPP_EARTH_STATE}
    COMMAND ${SL3DPP_POETRY} run python
            ${PROJECT_SOURCE_DIR}/scripts/make_example_earth_state.py
            --out ${SL3DPP_EARTH_STATE}
    WORKING_DIRECTORY ${SL3DPP_PYSLFP_DIR}
    DEPENDS ${PROJECT_SOURCE_DIR}/scripts/make_example_earth_state.py
    COMMENT "Extracting the present-day ICE-7G Earth state from pyslfp")
  add_custom_target(ExampleEarthState ALL DEPENDS ${SL3DPP_EARTH_STATE})
else()
  message(STATUS
      "sl3dpp examples: no pyslfp checkout + poetry found; the earth-state "
      "examples need earth_state_l256.txt beside them (set "
      "SL3DPP_PYSLFP_DIR to extract it during the build)")
endif()
configure_file(${PROJECT_SOURCE_DIR}/examples/README.md README.md COPYONLY)
file(GLOB SL3DPP_EXAMPLE_PLOTS ${CMAKE_CURRENT_SOURCE_DIR}/plots/*)
foreach(plot_file ${SL3DPP_EXAMPLE_PLOTS})
  get_filename_component(plot_name ${plot_file} NAME)
  configure_file(${plot_file} ${plot_name} COPYONLY)
endforeach()
