add_executable(fortran_gate fortran_gate.cpp)
target_link_libraries(fortran_gate PRIVATE sl3dpp::sl3dpp)
target_compile_definitions(fortran_gate PRIVATE
  $<$<BOOL:${BLAS_FOUND}>:GSHTRANS_HAVE_BLAS>)

add_executable(step_timing step_timing.cpp)
target_link_libraries(step_timing PRIVATE sl3dpp::sl3dpp)

# The E1 exchange writer records which commit computed it.
execute_process(COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE SL3DPP_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT SL3DPP_GIT_COMMIT)
  set(SL3DPP_GIT_COMMIT "unknown")
endif()
add_executable(traction_numbers traction_numbers.cpp)
target_link_libraries(traction_numbers PRIVATE sl3dpp::sl3dpp)
target_compile_definitions(traction_numbers PRIVATE
  SL3DPP_GIT_COMMIT="${SL3DPP_GIT_COMMIT}")

add_executable(love_numbers love_numbers.cpp)
target_link_libraries(love_numbers PRIVATE sl3dpp::sl3dpp)
target_compile_definitions(love_numbers PRIVATE
  SL3DPP_GIT_COMMIT="${SL3DPP_GIT_COMMIT}")
