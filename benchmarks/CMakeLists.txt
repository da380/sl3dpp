add_executable(fortran_gate fortran_gate.cpp)
target_link_libraries(fortran_gate PRIVATE sl3dpp::sl3dpp)
target_compile_definitions(fortran_gate PRIVATE
  $<$<BOOL:${BLAS_FOUND}>:GSHTRANS_HAVE_BLAS>)

add_executable(step_timing step_timing.cpp)
target_link_libraries(step_timing PRIVATE sl3dpp::sl3dpp)

# The serial prototype of the radially substructured degree solve
# (docs/dev/radial-substructuring.md).
add_executable(substructure_prototype substructure_prototype.cpp)
target_link_libraries(substructure_prototype PRIVATE sl3dpp::sl3dpp)
# ... and the library's substructured solves on the real degree systems.
add_executable(substructure_timing substructure_timing.cpp)
target_link_libraries(substructure_timing PRIVATE sl3dpp::sl3dpp)
# ... and distributed over MPI ranks (mpiexec), with SL3DPP_WITH_MPI.
if(SL3DPP_WITH_MPI)
  add_executable(distributed_solves distributed_solves.cpp)
  target_link_libraries(distributed_solves PRIVATE sl3dpp::sl3dpp)
  # ... and the phases of the distributed 3D step, ranks x threads.
  add_executable(distributed_step distributed_step.cpp)
  target_link_libraries(distributed_step PRIVATE sl3dpp::sl3dpp)
endif()

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

add_executable(love_histories love_histories.cpp)
target_link_libraries(love_histories PRIVATE sl3dpp::sl3dpp)
target_compile_definitions(love_histories PRIVATE
  SL3DPP_GIT_COMMIT="${SL3DPP_GIT_COMMIT}")
