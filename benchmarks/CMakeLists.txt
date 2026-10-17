add_executable(fortran_gate fortran_gate.cpp)
target_link_libraries(fortran_gate PRIVATE sl3dpp::sl3dpp)
target_compile_definitions(fortran_gate PRIVATE
  $<$<BOOL:${BLAS_FOUND}>:GSHTRANS_HAVE_BLAS>)

add_executable(step_timing step_timing.cpp)
target_link_libraries(step_timing PRIVATE sl3dpp::sl3dpp)
