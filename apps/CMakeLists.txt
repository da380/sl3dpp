# The sl3d program: the library's common capabilities from JSON configs.
# The command readers and runners live in a small static library so the
# tests can hold the configs' validation to its promises; the program is
# the thin dispatch of sl3d.cpp. Only the program is installed.
add_library(sl3d_app STATIC
  Input.cpp
  Common.cpp
  EvolveCommand.cpp
  KernelCommand.cpp
  LoveNumbersCommand.cpp
  FingerprintCommand.cpp)
target_include_directories(sl3d_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sl3d_app PUBLIC sl3dpp::sl3dpp planetmodel::planetmodel)

add_executable(sl3d sl3d.cpp)
target_link_libraries(sl3d PRIVATE sl3d_app)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target IN ITEMS sl3d_app sl3d)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endforeach()
endif()

include(GNUInstallDirs)
install(TARGETS sl3d RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
