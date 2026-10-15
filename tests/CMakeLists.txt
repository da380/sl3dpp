FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(sl3dpp_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE sl3dpp::sl3dpp GTest::gtest_main)
  target_compile_definitions(${name} PRIVATE
    SL3DPP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  gtest_discover_tests(${name})
endfunction()

sl3dpp_add_test(TestToolkit)
sl3dpp_add_test(TestDeck)
sl3dpp_add_test(TestEarthModel)
sl3dpp_add_test(TestRadialMesh)
sl3dpp_add_test(TestDegreeOperator)
sl3dpp_add_test(TestSeaLevel)
sl3dpp_add_test(TestViscoelastic)
sl3dpp_add_test(TestEvolution)
sl3dpp_add_test(TestFixtures)
