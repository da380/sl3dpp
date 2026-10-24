# A code consuming an installed sl3dpp, as the install rules promise: only
# find_package and the prefix, nothing of the source tree. CI builds it
# against a fresh `cmake --install` (.github/workflows/ci.yml, job package).
cmake_minimum_required(VERSION 3.24)
project(sl3dpp_package_test LANGUAGES CXX)

find_package(sl3dpp REQUIRED)

# No CMAKE_CXX_STANDARD: the C++23 requirement must come with the target.
add_executable(PackageTest main.cpp)
target_link_libraries(PackageTest PRIVATE sl3dpp::sl3dpp)

enable_testing()
add_test(NAME PackageTest
         COMMAND PackageTest ${CMAKE_CURRENT_SOURCE_DIR}/../data/prem8.elastic.deck)
set_tests_properties(PackageTest PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
