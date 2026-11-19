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
  # The library threads by default; each test runs on one thread, so
  # that ctest -jN does not oversubscribe the machine (a test that needs
  # threads sets its own count).
  gtest_discover_tests(${name} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endfunction()

sl3dpp_add_test(TestToolkit)
sl3dpp_add_test(TestBandSolvers)
sl3dpp_add_test(TestDeck)
sl3dpp_add_test(TestEarthModel)
sl3dpp_add_test(TestRadialMesh)
sl3dpp_add_test(TestDegreeOperator)
sl3dpp_add_test(TestTractionNumbers)
sl3dpp_add_test(TestSeaLevel)
sl3dpp_add_test(TestViscoelastic)
sl3dpp_add_test(TestEvolution)
sl3dpp_add_test(TestAdjoint)
sl3dpp_add_test(TestAdjoint3D)
sl3dpp_add_test(TestTangentLinear)
sl3dpp_add_test(TestSecondOrderAdjoint)
sl3dpp_add_test(TestTrajectoryCursor)
sl3dpp_add_test(TestStaticDerivatives)
sl3dpp_add_test(TestLateralKernels)
sl3dpp_add_test(TestUnevenSteps)
sl3dpp_add_test(TestSdirk2Derivatives)
sl3dpp_add_test(TestSdirk2Continuous)
sl3dpp_add_test(TestMigratingDerivatives)
sl3dpp_add_test(TestMigratingTide)
sl3dpp_add_test(TestTangentLinear3D)
sl3dpp_add_test(TestSecondOrderAdjoint3D)
sl3dpp_add_test(TestDiscreteAdjoint3D)
sl3dpp_add_test(TestMaterialDerivatives)
sl3dpp_add_test(TestElasticPerturbation3D)
sl3dpp_add_test(TestSpinMachinery)
sl3dpp_add_test(TestGridCoupling)
sl3dpp_add_test(TestViscoelastic3D)
sl3dpp_add_test(TestLateralModuli3D)
sl3dpp_add_test(TestEvolution3D)
sl3dpp_add_test(TestFixtures)
sl3dpp_add_test(TestMemoryPoints)
sl3dpp_add_test(TestEvolutionResume)
sl3dpp_add_test(TestMeasurements)
sl3dpp_add_test(TestPointSource)
sl3dpp_add_test(TestFault)
sl3dpp_add_test(TestWithoutGravity)
if(TARGET sl3d_app)
  sl3dpp_add_test(TestApps)
  target_link_libraries(TestApps PRIVATE sl3d_app)
  sl3dpp_add_test(TestAppsFaultFile)
  target_link_libraries(TestAppsFaultFile PRIVATE sl3d_app)
  sl3dpp_add_test(TestAppsDecks)
  target_link_libraries(TestAppsDecks PRIVATE sl3d_app)
  sl3dpp_add_test(TestAppsTangent)
  target_link_libraries(TestAppsTangent PRIVATE sl3d_app)
  sl3dpp_add_test(TestAppsSecondOrder)
  target_link_libraries(TestAppsSecondOrder PRIVATE sl3d_app)
endif()
# The program refuses the words that are not commands but name what one
# runs, the message naming the command to run instead.
if(TARGET sl3d)
  foreach(pair IN ITEMS sea-level:solve response:solve kernel1d:kernel)
    string(REPLACE ":" ";" pair "${pair}")
    list(GET pair 0 word)
    list(GET pair 1 command)
    add_test(NAME AppsProgram.RefusesTheWord.${word}
             COMMAND $<TARGET_FILE:sl3d> ${word} config.json)
    set_tests_properties(AppsProgram.RefusesTheWord.${word} PROPERTIES
      PASS_REGULAR_EXPRESSION
        "^sl3d: \"${word}\" is not a command: [^\n]*`sl3d ${command} <config.json>`")
  endforeach()
endif()
sl3dpp_add_test(TestFieldHistory)
sl3dpp_add_test(TestModelViscosity)
sl3dpp_add_test(TestModelModuli)
sl3dpp_add_test(TestModelField)
sl3dpp_add_test(TestKernelWriter)
sl3dpp_add_test(TestTrajectoryStore)
sl3dpp_add_test(TestUnitsParity)
sl3dpp_add_test(TestGaussMemory)
sl3dpp_add_test(TestGaussMemory3D)
sl3dpp_add_test(TestInitialSeaLevel)
sl3dpp_add_test(TestCalibratedKernels)
sl3dpp_add_test(TestSubstructure)
sl3dpp_add_test(TestNodeBlocks)
sl3dpp_add_test(TestRadialRanks)

# The distributed solves, under mpiexec on 2, 3 and 4 ranks (one thread
# each), when the library is built with SL3DPP_WITH_MPI.
if(SL3DPP_WITH_MPI)
  add_executable(TestDistributed TestDistributed.cpp)
  target_link_libraries(TestDistributed PRIVATE sl3dpp::sl3dpp GTest::gtest)
  target_compile_definitions(TestDistributed PRIVATE
    SL3DPP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  foreach(ranks 2 3 4)
    add_test(NAME Distributed.Ranks${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              ${MPIEXEC_PREFLAGS} $<TARGET_FILE:TestDistributed>
              ${MPIEXEC_POSTFLAGS})
    set_tests_properties(Distributed.Ranks${ranks} PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS ${ranks})
  endforeach()
  # ... and sl3d's 3D commands over 2 and 3 ranks.
  if(TARGET sl3d_app)
    add_executable(TestAppsDistributed TestAppsDistributed.cpp)
    target_link_libraries(TestAppsDistributed PRIVATE sl3d_app GTest::gtest)
    target_compile_definitions(TestAppsDistributed PRIVATE
      SL3DPP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    foreach(ranks 2 3)
      add_test(NAME AppsDistributed.Ranks${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:TestAppsDistributed>
                ${MPIEXEC_POSTFLAGS})
      set_tests_properties(AppsDistributed.Ranks${ranks} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS ${ranks})
    endforeach()
  endif()
endif()
