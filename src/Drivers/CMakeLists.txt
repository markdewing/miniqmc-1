if(QMC_BUILD_LEVEL GREATER 4)
# add apps XYZ.cpp, e.g., qmc_particles.cpp
#SET(ESTEST einspline_smp einspline_spo qmc_particles moveonsphere twobody ptclset)
SET(ESTEST check_wfc check_spo check_determinant)

FOREACH(p ${ESTEST})
  ADD_EXECUTABLE( ${p}  ${p}.cpp)
  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ENDFOREACH(p ${ESTEST})

ADD_LIBRARY(miniwfs ../QMCWaveFunctions/WaveFunction.cpp)

ADD_EXECUTABLE(miniqmc miniqmc.cpp)
TARGET_LINK_LIBRARIES(miniqmc miniwfs qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

ADD_EXECUTABLE(unit_test ../Particle/test/test_bconds.cpp)
TARGET_LINK_LIBRARIES(unit_test miniwfs qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ADD_TEST(NAME unit_test COMMAND unit_test)
SET_TESTS_PROPERTIES(unit_test PROPERTIES LABELS "unit")

endif()

#SET(boost_test exchange_walker)
#FOREACH(p ${boost_test})
#  ADD_EXECUTABLE( ${p}  ${p}.cpp)
#  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} boost ${MPI_LIBRARY})
#ENDFOREACH(p ${boost_test})

