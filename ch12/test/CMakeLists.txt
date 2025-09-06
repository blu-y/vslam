set(TEST_SOURCES testTriangulation)

foreach (test_src ${TEST_SOURCES})
    add_executable(${test_src} ${test_src}.cpp)
    target_link_libraries(${test_src} ${THIRD_PARTY_LIBS} mySLAM)
    add_test(${test_src} ${test_src})
endforeach (test_src)