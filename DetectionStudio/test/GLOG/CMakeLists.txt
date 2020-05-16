
include_directories(${GLOG_INCLUDE_DIR})
add_executable(glogTest glogTest)

target_link_libraries(glogTest
        ${GLOG_LIBRARIES}
        )