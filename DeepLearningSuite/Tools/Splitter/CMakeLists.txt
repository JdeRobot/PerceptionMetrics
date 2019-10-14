
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${config_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}
        ${QT_INCLUDE_DIRS}
        ${DeepLearningSuite_INCLUDE_DIR}

)

add_executable (splitter splitter.cpp)

TARGET_LINK_LIBRARIES(splitter
        DeepLearningSuite
        ${EXTRA_LIBS}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${QT_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ssl
        crypto
        )
