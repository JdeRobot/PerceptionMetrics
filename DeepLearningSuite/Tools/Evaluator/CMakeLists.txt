
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${config_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}
        ${QT_INCLUDE_DIRS}
        ${DeepLearningSuite_INCLUDE_DIR}

)

add_executable (evaluator evaluator.cpp)

TARGET_LINK_LIBRARIES(evaluator
        DeepLearningSuite
        ${EXTRA_LIBS}
        ${OpenCV_LIBRARIES}
        ${QT_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ssl
        crypto
        )
