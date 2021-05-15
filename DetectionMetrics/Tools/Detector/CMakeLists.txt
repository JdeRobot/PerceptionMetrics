
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${config_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}
        ${QT_INCLUDE_DIRS}
        ${DetectionMetrics_INCLUDE_DIR}

)

add_executable (detector detector.cpp)

TARGET_LINK_LIBRARIES(detector
        DetectionMetrics
        ${EXTRA_LIBS}
        ${QT_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ssl
        crypto
        )
