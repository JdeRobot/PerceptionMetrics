include_directories(
        ${JdeRobot_INCLUDE_DIRS}
        ${config_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${config_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}
        ${QT_INCLUDE_DIRS}
        ${DetectionStudio_INCLUDE_DIR}
        ${depthLib_INCLUDE_DIRS}
)

add_executable (SampleGenerationApp generator.cpp)

TARGET_LINK_LIBRARIES(SampleGenerationApp
        DetectionStudio
        ${EXTRA_LIBS}
        ${config_LIBRARIES}
        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${QT_LIBRARIES}
        ${Ice_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ssl
        crypto
        )
