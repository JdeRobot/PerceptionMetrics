IF (DARKNET_ACTIVE)
    SET(EXTRA_LIBS ${EXTRA_LIBS}
         ${DARKNET_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY})

    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()


include_directories(
        ${JdeRobot_INCLUDE_DIRS}
        ${config_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${config_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}
        ${QT_INCLUDE_DIRS}
        ${DeepLearningSuite_INCLUDE_DIR}
        ${depthLib_INCLUDE_DIRS}
)

add_executable (SampleGenerationApp generator.cpp)

TARGET_LINK_LIBRARIES(SampleGenerationApp
        DeepLearningSuite
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
