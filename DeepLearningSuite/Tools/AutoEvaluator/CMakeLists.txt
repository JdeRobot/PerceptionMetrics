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
        ${OpenCV_INCLUDE_DIRS}
        ${config_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}
        ${QT_INCLUDE_DIRS}
        ${DeepLearningSuite_INCLUDE_DIR}

)

add_executable (autoEvaluator autoEvaluator.cpp)

TARGET_LINK_LIBRARIES(autoEvaluator
        DeepLearningSuite
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
