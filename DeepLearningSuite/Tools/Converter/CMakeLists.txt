IF (DARKNET_ACTIVE)
    SET(EXTRA_LIBS ${EXTRA_LIBS} ${DARKNET_LIBRARIES} ${CUDA_LIBRARIES} -L/usr/local/cuda/lib64 cudart cublas curand)
    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}
        ${DeepLearningSuite_INCLUDE_DIR}

)

add_executable (converter  converter.cpp)

message(${OpenCV_LIBRARIES})

TARGET_LINK_LIBRARIES(converter
        DeepLearningSuite
        ${EXTRA_LIBS}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ssl
        crypto
        )
