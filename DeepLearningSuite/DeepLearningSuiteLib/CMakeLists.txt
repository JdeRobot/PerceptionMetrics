IF (DARKNET_ACTIVE)
    SET(EXTRA_LIBS ${EXTRA_LIBS}
         ${DARKNET_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY})

    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()

add_subdirectory(Common)
add_subdirectory(DatasetConverters)
add_subdirectory(Detectors)
add_subdirectory(FrameworkEvaluator)
add_subdirectory(GenerationUtils)
add_subdirectory(Regions)
add_subdirectory(Utils)

add_library(DeepLearningSuite SHARED
              $<TARGET_OBJECTS:DeepLearningSuite_Common>
              $<TARGET_OBJECTS:DeepLearningSuite_DatasetConverters>
              $<TARGET_OBJECTS:DeepLearningSuite_Detectors>
              $<TARGET_OBJECTS:DeepLearningSuite_FrameworkEvaluator>
              $<TARGET_OBJECTS:DeepLearningSuite_GenerationUtils>
              $<TARGET_OBJECTS:DeepLearningSuite_Regions>
              $<TARGET_OBJECTS:DeepLearningSuite_Utils>

            )

TARGET_LINK_LIBRARIES(DeepLearningSuite
                    ${OpenCV_LIBRARIES}
                    ${EXTRA_LIBS}
                    ${depthLib_LIBRARIES}
                    ${JderobotInterfaces_LIBRARIES}
                    ${comm_LIBRARIES}
                    ${config_LIBRARIES}
                    ${PYTHON_LIBRARIES}
                    ${GLOG_LIBRARIES}
                    )
