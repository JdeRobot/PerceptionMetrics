
add_subdirectory(Common)
add_subdirectory(DatasetConverters)
add_subdirectory(Detectors)
add_subdirectory(FrameworkEvaluator)
add_subdirectory(GenerationUtils)
add_subdirectory(Regions)
add_subdirectory(Utils)

add_library(DetectionMetrics SHARED
              $<TARGET_OBJECTS:DetectionMetrics_Common>
              $<TARGET_OBJECTS:DetectionMetrics_DatasetConverters>
              $<TARGET_OBJECTS:DetectionMetrics_Detectors>
              $<TARGET_OBJECTS:DetectionMetrics_FrameworkEvaluator>
              $<TARGET_OBJECTS:DetectionMetrics_GenerationUtils>
              $<TARGET_OBJECTS:DetectionMetrics_Regions>
              $<TARGET_OBJECTS:DetectionMetrics_Utils>
            )


TARGET_LINK_LIBRARIES(DetectionMetrics
                    ${OpenCV_LIBRARIES}
                    ${EXTRA_LIBS}
                    ${depthLib_LIBRARIES}
                    ${JderobotInterfaces_LIBRARIES}
                    ${comm_LIBRARIES}
                    ${config_LIBRARIES}
                    ${PYTHON_LIBRARIES}
                    ${GLOG_LIBRARIES}
                    )
