
add_subdirectory(Common)
add_subdirectory(DatasetConverters)
add_subdirectory(Detectors)
add_subdirectory(FrameworkEvaluator)
add_subdirectory(GenerationUtils)
add_subdirectory(Regions)
add_subdirectory(Utils)

add_library(DetectionStudio SHARED
              $<TARGET_OBJECTS:DetectionStudio_Common>
              $<TARGET_OBJECTS:DetectionStudio_DatasetConverters>
              $<TARGET_OBJECTS:DetectionStudio_Detectors>
              $<TARGET_OBJECTS:DetectionStudio_FrameworkEvaluator>
              $<TARGET_OBJECTS:DetectionStudio_GenerationUtils>
              $<TARGET_OBJECTS:DetectionStudio_Regions>
              $<TARGET_OBJECTS:DetectionStudio_Utils>
            )


TARGET_LINK_LIBRARIES(DetectionStudio
                    ${OpenCV_LIBRARIES}
                    ${EXTRA_LIBS}
                    ${depthLib_LIBRARIES}
                    ${JderobotInterfaces_LIBRARIES}
                    ${comm_LIBRARIES}
                    ${config_LIBRARIES}
                    ${PYTHON_LIBRARIES}
                    ${GLOG_LIBRARIES}
                    )
