set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


IF (DARKNET_ACTIVE)
    SET(EXTRA_LIBS ${EXTRA_LIBS} ${DARKNET_LIBRARIES} ${CUDA_LIBRARIES} -L/usr/local/cuda/lib64 cudart cublas curand)
    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()


set (SOURCES main.cpp
        mainwindow
        gui/ListViewConfig
        SamplerGeneratorHandler/Viewer
        gui/TabHandler.cpp gui/TabHandler.h
        SamplerGeneratorHandler/Converter
        SamplerGeneratorHandler/SamplerGenerationHandler
        gui/Utils
        SamplerGeneratorHandler/Detector
        SamplerGeneratorHandler/Evaluator
        SamplerGeneratorHandler/Deployer
        )


include_directories(
        ${JdeRobot_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${QT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}
        ${DeepLearningSuite_INCLUDE_DIR}
)



add_executable(DatasetEvaluationApp ${SOURCES})


TARGET_LINK_LIBRARIES(DatasetEvaluationApp
        DeepLearningSuite
        ${EXTRA_LIBS}
        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${QT_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ssl
        crypto
        )
