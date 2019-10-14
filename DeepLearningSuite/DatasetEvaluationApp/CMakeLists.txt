set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


IF (DARKNET_ACTIVE)
    SET(EXTRA_LIBS ${EXTRA_LIBS}
         ${DARKNET_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY})

    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()


set (SOURCES main.cpp
        mainwindow
        gui/Appcfg
        gui/appconfig
        gui/ListViewConfig
        gui/TabHandler.cpp gui/TabHandler.h
        SamplerGeneratorHandler/Converter
        SamplerGeneratorHandler/SamplerGenerationHandler
        gui/Utils
        SamplerGeneratorHandler/Detector
        SamplerGeneratorHandler/Evaluator
        SamplerGeneratorHandler/Deployer
        SamplerGeneratorHandler/Viewer
        # gui/pop_up
        )


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${config_INCLUDE_DIRS}
        ${comm_INCLUDE_DIRS}
        ${utils_INCLUDE_DIRS}
        ${ros_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${INTERFACES_CPP_DIR}
        ${jderobottypes_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${QT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}
        ${DeepLearningSuite_INCLUDE_DIR}
)


add_executable(DatasetEvaluationApp ${SOURCES})


TARGET_LINK_LIBRARIES(DatasetEvaluationApp
        DeepLearningSuite
        ${OpenCV_LIBRARIES}
        ${JderobotInterfaces_LIBRARIES}
        ${EXTRA_LIBS}
        ${Boost_LIBRARIES}
        ${QT_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ssl
        crypto
        )
