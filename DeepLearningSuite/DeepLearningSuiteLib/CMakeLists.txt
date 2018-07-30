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
    ${PYTHON_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${config_INCLUDE_DIRS}
    ${depthLib_INCLUDE_DIRS}
    ${comm_INCLUDE_DIRS}
    ${ros_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${utils_INCLUDE_DIRS}
    ${INTERFACES_CPP_DIR}
    ${jderobottypes_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}
    ${QT_INCLUDE_DIRS}
    ${DeepLearningSuite_INCLUDE_DIR}
)

SET( SOURCE_FILES
        GenerationUtils/DepthForegroundSegmentator
        GenerationUtils/BoundingValidator
        GenerationUtils/BoundingRectGuiMover
        GenerationUtils/DetectionsValidator
        Regions/RectRegions
        Regions/RleRegions
        Regions/ContourRegions
        Regions/Regions.h
        DatasetConverters/DatasetConverter
        DatasetConverters/readers/DatasetReader
        Common/Matrix.h
        Common/EvalMatrix
        Common/Sample
        DatasetConverters/readers/OwnDatasetReader
        DatasetConverters/writers/YoloDatasetWriter
        DatasetConverters/readers/YoloDatasetReader
        DatasetConverters/writers/COCODatasetWriter
        DatasetConverters/readers/COCODatasetReader
        DatasetConverters/readers/PascalVOCDatasetReader
        DatasetConverters/writers/PascalVOCDatasetWriter
        DatasetConverters/readers/ImageNetDatasetReader
        DatasetConverters/ClassTypeGeneric
        Regions/Region
        Regions/RectRegion
        Regions/RleRegion
        Regions/ContourRegion
        Regions/maskApi
        Detectors/Detector.h
        DatasetConverters/readers/SpinelloDatasetReader
        FrameworkEvaluator/FrameworkInferencer
        Utils/StringHandler.h
        Utils/Normalizations
        FrameworkEvaluator/ClassStatistics
        FrameworkEvaluator/DetectionsEvaluator
        FrameworkEvaluator/MassInferencer
        FrameworkEvaluator/StatsWriter
        DatasetConverters/ClassTypeOwn
        DatasetConverters/ClassTypeMapper
        DatasetConverters/ClassType
        DatasetConverters/Tree
        Utils/Configuration
        Utils/Key
        Utils/SampleGenerationApp
        DatasetConverters/readers/GenericDatasetReader
        FrameworkEvaluator/GenericInferencer
        DatasetConverters/writers/DatasetWriter
        DatasetConverters/writers/OwnDatasetWriter
        DatasetConverters/writers/GenericDatasetWriter
        Utils/StatsUtils
        DatasetConverters/liveReaders/GenericLiveReader
        DatasetConverters/liveReaders/JderobotReader
        DatasetConverters/liveReaders/RecorderReader
        DatasetConverters/liveReaders/CameraReader
        DatasetConverters/liveReaders/VideoReader
        FrameworkEvaluator/GlobalStats
        DatasetConverters/readers/PrincetonDatasetReader
        Utils/JsonHelper.h
        Utils/DepthUtils
        FrameworkEvaluator/TensorFlowInferencer
        FrameworkEvaluator/KerasInferencer
        DatasetConverters/liveReaders/VideoReader
        Utils/PathHelper
        )


IF(DARKNET_ACTIVE)
    SET( SOURCE_FILES ${SOURCE_FILES} FrameworkEvaluator/DarknetInferencer )
    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()

IF(ENABLE_DNN_CAFFE)
    SET( SOURCE_FILES ${SOURCE_FILES} FrameworkEvaluator/CaffeInferencer )
ENDIF()

#message("WARNING: DeepLearningSuite_INCLUDE_DIR ${DeepLearningSuite_INCLUDE_DIR}")


add_library (DeepLearningSuite SHARED ${SOURCE_FILES})

message("Python: ${PYTHON_LIBRARIES}")

TARGET_LINK_LIBRARIES(DeepLearningSuite
                    ${EXTRA_LIBS}
                    ${depthLib_LIBRARIES}
                    ${JderobotInterfaces_LIBRARIES}
                    ${comm_LIBRARIES}
                    ${config_LIBRARIES}
                    ${PYTHON_LIBRARIES}
                    ${GLOG_LIBRARIES}
                    )
