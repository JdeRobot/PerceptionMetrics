
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${config_INCLUDE_DIRS}
    ${depthLib_INCLUDE_DIRS}
    ${comm_INCLUDE_DIRS}
    ${roscpp_INCLUDE_DIRS}
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
        Regions/ContourRegions
        Regions/Regions.h
        DatasetConverters/DatasetConverter
        DatasetConverters/readers/DatasetReader
        Common/Matrix.h
        Common/EvalMatrix
        Common/Sample
        DatasetConverters/readers/OwnDatasetReader
	DatasetConverters/readers/PrincetonDatasetReader
        DatasetConverters/writers/YoloDatasetWriter
        DatasetConverters/readers/YoloDatasetReader
        DatasetConverters/writers/COCODatasetWriter
        DatasetConverters/readers/COCODatasetReader
        DatasetConverters/readers/PascalVOCDatasetReader
        DatasetConverters/writers/PascalVOCDatasetWriter
        DatasetConverters/readers/ImageNetDatasetReader
        DatasetConverters/readers/SamplesReader
        DatasetConverters/ClassTypeGeneric
        Regions/Region
        Regions/RectRegion
        Regions/ContourRegion
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
<<<<<<< HEAD
        FrameworkEvaluator/GlobalStats
        DatasetConverters/readers/PrincetonDatasetReader
        Utils/JsonHelper.h
        Utils/DepthUtils
=======
	FrameworkEvaluator/GlobalStats
>>>>>>> readme
        FrameworkEvaluator/TensorFlowInferencer
        FrameworkEvaluator/KerasInferencer
        FrameworkEvaluator/CaffeInferencer
        DatasetConverters/liveReaders/VideoReader
        Utils/PathHelper
	Utils/DepthUtils
        )


IF(DARKNET_ACTIVE)
<<<<<<< HEAD
    message("add darknet files")
    SET( SOURCE_FILES ${SOURCE_FILES} FrameworkEvaluator/DarknetInferencer )
=======
    SET( SOURCE_FILES ${SOURCE_FILES} FrameworkEvaluator/DarknetInferencer Utils/JsonHelper.h)
>>>>>>> readme
    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()

IF(ENABLE_DNN_CAFFE)
    message("added caffe files")
    SET( SOURCE_FILES ${SOURCE_FILES} FrameworkEvaluator/CaffeInferencer )
ENDIF()

message("WARNING: DeepLearningSuite_INCLUDE_DIR ${DeepLearningSuite_INCLUDE_DIR}")


add_library (DeepLearningSuite SHARED ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(DeepLearningSuite ${depthLib_LIBRARIES} ${JderobotInterfaces_LIBRARIES} ${comm_LIBRARIES} ${config_LIBRARIES})
