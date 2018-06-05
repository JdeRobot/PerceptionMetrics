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
        Common/Sample
        DatasetConverters/readers/OwnDatasetReader
        DatasetConverters/writers/YoloDatasetWriter
        DatasetConverters/readers/YoloDatasetReader
        DatasetConverters/writers/COCODatasetWriter
        DatasetConverters/readers/COCODatasetReader
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
        DatasetConverters/ClassTypeOwn
        DatasetConverters/ClassType
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
        DatasetConverters/liveReaders/VideoReader
        FrameworkEvaluator/DarknetInferencer
        FrameworkEvaluator/TensorFlowInferencer
        FrameworkEvaluator/KerasInferencer
        DatasetConverters/liveReaders/VideoReader
        Utils/PathHelper
        )


IF(DARKNET_ACTIVE)
    SET( SOURCE_FILES ${SOURCE_FILES} FrameworkEvaluator/GlobalStats.cpp FrameworkEvaluator/GlobalStats.h DatasetConverters/readers/PrincetonDatasetReader.cpp DatasetConverters/readers/PrincetonDatasetReader.h Utils/JsonHelper.h Utils/DepthUtils.cpp Utils/DepthUtils.h)
    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()


include_directories(
    ${JdeRobot_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
    ${DeepLearningSuite_INCLUDE_DIR}
)

message("WARNING: DeepLearningSuite_INCLUDE_DIR ${DeepLearningSuite_INCLUDE_DIR}")


add_library (DeepLearningSuite  ${SOURCE_FILES})
