SET (DatasetConverters_SOURCE_FILES
  DatasetConverter
  readers/DatasetReader
  readers/OwnDatasetReader
  writers/YoloDatasetWriter
  readers/YoloDatasetReader
  writers/COCODatasetWriter
  readers/COCODatasetReader
  readers/PascalVOCDatasetReader
  readers/PrincetonDatasetReader
  writers/PascalVOCDatasetWriter
  readers/ImageNetDatasetReader
  readers/SpinelloDatasetReader
  readers/OpenImagesDatasetReader
  writers/OpenImagesDatasetWriter
  ClassTypeGeneric
  ClassTypeOwn
  ClassTypeMapper
  ClassType
  Tree
  readers/GenericDatasetReader
  writers/DatasetWriter
  writers/OwnDatasetWriter
  writers/GenericDatasetWriter
  liveReaders/GenericLiveReader
  liveReaders/JderobotReader
  liveReaders/RecorderReader
  liveReaders/CameraReader
  liveReaders/VideoReader
  liveReaders/VideoReader
)

ADD_LIBRARY(DetectionStudio_DatasetConverters OBJECT ${DatasetConverters_SOURCE_FILES})

TARGET_INCLUDE_DIRECTORIES( DetectionStudio_DatasetConverters PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${config_INCLUDE_DIRS}
  ${comm_INCLUDE_DIRS}
  ${ros_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${utils_INCLUDE_DIRS}
  ${INTERFACES_CPP_DIR}
  ${jderobottypes_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${DetectionStudio_INCLUDE_DIR}
  ${QT_INCLUDE_DIRS}
)
