SET(FrameworkEvaluator_SOURCE_FILES
  ClassStatistics
  FrameworkInferencer
  GlobalStats
  DetectionsEvaluator
  MassInferencer
  StatsWriter
  GenericInferencer
  TensorFlowInferencer
  KerasInferencer
  Labelling
  DarknetInferencer
)

IF(ENABLE_DNN_CAFFE)
    SET( FrameworkEvaluator_SOURCE_FILES ${FrameworkEvaluator_SOURCE_FILES} CaffeInferencer )
ENDIF()

ADD_LIBRARY(DeepLearningSuite_FrameworkEvaluator OBJECT ${FrameworkEvaluator_SOURCE_FILES})


message("OPENCV DIRS:")
message(${OpenCV_INCLUDE_DIRS})
set(OpenCV_INCLUDE_DIRS /usr/local/include/opencv2)
message("OPENCV DIRS:")
message(${OpenCV_INCLUDE_DIRS})

TARGET_INCLUDE_DIRECTORIES ( DeepLearningSuite_FrameworkEvaluator PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${DeepLearningSuite_INCLUDE_DIR}
  ${QT_INCLUDE_DIRS}
)
