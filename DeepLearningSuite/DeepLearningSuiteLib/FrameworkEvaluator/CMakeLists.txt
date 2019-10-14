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
  labeling
)

IF(DARKNET_ACTIVE)
    SET( FrameworkEvaluator_SOURCE_FILES ${FrameworkEvaluator_SOURCE_FILES} DarknetInferencer )
ENDIF()

IF(ENABLE_DNN_CAFFE)
    SET( FrameworkEvaluator_SOURCE_FILES ${FrameworkEvaluator_SOURCE_FILES} CaffeInferencer )
ENDIF()

ADD_LIBRARY(DeepLearningSuite_FrameworkEvaluator OBJECT ${FrameworkEvaluator_SOURCE_FILES})



TARGET_INCLUDE_DIRECTORIES ( DeepLearningSuite_FrameworkEvaluator PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${DeepLearningSuite_INCLUDE_DIR}
  ${QT_INCLUDE_DIRS}
)

IF(DARKNET_ACTIVE)
  TARGET_INCLUDE_DIRECTORIES( DeepLearningSuite_FrameworkEvaluator PUBLIC ${DARKNET_INCLUDE_DIR})
ENDIF()
