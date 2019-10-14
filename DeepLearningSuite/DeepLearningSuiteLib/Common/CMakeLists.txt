SET ( Common_SOURCE_FILES
  Matrix.h
  EvalMatrix
  Sample
)

ADD_LIBRARY(DeepLearningSuite_Common OBJECT ${Common_SOURCE_FILES})

TARGET_INCLUDE_DIRECTORIES( DeepLearningSuite_Common PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${DeepLearningSuite_INCLUDE_DIR}
  ${QT_INCLUDE_DIRS}
)
