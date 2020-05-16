SET ( Common_SOURCE_FILES
  Matrix.h
  EvalMatrix
  Sample
)

ADD_LIBRARY(DetectionStudio_Common OBJECT ${Common_SOURCE_FILES})

TARGET_INCLUDE_DIRECTORIES( DetectionStudio_Common PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${DetectionStudio_INCLUDE_DIR}
  ${QT_INCLUDE_DIRS}
)
