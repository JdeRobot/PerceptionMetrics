SET(Detectors_SOURCE_FILES
  Detector.h
)

ADD_LIBRARY(DeepLearningSuite_Detectors OBJECT ${Detectors_SOURCE_FILES})

TARGET_INCLUDE_DIRECTORIES ( DeepLearningSuite_Detectors PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
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
