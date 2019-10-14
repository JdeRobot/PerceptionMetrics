set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


# MESSAGE(${QT_INCLUDE_DIRS})

SET(Utils_SOURCE_FILES
  StringHandler.h
  Normalizations
  Configuration
  Key
  SampleGenerationApp
  StatsUtils
  JsonHelper.h
  DepthUtils
  PathHelper
  Playback
  pop_up
  setclass
  addclass
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

ADD_LIBRARY(DeepLearningSuite_Utils OBJECT ${Utils_SOURCE_FILES})


TARGET_INCLUDE_DIRECTORIES ( DeepLearningSuite_Utils PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${config_INCLUDE_DIRS}
  ${depthLib_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${utils_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${DeepLearningSuite_INCLUDE_DIR}
  ${QT_INCLUDE_DIRS}
)
