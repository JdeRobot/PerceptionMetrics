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

)
