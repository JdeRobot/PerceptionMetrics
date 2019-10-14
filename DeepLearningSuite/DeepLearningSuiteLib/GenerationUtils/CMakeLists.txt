SET(GenerationUtils_SOURCE_FILES
  DepthForegroundSegmentator
  BoundingValidator
  BoundingRectGuiMover
  DetectionsValidator
)

ADD_LIBRARY(DeepLearningSuite_GenerationUtils OBJECT ${GenerationUtils_SOURCE_FILES})



TARGET_INCLUDE_DIRECTORIES ( DeepLearningSuite_GenerationUtils PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${depthLib_INCLUDE_DIRS}
  ${utils_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${DeepLearningSuite_INCLUDE_DIR}
  ${QT_INCLUDE_DIRS}
)
