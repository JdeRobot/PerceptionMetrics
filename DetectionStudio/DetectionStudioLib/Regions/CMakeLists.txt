SET(Regions_SOURCE_FILES
  RectRegions
  RleRegions
  ContourRegions
  Regions.h
  Region
  RectRegion
  RleRegion
  ContourRegion
  maskApi
)

ADD_LIBRARY(DetectionStudio_Regions OBJECT ${Regions_SOURCE_FILES})



TARGET_INCLUDE_DIRECTORIES ( DetectionStudio_Regions PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${utils_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${QT_INCLUDE_DIRS}
  ${DetectionStudio_INCLUDE_DIR}

)
