include_directories(${SLICE_DIR})
include_directories(${INTERFACES_CPP_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR})

project(utils)

if (ZeroCIce_FOUND)
SET(UTILS_SOURCES
        CameraUtils
        )
endif()

set(${PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/colorspaces" CACHE PATH "Find(${PROJECT_NAME})")

ADD_LIBRARY (colorspaces STATIC colorspaces/rgb2hsv.c colorspaces/rgb2yuv.c colorspaces/colorspaces.h colorspaces/colorspacesmm.h)

TARGET_LINK_LIBRARIES(colorspaces ${OpenCV_LIBRARIES})

ADD_LIBRARY (colorspacesshare SHARED colorspaces/rgb2hsv.c colorspaces/rgb2yuv.c colorspaces/colorspaces.h colorspaces/colorspacesmm.h)

TARGET_LINK_LIBRARIES(colorspacesshare  ${OpenCV_LIBRARIES})

ADD_LIBRARY (colorspacesmm SHARED colorspaces/imagecv.cpp colorspaces/imagecv.h colorspaces/uncopyable.h)

TARGET_LINK_LIBRARIES(colorspacesmm ${OpenCV_LIBRARIES} ${OpenCVGUI_LIBRARIES})
if (ZeroCIce_FOUND)
  ADD_LIBRARY (DetectionSuiteUtils ${UTILS_SOURCES})

  TARGET_LINK_LIBRARIES(DetectionSuiteUtils colorspaces colorspacesmm colorspacesshare)

else()
  ADD_LIBRARY (DetectionSuiteUtils INTERFACE)

  TARGET_LINK_LIBRARIES(DetectionSuiteUtils INTERFACE colorspaces colorspacesmm colorspacesshare)

endif()
