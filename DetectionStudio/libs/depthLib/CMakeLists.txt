include_directories(${SLICE_DIR}) # Aquí se alojan las cabeceras de las interfaces ICE en C++
include_directories(${INTERFACES_CPP_DIR}) # Aquí se alojan las cabeceras de las interfaces ICE en C++
include_directories(${LIBS_DIR}) # Aquí se alojan las cabeceras de las interfaces ICE en C++

project(depthLib)

include_directories( ${OpenCV_INCLUDE_DIRS})

ADD_LIBRARY (depthLib STATIC DepthFilter.cpp  DepthFilter.h  DepthSampler.cpp  DepthSampler.h)
TARGET_LINK_LIBRARIES(depthLib ${Boost_LIBRARIES})

ADD_LIBRARY (depthLibshare SHARED DepthFilter.cpp  DepthFilter.h  DepthSampler.cpp  DepthSampler.h)
TARGET_LINK_LIBRARIES(depthLibshare ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${GLOG_LIBRARIES})


## Export library variables (like find_package)
set(${PROJECT_NAME}_FOUND 1 CACHE BOOL "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_LIBRARY_DIRS "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_LIBRARIES "${PROJECT_NAME}" CACHE STRINGS "Find(${PROJECT_NAME})")
