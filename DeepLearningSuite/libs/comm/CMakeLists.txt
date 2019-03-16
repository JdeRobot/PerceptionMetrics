cmake_minimum_required(VERSION 2.8)
if(roscpp_FOUND)
    MESSAGE("ROSCPP found")
    SET(JDEROBOTCOM_ROS ON)
ELSE()
    SET(JDEROBOTCOM_ROS OFF)
endif()


project(comm)


IF (JDEROBOTCOM_ROS)
    catkin_package(
    CATKIN_DEPENDS roscpp std_msgs message_runtime
    )
ENDIF()

set(binname ${PROJECT_NAME})

### Project config
include_directories(
    include
    ${jderobottypes_INCLUDE_DIRS}
    ${Ice_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${DeepLearningSuite_LIBS_DIR}
    ${ros_INCLUDE_DIRS}
    ${INTERFACES_CPP_DIR}
    ${config_INCLUDE_DIRS}
    ${utils_INCLUDE_DIRS}
)


set(HEADERS
    include/comm/communicator.hpp
    include/comm/tools.hpp
    include/comm/cameraClient.hpp
    include/comm/interfaces/cameraClient.hpp
)

set(SOURCES
    src/communicator.cpp
    src/tools.cpp
    src/cameraClient.cpp
)


IF (ZeroCIce_FOUND)
    set(HEADERS
        ${HEADERS}
        include/comm/ice/cameraIceClient.hpp
    )
    set(SOURCES
        ${SOURCES}
        src/ice/cameraIceClient.cpp
    )
ENDIF()

IF(JDEROBOTCOM_ROS)
    set(HEADERS
        ${HEADERS}
        include/comm/ros/translators.hpp
        include/comm/ros/listenerCamera.hpp
    )

    set(SOURCES
        ${SOURCES}
        src/ros/translators.cpp
        src/ros/listenerCamera.cpp
    )

ENDIF()

## Adding shared library for common usage
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

#add_dependencies(${PROJECT_NAME} ${config_LIBRARIES})

target_link_libraries(${PROJECT_NAME}
    ${ZeroCIce_LIBRARIES}
    ${Boost_LIBRARIES}
    ${ros_LIBRARIES}
    ${GLOG_LIBRARIES}
    colorspacesmm
    config
    DetectionSuiteUtils
    )


## Export library variables (like find_package)
set(${PROJECT_NAME}_FOUND 1 CACHE BOOL "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_LIBRARY_DIRS "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_LIBRARIES "${PROJECT_NAME}" CACHE STRINGS "Find(${PROJECT_NAME})")
