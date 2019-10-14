set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

cmake_minimum_required(VERSION 2.8.3)
project(detection_suite)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  OpenCV
  sensor_msgs
  cv_bridge
  image_transport
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  object.msg
  objects.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES detection_suite
 CATKIN_DEPENDS  message_runtime
#  DEPENDS system_lib
)


# include_directories(
# # include
#   ${catkin_INCLUDE_DIRS}
#   # ${OpenCV_INCLUDE_DIRS}
# )


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
        ${CMAKE_CURRENT_LIST_DIR}/include/detectionsuite
        ${DeepLearningSuite_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
)

add_executable(code src/code.cpp)
target_link_libraries(code
  DeepLearningSuite
  ${OpenCV_LIBRARIES}
  ${JderobotInterfaces_LIBRARIES}
  ${EXTRA_LIBS}
  ${Boost_LIBRARIES}
  ${QT_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ssl
  crypto
)

add_executable(detection_node src/detection_node.cpp)
target_link_libraries(detection_node
  DeepLearningSuite
  ${OpenCV_LIBRARIES}
  ${JderobotInterfaces_LIBRARIES}
  ${EXTRA_LIBS}
  ${Boost_LIBRARIES}
  ${QT_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ssl
  crypto
)

add_executable(test src/test.cpp include/detectionsuite/DeployerNode.hpp src/DeployerNode.cpp)
add_dependencies(test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(test
  DeepLearningSuite
  ${OpenCV_LIBRARIES}
  ${JderobotInterfaces_LIBRARIES}
  ${EXTRA_LIBS}
  ${Boost_LIBRARIES}
  ${QT_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${catkin_LIBRARIES}
  ssl
  crypto
)
