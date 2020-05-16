 cmake_minimum_required(VERSION 2.8)

project(samplerGenerator C CXX)


#check compiller

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

message ("-- version: ${CMAKE_CXX_COMPILER_VERSION}")
if(COMPILER_SUPPORTS_CXX11)
  		set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
      MESSAGE("-- C++11 support enabled")
else()
	    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

IF (NOT CMAKE_BUILD_TYPE)
	MESSAGE("-- SETTING BUILD TYPE TO RELEASE")
	SET(CMAKE_BUILD_TYPE RELEASE)
ENDIF()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -fPIC")
set(CMAKE_MACOSX_RPATH 1)

#automated opencv
include(FindPkgConfig)

## FIND_PACKAGE(JdeRobot REQUIRED)

 include(Deps/glog/CMakeLists.txt)
 include(Deps/ice/CMakeLists.txt)
 include(Deps/numpy/CMakeLists.txt)
 include(Deps/yaml-cpp/CMakeLists.txt)
 include(Deps/ros/CMakeLists.txt)
 include(Deps/opencv/CMakeLists.txt)
 include(Deps/qt/CMakeLists.txt)

FIND_PACKAGE(Boost REQUIRED program_options filesystem)

set(DetectionStudio_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/DetectionStudioLib)
SET( INTERFACES_CPP_DIR ${CMAKE_CURRENT_BINARY_DIR}/libs/interfaces/cpp/jderobot ${CMAKE_CURRENT_BINARY_DIR}/libs/interfaces/cpp)

if(ZeroCIce_FOUND)
  add_definitions(-DICE)
endif()
if(roscpp_FOUND)
 	add_definitions(-DJDERROS)
endif()
if(OpenCV_FOUND)
	SET(ENABLE_DNN_CAFFE ON)
  add_definitions(-DENABLE_DNN_CAFFE)
endif(OpenCV_FOUND)

 add_subdirectory(libs)
 if (QT_FOUND)
   add_subdirectory(DetectionStudioLib)
   add_subdirectory(DatasetEvaluationApp)
 endif(QT_FOUND)
 add_subdirectory(SampleGenerationApp)
 add_subdirectory(Tools)

if(roscpp_FOUND)
 add_subdirectory(DetectionStudioROS)
endif()
 file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ClassMappingHierarchy.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
 file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ClassMappingHierarchy.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Tools)

 IF (BUILD_TEST)
	add_subdirectory(test)
ENDIF()
