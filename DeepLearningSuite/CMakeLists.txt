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

OPTION(USE_GPU_DARKNET "Compile with GPU Support" OFF)						## Only Applicable for Darknet

## FIND_PACKAGE(JdeRobot REQUIRED)

 include(Deps/glog/CMakeLists.txt)
 include(Deps/ice/CMakeLists.txt)
 include(Deps/numpy/CMakeLists.txt)
 include(Deps/yaml-cpp/CMakeLists.txt)
 include(Deps/ros/CMakeLists.txt)
 include(Deps/opencv/CMakeLists.txt)
 include(Deps/qt/CMakeLists.txt)

FIND_PACKAGE(Boost REQUIRED program_options filesystem)


#darknet

FIND_PATH( DARKNET_INCLUDE_DIR NAMES darknet.h  PATHS ${DARKNET_PATH}/include )
# MESSAGE(${DARKNET_PATH_DIR})
# file(GLOB DARKNET_LIBRARIES ${DARKNET_PATH}/obj
#     "*.o"
# )

SET(DARKNET_NAMES darknetAPILib darknetExamplesLib darknetLib)
# SET(DARKNET_NAMES darknetLib)

# FOREACH(N ${DARKNET_NAMES})
  # MESSAGE(${N})
  # MESSAGE(${DARKNET_LIBRARIES_${N}})
	# FIND_LIBRARY( DARKNET_LIBRARIES_${N} NAMES ${N} PATHS ${DARKNET_PATH}/lib )
	# SET(DARKNET_LIBRARIES ${DARKNET_LIBRARIES} ${DARKNET_LIBRARIES_${N}})
# ENDFOREACH()
# MESSAGE(${DARKNET_LIBRARIES})
#UNSET(DARKNET_INCLUDE_DIR CACHE)
#UNSET(DARKNET_LIBRARIES CACHE)


# if (DARKNET_INCLUDE_DIR AND DARKNET_LIBRARIES)				## Sometimes it may be found without specifing DARNKNET_PATH
if (DARKNET_INCLUDE_DIR)
	if (USE_GPU_DARKNET)
		MESSAGE("Darknet must be build with GPU support")
		FIND_PACKAGE(CUDA REQUIRED)
	endif(USE_GPU_DARKNET)

	message("DARKNET FOUND")
	message("DARKNET_INCLUDE_DIR: ${DARKNET_INCLUDE_DIR}")
	message("DARKNET_LIBRARIES: ${DARKNET_LIBRARIES}")
	SET(DARKNET_ACTIVE ON)
	ADD_DEFINITIONS(-DDARKNET_ACTIVE)
else()
	message("DARKNET_INCLUDE_DIR: ${DARKNET_INCLUDE_DIR}")
	message("DARKNET_LIBRARIES: ${DARKNET_LIBRARIES}")
	if (NOT DARKNET_PATH)
		MESSAGE("To enable Darknet define DARKNET_PATH equal to Darknet Installation Path")
		MESSAGE(WARNING "Darknet Inferencing support is Disabled")
	else()
		MESSAGE("Couldn't find darknet in the DARKNET_PATH specified")
		MESSAGE(WARNING "Darknet Inferencing support is Disabled")
	endif()
endif()

 set(DeepLearningSuite_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/DeepLearningSuiteLib)
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
   add_subdirectory(DeepLearningSuiteLib)
   add_subdirectory(DatasetEvaluationApp)
 endif(QT_FOUND)
 add_subdirectory(SampleGenerationApp)
 add_subdirectory(Tools)

add_subdirectory(detection_suite)

 file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ClassMappingHierarchy.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
 file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ClassMappingHierarchy.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Tools)

 IF (BUILD_TEST)
	add_subdirectory(test)
ENDIF()
