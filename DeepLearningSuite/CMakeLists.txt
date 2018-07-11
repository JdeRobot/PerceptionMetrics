 cmake_minimum_required(VERSION 2.8)

project(samplerGenerator C CXX)


#check compiller
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	message ("-- version: ${CMAKE_CXX_COMPILER_VERSION}")
	if(COMPILER_SUPPORTS_CXX11)
    		set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
	elseif(COMPILER_SUPPORTS_CXX0X)
        #boost have problems with gcc < 4.7 and c++0x
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.7.0")
    		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        endif()
	else()
		    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
	endif()
endif()

IF (NOT CMAKE_BUILD_TYPE)
	MESSAGE("-- SETTING BUILD TYPE TO RELEASE")
	SET(CMAKE_BUILD_TYPE RELEASE)
ENDIF()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -fPIC")


#automated opencv
include(FindPkgConfig)

FIND_PACKAGE(CUDA REQUIRED)
## FIND_PACKAGE(JdeRobot REQUIRED)

 include(Deps/glog/CMakeLists.txt)
 include(Deps/ice/CMakeLists.txt)
 include(Deps/numpy/CMakeLists.txt)
 include(Deps/yaml-cpp/CMakeLists.txt)
 include(Deps/ros/CMakeLists.txt)
 include(Deps/opencv/CMakeLists.txt)
FIND_PACKAGE(Boost REQUIRED program_options filesystem)
FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt5Core REQUIRED)
FIND_PACKAGE(Qt5Gui REQUIRED)
FIND_PACKAGE(Qt5Svg REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)

 SET(QT_INCLUDE_DIRS
	 Qt5Widgets_INCLUDE_DIRS}
	 ${Qt5Core_INCLUDE_DIRS}
	 ${Qt5Gui_INCLUDE_DIRS}
	 ${Qt5Svg_INCLUDE_DIRS}
	 ${Qt5OpenGL_INCLUDE_DIRS}
 )

 set(QT_LIBRARIES
	 ${Qt5Widgets_LIBRARIES}
	 ${Qt5Core_LIBRARIES}
	 ${Qt5Gui_LIBRARIES}
	 ${Qt5Svg_LIBRARIES}
	 ${Qt5OpenGL_LIBRARIES}
)


#darknet

FIND_PATH( DARKNET_INCLUDE_DIR NAMES DarknetAPI/DarknetAPI.h  PATHS ${DARKNET_PATH}/include )
SET(DARKNET_NAMES darknetAPILib darknetExamplesLib darknetLib)
#SET(DARKNET_NAMES darknetLib)

 FOREACH(N ${DARKNET_NAMES})
	FIND_LIBRARY( DARKNET_LIBRARIES_${N} NAMES ${N} PATHS ${DARKNET_PATH}/lib )
	SET(DARKNET_LIBRARIES ${DARKNET_LIBRARIES} ${DARKNET_LIBRARIES_${N}})

ENDFOREACH()

#UNSET(DARKNET_INCLUDE_DIR CACHE)
#UNSET(DARKNET_LIBRARIES CACHE)


if (DARKNET_INCLUDE_DIR AND DARKNET_LIBRARIES)				## Sometimes it may be found without specifing DARNKNET_PATH
	message("DARKNET FOUND")
	message("DARKNET_INCLUDE_DIR: ${DARKNET_INCLUDE_DIR}")
	message("DARKNET_LIBRARIES: ${DARKNET_LIBRARIES}")
	SET(DARKNET_ACTIVE ON)
	ADD_DEFINITIONS(-DDARKNET_ACTIVE)
else()
	#message("DARKNET_INCLUDE_DIR: ${DARKNET_INCLUDE_DIR}")
	#message("DARKNET_LIBRARIES: ${DARKNET_LIBRARIES}")
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
if(HAS_OPENCV_DNN)
	SET(ENABLE_DNN_CAFFE ON)
  add_definitions(-DENABLE_DNN_CAFFE)
endif(HAS_OPENCV_DNN)

 add_subdirectory(libs)
 add_subdirectory(DeepLearningSuiteLib)
 add_subdirectory(DatasetEvaluationApp)
 add_subdirectory(SampleGenerationApp)
 add_subdirectory(Tools)


 file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ClassMappingHierarchy.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
 file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ClassMappingHierarchy.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Tools)

 IF (BUILD_TEST)
	add_subdirectory(test)
ENDIF()
