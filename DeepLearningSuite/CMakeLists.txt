 cmake_minimum_required(VERSION 2.8)

project(samplerGenerator C CXX)


#check compiller
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	message (version: ${CMAKE_CXX_COMPILER_VERSION})
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
	MESSAGE("SETTING BUILD TYPE TO RELEASE")
	SET(CMAKE_BUILD_TYPE RELEASE)
ENDIF()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -fPIC")


#automated opencv
include(FindPkgConfig)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(JdeRobot REQUIRED)

 include(Deps/glog/CMakeLists.txt)
 include(Deps/ice/CMakeLists.txt)
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

IF (NOT DARKNET_PATH)
	MESSAGE("Please to use darknet please define DARKNET_PATH to the root installation")
	MESSAGE(WARNING "DARKNET DISABLE")
ENDIF()

FIND_PATH( DARKNET_INCLUDE_DIR NAMES DarknetAPI/DarknetAPI.h  PATHS ${DARKNET_PATH}/include )
SET(DARKNET_NAMES darknetAPILib darknetExamplesLib darknetLib)
#SET(DARKNET_NAMES darknetLib)

 FOREACH(N ${DARKNET_NAMES})
	FIND_LIBRARY( DARKNET_LIBRARIES_${N} NAMES ${N} PATHS ${DARKNET_PATH}/lib )
	SET(DARKNET_LIBRARIES ${DARKNET_LIBRARIES} ${DARKNET_LIBRARIES_${N}})

ENDFOREACH()

#UNSET(DARKNET_INCLUDE_DIR CACHE)
#UNSET(DARKNET_LIBRARIES CACHE)


if (DARKNET_INCLUDE_DIR AND DARKNET_LIBRARIES)
	message("DARKNET_INCLUDE_DIR: ${DARKNET_INCLUDE_DIR}")
	message("DARKNET_LIBRARIES: ${DARKNET_LIBRARIES}")
	SET(DARKNET_ACTIVE ON)
	ADD_DEFINITIONS(-DDARKNET_ACTIVE)
else()
	message("DARKNET_INCLUDE_DIR: ${DARKNET_INCLUDE_DIR}")
	message("DARKNET_LIBRARIES: ${DARKNET_LIBRARIES}")
	MESSAGE(WARNING "DARKNET DISABLE")
endif()

 set(DeepLearningSuite_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/DeepLearningSuiteLib)

 add_subdirectory(DeepLearningSuiteLib)
 add_subdirectory(DatasetEvaluationApp)
 add_subdirectory(SampleGenerationApp)
 add_subdirectory(Tools)




 IF (BUILD_TEST)
	add_subdirectory(test)
ENDIF()
