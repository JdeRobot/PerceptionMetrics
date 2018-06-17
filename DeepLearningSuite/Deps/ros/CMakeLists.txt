OPTION(ENABLE_ROS "Enable ROS compatibility modules" ON)

if (ENABLE_ROS)
	find_package(roscpp QUIET)

	if(roscpp_FOUND)

	    MESSAGE("*** ROS LIBRARIES FOUND : ${roscpp_INCLUDE_DIRS}")

	    SET (CMAKE_PREFIX_PATH /opt/ros/kinetic)

	    set (CATKIN_BUILD_BINARY_PACKAGE 1) #doesn't add ROS environment files to package

	    find_package(catkin REQUIRED COMPONENTS
	      roscpp
	      std_msgs
	      cv_bridge
	      image_transport
	    )

		list(APPEND DEPS ros-kinetic-roscpp ros-kinetic-std-msgs ros-kinetic-cv-bridge ros-kinetic-image-transport ros-kinetic-roscpp-core )


	#	list(APPEND DEPS_DEV )
	else()
		MESSAGE(WARNING "*** ROS LIBRARIES NOT FOUND. ROS SUPPORT WILL BE DISABLED")
	endif()
else()
	SET(roscpp_FOUND FALSE)
	MESSAGE("ROS is Disabled by default, Set ENABLE_ROS=ON to enable")
endif()
