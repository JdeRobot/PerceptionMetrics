OPTION(ENABLE_ROS "Enable ROS compatibility modules" ON)

SET (CMAKE_PREFIX_PATH "/opt/ros/kinetic;/opt/ros/lunar;/opt/ros/jade")

if (ENABLE_ROS)
	find_package(roscpp QUIET)

	if(roscpp_FOUND)

	    MESSAGE("*** ROS LIBRARIES FOUND : ${roscpp_INCLUDE_DIRS}")

	    set (CATKIN_BUILD_BINARY_PACKAGE 1) #doesn't add ROS environment files to package

	    find_package(catkin REQUIRED COMPONENTS
	      roscpp
	      std_msgs
	      cv_bridge
	      image_transport
	    )

	    SET(ros_INCLUDE_DIRS ${roscpp_INCLUDE_DIRS} ${std_msgs_INCLUDE_DIRS} ${cv_bridge_INCLUDE_DIRS} ${image_transport_INCLUDE_DIRS})
	    SET(ros_LIBRARIES ${roscpp_LIBRARIES} ${std_msgs_LIBRARIES} ${cv_bridge_LIBRARIES} ${image_transport_LIBRARIES})

		list(APPEND DEPS ros-kinetic-roscpp ros-kinetic-std-msgs ros-kinetic-cv-bridge ros-kinetic-image-transport ros-kinetic-roscpp-core )


	#	list(APPEND DEPS_DEV )
	else()
		MESSAGE(WARNING "*** ROS LIBRARIES NOT FOUND. ROS SUPPORT WILL BE DISABLED")
	endif()
else()
	SET(roscpp_FOUND FALSE)
	MESSAGE("ROS is Disabled by default, Set ENABLE_ROS=ON to enable")
endif()
