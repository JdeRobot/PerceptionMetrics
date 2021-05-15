OPTION(ENABLE_ICE "Enable ICE compatibility for streaming" ON)

IF (ENABLE_ICE)

	SET(CMAKE_MODULE_PATH
		${CMAKE_MODULE_PATH}
		"${CMAKE_CURRENT_LIST_DIR}/CMake"
	)

	SET(slice_path /usr/share/slice)

	#ICE installs c++11 libraries in this directory and we don't indicate this, cmake uses c++98 files
	SET(CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu/c++11)
	FIND_PACKAGE(ZeroCIceUtil)
	FIND_PACKAGE(ZeroCIceStorm)
	FIND_PACKAGE(ZeroCIce)
		IF( ZeroCIceStorm_LIBRARY )
			IF( ZeroCIceUtil_LIBRARY )
				MESSAGE("-- Found Ice")
				SET(ZeroCIce_FOUND TRUE)
				include_directories(${ZeroCIceUtil_INCLUDE_DIR})
				link_directories(${ZeroCIceUtil_LIBRARY})
				include_directories(${ZeroCIceCore_INCLUDE_DIR})
				link_directories(${ZeroCIceCore_LIBRARY})
				include_directories(${ZeroCIceStorm_INCLUDE_DIR})
				link_directories(${ZeroCIceStorm_LIBRARY})
				include_directories(${ZeroCIce_INCLUDE_DIR})
				link_directories(${ZeroCIce_LIBRARIES})

				list(APPEND DEPS libzeroc-ice3.6 zeroc-ice-utils libzeroc-icestorm3.6)
				list(APPEND DEPS_DEV zeroc-ice-slice libzeroc-ice-dev)
				set(Ice_LIBRARIES ${ZeroCIce_LIBRARIES})

			ENDIF( ZeroCIceUtil_LIBRARY )
		ENDIF( ZeroCIceStorm_LIBRARY )

	IF((NOT  ZeroCIceStorm_LIBRARY ) OR (NOT  ZeroCIceUtil_LIBRARY))
		SET(ZeroCIce_FOUND FALSE)
		MESSAGE(WARNING "*** ICE LIBRARIES NOT FOUND. ICE SUPPORT WILL BE DISABLED")
	ENDIF()
	
ELSE()
	SET(ZeroCIce_FOUND FALSE)
	MESSAGE("Ice is Disabled by default, Set ENABLE_Ice=ON to Enable")
ENDIF()
