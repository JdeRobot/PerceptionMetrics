OPTION(ENABLE_QT "Enable Qt support for GUI" ON)

IF(ENABLE_QT)

	FIND_PACKAGE(Qt5Core)                   # Just to print error if Qt isn't found
  FIND_PACKAGE(Qt5Widgets QUIET)
  FIND_PACKAGE(Qt5Gui QUIET)
	FIND_PACKAGE(Qt5Svg QUIET)
	FIND_PACKAGE(Qt5OpenGL QUIET)

	IF (Qt5Widgets_FOUND AND Qt5Core_FOUND AND Qt5Gui_FOUND AND Qt5Svg_FOUND AND Qt5OpenGL_FOUND)

  	 SET(QT_INCLUDE_DIRS
  		 ${Qt5Widgets_INCLUDE_DIRS}
  		 ${Qt5Core_INCLUDE_DIRS}
  		 ${Qt5Gui_INCLUDE_DIRS}
  		 ${Qt5Svg_INCLUDE_DIRS}
  		 ${Qt5OpenGL_INCLUDE_DIRS}
  	 )

  	 SET(QT_LIBRARIES
  		 ${Qt5Widgets_LIBRARIES}
  		 ${Qt5Core_LIBRARIES}
  		 ${Qt5Gui_LIBRARIES}
  		 ${Qt5Svg_LIBRARIES}
  		 ${Qt5OpenGL_LIBRARIES}
  	)

    SET(QT_FOUND TRUE)
  ELSE()

		MESSAGE("QT Libraries can't be found, Disabling GUI support. DatasetEvaluationApp will not be build.")
    SET(QT_FOUND FALSE)

	ENDIF()

ENDIF(ENABLE_QT)
